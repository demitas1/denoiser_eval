"""
KAIR版 DnCNN でデノイズを実行するスクリプト。
グレースケール画像を入力として受け取り、デノイズ済み画像を保存する。

使い方:
  python scripts/run_dncnn.py --input <画像パスまたはディレクトリ> --output results/DnCNN
  python scripts/run_dncnn.py --input test_inputs/ --output results/DnCNN
  python scripts/run_dncnn.py --input models/KAIR/testsets/set5/ --output results/DnCNN
"""

import argparse
import os
import sys
import glob
import time

import numpy as np
import torch
from PIL import Image

# KAIR の models/ を import パスに追加
KAIR_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)
from models.network_dncnn import DnCNN as net


def load_model(model_path, device):
    model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model.to(device)


def denoise_image(model, img_path, device):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    try:
        with torch.no_grad():
            y = model(x)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} skipped.')
        return None
    out = y.squeeze().cpu().numpy().clip(0, 1) * 255
    return out.astype(np.uint8)


def collect_inputs(input_path):
    if os.path.isdir(input_path):
        paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            paths += glob.glob(os.path.join(input_path, ext))
        return sorted(paths)
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise FileNotFoundError(f'Input not found: {input_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/DnCNN', help='Output directory')
    parser.add_argument('--model', default='models/KAIR/model_zoo/dncnn_gray_blind.pth',
                        help='Path to model weights')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    # スクリプトの場所を基準にパスを解決
    root = os.path.join(os.path.dirname(__file__), '..')
    model_path = os.path.join(root, args.model) if not os.path.isabs(args.model) else args.model
    output_dir = os.path.join(root, args.output) if not os.path.isabs(args.output) else args.output

    print(f'Loading model: {model_path}')
    model = load_model(model_path, device)
    print('Model loaded.')

    os.makedirs(output_dir, exist_ok=True)
    input_files = collect_inputs(args.input if os.path.isabs(args.input)
                                 else os.path.join(root, args.input))

    if not input_files:
        print('No input images found.')
        return

    for img_path in input_files:
        t0 = time.time()
        out = denoise_image(model, img_path, device)
        elapsed = time.time() - t0
        if out is None:
            continue

        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f'{basename}_dncnn.png')
        Image.fromarray(out).save(out_path)
        print(f'  {os.path.basename(img_path)} -> {out_path}  ({elapsed:.2f}s)')

    print(f'\nDone. {len(input_files)} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
