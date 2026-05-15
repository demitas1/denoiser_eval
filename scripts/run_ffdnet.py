"""
KAIR版 FFDNet でデノイズを実行するスクリプト。
sigma（ノイズレベル）を指定して推論する。複数値を指定するとすべて出力。

使い方:
  # 全 sigma でスイープ（デフォルト: 5 15 25 50 75）
  python scripts/run_ffdnet.py --input test_inputs/ --output results/FFDNet

  # sigma を指定
  python scripts/run_ffdnet.py --input test_inputs/ --output results/FFDNet --sigma 15

  # 複数 sigma を指定
  python scripts/run_ffdnet.py --input test_inputs/ --output results/FFDNet --sigma 5 15 25
"""

import argparse
import os
import sys
import glob
import time

import numpy as np
import torch
from PIL import Image

KAIR_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)
from models.network_ffdnet import FFDNet as net

DEFAULT_SIGMAS = [5, 10, 15, 20, 25, 50]


def load_model(model_path, device):
    model = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model.to(device)


def denoise_image(model, img_path, sigma_val, device):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    sigma = torch.full((1, 1, 1, 1), sigma_val / 255.0, dtype=torch.float32).to(device)
    with torch.no_grad():
        y = model(x, sigma)
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
    parser.add_argument('--output', default='results/FFDNet', help='Output directory')
    parser.add_argument('--model', default='models/KAIR/model_zoo/ffdnet_gray.pth',
                        help='Path to model weights')
    parser.add_argument('--sigma', type=int, nargs='+', default=DEFAULT_SIGMAS,
                        help='Noise level(s) in 0-255 scale (default: 5 15 25 50 75)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    root = os.path.join(os.path.dirname(__file__), '..')
    model_path = os.path.join(root, args.model) if not os.path.isabs(args.model) else args.model
    output_dir = os.path.join(root, args.output) if not os.path.isabs(args.output) else args.output

    print(f'Loading model: {model_path}')
    model = load_model(model_path, device)
    print(f'Model loaded. sigma sweep: {args.sigma}')

    os.makedirs(output_dir, exist_ok=True)
    input_files = collect_inputs(args.input if os.path.isabs(args.input)
                                 else os.path.join(root, args.input))

    if not input_files:
        print('No input images found.')
        return

    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        for sigma_val in args.sigma:
            t0 = time.time()
            out = denoise_image(model, img_path, sigma_val, device)
            elapsed = time.time() - t0
            out_path = os.path.join(output_dir, f'{basename}_ffdnet_s{sigma_val:02d}.png')
            Image.fromarray(out).save(out_path)
            print(f'  sigma={sigma_val:2d}  {os.path.basename(img_path)} -> {os.path.basename(out_path)}  ({elapsed:.2f}s)')

    print(f'\nDone. {len(input_files) * len(args.sigma)} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
