"""
SCUNet でデノイズを実行するスクリプト。
sys.path 追加でインストール不要。

使い方:
  # 実世界ノイズ（PSNR版、デフォルト）
  python scripts/run_scunet.py --input test_inputs/ --output results/SCUNet

  # 実世界ノイズ（GAN版、シャープだが捏造リスクあり）
  python scripts/run_scunet.py --input test_inputs/ --model scunet_color_real_gan

  # グレースケール3強度を一括出力（デフォルト: gray_15 gray_25 gray_50）
  python scripts/run_scunet.py --input test_inputs/ --model scunet_gray_15 scunet_gray_25 scunet_gray_50

  # 複数モデルを指定
  python scripts/run_scunet.py --input test_inputs/ --model scunet_color_real_psnr scunet_gray_25
"""

import argparse
import os
import sys
import glob
import time

import numpy as np
import torch
from PIL import Image

SCUNET_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'SCUNet')
sys.path.insert(0, SCUNET_DIR)
from models.network_scunet import SCUNet as net

# モデル名 -> (in_nc, グレースケール出力フラグ)
MODEL_CONFIGS = {
    'scunet_color_real_psnr': (3, False),
    'scunet_color_real_gan':  (3, False),
    'scunet_color_15':        (3, False),
    'scunet_color_25':        (3, False),
    'scunet_color_50':        (3, False),
    'scunet_gray_15':         (1, True),
    'scunet_gray_25':         (1, True),
    'scunet_gray_50':         (1, True),
}


def load_model(model_path, in_nc, device):
    model = net(in_nc=in_nc, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def denoise_image(model, img_path, in_nc, device):
    img = Image.open(img_path)

    if in_nc == 1:
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    else:
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        y = model(x)

    out = y.squeeze().cpu().numpy().clip(0, 1) * 255
    if in_nc == 1:
        return out.astype(np.uint8)
    else:
        # カラー出力をグレースケールに変換して返す
        rgb = out.transpose(1, 2, 0).astype(np.uint8)
        return np.array(Image.fromarray(rgb).convert('L'))


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


DEFAULT_MODELS = ['scunet_color_real_psnr']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/SCUNet', help='Output directory')
    parser.add_argument('--model', nargs='+', default=DEFAULT_MODELS,
                        choices=list(MODEL_CONFIGS.keys()), help='Model name(s)')
    parser.add_argument('--model_zoo', default='models/SCUNet/model_zoo',
                        help='Path to model_zoo directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}  Models: {args.model}')

    root = os.path.join(os.path.dirname(__file__), '..')
    model_zoo = os.path.join(root, args.model_zoo) if not os.path.isabs(args.model_zoo) else args.model_zoo
    output_dir = os.path.join(root, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    input_files = collect_inputs(args.input if os.path.isabs(args.input) else os.path.join(root, args.input))
    if not input_files:
        print('No input images found.')
        return

    total = 0
    for model_name in args.model:
        in_nc, _ = MODEL_CONFIGS[model_name]
        model_path = os.path.join(model_zoo, f'{model_name}.pth')
        print(f'Loading {model_name} (in_nc={in_nc})...')
        model = load_model(model_path, in_nc, device)

        for img_path in input_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f'{basename}_scunet_{model_name}.png')
            t0 = time.time()
            out = denoise_image(model, img_path, in_nc, device)
            elapsed = time.time() - t0
            Image.fromarray(out).save(out_path)
            print(f'  {model_name}  {os.path.basename(img_path)} -> {os.path.basename(out_path)}  ({elapsed:.2f}s)')
            total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
