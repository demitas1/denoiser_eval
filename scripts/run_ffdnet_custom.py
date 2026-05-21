"""
学習途中の FFDNet チェックポイントを使ってデノイズ推論を実行するスクリプト。

フルチェックポイント（iter_XXXXXX.pth）と
state_dict のみ（best.pth）の両方に対応。

使い方:
  # best.pth（state_dict のみ）で sigma=25 デノイズ
  python scripts/run_ffdnet_custom.py \
      --checkpoint results/train_ffdnet_gray/best.pth \
      --input test_inputs/ --sigma 25

  # iter チェックポイントで複数 sigma を適用
  python scripts/run_ffdnet_custom.py \
      --checkpoint results/train_ffdnet_gray/iter_010000.pth \
      --input test_inputs/ --sigma 10 25

  # 出力ディレクトリを指定
  python scripts/run_ffdnet_custom.py \
      --checkpoint results/train_ffdnet_gray/best.pth \
      --input test_inputs/ --output results/ffdnet_custom/ --sigma 10

  # CPU 推論
  python scripts/run_ffdnet_custom.py \
      --checkpoint results/train_ffdnet_gray/best.pth \
      --input test_inputs/ --cpu --sigma 25
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_ffdnet import FFDNet


def load_checkpoint(path, device):
    """フルチェックポイント（'state_dict' キー）と state_dict-only の両方を受け付ける。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        step = ckpt.get('step', None)
        best_psnr = ckpt.get('best_psnr', None)
        label = f'step={step}' if step is not None else 'step=unknown'
        if best_psnr is not None:
            label += f', best_psnr={best_psnr:.2f}'
        print(f'  Full checkpoint: {label}')
        return state_dict, step
    else:
        print('  State-dict-only checkpoint')
        return ckpt, None


def build_model(state_dict, device):
    model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def denoise_image(model, img_path, sigma_val, device):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    sigma = torch.full((1, 1, 1, 1), sigma_val / 255.0, dtype=torch.float32).to(device)
    try:
        with torch.no_grad():
            y = model(x, sigma)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} sigma={sigma_val} skipped.')
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
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth checkpoint (full or state_dict-only)')
    parser.add_argument('--input', required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', default='results/ffdnet_custom',
                        help='Output directory (default: results/ffdnet_custom)')
    parser.add_argument('--sigma', type=int, nargs='+', default=[10, 25],
                        help='Noise level(s) in 0-255 scale (default: 10 25)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f'Error: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Device:     {device}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Sigma:      {args.sigma}')
    print(f'Output:     {output_dir}')

    state_dict, step = load_checkpoint(ckpt_path, device)
    step_tag = f'step{step:06d}' if step is not None else 'custom'

    model = build_model(state_dict, device)
    print(f'Model built: FFDNet gray  (tag: {step_tag})')

    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    input_files = collect_inputs(input_path)
    if not input_files:
        print('No input images found.')
        return

    total = 0
    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        for sigma_val in args.sigma:
            t0 = time.time()
            out = denoise_image(model, img_path, sigma_val, device)
            elapsed = time.time() - t0
            if out is None:
                continue
            out_path = os.path.join(output_dir, f'{basename}_ffdnet_{step_tag}_s{sigma_val:02d}.png')
            Image.fromarray(out).save(out_path)
            print(f'  sigma={sigma_val:2d}  {os.path.basename(img_path)} -> {os.path.basename(out_path)}  ({elapsed:.2f}s)')
            total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
