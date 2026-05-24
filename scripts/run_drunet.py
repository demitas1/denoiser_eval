"""
DRUNet (DPIR) グレースケールデノイズスクリプト。

sigma（ノイズレベル）を明示的に指定することで補正強度を制御できる。
sigma が大きいほど強くデノイズされる（強度のつまみとして使う）。

使い方:
  # デフォルト sigma スイープ (5 10 15 25 50)
  python scripts/run_drunet.py --input test_inputs/ --output results/DRUNet

  # sigma を絞って実行
  python scripts/run_drunet.py --input test_inputs/ --sigma 10 15 25

  # カスタム重みを使う場合
  python scripts/run_drunet.py --input test_inputs/ --model results/trained_models/drunet_gray_scratch_unsplash_lite.pth

参考:
  公式重み: models/KAIR/model_zoo/drunet_gray.pth
  元論文: "Plug-and-Play Image Restoration with Deep Denoiser Prior" (Zhang et al., 2021)
  アーキテクチャ: UNetRes(in_nc=2, out_nc=1, nc=[64,128,256,512], nb=4)
                 入力 = [ノイズ画像(1ch) + σマップ(1ch)] を channel concat
"""

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)
from models.network_unet import UNetRes

DEFAULT_SIGMAS     = [5, 10, 15, 25, 50]
DEFAULT_MODEL_PATH = 'models/KAIR/model_zoo/drunet_gray.pth'

# UNetRes は 2^4 = 16 の倍数サイズを要求する
PAD_MULTIPLE = 16


def load_model(model_path, device):
    model = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
                    act_mode='R', downsample_mode='strideconv',
                    upsample_mode='convtranspose', bias=False)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model.to(device)


def pad_to_multiple(x, multiple):
    """右・下にゼロパディングして H,W を multiple の倍数にする。"""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w


def denoise_image(model, img_path, sigma_val, device):
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    x_pad, orig_h, orig_w = pad_to_multiple(x, PAD_MULTIPLE)
    sigma_map = torch.full_like(x_pad, sigma_val / 255.0)
    inp = torch.cat([x_pad, sigma_map], dim=1)  # (1,2,H',W')

    try:
        with torch.no_grad():
            y = model(inp)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} sigma={sigma_val} skipped.')
        return None

    out = y[:, :, :orig_h, :orig_w].squeeze().cpu().numpy().clip(0, 1) * 255
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
    parser = argparse.ArgumentParser(
        description='DRUNet grayscale denoising with explicit sigma control')
    parser.add_argument('--input',  required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/DRUNet', help='Output directory')
    parser.add_argument('--model',  default=DEFAULT_MODEL_PATH,
                        help='Path to model weights (default: drunet_gray.pth)')
    parser.add_argument('--sigma', type=int, nargs='+', default=DEFAULT_SIGMAS,
                        help='Noise level(s) 0-255 (default: 5 10 15 25 50); '
                             'larger = stronger denoising')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    input_path = args.input  if os.path.isabs(args.input)  else os.path.join(ROOT, args.input)

    print(f'Loading model: {model_path}')
    model = load_model(model_path, device)
    print(f'Model loaded. sigma sweep: {args.sigma}')

    os.makedirs(output_dir, exist_ok=True)
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
            out_path = os.path.join(output_dir, f'{basename}_drunet_s{sigma_val:02d}.png')
            Image.fromarray(out).save(out_path)
            print(f'  sigma={sigma_val:3d}  {os.path.basename(img_path)}'
                  f' -> {os.path.basename(out_path)}  ({elapsed:.2f}s)')
            total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
