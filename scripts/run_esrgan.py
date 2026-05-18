"""
ESRGAN / BSRGAN で超解像を実行するスクリプト。
sys.path 追加でインストール不要。

使い方:
  # BSRGAN x4（GAN版、実世界劣化合成、デフォルト）
  # デフォルトで元サイズに LANCZOS ダウンスケールした出力も保存される
  python scripts/run_esrgan.py --input test_inputs/ --output results/ESRGAN

  # 複数モデルを一括実行
  python scripts/run_esrgan.py --input test_inputs/ --model BSRGAN BSRNet

  # ESRGAN（古典的 GAN 版）
  python scripts/run_esrgan.py --input test_inputs/ --model ESRGAN

  # BSRGANx2（×2 アップスケール）
  python scripts/run_esrgan.py --input test_inputs/ --model BSRGANx2

  # アップスケール済み画像のみ保存（ダウンスケールしない）
  python scripts/run_esrgan.py --input test_inputs/ --downscale none

  # VRAM が厳しい場合はタイルサイズを小さく（デフォルト 512）
  python scripts/run_esrgan.py --input test_inputs/ --tile 256
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

from models.network_rrdbnet import RRDBNet
from models.network_rrdb import RRDB

RESAMPLE_FILTERS = {
    'lanczos':  Image.LANCZOS,
    'bicubic':  Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'nearest':  Image.NEAREST,
}

# モデル名 -> (net_type, scale)
#   rrdbnet: RRDBNet (BSRGAN 系)
#   rrdb:    RRDB    (ESRGAN、network_rrdb.py の実装)
MODEL_CONFIGS = {
    'BSRGAN':   ('rrdbnet', 4),
    'BSRNet':   ('rrdbnet', 4),
    'BSRGANx2': ('rrdbnet', 2),
    'ESRGAN':   ('rrdb',    4),
}


def tile_inference_sr(model, x, tile_size, tile_overlap, scale, device):
    """超解像用タイル分割推論。出力テンソルは入力の scale 倍サイズ。"""
    b, c, h, w = x.shape
    stride = tile_size - tile_overlap
    h_idx = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx = list(range(0, w - tile_size, stride)) + [w - tile_size]

    out_tile = tile_size * scale
    E = torch.zeros(b, c, h * scale, w * scale)
    W = torch.zeros(b, c, h * scale, w * scale)

    for hi in h_idx:
        for wi in w_idx:
            patch = x[:, :, hi:hi+tile_size, wi:wi+tile_size].to(device)
            with torch.no_grad():
                out = model(patch).cpu()
            ohi, owi = hi * scale, wi * scale
            E[:, :, ohi:ohi+out_tile, owi:owi+out_tile] += out
            W[:, :, ohi:ohi+out_tile, owi:owi+out_tile] += 1

    return E / W


def load_model(model_name, model_path, device):
    net_type, scale = MODEL_CONFIGS[model_name]
    if net_type == 'rrdbnet':
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)
    else:
        model = RRDB(in_nc=3, out_nc=3, nc=64, nb=23, gc=32, upscale=scale)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def upscale_image(model, img_path, model_name, tile_size, device):
    _, scale = MODEL_CONFIGS[model_name]
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    _, _, h, w = x.shape
    try:
        if tile_size and (h > tile_size or w > tile_size):
            out_t = tile_inference_sr(model, x, tile_size, tile_overlap=32, scale=scale, device=device)
        else:
            with torch.no_grad():
                out_t = model(x.to(device)).cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} skipped. Try --tile with a smaller value.')
        return None

    out = out_t.squeeze().permute(1, 2, 0).numpy().clip(0, 1) * 255
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


DEFAULT_MODELS = ['BSRGAN']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/ESRGAN', help='Output directory')
    parser.add_argument('--model', nargs='+', default=DEFAULT_MODELS,
                        choices=list(MODEL_CONFIGS.keys()), help='Model name(s)')
    parser.add_argument('--model_zoo', default='models/KAIR/model_zoo',
                        help='Path to model_zoo directory')
    parser.add_argument('--tile', type=int, default=512,
                        help='Tile size for large images (0 to disable)')
    parser.add_argument('--downscale', default='lanczos',
                        choices=['none'] + list(RESAMPLE_FILTERS.keys()),
                        help='Downscale algorithm to restore original size after SR '
                             '(default: lanczos). "none" keeps upscaled output only.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    tile_size = args.tile if args.tile > 0 else None
    print(f'Device: {device}  Models: {args.model}  Tile: {tile_size}')

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
        _, scale = MODEL_CONFIGS[model_name]
        model_path = os.path.join(model_zoo, f'{model_name}.pth')
        print(f'Loading {model_name} (x{scale})...')
        model = load_model(model_name, model_path, device)

        for img_path in input_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            t0 = time.time()
            out = upscale_image(model, img_path, model_name, tile_size, device)
            elapsed = time.time() - t0
            if out is None:
                continue

            sr_img = Image.fromarray(out)
            sr_path = os.path.join(output_dir, f'{basename}_{model_name}_x{scale}.png')
            sr_img.save(sr_path)
            sh, sw = out.shape[:2]
            print(f'  {model_name}  {os.path.basename(img_path)} -> {os.path.basename(sr_path)}  {sw}x{sh}  ({elapsed:.2f}s)')
            total += 1

            if args.downscale != 'none':
                orig_w, orig_h = Image.open(img_path).size
                ds_img = sr_img.resize((orig_w, orig_h), resample=RESAMPLE_FILTERS[args.downscale])
                ds_path = os.path.join(output_dir, f'{basename}_{model_name}_{args.downscale}.png')
                ds_img.save(ds_path)
                print(f'  {model_name}  {os.path.basename(img_path)} -> {os.path.basename(ds_path)}  {orig_w}x{orig_h}  ({args.downscale})')
                total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
