"""
SwinIR real-world SR x4 推論スクリプト。
sys.path 追加でインストール不要。

使い方:
  # SwinIR-M（デフォルト）— 元サイズに LANCZOS ダウンスケールして保存
  python scripts/run_swinir.py --input test_inputs/ --output results/SwinIR

  # SwinIR-L（大規模モデル、VRAM 多く必要）
  python scripts/run_swinir.py --input test_inputs/ --model SwinIR-L

  # x4 アップスケール版も保存
  python scripts/run_swinir.py --input test_inputs/ --save_upscaled

  # ダウンスケールなし（アップスケール版のみ）
  python scripts/run_swinir.py --input test_inputs/ --downscale none --save_upscaled

  # VRAM が厳しい場合はタイルサイズを小さく（デフォルト 512）
  python scripts/run_swinir.py --input test_inputs/ --tile 256

重みは model_zoo に存在しない場合、GitHub releases から自動ダウンロードする。
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import requests
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_swinir import SwinIR

WINDOW_SIZE = 8
SCALE = 4
WEIGHT_BASE_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'

MODEL_CONFIGS = {
    'SwinIR-M': dict(
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        resi_connection='1conv',
        weight='003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
    ),
    'SwinIR-L': dict(
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        resi_connection='3conv',
        weight='003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
    ),
}

RESAMPLE_FILTERS = {
    'lanczos':  Image.LANCZOS,
    'bicubic':  Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'nearest':  Image.NEAREST,
}


def download_weight(url, dest_path):
    print(f'Downloading {os.path.basename(dest_path)} ...')
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f'\r  {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB', end='', flush=True)
    print()


def ensure_weight(model_name, model_zoo):
    weight_name = MODEL_CONFIGS[model_name]['weight']
    weight_path = os.path.join(model_zoo, weight_name)
    if not os.path.isfile(weight_path):
        os.makedirs(model_zoo, exist_ok=True)
        url = WEIGHT_BASE_URL + weight_name
        download_weight(url, weight_path)
    return weight_path


def load_model(model_name, weight_path, device):
    cfg = MODEL_CONFIGS[model_name]
    model = SwinIR(
        upscale=SCALE,
        in_chans=3,
        img_size=64,
        window_size=WINDOW_SIZE,
        img_range=1.0,
        depths=cfg['depths'],
        embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection=cfg['resi_connection'],
    )
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if 'params_ema' in ckpt:
            state_dict = ckpt['params_ema']
        elif 'params' in ckpt:
            state_dict = ckpt['params']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def pad_reflect(x):
    """window_size の倍数になるようリフレクションパディング。(padded, h_old, w_old) を返す。"""
    _, _, h_old, w_old = x.size()
    h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
    w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old
    x = torch.cat([x, torch.flip(x, [2])], dim=2)[:, :, :h_old + h_pad, :]
    x = torch.cat([x, torch.flip(x, [3])], dim=3)[:, :, :, :w_old + w_pad]
    return x, h_old, w_old


def infer_single(model, x, device):
    """ウィンドウパディング → 推論 → クロップ。入力と同じ空間サイズ × SCALE を返す。"""
    x_padded, h_old, w_old = pad_reflect(x)
    with torch.no_grad():
        out = model(x_padded.to(device)).cpu()
    return out[:, :, :h_old * SCALE, :w_old * SCALE]


def tile_inference_sr(model, x, tile_size, tile_overlap, device):
    """タイル分割推論。tile_size は WINDOW_SIZE の倍数であること。"""
    b, c, h, w = x.shape
    stride = tile_size - tile_overlap
    h_idx = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx = list(range(0, w - tile_size, stride)) + [w - tile_size]

    out_tile = tile_size * SCALE
    E = torch.zeros(b, c, h * SCALE, w * SCALE)
    W = torch.zeros(b, c, h * SCALE, w * SCALE)

    for hi in h_idx:
        for wi in w_idx:
            patch = x[:, :, hi:hi + tile_size, wi:wi + tile_size]
            out = infer_single(model, patch, device)
            ohi, owi = hi * SCALE, wi * SCALE
            E[:, :, ohi:ohi + out_tile, owi:owi + out_tile] += out
            W[:, :, ohi:ohi + out_tile, owi:owi + out_tile] += 1

    return E / W


def upscale_image(model, img_path, tile_size, device):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    _, _, h, w = x.shape
    try:
        if tile_size and (h > tile_size or w > tile_size):
            out_t = tile_inference_sr(model, x, tile_size, tile_overlap=32, device=device)
        else:
            out_t = infer_single(model, x, device)
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


def main():
    parser = argparse.ArgumentParser(description='SwinIR real-world SR x4 推論')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/SwinIR', help='Output directory')
    parser.add_argument('--model', default='SwinIR-M', choices=list(MODEL_CONFIGS.keys()),
                        help='Model variant (default: SwinIR-M)')
    parser.add_argument('--model_zoo', default='models/KAIR/model_zoo',
                        help='Path to model_zoo directory')
    parser.add_argument('--tile', type=int, default=512,
                        help='Tile size for large images (0 to disable). '
                             'Auto-rounded down to a multiple of 8.')
    parser.add_argument('--downscale', default='lanczos',
                        choices=['none'] + list(RESAMPLE_FILTERS.keys()),
                        help='Downscale algorithm to restore original size after SR '
                             '(default: lanczos). "none" skips downscaling.')
    parser.add_argument('--save_upscaled', action='store_true',
                        help='Also save the x4 upscaled image in addition to downscaled.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    # タイルサイズを WINDOW_SIZE の倍数に丸める
    tile_size = None
    if args.tile > 0:
        tile_size = (args.tile // WINDOW_SIZE) * WINDOW_SIZE
        if tile_size != args.tile:
            print(f'Tile size rounded to multiple of {WINDOW_SIZE}: {args.tile} -> {tile_size}')

    print(f'Device: {device}  Model: {args.model}  Tile: {tile_size}')

    model_zoo = args.model_zoo if os.path.isabs(args.model_zoo) else os.path.join(ROOT, args.model_zoo)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    input_files = collect_inputs(input_path)
    if not input_files:
        print('No input images found.')
        return

    weight_path = ensure_weight(args.model, model_zoo)
    print(f'Loading {args.model} from {os.path.basename(weight_path)} ...')
    model = load_model(args.model, weight_path, device)

    total = 0
    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        t0 = time.time()
        out = upscale_image(model, img_path, tile_size, device)
        elapsed = time.time() - t0
        if out is None:
            continue

        sr_img = Image.fromarray(out)
        sh, sw = out.shape[:2]

        if args.save_upscaled or args.downscale == 'none':
            sr_path = os.path.join(output_dir, f'{basename}_{args.model}_x{SCALE}.png')
            sr_img.save(sr_path)
            print(f'  {os.path.basename(img_path)} -> {os.path.basename(sr_path)}  {sw}x{sh}  ({elapsed:.2f}s)')
            total += 1

        if args.downscale != 'none':
            orig_w, orig_h = Image.open(img_path).size
            ds_img = sr_img.resize((orig_w, orig_h), resample=RESAMPLE_FILTERS[args.downscale])
            ds_path = os.path.join(output_dir, f'{basename}_{args.model}_{args.downscale}.png')
            ds_img.save(ds_path)
            print(f'  {os.path.basename(img_path)} -> {os.path.basename(ds_path)}  {orig_w}x{orig_h}  ({args.downscale})')
            total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
