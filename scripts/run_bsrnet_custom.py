"""
学習途中の BSRNet チェックポイントを使って SR 推論を実行するスクリプト。

フルチェックポイント（iter_XXXXXX.pth, 256MB）と
state_dict のみ（best.pth / last_E.pth, 64MB）の両方に対応。

デフォルトではダウンスケール版（元解像度）のみ保存。
--save_upscaled を指定すると x4 アップスケール版も追加保存。

使い方:
  # best.pth（state_dict のみ）を使って SR（ダウンスケール版のみ保存）
  python scripts/run_bsrnet_custom.py \
      --checkpoint results/train_bsrgan_psnr/best.pth \
      --input test_inputs/ --output results/bsrnet_custom/

  # iter_012000.pth（フルチェックポイント）を使って SR
  python scripts/run_bsrnet_custom.py \
      --checkpoint results/train_bsrgan_psnr/iter_012000.pth \
      --input test_inputs/ --output results/bsrnet_custom/

  # アップスケール版も保存
  python scripts/run_bsrnet_custom.py \
      --checkpoint results/train_bsrgan_psnr/best.pth \
      --input test_inputs/ --save_upscaled

  # CPU 推論
  python scripts/run_bsrnet_custom.py \
      --checkpoint results/train_bsrgan_psnr/best.pth \
      --input test_inputs/ --cpu
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

from models.network_rrdbnet import RRDBNet

SCALE = 4

RESAMPLE_FILTERS = {
    'lanczos':  Image.LANCZOS,
    'bicubic':  Image.BICUBIC,
    'bilinear': Image.BILINEAR,
    'nearest':  Image.NEAREST,
}


def load_checkpoint(path, device):
    """フルチェックポイントと state_dict-only の両方を受け付ける。EMA 重み（netE）を優先。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'netE' in ckpt:
        state_dict = ckpt['netE']
        step = ckpt.get('step', None)
        best_psnr = ckpt.get('best_psnr', None)
        label = f'step={step}' if step is not None else 'step=unknown'
        if best_psnr is not None:
            label += f', best_psnr={best_psnr:.2f}'
        print(f'  Full checkpoint: {label}')
        return state_dict, step
    elif isinstance(ckpt, dict) and 'netG' in ckpt:
        state_dict = ckpt['netG']
        step = ckpt.get('step', None)
        print(f'  Full checkpoint (netG): step={step}')
        return state_dict, step
    else:
        print('  State-dict-only checkpoint')
        return ckpt, None


def build_model(state_dict, device):
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=SCALE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def tile_inference_sr(model, x, tile_size, tile_overlap, device):
    b, c, h, w = x.shape
    stride = tile_size - tile_overlap
    h_idx = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx = list(range(0, w - tile_size, stride)) + [w - tile_size]

    out_tile = tile_size * SCALE
    E = torch.zeros(b, c, h * SCALE, w * SCALE)
    W = torch.zeros(b, c, h * SCALE, w * SCALE)

    for hi in h_idx:
        for wi in w_idx:
            patch = x[:, :, hi:hi + tile_size, wi:wi + tile_size].to(device)
            with torch.no_grad():
                out = model(patch).cpu()
            ohi, owi = hi * SCALE, wi * SCALE
            E[:, :, ohi:ohi + out_tile, owi:owi + out_tile] += out
            W[:, :, ohi:ohi + out_tile, owi:owi + out_tile] += 1

    return E / W


def upscale_image(model, img_path, tile_size, device):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    try:
        if tile_size and (x.shape[2] > tile_size or x.shape[3] > tile_size):
            out_t = tile_inference_sr(model, x, tile_size, tile_overlap=32, device=device)
        else:
            with torch.no_grad():
                out_t = model(x.to(device)).cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} skipped. Try --tile with a smaller value.')
        return None

    return (out_t.squeeze().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


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
    parser.add_argument('--output', default='results/bsrnet_custom',
                        help='Output directory (default: results/bsrnet_custom)')
    parser.add_argument('--tile', type=int, default=512,
                        help='Tile size for large images (0 to disable, default: 512)')
    parser.add_argument('--downscale', default='lanczos',
                        choices=list(RESAMPLE_FILTERS.keys()),
                        help='Algorithm to restore original size after SR (default: lanczos)')
    parser.add_argument('--save_upscaled', action='store_true',
                        help='Also save the x4 upscaled image in addition to the downscaled one')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    tile_size = args.tile if args.tile > 0 else None

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(ROOT, args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f'Error: checkpoint not found: {ckpt_path}')
        sys.exit(1)

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Device:        {device}')
    print(f'Checkpoint:    {ckpt_path}')
    print(f'Tile:          {tile_size}')
    print(f'Downscale:     {args.downscale}')
    print(f'Save upscaled: {args.save_upscaled}')
    print(f'Output:        {output_dir}')

    state_dict, step = load_checkpoint(ckpt_path, device)
    step_tag = f'step{step:06d}' if step is not None else 'custom'

    model = build_model(state_dict, device)
    print(f'Model built: RRDBNet x{SCALE}  (tag: {step_tag})')

    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    input_files = collect_inputs(input_path)
    if not input_files:
        print('No input images found.')
        return

    total = 0
    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        t0 = time.time()
        out = upscale_image(model, img_path, tile_size, device)
        elapsed = time.time() - t0
        if out is None:
            continue

        sr_img = Image.fromarray(out)
        orig_img = Image.open(img_path)
        orig_w, orig_h = orig_img.size

        ds_img = sr_img.resize((orig_w, orig_h), resample=RESAMPLE_FILTERS[args.downscale])
        ds_path = os.path.join(output_dir, f'{basename}_bsrnet_{step_tag}_{args.downscale}.png')
        ds_img.save(ds_path)
        print(f'  {os.path.basename(img_path)} -> {os.path.basename(ds_path)}  {orig_w}x{orig_h}  ({args.downscale}, {elapsed:.2f}s)')
        total += 1

        if args.save_upscaled:
            sr_path = os.path.join(output_dir, f'{basename}_bsrnet_{step_tag}_x{SCALE}.png')
            sr_img.save(sr_path)
            sh, sw = out.shape[:2]
            print(f'  {os.path.basename(img_path)} -> {os.path.basename(sr_path)}  {sw}x{sh}')
            total += 1

    print(f'\nDone. {total} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
