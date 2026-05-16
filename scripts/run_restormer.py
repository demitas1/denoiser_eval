"""
Restormer でデノイズ／デブラーを実行するスクリプト。
sys.path 追加でインストール不要。

使い方:
  # Real Denoising（実世界ノイズ、デフォルト）
  python scripts/run_restormer.py --input test_inputs/ --output results/Restormer

  # Gaussian Gray Denoising（グレースケールブラインド）
  python scripts/run_restormer.py --input test_inputs/ --task Gaussian_Gray_Denoising

  # Motion Deblurring（モーションブラー除去）
  python scripts/run_restormer.py --input test_inputs/ --task Motion_Deblurring

  # Defocus Deblurring（ピンボケ除去）
  python scripts/run_restormer.py --input test_inputs/ --task Defocus_Deblurring

  # VRAM が厳しい場合はタイルサイズを小さく
  python scripts/run_restormer.py --input test_inputs/ --tile 256
"""

import argparse
import os
import sys
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RESTORMER_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'Restormer')
sys.path.insert(0, RESTORMER_DIR)
from basicsr.models.archs.restormer_arch import Restormer

_DENOISE_PARAMS_BIASFREE = {
    'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
    'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False,
}
_DEBLUR_PARAMS_WITHBIAS = {
    'dim': 48, 'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
    'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False,
}

TASK_CONFIGS = {
    'Real_Denoising': {
        'weights': 'Denoising/pretrained_models/real_denoising.pth',
        'params': {'inp_channels': 3, 'out_channels': 3, **_DENOISE_PARAMS_BIASFREE},
        'grayscale': False,
    },
    'Gaussian_Gray_Denoising': {
        'weights': 'Denoising/pretrained_models/gaussian_gray_denoising_blind.pth',
        'params': {'inp_channels': 1, 'out_channels': 1, **_DENOISE_PARAMS_BIASFREE},
        'grayscale': True,
    },
    'Motion_Deblurring': {
        'weights': 'Motion_Deblurring/pretrained_models/motion_deblurring.pth',
        'params': {'inp_channels': 3, 'out_channels': 3, **_DEBLUR_PARAMS_WITHBIAS},
        'grayscale': False,
    },
    'Defocus_Deblurring': {
        'weights': 'Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth',
        'params': {'inp_channels': 3, 'out_channels': 3, **_DEBLUR_PARAMS_WITHBIAS},
        'grayscale': False,
    },
}


def load_model(task, device):
    cfg = TASK_CONFIGS[task]
    model = Restormer(**cfg['params'])
    weights_path = os.path.join(RESTORMER_DIR, cfg['weights'])
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt['params'])
    model.eval()
    return model.to(device)


def tile_inference(model, x, tile_size, tile_overlap, device):
    """タイル分割推論。VRAM 節約のため大きい画像を分割して処理する。"""
    b, c, h, w = x.shape
    stride = tile_size - tile_overlap
    h_idx = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx = list(range(0, w - tile_size, stride)) + [w - tile_size]

    E = torch.zeros_like(x)
    W = torch.zeros_like(x)

    for hi in h_idx:
        for wi in w_idx:
            patch = x[:, :, hi:hi+tile_size, wi:wi+tile_size].to(device)
            with torch.no_grad():
                out = model(patch)
            E[:, :, hi:hi+tile_size, wi:wi+tile_size] += out.cpu()
            W[:, :, hi:hi+tile_size, wi:wi+tile_size] += 1

    return E / W


def denoise_image(model, img_path, task, tile_size, device):
    cfg = TASK_CONFIGS[task]
    img = Image.open(img_path)

    if cfg['grayscale']:
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    else:
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    _, _, h, w = x.shape
    try:
        if tile_size and (h > tile_size or w > tile_size):
            out = tile_inference(model, x, tile_size, tile_overlap=32, device=device)
        else:
            with torch.no_grad():
                out = model(x.to(device)).cpu()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  [OOM] {os.path.basename(img_path)} skipped. Try --tile with a smaller value.')
        return None

    out = out.squeeze().numpy().clip(0, 1) * 255
    if cfg['grayscale']:
        return out.astype(np.uint8)
    else:
        return out.transpose(1, 2, 0).astype(np.uint8)


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
    parser.add_argument('--output', default='results/Restormer', help='Output directory')
    parser.add_argument('--task', default='Real_Denoising',
                        choices=list(TASK_CONFIGS.keys()), help='Denoising task')
    parser.add_argument('--tile', type=int, default=512, help='Tile size for large images (0 to disable)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    tile_size = args.tile if args.tile > 0 else None
    print(f'Device: {device}  Task: {args.task}  Tile: {tile_size}')

    root = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(root, args.output, args.task) if not os.path.isabs(args.output) else os.path.join(args.output, args.task)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading model...')
    model = load_model(args.task, device)
    print('Model loaded.')

    input_files = collect_inputs(args.input if os.path.isabs(args.input) else os.path.join(root, args.input))
    if not input_files:
        print('No input images found.')
        return

    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        task_tag = args.task.lower().replace('_', '-')
        out_path = os.path.join(output_dir, f'{basename}_restormer_{task_tag}.png')
        t0 = time.time()
        out = denoise_image(model, img_path, args.task, tile_size, device)
        elapsed = time.time() - t0
        if out is None:
            continue
        Image.fromarray(out).save(out_path)
        print(f'  {os.path.basename(img_path)} -> {os.path.basename(out_path)}  ({elapsed:.2f}s)')

    print(f'\nDone. {len(input_files)} image(s) saved to {output_dir}')


if __name__ == '__main__':
    main()
