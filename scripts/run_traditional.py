"""
従来型画像フィルターによるベースライン比較スクリプト。

Gaussian / Median / Bilateral / NL-Means の4種をパラメータスイープで実行し、
ML ベースのデノイザとの比較用画像を生成する。

使い方:
  # 全フィルター・全デフォルトパラメータ
  python scripts/run_traditional.py --input test_inputs/ --output results/traditional/

  # フィルター指定
  python scripts/run_traditional.py --input test_inputs/ --filters gaussian median

  # パラメータ上書き
  python scripts/run_traditional.py --input test_inputs/ \
      --filters gaussian --gaussian_sigma 1.0 2.0
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_GAUSSIAN_SIGMA  = [1.0]
DEFAULT_MEDIAN_KSIZE    = [3]
DEFAULT_BILATERAL_SIGMA = [10, 20, 30, 40]
DEFAULT_NLMEANS_H       = [5, 10, 15, 20]
DEFAULT_UPSCALE_FACTOR  = [4]
DEFAULT_UPSCALE_METHODS = ['bicubic', 'lanczos']


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


def load_gray(img_path):
    return np.array(Image.open(img_path).convert('L'), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description='Traditional filter baseline (Gaussian / Median / Bilateral / NL-Means)')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='results/traditional', help='Output directory')
    parser.add_argument('--filters', nargs='+',
                        choices=['gaussian', 'median', 'bilateral', 'nlmeans', 'upscale', 'combine'],
                        default=['gaussian', 'median', 'bilateral', 'nlmeans'],
                        help='Filters to run (default: gaussian median bilateral nlmeans)')
    parser.add_argument('--gaussian_sigma', type=float, nargs='+',
                        default=DEFAULT_GAUSSIAN_SIGMA,
                        help='Gaussian sigma values (default: 1.0)')
    parser.add_argument('--median_ksize', type=int, nargs='+',
                        default=DEFAULT_MEDIAN_KSIZE,
                        help='Median kernel sizes (default: 3; must be odd)')
    parser.add_argument('--bilateral_sigma', type=float, nargs='+',
                        default=DEFAULT_BILATERAL_SIGMA,
                        help='Bilateral sigma_color=sigma_space (default: 10 20 30 40)')
    parser.add_argument('--nlmeans_h', type=float, nargs='+',
                        default=DEFAULT_NLMEANS_H,
                        help='NL-Means filter strength h (default: 5 10 15 20)')
    parser.add_argument('--upscale_factor', type=int, nargs='+',
                        default=DEFAULT_UPSCALE_FACTOR,
                        help='Upscale factor(s) (default: 4)')
    parser.add_argument('--upscale_method', nargs='+',
                        choices=['bicubic', 'lanczos'],
                        default=DEFAULT_UPSCALE_METHODS,
                        help='Upscale interpolation method(s) (default: bicubic lanczos); '
                             'downscale is always Lanczos')
    parser.add_argument('--combine_denoise', nargs='+',
                        choices=['gaussian', 'median', 'bilateral', 'nlmeans'],
                        default=['gaussian', 'median', 'bilateral', 'nlmeans'],
                        help='Denoise stage(s) for combine mode (default: all four)')
    parser.add_argument('--combine_upscale', nargs='+',
                        choices=['bicubic', 'lanczos'],
                        default=['bicubic', 'lanczos'],
                        help='Upscale stage(s) for combine mode (default: bicubic lanczos)')
    args = parser.parse_args()

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    input_files = collect_inputs(input_path)
    if not input_files:
        print('No input images found.')
        sys.exit(1)

    # median ksize must be odd
    median_ksizes = []
    for k in args.median_ksize:
        if k % 2 == 0:
            print(f'  [warn] median_ksize={k} is even; rounding up to {k+1}')
            k += 1
        median_ksizes.append(k)

    filters = set(args.filters)
    t0 = time.time()
    total_saved = 0

    for img_path in input_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        src = load_gray(img_path)

        if 'gaussian' in filters:
            for sigma in args.gaussian_sigma:
                out = cv2.GaussianBlur(src, (0, 0), sigma)
                out_path = os.path.join(output_dir, f'{basename}_gaussian_s{sigma:.1f}.png')
                Image.fromarray(out).save(out_path)
                total_saved += 1

        if 'median' in filters:
            for ksize in median_ksizes:
                out = cv2.medianBlur(src, ksize)
                out_path = os.path.join(output_dir, f'{basename}_median_k{ksize}.png')
                Image.fromarray(out).save(out_path)
                total_saved += 1

        if 'bilateral' in filters:
            for sigma in args.bilateral_sigma:
                out = cv2.bilateralFilter(src, -1, sigma, sigma)
                out_path = os.path.join(output_dir, f'{basename}_bilateral_s{int(sigma)}.png')
                Image.fromarray(out).save(out_path)
                total_saved += 1

        if 'nlmeans' in filters:
            for h in args.nlmeans_h:
                out = cv2.fastNlMeansDenoising(src, None, h, 7, 21)
                out_path = os.path.join(output_dir, f'{basename}_nlmeans_h{int(h)}.png')
                Image.fromarray(out).save(out_path)
                total_saved += 1

        if 'upscale' in filters:
            _up_flags = {
                'bicubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4,
            }
            h0, w0 = src.shape[:2]
            for factor in args.upscale_factor:
                for method in args.upscale_method:
                    up = cv2.resize(src, (w0 * factor, h0 * factor),
                                    interpolation=_up_flags[method])
                    out = cv2.resize(up, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
                    out_path = os.path.join(
                        output_dir, f'{basename}_upscale_x{factor}_{method}.png')
                    Image.fromarray(out).save(out_path)
                    total_saved += 1

        if 'combine' in filters:
            _up_flags = {
                'bicubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4,
            }
            h0, w0 = src.shape[:2]

            denoise_variants = []
            if 'gaussian' in args.combine_denoise:
                for sigma in args.gaussian_sigma:
                    denoise_variants.append(
                        (cv2.GaussianBlur(src, (0, 0), sigma), f'gaussian_s{sigma:.1f}'))
            if 'median' in args.combine_denoise:
                for ksize in median_ksizes:
                    denoise_variants.append(
                        (cv2.medianBlur(src, ksize), f'median_k{ksize}'))
            if 'bilateral' in args.combine_denoise:
                for sigma in args.bilateral_sigma:
                    denoise_variants.append(
                        (cv2.bilateralFilter(src, -1, sigma, sigma), f'bilateral_s{int(sigma)}'))
            if 'nlmeans' in args.combine_denoise:
                for h in args.nlmeans_h:
                    denoise_variants.append(
                        (cv2.fastNlMeansDenoising(src, None, h, 7, 21), f'nlmeans_h{int(h)}'))

            for denoised, denoise_tag in denoise_variants:
                for factor in args.upscale_factor:
                    for method in args.combine_upscale:
                        up = cv2.resize(denoised, (w0 * factor, h0 * factor),
                                        interpolation=_up_flags[method])
                        out = cv2.resize(up, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
                        out_path = os.path.join(
                            output_dir,
                            f'{basename}_combine_{denoise_tag}_{method}_x{factor}.png')
                        Image.fromarray(out).save(out_path)
                        total_saved += 1

        print(f'  {basename}')

    elapsed = time.time() - t0
    print(f'Done. {len(input_files)} image(s) × {total_saved // len(input_files)} variants'
          f' = {total_saved} files saved to {output_dir}  ({elapsed:.1f}s)')


if __name__ == '__main__':
    main()
