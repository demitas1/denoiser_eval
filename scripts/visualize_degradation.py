"""
BSRGAN 劣化パイプラインの各操作を可視化するスクリプト。

各インデックスが実際にどのような画像劣化を生成するかを単独または組み合わせで確認できる。

【操作インデックス】
  0, 1 : ブラー（等方性 or 非等方性ガウシアンカーネル、ランダム）
  2    : 中間ダウンサンプル（ランダム倍率・補間、縮小後に元サイズへ nearest 拡大）
  3    : 最終ダウンサンプル（×1/sf、縮小後に元サイズへ nearest 拡大）
  4    : ガウシアンノイズ（カラー / グレー / 相関ノイズをランダム選択）
  5    : JPEG 圧縮（品質 30〜95、ランダム）
  6    : ISP カメラノイズ（モデルなしのため no-op、警告を表示）

【使い方】
  # 単一操作
  python scripts/visualize_degradation.py --input image.png --index 2

  # 複数操作を個別に適用（ファイルを各インデックスごとに保存）
  python scripts/visualize_degradation.py --input image.png --index 0 1 4 5

  # シャッフルモード（指定インデックスをランダムな順序で連続適用）
  python scripts/visualize_degradation.py --input image.png --index 0 1 4 5 --shuffle

  # シャッフル複数サンプル
  python scripts/visualize_degradation.py --input image.png --index 0 1 2 3 4 5 \
      --shuffle --num_samples 5 --output results/degradation_vis/

  # 再現性が必要な場合はシードを固定
  python scripts/visualize_degradation.py --input image.png --index 0 1 4 --seed 42
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from utils.utils_blindsr import (
    add_blur,
    add_Gaussian_noise,
    add_JPEG_noise,
    fspecial,
    shift_pixel,
)

OP_NAMES = {
    0: 'blur_A',
    1: 'blur_B',
    2: 'downsample_mid',
    3: 'downsample_final',
    4: 'gaussian_noise',
    5: 'jpeg_noise',
    6: 'isp_noise',
}


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------

def load_and_crop(path, size=320):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = size / min(w, h)
    if scale > 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# 操作の適用
# ---------------------------------------------------------------------------

def apply_op(idx, img, sf=4):
    h, w = img.shape[:2]

    if idx in (0, 1):
        return add_blur(img, sf=sf)

    elif idx == 2:
        if random.random() < 0.75:
            sf1 = random.uniform(1, 2 * sf)
            out = cv2.resize(
                img,
                (max(1, int(img.shape[1] / sf1)), max(1, int(img.shape[0] / sf1))),
                interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]),
            )
        else:
            k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
            k_shifted = shift_pixel(k, sf)
            k_shifted = k_shifted / k_shifted.sum()
            out = ndimage.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
            out = out[0::sf, 0::sf, ...]
        out = np.clip(out, 0.0, 1.0)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    elif idx == 3:
        out = cv2.resize(
            img,
            (max(1, w // sf), max(1, h // sf)),
            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]),
        )
        out = np.clip(out, 0.0, 1.0)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    elif idx == 4:
        return add_Gaussian_noise(img, noise_level1=2, noise_level2=25)

    elif idx == 5:
        return add_JPEG_noise(img)

    elif idx == 6:
        return img.copy()

    else:
        raise ValueError(f'Unknown index: {idx}')


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------

def save_image(img_float, path):
    img_uint8 = (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualize individual BSRGAN degradation operations.'
    )
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--index', type=int, nargs='+', required=True,
                        help='Degradation index/indices (0-6)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: same directory as input)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Apply specified indices in random order (shuffle mode)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of shuffle samples to generate (shuffle mode only)')
    parser.add_argument('--sf', type=int, default=4,
                        help='Scale factor for downsampling operations (default: 4)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--patch_size', type=int, default=320,
                        help='Input patch size (default: 320)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    for idx in args.index:
        if idx not in range(7):
            parser.error(f'Index {idx} is out of range (valid: 0-6)')

    # --- 入力画像 ---
    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    if not os.path.isfile(input_path):
        parser.error(f'Input file not found: {input_path}')

    img = load_and_crop(input_path, size=args.patch_size)
    print(f'Input : {input_path}  →  patch {img.shape[1]}×{img.shape[0]}')

    stem = os.path.splitext(os.path.basename(input_path))[0]

    out_dir = args.output if args.output else os.path.dirname(input_path)
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- 処理 ---
    if not args.shuffle:
        # 単一 / 複数インデックスを個別に適用
        for idx in args.index:
            result = apply_op(idx, img.copy(), sf=args.sf)
            out_path = os.path.join(out_dir, f'{stem}_{idx}.png')
            save_image(result, out_path)
            print(f'  idx={idx} ({OP_NAMES[idx]})  →  {out_path}')

    else:
        # シャッフルモード
        for s in range(args.num_samples):
            order = random.sample(args.index, len(args.index))
            result = img.copy()
            ops_str = ' → '.join(f'{i}({OP_NAMES[i]})' for i in order)
            print(f'  sample {s+1}/{args.num_samples}: {ops_str}')
            for idx in order:
                result = apply_op(idx, result, sf=args.sf)

            suffix = '_'.join(str(i) for i in order)
            if args.num_samples > 1:
                suffix += f'_s{s + 1}'
            out_path = os.path.join(out_dir, f'{stem}_{suffix}.png')
            save_image(result, out_path)
            print(f'    saved: {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
