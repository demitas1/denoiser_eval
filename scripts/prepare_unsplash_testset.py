"""
Unsplash Lite 訓練セットからテスト用画像を分割するスクリプト。

trainsets/trainH/unsplash_lite/ の末尾 N 枚を testsets/unsplash_lite_test/ へ移動する。
一度実行すると移動元から画像が減るため、冪等性のため --dry_run で対象を確認してから実行すること。

【使い方】
  # 対象ファイルを確認（移動は行わない）
  python scripts/prepare_unsplash_testset.py --dry_run

  # 実際に移動（デフォルト: 末尾 100 枚）
  python scripts/prepare_unsplash_testset.py

  # 枚数を変えたい場合
  python scripts/prepare_unsplash_testset.py --n_test 68
"""

import argparse
import glob
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def abs_path(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='trainsets/trainH/unsplash_lite',
                        help='Source directory (Unsplash Lite training images)')
    parser.add_argument('--dst', default='testsets/unsplash_lite_test',
                        help='Destination directory for test images')
    parser.add_argument('--n_test', type=int, default=100,
                        help='Number of images to move to test set (default: 100)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be moved without actually moving')
    args = parser.parse_args()

    src = abs_path(args.src)
    dst = abs_path(args.dst)

    if not os.path.isdir(src):
        print(f'Error: source directory not found: {src}')
        sys.exit(1)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(src, ext))
    paths = sorted(paths)

    if len(paths) == 0:
        print(f'Error: no images found in {src}')
        sys.exit(1)

    if args.n_test >= len(paths):
        print(f'Error: n_test={args.n_test} >= total images={len(paths)}')
        sys.exit(1)

    test_paths = paths[-args.n_test:]

    if args.dry_run:
        print(f'[dry_run] Source: {src}  ({len(paths)} images)')
        print(f'[dry_run] Destination: {dst}')
        print(f'[dry_run] Would move {len(test_paths)} images:')
        for p in test_paths:
            print(f'  {os.path.basename(p)}')
        print(f'[dry_run] Remaining train images: {len(paths) - len(test_paths)}')
        return

    os.makedirs(dst, exist_ok=True)
    existing = os.listdir(dst)
    if existing:
        print(f'Error: {dst} already contains {len(existing)} files. '
              f'Remove them or choose a different --dst to avoid overwriting.')
        sys.exit(1)

    for p in test_paths:
        shutil.move(p, os.path.join(dst, os.path.basename(p)))

    print(f'Moved {len(test_paths)} images → {dst}')
    print(f'Remaining train images in {src}: {len(paths) - len(test_paths)}')


if __name__ == '__main__':
    main()
