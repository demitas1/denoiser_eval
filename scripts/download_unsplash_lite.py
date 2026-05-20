"""
Unsplash Lite データセットから HR 画像をダウンロードするスクリプト。

photos.csv000（TSV）の photo_image_url を使い、
解像度フィルタを通過した画像を trainsets/trainH/unsplash_lite/ に保存する。
既にダウンロード済みのファイルはスキップするため中断後の再開が可能。

使い方:
  python scripts/download_unsplash_lite.py

  # 枚数を絞る
  python scripts/download_unsplash_lite.py --max_images 500

  # 最小解像度を変更（デフォルト: 1024）
  python scripts/download_unsplash_lite.py --min_size 2000

  # ダウンロード画像サイズを変更（デフォルト: 1080px）
  python scripts/download_unsplash_lite.py --download_width 2048
"""

import argparse
import os
import sys
import time

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

META_TSV = os.path.join(ROOT, 'trainsets', 'unsplash_lite_meta', 'photos.csv000')
OUT_DIR  = os.path.join(ROOT, 'trainsets', 'trainH', 'unsplash_lite')

HEADERS = {
    'User-Agent': 'denoiser-eval-training/1.0 (ML research; unsplash dataset terms accepted)',
}


def load_photos(tsv_path, min_size, max_images):
    photos = []
    with open(tsv_path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        idx_id    = header.index('photo_id')
        idx_url   = header.index('photo_image_url')
        idx_w     = header.index('photo_width')
        idx_h     = header.index('photo_height')

        for line in f:
            cols = line.rstrip('\n').split('\t')
            if len(cols) <= max(idx_id, idx_url, idx_w, idx_h):
                continue
            try:
                w, h = int(cols[idx_w]), int(cols[idx_h])
            except ValueError:
                continue
            if w < min_size or h < min_size:
                continue
            photos.append((cols[idx_id].strip(), cols[idx_url].strip()))
            if max_images and len(photos) >= max_images:
                break
    return photos


def download(photo_id, base_url, out_dir, width, delay):
    out_path = os.path.join(out_dir, f'{photo_id}.jpg')
    if os.path.exists(out_path):
        return 'skip'

    url = f'{base_url}?w={width}&q=85&fm=jpg&fit=max'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type:
            return f'err:unexpected content-type {content_type}'
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        time.sleep(delay)
        return 'ok'
    except requests.RequestException as e:
        return f'err:{e}'


def main():
    parser = argparse.ArgumentParser(description='Download Unsplash Lite HR images.')
    parser.add_argument('--meta',          default=META_TSV,
                        help='Path to photos.csv000 TSV (default: trainsets/unsplash_lite_meta/photos.csv000)')
    parser.add_argument('--out_dir',       default=OUT_DIR,
                        help='Output directory (default: trainsets/trainH/unsplash_lite)')
    parser.add_argument('--max_images',    type=int, default=2000,
                        help='Maximum number of images to download (default: 2000, 0=unlimited)')
    parser.add_argument('--min_size',      type=int, default=1024,
                        help='Minimum width AND height in pixels (default: 1024)')
    parser.add_argument('--download_width', type=int, default=1080,
                        help='Download width in pixels via Unsplash dynamic URL (default: 1080)')
    parser.add_argument('--delay',         type=float, default=0.5,
                        help='Delay between requests in seconds (default: 0.5)')
    args = parser.parse_args()

    max_images = args.max_images if args.max_images > 0 else None

    if not os.path.isfile(args.meta):
        print(f'Error: metadata not found: {args.meta}')
        print('Run: unzip unsplash-research-dataset-lite-latest.zip -d trainsets/unsplash_lite_meta/')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading metadata from {args.meta} ...')
    photos = load_photos(args.meta, args.min_size, max_images)
    print(f'  Eligible photos (>={args.min_size}px): {len(photos)}')

    already = sum(1 for pid, _ in photos if os.path.exists(os.path.join(args.out_dir, f'{pid}.jpg')))
    print(f'  Already downloaded: {already}  Remaining: {len(photos) - already}')

    ok = skip = err = 0
    for i, (photo_id, url) in enumerate(photos, 1):
        result = download(photo_id, url, args.out_dir, args.download_width, args.delay)
        if result == 'ok':
            ok += 1
        elif result == 'skip':
            skip += 1
        else:
            err += 1
            print(f'  [{i}/{len(photos)}] FAIL {photo_id}: {result}')

        if i % 100 == 0 or i == len(photos):
            print(f'  [{i}/{len(photos)}] ok={ok} skip={skip} err={err}')

    print(f'\nDone. Saved to {args.out_dir}')
    print(f'  Downloaded: {ok}  Skipped: {skip}  Errors: {err}')


if __name__ == '__main__':
    main()
