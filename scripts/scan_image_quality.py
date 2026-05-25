#!/usr/bin/env python3
"""
Scan images for JPEG compression artifacts.

Two metrics:
  blocking  -- 8px block boundary artifact ratio (works on PNG/JPEG)
  jpeg_q    -- JPEG quantization table mean (JPEG source files only; lower = higher quality)

Output (TSV by default):
  path <TAB> blocking <TAB> jpeg_q

Usage examples:
  # Scan a directory, sort by worst blocking score
  python scripts/scan_image_quality.py testsets/custom_natural/pexels-cc0-100-1/

  # Show only images with blocking >= 1.4, pipe paths to another script
  python scripts/scan_image_quality.py dir/ --min-blocking 1.4 --no-header | cut -f1

  # Scan original JPEGs (both metrics available)
  python scripts/scan_image_quality.py dir/original/ --ext jpg jpeg

  # Top 20 worst blocking images
  python scripts/scan_image_quality.py dir/ --top 20

  # JSON output
  python scripts/scan_image_quality.py dir/ --format json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

JPEG_EXTS = {'.jpg', '.jpeg', '.jpe', '.jfif'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def compute_blocking_score(img_gray: np.ndarray) -> float:
    """
    Ratio of mean absolute difference at 8px block boundaries vs. interior pixels.
    Score near 1.0 = no artifacts. Score > 1.3~1.5 = noticeable blocking.
    """
    h, w = img_gray.shape
    img = img_gray.astype(np.float32)
    eps = 1e-6
    scores = []

    if h > 16:
        dh = np.abs(np.diff(img, axis=0))   # shape: (h-1, w)
        boundary_mask = np.zeros(h - 1, dtype=bool)
        boundary_mask[7::8] = True
        d_block = dh[boundary_mask].mean()
        d_inner = dh[~boundary_mask].mean()
        scores.append(d_block / (d_inner + eps))

    if w > 16:
        dv = np.abs(np.diff(img, axis=1))   # shape: (h, w-1)
        boundary_mask = np.zeros(w - 1, dtype=bool)
        boundary_mask[7::8] = True
        d_block = dv[:, boundary_mask].mean()
        d_inner = dv[:, ~boundary_mask].mean()
        scores.append(d_block / (d_inner + eps))

    return float(np.mean(scores)) if scores else float('nan')


def compute_jpeg_quality_proxy(path: Path) -> float:
    """
    Mean of all JPEG quantization table coefficients.
    Higher value = more aggressive compression (lower JPEG quality).
    Returns nan for non-JPEG or unreadable files.

    Rough mapping: ~2-5 = quality 95+, ~8-12 = quality 85, ~15-25 = quality 75
    """
    try:
        img = Image.open(path)
        tables = getattr(img, 'quantization', None)
        if not tables:
            return float('nan')
        return float(np.mean([np.mean(t) for t in tables.values()]))
    except Exception:
        return float('nan')


def scan_file(path: Path) -> dict | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blocking = compute_blocking_score(img)
    is_jpeg = path.suffix.lower() in JPEG_EXTS
    jpeg_q = compute_jpeg_quality_proxy(path) if is_jpeg else float('nan')

    return {
        'path': str(path),
        'blocking': None if np.isnan(blocking) else round(blocking, 4),
        'jpeg_q': None if np.isnan(jpeg_q) else round(jpeg_q, 2),
    }


def collect_files(input_path: Path, recursive: bool, exts: set) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = '**/*' if recursive else '*'
    files = sorted(
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in exts
    )
    return files


def format_value(v, nan_str: str) -> str:
    return nan_str if v is None else str(v)


def main():
    parser = argparse.ArgumentParser(
        description='Scan images for JPEG compression artifacts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('input',
                        help='Image file or directory to scan')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recurse into subdirectories')
    parser.add_argument('--ext', nargs='+',
                        default=['jpg', 'jpeg', 'png'],
                        metavar='EXT',
                        help='File extensions to include (default: jpg jpeg png)')
    parser.add_argument('--sort', choices=['blocking', 'jpeg_q', 'path'],
                        default='blocking',
                        help='Sort key (default: blocking, descending)')
    parser.add_argument('--top', type=int, default=None,
                        metavar='N',
                        help='Show only top N results after sorting')
    parser.add_argument('--min-blocking', type=float, default=None,
                        metavar='SCORE',
                        help='Keep only entries with blocking >= SCORE')
    parser.add_argument('--min-jpeg-q', type=float, default=None,
                        metavar='SCORE',
                        help='Keep only JPEG entries with jpeg_q >= SCORE')
    parser.add_argument('--format', choices=['tsv', 'csv', 'json'],
                        default='tsv',
                        help='Output format (default: tsv)')
    parser.add_argument('--no-header', action='store_true',
                        help='Suppress header row (TSV/CSV output)')
    parser.add_argument('--nan', default='',
                        metavar='STR',
                        help='Placeholder for N/A values in TSV/CSV (default: empty)')
    args = parser.parse_args()

    exts = {'.' + e.lstrip('.').lower() for e in args.ext}
    input_path = Path(args.input)

    if not input_path.exists():
        print(f'Error: {input_path} does not exist', file=sys.stderr)
        sys.exit(1)

    files = collect_files(input_path, args.recursive, exts)
    if not files:
        print('No image files found.', file=sys.stderr)
        sys.exit(0)

    print(f'Scanning {len(files)} file(s)...', file=sys.stderr)

    results = []
    for p in files:
        r = scan_file(p)
        if r is None:
            print(f'Warning: could not read {p}', file=sys.stderr)
            continue
        results.append(r)

    # Filter
    if args.min_blocking is not None:
        results = [r for r in results
                   if r['blocking'] is not None and r['blocking'] >= args.min_blocking]
    if args.min_jpeg_q is not None:
        results = [r for r in results
                   if r['jpeg_q'] is not None and r['jpeg_q'] >= args.min_jpeg_q]

    # Sort (path ascending, scores descending)
    if args.sort == 'path':
        results.sort(key=lambda r: r['path'])
    else:
        results.sort(
            key=lambda r: r[args.sort] if r[args.sort] is not None else -1,
            reverse=True,
        )

    if args.top is not None:
        results = results[:args.top]

    # Output
    if args.format == 'json':
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    delimiter = '\t' if args.format == 'tsv' else ','
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator='\n')

    if not args.no_header:
        writer.writerow(['path', 'blocking', 'jpeg_q'])

    for r in results:
        writer.writerow([
            r['path'],
            format_value(r['blocking'], args.nan),
            format_value(r['jpeg_q'], args.nan),
        ])


if __name__ == '__main__':
    main()
