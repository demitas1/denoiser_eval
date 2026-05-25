"""
FFDNet PSNR 評価スクリプト。

クリーン HR 画像にガウシアンノイズ（指定 sigma）を加算し、FFDNet でデノイズして
クリーン画像との PSNR を計算する。--seed でノイズ乱数を固定するため、
異なるモデル間の公平な比較が可能。

PSNR はグレースケールの MSE から計算（ffdnet_gray 系モデルと同一チャンネル）。
カラーモデル（ffdnet_color 系）の場合も RGB→グレースケール変換して評価する。

使い方:
  # 公式 ffdnet_gray で全デフォルト sigma を評価
  python scripts/eval_ffdnet_psnr.py \\
      --model ffdnet_gray \\
      --testset testsets/unsplash_lite_test/

  # カスタムチェックポイントで sigma=10,25 のみ評価
  python scripts/eval_ffdnet_psnr.py \\
      --checkpoint results/train_ffdnet_gray/best.pth \\
      --testset testsets/unsplash_lite_test/ --sigma 10 25

  # サブディレクトリを含めて評価
  python scripts/eval_ffdnet_psnr.py \\
      --model ffdnet_gray --testset testsets/ --recursive

  # TSV 出力
  python scripts/eval_ffdnet_psnr.py \\
      --model ffdnet_gray --testset testsets/unsplash_lite_test/ --tsv

  # seed を変えて複数回評価し平均をとる
  for s in 0 1 2; do
    python scripts/eval_ffdnet_psnr.py --model ffdnet_gray \\
        --testset testsets/unsplash_lite_test/ --sigma 25 --seed $s | tail -1
  done
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_ffdnet import FFDNet

DEFAULT_SIGMAS = [5, 10, 15, 25, 50]

NAMED_MODELS = {
    'ffdnet_gray':      (os.path.join(KAIR_DIR, 'model_zoo', 'ffdnet_gray.pth'),      'gray'),
    'ffdnet_gray_clip': (os.path.join(KAIR_DIR, 'model_zoo', 'ffdnet_gray_clip.pth'), 'gray'),
    'ffdnet_color':     (os.path.join(KAIR_DIR, 'model_zoo', 'ffdnet_color.pth'),     'color'),
    'ffdnet_color_clip':(os.path.join(KAIR_DIR, 'model_zoo', 'ffdnet_color_clip.pth'),'color'),
}


def load_checkpoint(path, device):
    """state_dict-only と完全チェックポイント（state_dict / step / best_psnr キー）を受け付ける。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict'], ckpt.get('step'), ckpt.get('best_psnr')
    return ckpt, None, None


def build_model(state_dict, mode, device):
    if mode == 'gray':
        model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
    else:
        model = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def collect_images(directory, recursive, exts=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    directory = os.path.normpath(directory)
    paths = []
    if recursive:
        for ext in exts:
            paths += glob.glob(os.path.join(directory, '**', f'*{ext}'), recursive=True)
            paths += glob.glob(os.path.join(directory, '**', f'*{ext.upper()}'), recursive=True)
    else:
        for ext in exts:
            paths += glob.glob(os.path.join(directory, f'*{ext}'))
            paths += glob.glob(os.path.join(directory, f'*{ext.upper()}'))
    return sorted(set(paths))


def load_image_gray(path):
    """グレースケール float32 テンソル (1,1,H,W) を返す。値域 [0,1]。"""
    img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


def evaluate(model, paths, device, sigmas, seed):
    """各 sigma ごとに全画像を評価し、{sigma: [(name, psnr), ...]} を返す。"""
    rng = np.random.default_rng(seed)

    results = {s: [] for s in sigmas}
    for path in paths:
        clean = load_image_gray(path).to(device)
        name = os.path.basename(path)
        noise_seeds = {s: int(rng.integers(0, 2**31)) for s in sigmas}

        for sigma in sigmas:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(noise_seeds[sigma])
            noise = torch.randn(clean.shape, generator=gen) * (sigma / 255.0)
            noisy = (clean.cpu() + noise).clamp(0.0, 1.0).to(device)

            sigma_map = torch.full((1, 1, 1, 1), sigma / 255.0,
                                   dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(noisy, sigma_map).clamp(0.0, 1.0)

            mse = F.mse_loss(pred, clean).item()
            psnr = 10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0
            results[sigma].append((name, psnr))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='FFDNet PSNR 評価',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--model', choices=list(NAMED_MODELS),
                     metavar='NAME',
                     help=f'公式モデル名: {", ".join(NAMED_MODELS)}')
    src.add_argument('--checkpoint', metavar='PATH',
                     help='.pth ファイル（state_dict / 完全チェックポイント）')

    parser.add_argument('--testset', required=True, metavar='DIR',
                        help='クリーン HR 画像のディレクトリ')
    parser.add_argument('--recursive', action='store_true',
                        help='サブディレクトリも含める（デフォルト: top-level のみ）')
    parser.add_argument('--sigma', type=int, nargs='+', default=DEFAULT_SIGMAS,
                        metavar='N',
                        help=f'評価する sigma 値（複数指定可、デフォルト: {DEFAULT_SIGMAS}）')
    parser.add_argument('--seed', type=int, default=0,
                        help='ノイズ生成の乱数シード（デフォルト: 0）')
    parser.add_argument('--mode', choices=['gray', 'color'], default=None,
                        help='モデルモード（--model 指定時は自動設定; --checkpoint 時のデフォルト: gray）')
    parser.add_argument('--cpu', action='store_true',
                        help='CPU 推論を強制する')
    parser.add_argument('--tsv', action='store_true',
                        help='TSV 形式で出力（ヘッダー付き、パイプ処理向け）')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    # --- モデルのロード ---
    if args.model:
        model_path, mode = NAMED_MODELS[args.model]
        if not os.path.isfile(model_path):
            print(f'Error: weight not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        label = f'{args.model} (official)'
        step = best_psnr = None
    else:
        ckpt_path = (args.checkpoint if os.path.isabs(args.checkpoint)
                     else os.path.join(ROOT, args.checkpoint))
        if not os.path.isfile(ckpt_path):
            print(f'Error: checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)
        state_dict, step, best_psnr = load_checkpoint(ckpt_path, device)
        mode = args.mode or 'gray'
        label = os.path.basename(ckpt_path)
        if step is not None:
            label += f'  step={step}'
            if best_psnr is not None:
                label += f'  best_train_psnr={best_psnr:.2f} dB'

    if args.mode:
        mode = args.mode

    model = build_model(state_dict, mode, device)

    # --- テストセットのロード ---
    testset_dir = os.path.normpath(
        args.testset if os.path.isabs(args.testset) else os.path.join(ROOT, args.testset)
    )
    if not os.path.isdir(testset_dir):
        print(f'Error: testset not found: {testset_dir}', file=sys.stderr)
        sys.exit(1)

    paths = collect_images(testset_dir, args.recursive)
    if not paths:
        print(f'Error: no images found in {testset_dir}', file=sys.stderr)
        print('  Use --recursive to include subdirectories.', file=sys.stderr)
        sys.exit(1)

    # --- ヘッダー（TSV 以外） ---
    if not args.tsv:
        print(f'Model:   {label}')
        print(f'Testset: {testset_dir}  [{len(paths)} images]')
        print(f'Seed: {args.seed}  mode: {mode}  sigma: {args.sigma}  device: {device}')
        print()

    # --- 評価 ---
    results = evaluate(model, paths, device, args.sigma, seed=args.seed)

    # --- 出力 ---
    if args.tsv:
        header_sigmas = '\t'.join(f'psnr_s{s}' for s in args.sigma)
        print(f'filename\t{header_sigmas}')
        names = [n for n, _ in results[args.sigma[0]]]
        for i, name in enumerate(names):
            row = '\t'.join(f'{results[s][i][1]:.4f}' for s in args.sigma)
            print(f'{name}\t{row}')
    else:
        col = max(len(os.path.basename(p)) for p in paths)
        for sigma in args.sigma:
            print(f'--- sigma={sigma} ---')
            for name, psnr in results[sigma]:
                print(f'  {name:<{col}}  {psnr:.2f} dB')
            psnrs = [p for _, p in results[sigma]]
            avg = float(np.mean(psnrs))
            print(f'  Average: {avg:.2f} dB'
                  f'  (min: {min(psnrs):.2f}  max: {max(psnrs):.2f}  n={len(psnrs)})')
            print()

    if args.tsv:
        avg_row = '\t'.join(f'{float(np.mean([p for _, p in results[s]])):.4f}'
                            for s in args.sigma)
        print(f'Average\t{avg_row}')


if __name__ == '__main__':
    main()
