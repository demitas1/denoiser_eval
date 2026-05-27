"""
BSRNet 学習曲線プロットスクリプト。

チェックポイントディレクトリ内の iter_*.pth を順番に評価し、
PSNR vs. iter のグラフを生成する。

使い方:
  python scripts/plot_bsrnet_psnr_curve.py \
      --ckpt_dir results/train_bsrgan_psnr_v2/ \
      --testset testsets/custom_natural/pexels-cc0-100-1/ \
      --output results/train_bsrgan_psnr_v2/psnr_curve.png

  # 評価済み TSV を再利用してプロットのみ
  python scripts/plot_bsrnet_psnr_curve.py \
      --ckpt_dir results/train_bsrgan_psnr_v2/ \
      --testset testsets/custom_natural/pexels-cc0-100-1/ \
      --output results/train_bsrgan_psnr_v2/psnr_curve.png \
      --tsv_cache results/train_bsrgan_psnr_v2/psnr_curve.tsv
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_rrdbnet import RRDBNet
from data.dataset_blindsr import DatasetBlindSR


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'netE' in ckpt:
        return ckpt['netE']
    if isinstance(ckpt, dict) and 'netG' in ckpt:
        return ckpt['netG']
    return ckpt


def build_model(state_dict, device):
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def evaluate_avg_psnr(model, loader, device, seed):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    psnrs = []
    with torch.no_grad():
        for batch in loader:
            L = batch['L'].to(device)
            H = batch['H'].to(device)
            pred = model(L).clamp(0.0, 1.0)
            mse = F.mse_loss(pred, H).item()
            psnrs.append(10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0)
    np.random.set_state(rng_state)
    return float(np.mean(psnrs))


def make_loader(testset_dir, lq_patchsize, scale, n_channels):
    ds_opt = {
        'phase': 'test',
        'n_channels': n_channels,
        'scale': scale,
        'shuffle_prob': 0.1,
        'use_sharp': False,
        'degradation_type': 'bsrgan',
        'lq_patchsize': lq_patchsize,
        'H_size': lq_patchsize * scale,
        'dataroot_H': testset_dir,
    }
    dataset = DatasetBlindSR(ds_opt)
    dataset.paths_H = [
        p for p in dataset.paths_H
        if os.path.dirname(os.path.normpath(p)) == os.path.normpath(testset_dir)
    ]
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def collect_checkpoints(ckpt_dir):
    pattern = os.path.join(ckpt_dir, 'iter_*.pth')
    paths = sorted(glob.glob(pattern))
    result = []
    for p in paths:
        m = re.search(r'iter_(\d+)\.pth$', p)
        if m:
            result.append((int(m.group(1)), p))
    return result


def load_tsv_cache(path):
    results = {}
    if not os.path.isfile(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('iter'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                results[int(parts[0])] = float(parts[1])
    return results


def save_tsv_cache(path, results):
    with open(path, 'w') as f:
        f.write('iter\tpsnr\n')
        for it, psnr in sorted(results.items()):
            f.write(f'{it}\t{psnr:.4f}\n')


def plot(iters, psnrs, best_iter, best_psnr, output_path, testset_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, psnrs, color='steelblue', linewidth=1.5, label='EMA PSNR')
    ax.axvline(best_iter, color='tomato', linestyle='--', linewidth=1.0,
               label=f'Best: {best_psnr:.2f} dB @ iter {best_iter:,}')
    ax.scatter([best_iter], [best_psnr], color='tomato', zorder=5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'BSRNet v2 — PSNR curve ({testset_name}, seed=0)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Plot saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='BSRNet PSNR 学習曲線プロット')
    parser.add_argument('--ckpt_dir', required=True, metavar='DIR',
                        help='iter_*.pth を含むチェックポイントディレクトリ')
    parser.add_argument('--testset', required=True, metavar='DIR',
                        help='HR 画像のテストセットディレクトリ')
    parser.add_argument('--output', required=True, metavar='PATH',
                        help='出力 PNG パス')
    parser.add_argument('--tsv_cache', metavar='PATH', default=None,
                        help='評価結果の TSV キャッシュ（指定時は未評価分のみ実行）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lq_patchsize', type=int, default=72)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    ckpt_dir = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(ROOT, args.ckpt_dir)
    testset_dir = args.testset if os.path.isabs(args.testset) else os.path.join(ROOT, args.testset)
    testset_dir = os.path.normpath(testset_dir)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)

    tsv_cache_path = args.tsv_cache or os.path.join(ckpt_dir, 'psnr_curve.tsv')

    checkpoints = collect_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f'Error: no iter_*.pth found in {ckpt_dir}', file=sys.stderr)
        sys.exit(1)

    cached = load_tsv_cache(tsv_cache_path)
    todo = [(it, p) for it, p in checkpoints if it not in cached]

    print(f'Checkpoints: {len(checkpoints)}  Cached: {len(cached)}  To evaluate: {len(todo)}')
    print(f'Testset: {testset_dir}')
    print(f'Device: {device}')

    if todo:
        loader = make_loader(testset_dir, args.lq_patchsize, args.scale, args.n_channels)
        print(f'Images: {len(loader.dataset)}')
        print()

        for i, (it, ckpt_path) in enumerate(todo, 1):
            print(f'[{i}/{len(todo)}] iter {it:,}  {os.path.basename(ckpt_path)} ...', end=' ', flush=True)
            state_dict = load_checkpoint(ckpt_path, device)
            model = build_model(state_dict, device)
            psnr = evaluate_avg_psnr(model, loader, device, args.seed)
            cached[it] = psnr
            print(f'{psnr:.2f} dB', flush=True)
            del model
            torch.cuda.empty_cache()
            save_tsv_cache(tsv_cache_path, cached)

        print(f'\nResults saved: {tsv_cache_path}')

    # 全チェックポイント分のデータでプロット
    iters = sorted(cached.keys())
    psnrs = [cached[it] for it in iters]
    best_iter = iters[int(np.argmax(psnrs))]
    best_psnr = max(psnrs)

    print(f'\nBest: {best_psnr:.2f} dB @ iter {best_iter:,}')

    plot(iters, psnrs, best_iter, best_psnr, output_path,
         os.path.basename(testset_dir))


if __name__ == '__main__':
    main()
