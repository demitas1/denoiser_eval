"""
BSRNet 学習曲線プロットスクリプト（PSNR + SSIM）。

チェックポイントディレクトリ内の iter_*.pth を順番に評価し、
PSNR / SSIM それぞれの平均・min・max を TSV に記録してグラフを生成する。

使い方:
  # 全チェックポイントを評価
  python scripts/plot_bsrnet_psnr_curve.py \\
      --ckpt_dir results/train_bsrgan_psnr_v2/ \\
      --testset testsets/custom_natural/pexels-cc0-100-1/ \\
      --output results/train_bsrgan_psnr_v2/quality_curve.png

  # iter 範囲と間引きを指定（10k ごと、2k–200k）
  python scripts/plot_bsrnet_psnr_curve.py \\
      --ckpt_dir results/train_bsrgan_psnr_v2/ \\
      --testset testsets/custom_natural/pexels-cc0-100-1/ \\
      --output results/train_bsrgan_psnr_v2/quality_curve.png \\
      --iter_min 2000 --iter_max 200000 --iter_step 10000

  # 評価済み TSV を再利用してプロットのみ
  python scripts/plot_bsrnet_psnr_curve.py \\
      --ckpt_dir results/train_bsrgan_psnr_v2/ \\
      --testset testsets/custom_natural/pexels-cc0-100-1/ \\
      --output results/train_bsrgan_psnr_v2/quality_curve.png \\
      --tsv_cache results/train_bsrgan_psnr_v2/quality_curve.tsv

TSV フォーマット:
  iter  psnr_mean  psnr_min  psnr_max  ssim_mean  ssim_min  ssim_max
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
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_rrdbnet import RRDBNet
from data.dataset_blindsr import DatasetBlindSR

TSV_HEADER = 'iter\tpsnr_mean\tpsnr_min\tpsnr_max\tssim_mean\tssim_min\tssim_max'


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


def evaluate(model, loader, device, seed):
    """PSNR と SSIM の mean / min / max を返す。"""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    psnrs, ssims = [], []
    with torch.no_grad():
        for batch in loader:
            L = batch['L'].to(device)
            H = batch['H'].to(device)
            pred = model(L).clamp(0.0, 1.0)

            mse = F.mse_loss(pred, H).item()
            psnrs.append(10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0)

            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            h_np = H.squeeze(0).permute(1, 2, 0).cpu().numpy()
            ssims.append(ssim_metric(h_np, pred_np, data_range=1.0, channel_axis=-1))

    np.random.set_state(rng_state)
    return {
        'psnr_mean': float(np.mean(psnrs)),
        'psnr_min':  float(np.min(psnrs)),
        'psnr_max':  float(np.max(psnrs)),
        'ssim_mean': float(np.mean(ssims)),
        'ssim_min':  float(np.min(ssims)),
        'ssim_max':  float(np.max(ssims)),
    }


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


def collect_checkpoints(ckpt_dir, iter_min, iter_max, iter_step):
    pattern = os.path.join(ckpt_dir, 'iter_*.pth')
    paths = sorted(glob.glob(pattern))
    result = []
    for p in paths:
        m = re.search(r'iter_(\d+)\.pth$', p)
        if not m:
            continue
        it = int(m.group(1))
        if iter_min is not None and it < iter_min:
            continue
        if iter_max is not None and it > iter_max:
            continue
        if iter_step is not None and it % iter_step != 0:
            continue
        result.append((it, p))
    return result


def load_tsv_cache(path):
    """TSV からキャッシュを読み込む。7列フォーマットのみ受け付ける。"""
    results = {}
    if not os.path.isfile(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('iter'):
                continue
            parts = line.split('\t')
            if len(parts) != 7:
                continue  # 旧 2列フォーマットは無視
            try:
                it = int(parts[0])
                results[it] = {
                    'psnr_mean': float(parts[1]),
                    'psnr_min':  float(parts[2]),
                    'psnr_max':  float(parts[3]),
                    'ssim_mean': float(parts[4]),
                    'ssim_min':  float(parts[5]),
                    'ssim_max':  float(parts[6]),
                }
            except ValueError:
                continue
    return results


def save_tsv_cache(path, results):
    with open(path, 'w') as f:
        f.write(TSV_HEADER + '\n')
        for it, r in sorted(results.items()):
            f.write(f"{it}\t{r['psnr_mean']:.4f}\t{r['psnr_min']:.4f}\t{r['psnr_max']:.4f}"
                    f"\t{r['ssim_mean']:.4f}\t{r['ssim_min']:.4f}\t{r['ssim_max']:.4f}\n")


def plot(cached, output_path, testset_name, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    iters = sorted(cached.keys())
    psnr_mean = [cached[it]['psnr_mean'] for it in iters]
    psnr_min  = [cached[it]['psnr_min']  for it in iters]
    psnr_max  = [cached[it]['psnr_max']  for it in iters]
    ssim_mean = [cached[it]['ssim_mean'] for it in iters]
    ssim_min  = [cached[it]['ssim_min']  for it in iters]
    ssim_max  = [cached[it]['ssim_max']  for it in iters]

    best_psnr_iter = iters[int(np.argmax(psnr_mean))]
    best_psnr_val  = max(psnr_mean)
    best_ssim_iter = iters[int(np.argmax(ssim_mean))]
    best_ssim_val  = max(ssim_mean)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fmt = plt.FuncFormatter(lambda x, _: f'{int(x):,}')

    # --- PSNR ---
    ax1.fill_between(iters, psnr_min, psnr_max, alpha=0.2, color='steelblue', label='min–max range')
    ax1.plot(iters, psnr_mean, color='steelblue', linewidth=1.5, label='mean PSNR')
    ax1.axvline(best_psnr_iter, color='tomato', linestyle='--', linewidth=1.0,
                label=f'Best: {best_psnr_val:.2f} dB @ {best_psnr_iter:,}')
    ax1.scatter([best_psnr_iter], [best_psnr_val], color='tomato', zorder=5)
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- SSIM ---
    ax2.fill_between(iters, ssim_min, ssim_max, alpha=0.2, color='seagreen', label='min–max range')
    ax2.plot(iters, ssim_mean, color='seagreen', linewidth=1.5, label='mean SSIM')
    ax2.axvline(best_ssim_iter, color='tomato', linestyle='--', linewidth=1.0,
                label=f'Best: {best_ssim_val:.4f} @ {best_ssim_iter:,}')
    ax2.scatter([best_ssim_iter], [best_ssim_val], color='tomato', zorder=5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('SSIM')
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.xaxis.set_major_formatter(fmt)

    fig.suptitle(f'BSRNet — PSNR / SSIM curve ({testset_name}, seed={seed})', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Plot saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='BSRNet PSNR+SSIM 学習曲線プロット')
    parser.add_argument('--ckpt_dir', required=True, metavar='DIR',
                        help='iter_*.pth を含むチェックポイントディレクトリ')
    parser.add_argument('--testset', required=True, metavar='DIR',
                        help='HR 画像のテストセットディレクトリ')
    parser.add_argument('--output', required=True, metavar='PATH',
                        help='出力 PNG パス')
    parser.add_argument('--tsv_cache', metavar='PATH', default=None,
                        help='評価結果の TSV キャッシュ（指定時は未評価分のみ実行）')
    parser.add_argument('--iter_min', type=int, default=None,
                        help='評価する iter の下限（未満は除外）')
    parser.add_argument('--iter_max', type=int, default=None,
                        help='評価する iter の上限（超過は除外）')
    parser.add_argument('--iter_step', type=int, default=None,
                        help='この値の倍数の iter のみ評価（例: 10000 → 10k 刻み）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lq_patchsize', type=int, default=72)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    ckpt_dir    = args.ckpt_dir if os.path.isabs(args.ckpt_dir) else os.path.join(ROOT, args.ckpt_dir)
    testset_dir = args.testset  if os.path.isabs(args.testset)  else os.path.join(ROOT, args.testset)
    testset_dir = os.path.normpath(testset_dir)
    output_path = args.output   if os.path.isabs(args.output)   else os.path.join(ROOT, args.output)

    tsv_cache_path = (args.tsv_cache if args.tsv_cache
                      else os.path.join(ckpt_dir, 'quality_curve.tsv'))
    if not os.path.isabs(tsv_cache_path):
        tsv_cache_path = os.path.join(ROOT, tsv_cache_path)

    checkpoints = collect_checkpoints(ckpt_dir, args.iter_min, args.iter_max, args.iter_step)
    if not checkpoints:
        print(f'Error: no matching iter_*.pth found in {ckpt_dir}', file=sys.stderr)
        sys.exit(1)

    cached = load_tsv_cache(tsv_cache_path)
    todo = [(it, p) for it, p in checkpoints if it not in cached]

    print(f'Checkpoints (filtered): {len(checkpoints)}  Cached: {len(cached)}  To evaluate: {len(todo)}')
    if args.iter_min or args.iter_max or args.iter_step:
        print(f'Filter: iter_min={args.iter_min}  iter_max={args.iter_max}  iter_step={args.iter_step}')
    print(f'Testset: {testset_dir}')
    print(f'Device:  {device}  Seed: {args.seed}')

    if todo:
        loader = make_loader(testset_dir, args.lq_patchsize, args.scale, args.n_channels)
        print(f'Images:  {len(loader.dataset)}')
        print()

        for i, (it, ckpt_path) in enumerate(todo, 1):
            print(f'[{i}/{len(todo)}] iter {it:,}  {os.path.basename(ckpt_path)} ...', end=' ', flush=True)
            state_dict = load_checkpoint(ckpt_path, device)
            model = build_model(state_dict, device)
            r = evaluate(model, loader, device, args.seed)
            cached[it] = r
            print(f"PSNR {r['psnr_mean']:.2f} dB [{r['psnr_min']:.2f}–{r['psnr_max']:.2f}]"
                  f"  SSIM {r['ssim_mean']:.4f} [{r['ssim_min']:.4f}–{r['ssim_max']:.4f}]",
                  flush=True)
            del model
            torch.cuda.empty_cache()
            save_tsv_cache(tsv_cache_path, cached)

        print(f'\nResults saved: {tsv_cache_path}')

    # プロット対象は checkpoints に含まれる iter のみ
    plot_iters = {it for it, _ in checkpoints}
    plot_cached = {it: v for it, v in cached.items() if it in plot_iters}

    if not plot_cached:
        print('Error: no data to plot.', file=sys.stderr)
        sys.exit(1)

    iters = sorted(plot_cached.keys())
    best_psnr = max(plot_cached[it]['psnr_mean'] for it in iters)
    best_psnr_iter = max(iters, key=lambda it: plot_cached[it]['psnr_mean'])
    best_ssim = max(plot_cached[it]['ssim_mean'] for it in iters)
    best_ssim_iter = max(iters, key=lambda it: plot_cached[it]['ssim_mean'])
    print(f'\nBest PSNR: {best_psnr:.2f} dB @ iter {best_psnr_iter:,}')
    print(f'Best SSIM: {best_ssim:.4f} @ iter {best_ssim_iter:,}')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plot(plot_cached, output_path, os.path.basename(testset_dir), args.seed)


if __name__ == '__main__':
    main()
