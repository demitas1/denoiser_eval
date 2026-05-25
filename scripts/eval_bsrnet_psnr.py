"""
BSRNet / BSRGAN PSNR 評価スクリプト。

BSRGAN 劣化パイプライン（DatasetBlindSR）で LR を生成し、SR → PSNR を計算する。
評価方法は学習スクリプト（train_bsrgan_psnr.py）と同一のため、
ログに記録された PSNR 値と直接比較できる。

PSNR は RGB 全チャンネルの MSE から計算（学習スクリプトの evaluate_psnr と同一）。
劣化はランダムだが --seed で固定するため、異なるモデル間の公平な比較が可能。

使い方:
  # 公式 BSRNet で評価
  python scripts/eval_bsrnet_psnr.py \\
      --model BSRNet \\
      --testset testsets/custom_natural/pexels-cc0-100-2/

  # カスタムチェックポイントで評価（best.pth / last_E.pth / iter_*.pth 共通）
  python scripts/eval_bsrnet_psnr.py \\
      --checkpoint results/train_bsrgan_psnr/best.pth \\
      --testset testsets/custom_natural/pexels-cc0-100-2/

  # サブディレクトリを含めて評価
  python scripts/eval_bsrnet_psnr.py \\
      --model BSRNet --testset testsets/custom_natural/ --recursive

  # TSV 出力（別スクリプトでのパイプ処理向け）
  python scripts/eval_bsrnet_psnr.py \\
      --model BSRNet --testset testsets/custom_natural/pexels-cc0-100-2/ --tsv

  # seed を変えて複数回評価し平均をとる
  for s in 0 1 2; do
    python scripts/eval_bsrnet_psnr.py --model BSRNet \\
        --testset testsets/custom_natural/pexels-cc0-100-2/ --seed $s | tail -1
  done
"""

import argparse
import math
import os
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

NAMED_MODELS = {
    'BSRNet':   (os.path.join(KAIR_DIR, 'model_zoo', 'BSRNet.pth'),   4),
    'BSRGAN':   (os.path.join(KAIR_DIR, 'model_zoo', 'BSRGAN.pth'),   4),
    'BSRGANx2': (os.path.join(KAIR_DIR, 'model_zoo', 'BSRGANx2.pth'), 2),
}


def load_checkpoint(path, device):
    """state_dict-only と完全チェックポイント（netE / netG キー）の両方を受け付ける。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'netE' in ckpt:
        return ckpt['netE'], ckpt.get('step'), ckpt.get('best_psnr')
    if isinstance(ckpt, dict) and 'netG' in ckpt:
        return ckpt['netG'], ckpt.get('step'), None
    return ckpt, None, None


def build_model(state_dict, scale, device):
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def evaluate(model, loader, device, seed):
    """train_bsrgan_psnr.py の evaluate_psnr と同一の評価ロジック。"""
    rng_state = np.random.get_state()
    np.random.seed(seed)

    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            L = batch['L'].to(device)
            H = batch['H'].to(device)
            pred = model(L).clamp(0.0, 1.0)
            mse = F.mse_loss(pred, H).item()
            psnr = 10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0
            name = os.path.basename(loader.dataset.paths_H[i])
            results.append((name, psnr))

    np.random.set_state(rng_state)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='BSRNet / BSRGAN PSNR 評価',
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
                        help='HR 画像のディレクトリ')
    parser.add_argument('--recursive', action='store_true',
                        help='サブディレクトリも含める（デフォルト: top-level のみ）')
    parser.add_argument('--seed', type=int, default=0,
                        help='劣化パイプラインの乱数シード（デフォルト: 0）')
    parser.add_argument('--lq_patchsize', type=int, default=72,
                        help='LQ パッチサイズ（デフォルト: 72、学習設定 train_bsrgan_x4_psnr_unsplash.json と同値）')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='チャンネル数（デフォルト: 3）')
    parser.add_argument('--scale', type=int, default=None,
                        help='スケール倍率（--model 指定時は自動設定; --checkpoint 時のデフォルト: 4）')
    parser.add_argument('--cpu', action='store_true',
                        help='GPU が使えない環境や動作確認で CPU 推論を強制する')
    parser.add_argument('--tsv', action='store_true',
                        help='TSV 形式で出力（ヘッダー付き、パイプ処理向け）')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    # --- モデルのロード ---
    if args.model:
        model_path, scale = NAMED_MODELS[args.model]
        if not os.path.isfile(model_path):
            print(f'Error: weight not found: {model_path}', file=sys.stderr)
            sys.exit(1)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        label = f'{args.model} (official)'
    else:
        ckpt_path = (args.checkpoint if os.path.isabs(args.checkpoint)
                     else os.path.join(ROOT, args.checkpoint))
        if not os.path.isfile(ckpt_path):
            print(f'Error: checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)
        state_dict, step, best_psnr = load_checkpoint(ckpt_path, device)
        scale = args.scale or 4
        label = os.path.basename(ckpt_path)
        if step is not None:
            label += f'  step={step}'
            if best_psnr is not None:
                label += f'  best_train_psnr={best_psnr:.2f} dB'

    if args.scale:
        scale = args.scale

    model = build_model(state_dict, scale, device)

    # --- テストセットのロード ---
    testset_dir = os.path.normpath(
        args.testset if os.path.isabs(args.testset) else os.path.join(ROOT, args.testset)
    )
    if not os.path.isdir(testset_dir):
        print(f'Error: testset not found: {testset_dir}', file=sys.stderr)
        sys.exit(1)

    ds_opt = {
        'phase': 'test',
        'n_channels': args.n_channels,
        'scale': scale,
        'shuffle_prob': 0.1,
        'use_sharp': False,
        'degradation_type': 'bsrgan',
        'lq_patchsize': args.lq_patchsize,
        'H_size': args.lq_patchsize * scale,
        'dataroot_H': testset_dir,
    }
    dataset = DatasetBlindSR(ds_opt)

    if not args.recursive:
        dataset.paths_H = [
            p for p in dataset.paths_H
            if os.path.dirname(os.path.normpath(p)) == testset_dir
        ]

    if not dataset.paths_H:
        print(f'Error: no images found in {testset_dir}', file=sys.stderr)
        print('  Use --recursive to include subdirectories.', file=sys.stderr)
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- ヘッダー（TSV 以外） ---
    if not args.tsv:
        print(f'Model:   {label}')
        print(f'Testset: {testset_dir}  [{len(dataset)} images]')
        print(f'Seed: {args.seed}  lq_patchsize: {args.lq_patchsize}  scale: x{scale}  device: {device}')
        print()

    # --- 評価 ---
    results = evaluate(model, loader, device, seed=args.seed)

    # --- 出力 ---
    if args.tsv:
        print('filename\tpsnr')
        for name, psnr in results:
            print(f'{name}\t{psnr:.4f}')
    else:
        col = max(len(n) for n, _ in results)
        for name, psnr in results:
            print(f'  {name:<{col}}  {psnr:.2f} dB')

    psnrs = [p for _, p in results]
    avg = float(np.mean(psnrs))

    if args.tsv:
        print(f'Average\t{avg:.4f}')
    else:
        print()
        print(f'Average PSNR: {avg:.2f} dB'
              f'  (min: {min(psnrs):.2f}  max: {max(psnrs):.2f}  n={len(psnrs)})')


if __name__ == '__main__':
    main()
