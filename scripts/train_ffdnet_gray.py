"""
FFDNet グレースケールモデルの学習スクリプト。

ランダムな sigma ([sigma_min, sigma_max]) でガウシアンノイズを on-the-fly に生成し、
FFDNet を L1 損失で学習する。公式 ffdnet_gray.pth からの fine-tuning または
ゼロからの学習に対応。

【使い方】
  # 動作確認（200 iters）
  python scripts/train_ffdnet_gray.py \
      --config options/train_ffdnet_gray_unsplash.json \
      --max_iters 200 --datasets unsplash_lite

  # 本番実行（500k iters、Unsplash Lite）
  python scripts/train_ffdnet_gray.py \
      --config options/train_ffdnet_gray_unsplash.json \
      --datasets unsplash_lite

  # 公式重みから fine-tuning
  python scripts/train_ffdnet_gray.py \
      --config options/train_ffdnet_gray_unsplash.json \
      --datasets unsplash_lite \
      --pretrained models/KAIR/model_zoo/ffdnet_gray.pth

  # チェックポイントから再開（同じ config を使えばログは自動追記）
  python scripts/train_ffdnet_gray.py \
      --config options/train_ffdnet_gray_unsplash.json \
      --datasets unsplash_lite \
      --resume results/train_ffdnet_gray/iter_005000.pth

【学習完了後】
  cp results/train_ffdnet_gray/best.pth models/KAIR/model_zoo/ffdnet_gray_unsplash.pth
  python scripts/run_ffdnet.py --input test_inputs/ --sigma 25
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_ffdnet import FFDNet


# ---------------------------------------------------------------------------
# データ拡張（8パターン flip/rotate）
# ---------------------------------------------------------------------------

def random_augment(img):
    mode = random.randint(0, 7)
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.fliplr(img)
    elif mode == 3:
        return np.rot90(img, 1)
    elif mode == 4:
        return np.rot90(img, 2)
    elif mode == 5:
        return np.rot90(img, 3)
    elif mode == 6:
        return np.flipud(np.rot90(img, 1))
    else:
        return np.fliplr(np.rot90(img, 1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FFDNetGrayDataset(Dataset):
    """グレースケール画像のランダムパッチ + ランダム sigma ノイズ。

    data_dirs: str または list[str]。複数ディレクトリの画像を混合して使う。
    sigma_min/sigma_max は 0–255 スケール。forward 時に /255 して [0,1] に変換する。
    """

    def __init__(self, data_dirs, patch_size, sigma_min, sigma_max, phase='train'):
        dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        self.paths = []
        for d in dirs:
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                self.paths += glob.glob(os.path.join(d, ext))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f'No images found in {dirs}')
        self.patch_size = patch_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.phase = phase

    def __len__(self):
        return len(self.paths) * (50 if self.phase == 'train' else 1)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx % len(self.paths)]).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.phase == 'train':
            h, w = arr.shape
            if h < self.patch_size or w < self.patch_size:
                arr = np.pad(arr,
                             ((0, max(0, self.patch_size - h)),
                              (0, max(0, self.patch_size - w))),
                             mode='reflect')
                h, w = arr.shape
            rh = random.randint(0, h - self.patch_size)
            rw = random.randint(0, w - self.patch_size)
            clean = arr[rh:rh + self.patch_size, rw:rw + self.patch_size].copy()
            clean = np.ascontiguousarray(random_augment(clean))
            # バッチ内でアイテムごとに独立した sigma をサンプル
            sigma_val = random.uniform(self.sigma_min, self.sigma_max) / 255.0
        else:
            clean = arr
            sigma_val = self.sigma_max / 255.0  # テスト時は sigma_max で固定（呼び出し側で上書き）

        noise = np.random.randn(*clean.shape).astype(np.float32) * sigma_val
        noisy = (clean + noise).astype(np.float32)

        clean_t = torch.from_numpy(clean).unsqueeze(0)   # [1, H, W]
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)   # [1, H, W]
        sigma_t = torch.tensor([sigma_val], dtype=torch.float32)  # [1]
        return noisy_t, clean_t, sigma_t


# ---------------------------------------------------------------------------
# PSNR 評価
# ---------------------------------------------------------------------------

def evaluate_psnr(model, test_paths, patch_size, sigma_test, device, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    model.eval()
    sigma_val = sigma_test / 255.0
    psnrs = []
    with torch.no_grad():
        for path in test_paths:
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
            noise = np.random.randn(*arr.shape).astype(np.float32) * sigma_val
            noisy = (arr + noise).astype(np.float32)

            x = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
            s = torch.full((1, 1, 1, 1), sigma_val, dtype=torch.float32).to(device)
            pred = model(x, s).cpu().clamp(0.0, 1.0)
            clean_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            mse = F.mse_loss(pred, clean_t).item()
            psnrs.append(10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0)

    model.train()
    np.random.set_state(rng_state)
    return float(np.mean(psnrs))


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON config file path')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override total_iters (for quick test)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset subdirectory names under dataroot_H '
                             '(e.g. unsplash_lite). Default: use all subdirectories.')
    parser.add_argument('--pretrained', default=None,
                        help='Load pretrained weights as starting point (state_dict only)')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint (iter_XXXXXX.pth)')
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(config_path) as f:
        opt = json.load(f)

    total_iters = args.max_iters if args.max_iters is not None else opt['total_iters']
    output_dir = os.path.join(ROOT, opt['output_dir']) if not os.path.isabs(opt['output_dir']) else opt['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    log_path = os.path.join(output_dir, f'{config_stem}.log')

    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}]  {msg}'
        print(line)
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(line + '\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Device: {device}')
    log(f'Config: {config_stem}')
    log(f'Total iters: {total_iters}  sigma=[{opt["sigma_min"]},{opt["sigma_max"]}]'
        f'  sigma_test={opt["sigma_test"]}  lr={opt["lr"]}')
    log(f'Output: {output_dir}')
    log(f'Log: {log_path}')

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    base_train_dir = abs_path(opt['dataroot_H'])
    test_dir = abs_path(opt['dataroot_test'])

    if args.datasets:
        train_dirs = [os.path.join(base_train_dir, name) for name in args.datasets]
        missing = [d for d in train_dirs if not os.path.isdir(d)]
        if missing:
            log(f'Error: dataset directories not found: {missing}')
            sys.exit(1)
        train_dir = train_dirs
    else:
        train_dir = base_train_dir
    log(f'Train data: {train_dir}')
    log(f'Test  data: {test_dir}')

    train_set = FFDNetGrayDataset(train_dir, patch_size=opt['H_size'],
                                  sigma_min=opt['sigma_min'], sigma_max=opt['sigma_max'],
                                  phase='train')
    test_paths = sorted(
        sum([glob.glob(os.path.join(test_dir, ext)) for ext in ('*.png', '*.jpg', '*.bmp')], [])
    )
    if not test_paths:
        raise FileNotFoundError(f'No test images found in {test_dir}')
    log(f'Train images: {len(train_set.paths)}  Test images: {len(test_paths)}')

    train_loader = DataLoader(train_set, batch_size=opt['batch_size'],
                              shuffle=True, num_workers=opt['num_workers'],
                              drop_last=True, pin_memory=(device.type == 'cuda'))

    # FFDNet grayscale: in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'
    # — 公式 ffdnet_gray.pth と一致する唯一の設定
    model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R').to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt['lr'])
    scheduler = MultiStepLR(optimizer, milestones=opt['lr_milestones'], gamma=opt['lr_gamma'])

    start_step = 0
    best_psnr = 0.0

    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
        log(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        log(f'  Resumed at step={start_step}  best_psnr={best_psnr:.2f}')
    elif args.pretrained:
        pretrained_path = args.pretrained if os.path.isabs(args.pretrained) else os.path.join(ROOT, args.pretrained)
        log(f'Loading pretrained weights: {pretrained_path}')
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False), strict=True)
        log('  Pretrained weights loaded.')
    else:
        log('Training from scratch (random init).')

    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()

    log(f'--- Training start (step {start_step} → {total_iters}) ---')

    for step in range(start_step, total_iters):
        try:
            noisy, clean, sigma_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy, clean, sigma_batch = next(train_iter)

        noisy = noisy.to(device)
        clean = clean.to(device)
        # sigma: [B,1] → [B,1,1,1] として FFDNet に渡す
        sigma_map = sigma_batch.view(-1, 1, 1, 1).to(device)

        pred = model(noisy, sigma_map)
        loss = criterion(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if opt['checkpoint_print'] > 0 and step % opt['checkpoint_print'] == 0:
            elapsed = time.time() - t_start
            iters_done = step - start_step + 1
            eta = elapsed / iters_done * (total_iters - step - 1) if iters_done > 0 else 0
            log(f'[{step:6d}/{total_iters}]'
                f'  loss={loss.item():.4f}'
                f'  lr={scheduler.get_last_lr()[0]:.2e}'
                f'  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')

        if opt['checkpoint_test'] > 0 and step % opt['checkpoint_test'] == 0 and step > 0:
            psnr = evaluate_psnr(model, test_paths, opt['H_size'], opt['sigma_test'], device)
            model.train()
            flag = ' *** best ***' if psnr > best_psnr else ''
            log(f'  >> PSNR (σ={opt["sigma_test"]}, {len(test_paths)} imgs): {psnr:.2f} dB{flag}')
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))

        if opt['checkpoint_save'] > 0 and step % opt['checkpoint_save'] == 0 and step > 0:
            ckpt_path = os.path.join(output_dir, f'iter_{step:06d}.pth')
            torch.save({
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_path)
            log(f'  >> Checkpoint saved: {ckpt_path}')

    total_time = time.time() - t_start
    log(f'--- Done. Total time: {total_time/3600:.2f}h ---')

    if best_psnr == 0.0:
        log('Running final PSNR evaluation...')
        psnr = evaluate_psnr(model, test_paths, opt['H_size'], opt['sigma_test'], device)
        model.train()
        log(f'  >> PSNR (σ={opt["sigma_test"]}, {len(test_paths)} imgs): {psnr:.2f} dB')
        best_psnr = psnr
        torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
        log('  >> best.pth saved')

    log(f'Best PSNR: {best_psnr:.2f} dB')
    final_path = os.path.join(output_dir, f'final_iter{total_iters}.pth')
    torch.save(model.state_dict(), final_path)
    log(f'Final model: {final_path}')
    log(f'To use the trained model:')
    log(f'  cp {os.path.join(output_dir, "best.pth")} models/KAIR/model_zoo/ffdnet_gray_unsplash.pth')


if __name__ == '__main__':
    main()
