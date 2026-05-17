"""
SCUNet グレースケールモデル (sigma=10) の学習スクリプト。

【データ準備】（初回のみ）
  # BSD400 学習データ（GitHub から sparse clone、追加ツール不要）
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/smartboy110/denoising-datasets.git /tmp/bsd400_tmp
  cd /tmp/bsd400_tmp && git sparse-checkout set BSD400
  mkdir -p trainsets/trainH_BSD400
  cp BSD400/*.png trainsets/trainH_BSD400/
  cd /tmp && rm -rf bsd400_tmp

  # BSD68 検証データ（BSD400 と同じリポジトリの BSD68/original/）
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/smartboy110/denoising-datasets.git /tmp/bsd68_tmp
  cd /tmp/bsd68_tmp && git sparse-checkout set BSD68/original
  mkdir -p models/KAIR/testsets/bsd68
  cp BSD68/original/*.png models/KAIR/testsets/bsd68/
  cd /tmp && rm -rf bsd68_tmp

【使い方】
  # 試験実行（1k イテレーション、所要時間を計測）
  python scripts/train_scunet_gray.py \
      --config options/train_scunet_gray_finetune.json \
      --max_iters 1000

  # fine-tuning 本番実行（100k イテレーション、~6-8時間）
  python scripts/train_scunet_gray.py \
      --config options/train_scunet_gray_finetune.json

  # チェックポイントから再開
  python scripts/train_scunet_gray.py \
      --config options/train_scunet_gray_finetune.json \
      --resume results/train_scunet_gray/iter_010000.pth

  # フルトレーニング（ランダム初期化、300k イテレーション、~18-24時間）
  python scripts/train_scunet_gray.py \
      --config options/train_scunet_gray_full.json

【学習完了後】
  cp results/train_scunet_gray/best.pth models/SCUNet/model_zoo/scunet_gray_10.pth
  python scripts/run_scunet.py --input test_inputs/ --model scunet_gray_10 scunet_gray_15
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCUNET_DIR = os.path.join(ROOT, 'models', 'SCUNet')
sys.path.insert(0, SCUNET_DIR)
from models.network_scunet import SCUNet


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

class SCUNetGrayDataset(Dataset):
    def __init__(self, data_dir, patch_size, sigma, phase='train'):
        self.paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.paths += glob.glob(os.path.join(data_dir, ext))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise FileNotFoundError(f'No images found in {data_dir}')
        self.patch_size = patch_size
        self.sigma = sigma
        self.phase = phase

    def __len__(self):
        # train: 仮想的に大きい長さにして DataLoader を長く使い回す
        return len(self.paths) * (50 if self.phase == 'train' else 1)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx % len(self.paths)]).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.phase == 'train':
            h, w = arr.shape
            if h < self.patch_size or w < self.patch_size:
                # 小さい画像はパディング（BSD400 には該当なし）
                arr = np.pad(arr,
                             ((0, max(0, self.patch_size - h)),
                              (0, max(0, self.patch_size - w))),
                             mode='reflect')
                h, w = arr.shape
            rh = random.randint(0, h - self.patch_size)
            rw = random.randint(0, w - self.patch_size)
            clean = arr[rh:rh+self.patch_size, rw:rw+self.patch_size].copy()
            clean = np.ascontiguousarray(random_augment(clean))
        else:
            clean = arr

        noise = np.random.randn(*clean.shape).astype(np.float32) * (self.sigma / 255.0)
        noisy = (clean + noise).astype(np.float32)

        return (torch.from_numpy(noisy).unsqueeze(0),
                torch.from_numpy(clean).unsqueeze(0))


# ---------------------------------------------------------------------------
# PSNR 評価
# ---------------------------------------------------------------------------

def evaluate_psnr(model, test_set, device, sigma, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    model.eval()
    psnrs = []
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for noisy, clean in loader:
            pred = model(noisy.to(device)).cpu().clamp(0.0, 1.0)
            mse = F.mse_loss(pred, clean).item()
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
                        help='Override total_iters (for quick timing test)')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint .pth (optimizer state included)')
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(config_path) as f:
        opt = json.load(f)

    total_iters = args.max_iters if args.max_iters is not None else opt['total_iters']
    output_dir = os.path.join(ROOT, opt['output_dir']) if not os.path.isabs(opt['output_dir']) else opt['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Config: {config_path}')
    print(f'Total iters: {total_iters}  sigma={opt["sigma"]}  lr={opt["lr"]}')
    print(f'Output: {output_dir}')

    # --- データセット ---
    train_dir = os.path.join(ROOT, opt['dataroot_train']) if not os.path.isabs(opt['dataroot_train']) else opt['dataroot_train']
    test_dir  = os.path.join(ROOT, opt['dataroot_test'])  if not os.path.isabs(opt['dataroot_test'])  else opt['dataroot_test']

    print(f'Train data: {train_dir}')
    print(f'Test  data: {test_dir}')

    train_set = SCUNetGrayDataset(train_dir, patch_size=opt['patch_size'],
                                  sigma=opt['sigma'], phase='train')
    test_set  = SCUNetGrayDataset(test_dir,  patch_size=opt['patch_size'],
                                  sigma=opt['sigma_test'], phase='test')
    print(f'Train images: {len(train_set.paths)}  Test images: {len(test_set.paths)}')

    train_loader = DataLoader(train_set, batch_size=opt['batch_size'],
                              shuffle=True, num_workers=4,
                              drop_last=True, pin_memory=(device.type == 'cuda'))

    # --- モデル ---
    # config=[4,4,4,4,4,4,4] は公式 gray_15/25/50 重みと一致。変更不可。
    model = SCUNet(in_nc=1, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt['lr'])
    scheduler = MultiStepLR(optimizer, milestones=opt['lr_milestones'], gamma=opt['lr_gamma'])

    start_step = 0
    best_psnr = 0.0

    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f'  Resumed at step={start_step}  best_psnr={best_psnr:.2f}')
    elif opt.get('pretrained'):
        pretrained_path = opt['pretrained'] if os.path.isabs(opt['pretrained']) else os.path.join(ROOT, opt['pretrained'])
        print(f'Loading pretrained weights: {pretrained_path}')
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=True)
        print('  Pretrained weights loaded.')

    # --- 学習ループ ---
    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()
    t_log = t_start

    print(f'\n--- Training start (step {start_step} → {total_iters}) ---')

    for step in range(start_step, total_iters):
        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy, clean = next(train_iter)

        noisy = noisy.to(device)
        clean = clean.to(device)

        pred = model(noisy)
        loss = criterion(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            now = time.time()
            elapsed = now - t_start
            iters_done = step - start_step + 1
            iters_left = total_iters - step - 1
            eta = elapsed / iters_done * iters_left if iters_done > 0 else 0
            print(f'[{step:6d}/{total_iters}] loss={loss.item():.4f}'
                  f'  lr={scheduler.get_last_lr()[0]:.2e}'
                  f'  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')
            t_log = now

        # 検証 PSNR
        if opt['checkpoint_test'] > 0 and step % opt['checkpoint_test'] == 0 and step > 0:
            psnr = evaluate_psnr(model, test_set, device, opt['sigma_test'])
            print(f'  >> PSNR (σ={opt["sigma_test"]}, {len(test_set.paths)} imgs): {psnr:.2f} dB')
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
                print(f'  >> Best model saved ({best_psnr:.2f} dB)')

        # チェックポイント保存
        if opt['checkpoint_save'] > 0 and step % opt['checkpoint_save'] == 0 and step > 0:
            ckpt_path = os.path.join(output_dir, f'iter_{step:06d}.pth')
            torch.save({
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_path)
            print(f'  >> Checkpoint saved: {ckpt_path}')

    # --- 終了処理 ---
    total_time = time.time() - t_start
    print(f'\n--- Done. Total time: {total_time/3600:.2f}h ---')

    # ループ中に PSNR 評価が一度も走らなかった場合（--max_iters が小さいとき等）は終了時に実行
    if best_psnr == 0.0:
        print('Running final PSNR evaluation...')
        psnr = evaluate_psnr(model, test_set, device, opt['sigma_test'])
        print(f'  >> PSNR (σ={opt["sigma_test"]}, {len(test_set.paths)} imgs): {psnr:.2f} dB')
        best_psnr = psnr
        torch.save(model.state_dict(), os.path.join(output_dir, 'best.pth'))
        print(f'  >> best.pth saved')

    print(f'Best PSNR: {best_psnr:.2f} dB')

    # 最終モデルを保存（best と別に）
    final_path = os.path.join(output_dir, f'final_iter{total_iters}.pth')
    torch.save(model.state_dict(), final_path)
    print(f'Final model: {final_path}')
    print(f'\nTo use the trained model:')
    print(f'  cp {os.path.join(output_dir, "best.pth")} models/SCUNet/model_zoo/scunet_gray_10.pth')
    print(f'  python scripts/run_scunet.py --input test_inputs/ --model scunet_gray_10 scunet_gray_15')


if __name__ == '__main__':
    main()
