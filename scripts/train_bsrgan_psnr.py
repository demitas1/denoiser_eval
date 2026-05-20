"""
BSRGAN PSNR フェーズ学習スクリプト。

ランダム劣化（BSRGAN degradation pipeline）を on-the-fly で適用しながら
RRDBNet を L1 損失で学習する。GAN フェーズの前段として BSRNet.pth 相当の
モデルをゼロから生成する用途を想定。

【使い方】
  # 動作確認（100 iters）
  python scripts/train_bsrgan_psnr.py \
      --config options/train_bsrgan_x4_psnr_unsplash.json \
      --max_iters 100

  # 本番実行（500k iters、Unsplash Lite のみ）
  python scripts/train_bsrgan_psnr.py \
      --config options/train_bsrgan_x4_psnr_unsplash.json \
      --datasets unsplash_lite

  # チェックポイントから再開
  python scripts/train_bsrgan_psnr.py \
      --config options/train_bsrgan_x4_psnr_unsplash.json \
      --datasets unsplash_lite \
      --resume results/train_bsrgan_psnr/iter_001000.pth

【学習完了後】
  cp results/train_bsrgan_psnr/last_E.pth models/KAIR/model_zoo/BSRNet_unsplash.pth
  # GAN フェーズの出発点として使う場合:
  # options/train_bsrgan_x4_gan_finetune.json の pretrained_netG を上記パスに変更
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAIR_DIR = os.path.join(ROOT, 'models', 'KAIR')
sys.path.insert(0, KAIR_DIR)

from models.network_rrdbnet import RRDBNet
from data.dataset_blindsr import DatasetBlindSR


def evaluate_psnr(netE, test_loader, device, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)

    netE.eval()
    psnrs = []
    with torch.no_grad():
        for batch in test_loader:
            L = batch['L'].to(device)
            H = batch['H'].to(device)
            pred = netE(L).clamp(0.0, 1.0)
            mse = F.mse_loss(pred, H).item()
            psnrs.append(10.0 * math.log10(1.0 / mse) if mse > 1e-10 else 100.0)

    np.random.set_state(rng_state)
    return float(np.mean(psnrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='JSON config file path')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Override total_iters (for quick test)')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint (iter_XXXXXX.pth)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Dataset subdirectory names under dataroot_H to use '
                             '(e.g. unsplash_lite). Default: use all subdirectories.')
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
    print(f'Total iters: {total_iters}')
    print(f'Output: {output_dir}')

    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    base_train_dir = abs_path(opt['dataroot_H'])
    test_dir       = abs_path(opt['dataroot_test'])

    if args.datasets:
        train_dir = [os.path.join(base_train_dir, name) for name in args.datasets]
        missing = [d for d in train_dir if not os.path.isdir(d)]
        if missing:
            print(f'Error: dataset directories not found: {missing}')
            sys.exit(1)
        print(f'Train data: {train_dir}')
    else:
        train_dir = base_train_dir
        print(f'Train data: {train_dir} (all subdirectories)')
    print(f'Test  data: {test_dir}')

    ds_opt_train = {
        'phase': 'train',
        'n_channels': opt['n_channels'],
        'scale': opt['sf'],
        'shuffle_prob': 0.1,
        'use_sharp': False,
        'degradation_type': 'bsrgan',
        'lq_patchsize': opt['lq_patchsize'],
        'H_size': opt['H_size'],
        'dataroot_H': train_dir,
    }
    ds_opt_test = {
        'phase': 'test',
        'n_channels': opt['n_channels'],
        'scale': opt['sf'],
        'shuffle_prob': 0.1,
        'use_sharp': False,
        'degradation_type': 'bsrgan',
        'lq_patchsize': opt['lq_patchsize'],
        'H_size': opt['H_size'],
        'dataroot_H': test_dir,
    }

    train_set = DatasetBlindSR(ds_opt_train)
    test_set  = DatasetBlindSR(ds_opt_test)
    print(f'Train images: {len(train_set)}  Test images: {len(test_set)}')

    train_loader = DataLoader(
        train_set,
        batch_size=opt['batch_size'],
        shuffle=True,
        num_workers=opt['num_workers'],
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    netG = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=opt['sf']).to(device)
    netE = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=opt['sf']).to(device)
    netE.eval()

    pixel_loss = nn.L1Loss().to(device)
    E_decay = opt['E_decay']

    G_opt = Adam(netG.parameters(), lr=opt['G_lr'], betas=(0.9, 0.999))
    G_sch = MultiStepLR(G_opt, milestones=opt['G_milestones'], gamma=opt['lr_gamma'])

    start_step = 0
    best_psnr  = 0.0

    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(ROOT, args.resume)
        print(f'Resuming from {resume_path}')
        ckpt = torch.load(resume_path, map_location=device)
        netG.load_state_dict(ckpt['netG'])
        netE.load_state_dict(ckpt['netE'])
        G_opt.load_state_dict(ckpt['G_optimizer'])
        G_sch.load_state_dict(ckpt['G_scheduler'])
        start_step = ckpt['step'] + 1
        best_psnr  = ckpt.get('best_psnr', 0.0)
        print(f'  Resumed at step={start_step}  best_psnr={best_psnr:.2f}')
    else:
        netE.load_state_dict(netG.state_dict())
        print('Training from scratch (random init).')

    netG.train()
    train_iter = iter(train_loader)
    t_start = time.time()

    print(f'\n--- Training start (step {start_step} → {total_iters}) ---')

    for step in range(start_step, total_iters):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        L = batch['L'].to(device)
        H = batch['H'].to(device)

        E = netG(L)
        loss_G = pixel_loss(E, H)

        G_opt.zero_grad()
        loss_G.backward()
        G_opt.step()

        with torch.no_grad():
            for e_p, g_p in zip(netE.parameters(), netG.parameters()):
                e_p.data.mul_(E_decay).add_(g_p.data, alpha=1.0 - E_decay)
            for e_b, g_b in zip(netE.buffers(), netG.buffers()):
                e_b.copy_(g_b)

        G_sch.step()

        if opt['checkpoint_print'] > 0 and step % opt['checkpoint_print'] == 0:
            elapsed = time.time() - t_start
            iters_done = step - start_step + 1
            eta = elapsed / iters_done * (total_iters - step - 1) if iters_done > 0 else 0
            print(
                f'[{step:6d}/{total_iters}]'
                f'  loss={loss_G.item():.4f}'
                f'  lr={G_sch.get_last_lr()[0]:.2e}'
                f'  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m'
            )

        if opt['checkpoint_test'] > 0 and step % opt['checkpoint_test'] == 0 and step > 0:
            psnr = evaluate_psnr(netE, test_loader, device)
            netE.eval()
            netG.train()
            flag = ' *** best ***' if psnr > best_psnr else ''
            print(f'  >> PSNR (EMA, Set5): {psnr:.2f} dB{flag}')
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(netE.state_dict(), os.path.join(output_dir, 'best.pth'))

        if opt['checkpoint_save'] > 0 and step % opt['checkpoint_save'] == 0 and step > 0:
            ckpt_path = os.path.join(output_dir, f'iter_{step:06d}.pth')
            torch.save({
                'step': step,
                'netG': netG.state_dict(),
                'netE': netE.state_dict(),
                'G_optimizer': G_opt.state_dict(),
                'G_scheduler': G_sch.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_path)
            torch.save(netE.state_dict(), os.path.join(output_dir, 'last_E.pth'))
            print(f'  >> Checkpoint saved: {ckpt_path}')

    total_time = time.time() - t_start
    print(f'\n--- Done. Total time: {total_time/3600:.2f}h ---')

    torch.save(netE.state_dict(), os.path.join(output_dir, 'last_E.pth'))
    print(f'last_E.pth saved to {output_dir}')
    print(f'\nTo use as GAN pretrain:')
    print(f'  cp {os.path.join(output_dir, "best.pth")} models/KAIR/model_zoo/BSRNet_unsplash.pth')


if __name__ == '__main__':
    main()
