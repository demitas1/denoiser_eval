"""
DnCNN / FFDNet / Restormer / SCUNet を test_inputs/ に対して一括実行するスクリプト。

使い方:
  python scripts/run_all.py
  python scripts/run_all.py --input test_inputs/
  python scripts/run_all.py --cpu
"""

import argparse
import subprocess
import sys
import time
import os

SCRIPTS_DIR = os.path.dirname(__file__)

RESTORMER_TASKS = [
    'Real_Denoising',
    'Gaussian_Gray_Denoising',
    'Motion_Deblurring',
    'Defocus_Deblurring',
]

SCUNET_MODELS = [
    'scunet_color_real_psnr',
    'scunet_gray_15',
    'scunet_gray_25',
    'scunet_gray_50',
]


def build_jobs(input_path, cpu):
    cpu_flag = ['--cpu'] if cpu else []
    py = sys.executable
    jobs = []

    jobs.append((
        'DnCNN (gray blind)',
        [py, os.path.join(SCRIPTS_DIR, 'run_dncnn.py'),
         '--input', input_path, '--output', 'results/DnCNN'] + cpu_flag,
    ))

    jobs.append((
        'FFDNet (σ=5,10,15,20,25,50)',
        [py, os.path.join(SCRIPTS_DIR, 'run_ffdnet.py'),
         '--input', input_path, '--output', 'results/FFDNet',
         '--sigma', '5', '10', '15', '20', '25', '50'] + cpu_flag,
    ))

    for task in RESTORMER_TASKS:
        jobs.append((
            f'Restormer: {task}',
            [py, os.path.join(SCRIPTS_DIR, 'run_restormer.py'),
             '--input', input_path, '--task', task] + cpu_flag,
        ))

    jobs.append((
        'SCUNet: ' + ', '.join(SCUNET_MODELS),
        [py, os.path.join(SCRIPTS_DIR, 'run_scunet.py'),
         '--input', input_path, '--output', 'results/SCUNet',
         '--model'] + SCUNET_MODELS + cpu_flag,
    ))

    return jobs


def run_job(label, cmd):
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  {label}')
    print(sep)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    status = 'OK' if result.returncode == 0 else f'FAILED (code {result.returncode})'
    return elapsed, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='test_inputs/', help='Input image directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference for all models')
    args = parser.parse_args()

    jobs = build_jobs(args.input, args.cpu)
    results = []
    t_total = time.time()

    for label, cmd in jobs:
        elapsed, status = run_job(label, cmd)
        results.append((label, elapsed, status))

    total_elapsed = time.time() - t_total

    sep = '=' * 60
    print(f'\n{sep}')
    print('  Summary')
    print(sep)
    for label, elapsed, status in results:
        print(f'  {status:<10}  {elapsed:6.1f}s  {label}')
    print(sep)
    print(f'  Total: {total_elapsed:.1f}s')

    if any(s != 'OK' for _, _, s in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
