# denoiser_eval

鉛筆スケッチ→インク線画変換パイプライン向けの画像デノイザ・超解像モデルを研究するためのプロジェクト。既存のモデルの評価環境と独自データセットでのトレーニング環境を用意する。また、複数のモデルをパラメータスイープで一括実行し、出力画像を比較する。

詳細なモデル比較・選定の背景は [`docs/denoiser_setup_guide.md`](docs/denoiser_setup_guide.md) を参照。

---

## 利用可能なモデル（推論済み重みあり）

| モデル | スクリプト | 特性 |
|---|---|---|
| DnCNN | `run_dncnn.py` | グレースケールブラインドデノイザ |
| FFDNet | `run_ffdnet.py` | sigma 指定デノイザ（補正強度を直接調整可） |
| DRUNet | `run_drunet.py` | UNet ベース sigma 指定デノイザ（32M params、FFDNet より高精度） |
| Restormer | `run_restormer.py` | Transformer デノイザ、4タスク対応（実世界ノイズ / ガウシアン / モーションブラー / ピンボケ） |
| SCUNet | `run_scunet.py` | 実世界ブラインドデノイザ（カラー・グレースケール各強度） |
| BSRGAN | `run_esrgan.py` | ×4 超解像 GAN（実世界劣化に強い） |
| BSRNet | `run_esrgan.py` | ×4 超解像 PSNR（ハルシネーションリスクが低い安全版） |

重みのダウンロード方法・推論コマンドの詳細は [推論ガイド](docs/inference_guide.md) を参照。

### カスタム学習済みモデル（Unsplash Lite で再学習）

| ファイル | 内容 | PSNR |
|---|---|---|
| `ffdnet_gray_scratch_unsplash_lite.pth` | FFDNet gray、Unsplash Lite でゼロ学習、500k iters | 33.88 dB @ σ=25 |

`results/trained_models/` に保存、Git LFS で管理。

> **ライセンス注記**: Unsplash Lite の利用規約（Dataset Terms Section 2.A）では、学習・利用は**内部業務目的に限定**されます。外部配布・商用プロダクトへの組み込みは Unsplash への確認が必要です。

---

## 環境セットアップ（Linux）

### 前提

- Miniforge3（conda-forge）インストール済み
- NVIDIA GPU + CUDA 12.x ドライバ

### conda 環境の作成

```bash
conda create -n denoiser python=3.11 -y
conda activate denoiser

# pip が環境に含まれていない場合は先にインストール
conda install -n denoiser pip -y
```

### パッケージのインストール

```bash
# PyTorch（CUDA 12.4 ビルド。ドライバが 12.x なら動作する）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 共通ライブラリ
pip install numpy pillow opencv-python requests

# Restormer / SCUNet に必要な追加パッケージ
pip install einops gdown natsort lpips thop timm
```

### 動作確認

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## モデルリポジトリのセットアップ

```bash
cd /path/to/denoiser_eval

# KAIR（DnCNN・FFDNet・DRUNet・SwinIR 等）
git clone https://github.com/cszn/KAIR.git models/KAIR

# Restormer
git clone https://github.com/swz30/Restormer.git models/Restormer

# SCUNet
git clone https://github.com/cszn/SCUNet.git models/SCUNet
```

---

## クイックスタート（一括実行）

```bash
cd /path/to/denoiser_eval

# DnCNN / FFDNet / Restormer / SCUNet を一括実行
python scripts/run_all.py

# 入力ディレクトリを指定
python scripts/run_all.py --input test_inputs/

# CPU 推論（VRAM 不足時）
python scripts/run_all.py --cpu
```

出力は各モデルのデフォルト先（`results/DnCNN/`, `results/FFDNet/`, `results/Restormer/`, `results/SCUNet/`）に保存されます。

---

## ドキュメント

| ガイド | 内容 |
|---|---|
| [推論ガイド](docs/inference_guide.md) | 各モデルの重みダウンロード・実行コマンド・オプション一覧 |
| [学習ガイド](docs/training_guide.md) | Unsplash Lite でのカスタム学習手順（FFDNet / SCUNet / BSRNet / BSRGAN） |
| [従来型フィルターガイド](docs/traditional_filters_guide.md) | ML を使わない古典フィルターベースライン（Gaussian / Median / Bilateral / NL-Means） |
| [デノイザセットアップガイド](docs/denoiser_setup_guide.md) | モデル選定の経緯・NAFNet スキップ理由等 |
| [BSRGAN 学習プラン](docs/bsrgan_training_plan.md) | BSRGAN カスタム学習の設計・ライセンス・時間見積もり |

---

## ライセンス

本プロジェクトは **MIT License** で公開しています。詳細は [LICENSE.txt](LICENSE.txt) を参照してください。

本プロジェクトは [KAIR](https://github.com/cszn/KAIR)（MIT License, Copyright © 2019 Kai Zhang）のコードを利用しています。

> **データセットのライセンスについて**: 各学習データセット（Unsplash Lite 等）のライセンスは配布元で確認してください。Unsplash Lite は Unsplash Dataset Terms（内部業務目的に限定）が適用されます。
