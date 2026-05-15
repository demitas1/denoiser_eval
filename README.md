# denoiser_eval

鉛筆スケッチ→インク線画変換プロジェクト用の画像デノイザ評価環境。
詳細なモデル比較・選定の背景は [`docs/denoiser_setup_guide.md`](docs/denoiser_setup_guide.md) を参照。

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
pip install numpy pillow opencv-python
```

### 動作確認

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

---

## モデルリポジトリのセットアップ

```bash
cd /path/to/denoiser_eval

# KAIR（DnCNN・FFDNet・SwinIR 等をまとめて管理）
git clone https://github.com/cszn/KAIR.git models/KAIR
```

---

## DnCNN の実行

### 1. モデル重みのダウンロード

```bash
cd models/KAIR
python main_download_pretrained_models.py --models "DnCNN"
# model_zoo/ に dncnn_gray_blind.pth 等がダウンロードされる
```

> `--models` の引数は大文字小文字を区別します。`DnCNN`・`FFDNet` のように正式表記で指定してください。また、ダウンロードスクリプトは `requests` を使用するため、未インストールの場合は先に `pip install requests` を実行してください。

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# test_inputs/ 内のすべての画像を処理
python scripts/run_dncnn.py --input test_inputs/ --output results/DnCNN

# KAIR 付属のサンプル画像で動作確認
python scripts/run_dncnn.py --input models/KAIR/testsets/set5/ --output results/DnCNN
```

出力は `results/DnCNN/<元のファイル名>_dncnn.png` に保存されます。

---

## FFDNet の実行

DnCNN の発展版。推論時にノイズレベル（sigma）を指定できるため、除去強度を調整可能。

### 1. モデル重みのダウンロード

```bash
cd models/KAIR
python main_download_pretrained_models.py --models "FFDNet"
# model_zoo/ に ffdnet_gray.pth 等がダウンロードされる
```

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# デフォルト: sigma = 5, 10, 15, 20, 25, 50 の6種を一括出力
python scripts/run_ffdnet.py --input test_inputs/ --output results/FFDNet

# sigma を絞って実行
python scripts/run_ffdnet.py --input test_inputs/ --output results/FFDNet --sigma 15 25
```

出力は `results/FFDNet/<元のファイル名>_ffdnet_s<sigma>.png` に保存されます。

### sigma の目安

| sigma | 効果 |
|---|---|
| 5–10 | 軽微なノイズのみ除去。線の保持が高い |
| 15–25 | DnCNN-S-15〜25 相当の除去強度 |
| 50 | 強めのノイズ除去。細い線が失われやすい |

鉛筆スケッチには **sigma=10 前後が良好**（現テストデータでの評価結果）。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/FFDNet` | 出力ディレクトリ |
| `--model` | `models/KAIR/model_zoo/ffdnet_gray.pth` | 重みファイルのパス |
| `--sigma` | `5 10 15 20 25 50` | ノイズレベル（複数指定可） |
| `--cpu` | off | GPU が使えない場合に CPU 推論を強制 |

利用可能な重み（`models/KAIR/model_zoo/`）:

| ファイル | 用途 |
|---|---|
| `dncnn_gray_blind.pth` | グレースケール・ブラインド（ノイズレベル不問） |
| `dncnn_15.pth` / `dncnn_25.pth` / `dncnn_50.pth` | グレースケール・固定ノイズレベル |
| `dncnn_color_blind.pth` | カラー・ブラインド |

### 入力画像の仕様

- **フォーマット**: PNG / JPG / JPEG / BMP / TIF（カラーも可、内部でグレースケール変換）
- **サイズ**: 制限なし（全畳み込みネットワーク）。RTX 3060 12GB で 3000×3000 前後まで一括処理可能

---

## Restormer の実行

Transformer ベースの高性能デノイザ。`Real_Denoising`（実世界ノイズ）と `Gaussian_Gray_Denoising`（グレースケールブラインド）の2タスクに対応。

### 1. モデルリポジトリのクローンと重みのダウンロード

```bash
cd /path/to/denoiser_eval

git clone https://github.com/swz30/Restormer.git models/Restormer

# 依存パッケージのインストール
pip install einops gdown natsort lpips

# 重みのダウンロード（gdown を使用）
# Real Denoising
gdown 1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH \
  -O models/Restormer/Denoising/pretrained_models/real_denoising.pth

# Gaussian Denoising（フォルダごと）
gdown --folder 1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0 \
  -O models/Restormer/Denoising/pretrained_models/
# ダウンロード後、ネストされた pretrained_models/ 内のファイルを親ディレクトリに移動:
# mv models/Restormer/Denoising/pretrained_models/pretrained_models/*.pth \
#    models/Restormer/Denoising/pretrained_models/
```

> `setup.py develop` は不要。`scripts/run_restormer.py` が `sys.path` で自動的に `models/Restormer` を参照します。

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# Real Denoising（実世界ノイズ、デフォルト）
python scripts/run_restormer.py --input test_inputs/ --output results/Restormer

# Gaussian Gray Denoising（グレースケールブラインド）
python scripts/run_restormer.py --input test_inputs/ --task Gaussian_Gray_Denoising
```

出力は `results/Restormer/<タスク名>/<元のファイル名>_restormer_<タスク>.png` に保存されます。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/Restormer` | 出力ディレクトリ |
| `--task` | `Real_Denoising` | `Real_Denoising` または `Gaussian_Gray_Denoising` |
| `--tile` | `512` | タイルサイズ（0 で無効化） |
| `--cpu` | off | CPU 推論を強制 |

---

## SCUNet の実行

実世界ブラインドデノイザ。多様な劣化を含む合成データで学習。`scunet_color_real_psnr`（PSNR版）と `scunet_color_real_gan`（GAN版）に加え、グレースケール固定ノイズレベルモデルも利用可能。

### 1. モデルリポジトリのクローンと重みのダウンロード

```bash
cd /path/to/denoiser_eval

git clone https://github.com/cszn/SCUNet.git models/SCUNet

# 依存パッケージのインストール
pip install thop timm

# 重みのダウンロード（GitHub releases から自動取得）
conda run -n denoiser python models/SCUNet/main_download_pretrained_models.py \
  --models "SCUNet" --model_dir models/SCUNet/model_zoo
```

> `setup.py` のインストールは不要。`scripts/run_scunet.py` が `sys.path` で自動的に `models/SCUNet` を参照します。

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# 実世界ノイズ（PSNR版、デフォルト）
python scripts/run_scunet.py --input test_inputs/ --output results/SCUNet

# 実世界ノイズ（GAN版）
python scripts/run_scunet.py --input test_inputs/ --model scunet_color_real_gan

# グレースケール3強度を一括出力
python scripts/run_scunet.py --input test_inputs/ --model scunet_gray_15 scunet_gray_25 scunet_gray_50

# 複数モデルを任意に組み合わせ
python scripts/run_scunet.py --input test_inputs/ --model scunet_color_real_psnr scunet_gray_25
```

出力は `results/SCUNet/<元のファイル名>_scunet_<モデル名>.png` に保存されます。

### モデルの選択指針

| モデル | 特性 |
|---|---|
| `scunet_color_real_psnr` | ピクセル誤差学習。安全だが線がぼやけがち |
| `scunet_color_real_gan` | 敵対学習。シャープだが線の捏造リスクあり |
| `scunet_gray_15` | グレースケール固定 sigma=15 相当（弱） |
| `scunet_gray_25` | グレースケール固定 sigma=25 相当（中） |
| `scunet_gray_50` | グレースケール固定 sigma=50 相当（強） |

鉛筆スケッチには **PSNR版から先に試す** のが妥当（捏造リスク低）。PSNR版より強め・GAN版より安全な中間が欲しい場合は `scunet_gray_*` の3強度を一括比較するとよい。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/SCUNet` | 出力ディレクトリ |
| `--model` | `scunet_color_real_psnr` | モデル名（複数指定可） |
| `--model_zoo` | `models/SCUNet/model_zoo` | 重みディレクトリ |
| `--cpu` | off | CPU 推論を強制 |
