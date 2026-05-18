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

## 一括実行

DnCNN / FFDNet / Restormer（全タスク）/ SCUNet（主要4モデル）を1コマンドで実行します。

```bash
cd /path/to/denoiser_eval

python scripts/run_all.py
```

完了後にサマリーテーブル（各モデルの処理時間・成否）が表示されます。

| オプション | 説明 |
|---|---|
| `--input` | 入力ディレクトリ（デフォルト: `test_inputs/`） |
| `--cpu` | 全モデルを CPU 推論で実行 |

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

# Motion Deblurring（モーションブラー除去）
python scripts/run_restormer.py --input test_inputs/ --task Motion_Deblurring

# Defocus Deblurring（ピンボケ除去）
python scripts/run_restormer.py --input test_inputs/ --task Defocus_Deblurring
```

出力は `results/Restormer/<タスク名>/<元のファイル名>_restormer_<タスク>.png` に保存されます。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/Restormer` | 出力ディレクトリ |
| `--task` | `Real_Denoising` | `Real_Denoising`, `Gaussian_Gray_Denoising`, `Motion_Deblurring`, `Defocus_Deblurring` |
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

# 実世界ノイズ（GAN版）※ 敵対学習により出力がシャープになるが、存在しない線の捏造（hallucination）が起こりやすい
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
| `--tile` | `512` | タイルサイズ（0 で無効化）。大画像で VRAM 不足の場合は小さくする |
| `--cpu` | off | CPU 推論を強制 |

---

## ESRGAN / BSRGAN の実行

超解像モデル。入力画像を x2 または x4 にアップスケールする。デフォルトでアップスケール済み画像に加えて元サイズへ LANCZOS でダウンスケールした画像も保存されるため、デノイザ出力と直接比較しやすい。

### 1. モデル重みのダウンロード

```bash
cd models/KAIR

# BSRGAN / BSRNet / BSRGANx2
python main_download_pretrained_models.py --models "BSRGAN"

# ESRGAN（"others" キーに含まれる。他のモデルも一緒にダウンロードされる）
python main_download_pretrained_models.py --models "others"
```

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# BSRGAN x4（GAN版、実世界劣化に強い、デフォルト）
# → sketch_BSRGAN_x4.png（4096²）と sketch_BSRGAN_lanczos.png（元サイズ）の2ファイルが生成される
python scripts/run_esrgan.py --input test_inputs/ --output results/ESRGAN

# BSRNet x4（PSNR版、GAN版より安全）
python scripts/run_esrgan.py --input test_inputs/ --model BSRNet

# 複数モデルを一括実行
python scripts/run_esrgan.py --input test_inputs/ --model BSRGAN BSRNet

# ESRGAN x4（古典的 GAN 版）
python scripts/run_esrgan.py --input test_inputs/ --model ESRGAN

# BSRGANx2（×2 アップスケール）
python scripts/run_esrgan.py --input test_inputs/ --model BSRGANx2

# アップスケール済み画像のみ保存（ダウンスケールしない）
python scripts/run_esrgan.py --input test_inputs/ --downscale none
```

出力は `results/ESRGAN/` に保存されます。ファイル名の例（BSRGAN x4、元サイズ 1024²）:

| ファイル名 | サイズ | 説明 |
|---|---|---|
| `sketch_BSRGAN_x4.png` | 4096² | アップスケール済み |
| `sketch_BSRGAN_lanczos.png` | 1024²（元サイズ） | LANCZOS でダウンスケール済み |

### モデルの選択指針

| モデル | スケール | 学習ロス | 特性 |
|---|---|---|---|
| `BSRGAN` | ×4 | GAN | 実世界劣化合成データで学習。シャープだが捏造リスクあり |
| `BSRNet` | ×4 | PSNR (L1) | BSRGAN の PSNR 版。安全だが出力がやや滑らか |
| `BSRGANx2` | ×2 | GAN | BSRGAN の ×2 版 |
| `ESRGAN` | ×4 | GAN | 古典的 ESRGAN。鮮明だが BSRGAN より実世界劣化への汎化が弱い |

鉛筆スケッチには **BSRNet から先に試す** のが妥当（捏造リスク低）。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/ESRGAN` | 出力ディレクトリ |
| `--model` | `BSRGAN` | モデル名（複数指定可） |
| `--model_zoo` | `models/KAIR/model_zoo` | 重みディレクトリ |
| `--tile` | `512` | タイルサイズ（0 で無効化）。大画像で VRAM 不足の場合は小さくする |
| `--downscale` | `lanczos` | アップスケール後に元サイズへ戻すアルゴリズム。`lanczos` / `bicubic` / `bilinear` / `nearest` / `none`（無効化） |
| `--cpu` | off | CPU 推論を強制 |

> **VRAM メモ（RTX 3060 12GB）**: 1024² 入力を x4 すると出力が 4096² になるため、`--tile 512` のタイル推論を推奨。OOM 発生時は該当画像をスキップして処理を継続する。

---

## SCUNet gray sigma=10 モデルの学習

公式配布モデルは sigma=15/25/50 の 3 種のみ。鉛筆スケッチに最適な sigma=10 相当のモデルを `scunet_gray_15` から fine-tuning で作成する手順です。

### 1. データ準備（初回のみ）

BSD400（学習用 400 枚）と BSD68（検証用 68 枚）を同一リポジトリから取得します。

```bash
cd /path/to/denoiser_eval

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/smartboy110/denoising-datasets.git /tmp/ds_tmp
cd /tmp/ds_tmp && git sparse-checkout set BSD400 BSD68/original
mkdir -p trainsets/trainH_BSD400 models/KAIR/testsets/bsd68
cp BSD400/*.png trainsets/trainH_BSD400/
cp BSD68/original/*.png models/KAIR/testsets/bsd68/
cd /tmp && rm -rf ds_tmp
```

### 2. 試験実行（所要時間の確認）

1,000 イテレーション（約 7 分）だけ実行して速度を確認します。

```bash
cd /path/to/denoiser_eval

python scripts/train_scunet_gray.py \
    --config options/train_scunet_gray_finetune.json \
    --max_iters 1000
```

表示される `eta=` の値から本番実行の所要時間を見積もれます（RTX 3060 で 100k iters ≈ 11 時間）。

### 3. 本番 fine-tuning

```bash
# 100k イテレーション（約 11 時間、放置実行）
python scripts/train_scunet_gray.py \
    --config options/train_scunet_gray_finetune.json

# 中断後の再開
python scripts/train_scunet_gray.py \
    --config options/train_scunet_gray_finetune.json \
    --resume results/train_scunet_gray/iter_010000.pth
```

学習中は 5,000 イテレーションごとに BSD68 で PSNR を評価し、最良モデルを `results/train_scunet_gray/best.pth` に保存します。

### 4. 学習済みモデルの配置と推論

```bash
cp results/train_scunet_gray/best.pth models/SCUNet/model_zoo/scunet_gray_10.pth

# gray_10 と gray_15 を並べて比較
python scripts/run_scunet.py \
    --input test_inputs/ \
    --model scunet_gray_10 scunet_gray_15
```

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--config` | （必須） | JSON 設定ファイルのパス |
| `--max_iters` | — | イテレーション数を上書き（試験実行用） |
| `--resume` | — | チェックポイントから再開（optimizer 状態も復元） |

利用可能な設定ファイル:

| ファイル | 用途 |
|---|---|
| `options/train_scunet_gray_finetune.json` | sigma=10, lr=5e-5, 100k iters（**推奨**） |
| `options/train_scunet_gray_full.json` | sigma=10, lr=1e-4, 300k iters（ランダム初期化） |
