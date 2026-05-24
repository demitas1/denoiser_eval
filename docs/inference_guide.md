# 推論ガイド

学習済み重みを使って各モデルで推論を実行する手順です。

---

## 一括実行（推奨）

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

## DnCNN

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

利用可能な重み（`models/KAIR/model_zoo/`）:

| ファイル | 用途 |
|---|---|
| `dncnn_gray_blind.pth` | グレースケール・ブラインド（ノイズレベル不問） |
| `dncnn_15.pth` / `dncnn_25.pth` / `dncnn_50.pth` | グレースケール・固定ノイズレベル |
| `dncnn_color_blind.pth` | カラー・ブラインド |

---

## FFDNet

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

---

## DRUNet

FFDNet より強力な U-Net バックボーン（32M params）。sigma マップを channel concat で入力するため、sigma 値が補正強度の直接的なつまみとなる。グレースケール専用。

### 1. モデル重みのダウンロード

```bash
cd models/KAIR
python main_download_pretrained_models.py --models "DPIR"
# model_zoo/ に drunet_gray.pth 等がダウンロードされる
```

### 2. 推論の実行

```bash
cd /path/to/denoiser_eval

# デフォルト: sigma = 5, 10, 15, 25, 50 の5種を一括出力
python scripts/run_drunet.py --input test_inputs/ --output results/DRUNet

# sigma を指定（大きいほど強くデノイズ）
python scripts/run_drunet.py --input test_inputs/ --sigma 10 25
```

出力は `results/DRUNet/<元のファイル名>_drunet_s<sigma:02d>.png` に保存されます。

### オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/DRUNet` | 出力ディレクトリ |
| `--model` | `models/KAIR/model_zoo/drunet_gray.pth` | 重みファイルのパス |
| `--sigma` | `5 10 15 25 50` | ノイズレベル（複数指定可） |
| `--cpu` | off | CPU 推論を強制 |

> **VRAM**: 1024² 推論で約 1.9 GB（RTX 3060 12GB では余裕あり）

---

## Restormer

Transformer ベースの高性能デノイザ。Real Denoising / Gaussian Gray Denoising / Motion Deblurring / Defocus Deblurring の4タスクに対応。

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

## SCUNet

実世界ブラインドデノイザ。多様な劣化を含む合成データで学習。カラー・グレースケール・実世界・固定ノイズレベルのモデルを選択できる。

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

## ESRGAN / BSRGAN

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

## カスタム学習済み重みでの推論

[学習ガイド](training_guide.md) で生成したチェックポイントを試験・評価するためのスクリプトです。`scripts/run_ffdnet_custom.py` / `scripts/run_bsrnet_custom.py` で学習途中のチェックポイントを使って推論を実行できます。

```bash
# FFDNet: best.pth（state_dict のみ）で sigma=10, 25 デノイズ
python scripts/run_ffdnet_custom.py \
    --checkpoint results/trained_models/ffdnet_gray_scratch_unsplash_lite.pth \
    --input test_inputs/ --sigma 10 25

# BSRNet: best.pth で推論 → ダウンスケール版のみ保存
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input test_inputs/ --output results/bsrnet_custom/

# x4 アップスケール版も追加保存
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input test_inputs/ --save_upscaled
```

---

## VRAM 目安（RTX 3060 12GB）

| モデル | 1024² VRAM | 備考 |
|---|---|---|
| DnCNN / FFDNet | 1–2 GB | 問題なし |
| DRUNet | 約 1.9 GB | 問題なし |
| SCUNet | 4–6 GB | `--tile 512`（デフォルト）で大画像対応 |
| Restormer | 6–8 GB | `--tile 512`（デフォルト）で大画像対応 |
| ESRGAN / BSRGAN | 4–8 GB | `--tile 512`（デフォルト）; 1024² 入力 → 4096² 出力（×4） |

全スクリプトに `torch.cuda.OutOfMemoryError` のキャッチを実装済み。OOM 時は該当画像をスキップし、残りの処理を継続する。
