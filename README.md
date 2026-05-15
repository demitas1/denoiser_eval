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
| `--output` | `results/DnCNN` | 出力ディレクトリ |
| `--model` | `models/KAIR/model_zoo/dncnn_gray_blind.pth` | 重みファイルのパス |
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
