# 一般画像デノイザ事前学習モデル 評価セットアップガイド

鉛筆スケッチ→インク線画変換プロジェクトのベースライン評価用。
Linux 環境前提、入力 1024×1024 グレースケール想定。

---

## モデル一覧

| モデル | 学習済み即利用 | 推論VRAM (1024²) | 学習VRAM (1024²) | グレー対応 | 推論時間 (RTX 3060) | 第一印象 |
|---|---|---|---|---|---|---|
| DnCNN | ✓ | 1〜2 GB | 4〜6 GB | ネイティブ | 約 0.1 秒 | 軽い・古典・実験用 |
| FFDNet | ✓ | 1〜2 GB | 4〜6 GB | ネイティブ | 約 0.1 秒 | 強度可変・実験用 |
| Restormer | ✓ | 6〜8 GB | 厳しい (12GB) | Real: 3ch, Gaussian Gray: ネイティブ | 約 1〜2 秒 | 重いが高品質 |
| SCUNet | ✓ | 4〜6 GB | 10〜12 GB | color: 3ch, gray_*: ネイティブ | 約 0.5 秒 | ブラインド設計・本命候補 |
| BSRGAN | ✓ | 4〜8 GB | — | 3ch | 約 2〜5 秒 | ×4 SR GAN・実世界劣化に強い |
| BSRNet | ✓ | 4〜8 GB | — | 3ch | 約 2〜5 秒 | ×4 SR PSNR・ハルシネーションリスク低 |

VRAM 数値は目安。バッチサイズ・モデルバリアントにより変動。

---

## 共通の前提条件

### システム要件

- OS: Ubuntu 22.04 LTS または 24.04 LTS (CUDA ドライバが整備されているもの)
- GPU: NVIDIA RTX 3060 12GB 以上
- CUDA: 11.8 または 12.1 (PyTorch のサポート版に合わせる)
- Python: 3.10 または 3.11 (3.12 は一部ライブラリで未対応の可能性)
- ディスク: 各モデル 100MB〜500MB、合計 5GB 程度の余裕

> **現環境メモ (Ubuntu 24.04 / RTX 3060 12GB)**
> - CUDA 12.8 インストール済み → PyTorch インストール時は `cu128` を指定: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
> - Python 3.12.3 → 主要ライブラリは 3.12 対応済みのため実用上問題ない見込み。エラー時は conda で 3.11 環境を作成

### 共通環境構築

すべてのモデルを 1 つの conda 環境で動かす運用を推奨。モデルごとに環境を分けると依存衝突を避けやすいが、最初は統一環境の方が回しやすい。

```bash
# Miniforge (conda の軽量版) で環境作成
conda create -n denoiser python=3.11 -y
conda activate denoiser

# PyTorch (CUDA 12.1 版)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 共通ライブラリ
pip install numpy pillow opencv-python scikit-image matplotlib tqdm scipy einops timm
```

### 作業ディレクトリ構成

```
~/denoiser_eval/
├── models/              # 各モデルのリポジトリと重み
│   ├── DnCNN/
│   ├── FFDNet/
│   ├── KAIR/          # DnCNN / FFDNet / BSRGAN / BSRNet 共用
│   ├── Restormer/
│   └── SCUNet/
├── test_inputs/         # 評価用の鉛筆スケッチ画像
├── results/             # 各モデルの出力
│   ├── DnCNN/
│   ├── FFDNet/
│   └── ...
└── scripts/             # 共通の推論・比較スクリプト
```

```bash
mkdir -p ~/denoiser_eval/{models,test_inputs,results,scripts}
cd ~/denoiser_eval
```

---

## モデル別セットアップ

### 1. DnCNN

最も軽量・実装単純。最初に試すのに最適。KAIR 統合版を使用（FFDNet / DRUNet / BSRGAN と同リポジトリ）。

```bash
cd ~/denoiser_eval/models
git clone https://github.com/cszn/KAIR.git
cd KAIR
python main_download_pretrained_models.py --models "DnCNN"
# model_zoo/ にダウンロードされる
```

**推論サンプル** (グレースケール直接対応):

```python
import torch
import numpy as np
from PIL import Image
from models.network_dncnn import DnCNN as net

device = torch.device('cuda')
model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
model.load_state_dict(torch.load('model_zoo/dncnn_gray_blind.pth'), strict=True)
model.eval().to(device)

img = np.array(Image.open('input.png').convert('L'), dtype=np.float32) / 255.0
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

with torch.no_grad():
    y = model(x)

out = y.squeeze().cpu().numpy().clip(0, 1) * 255
Image.fromarray(out.astype(np.uint8)).save('output.png')
```

**所要時間目安**: 環境構築 10分 + 動作確認 5分

---

### 2. FFDNet

DnCNN の発展版。ノイズレベルマップを推論時に与えられる。

**実装**: KAIR 統合版を推奨 (オリジナル `cszn/FFDNet` でも可)

```bash
cd ~/denoiser_eval/models/KAIR  # DnCNN と同じリポジトリ
python main_download_pretrained_models.py --models "FFDNet"
# model_zoo/ffdnet_gray.pth, ffdnet_color.pth が得られる
```

**推論サンプル** (グレースケール、ノイズレベル可変):

```python
import torch
import numpy as np
from PIL import Image
from models.network_ffdnet import FFDNet as net

device = torch.device('cuda')
model = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
model.load_state_dict(torch.load('model_zoo/ffdnet_gray.pth'), strict=True)
model.eval().to(device)

img = np.array(Image.open('input.png').convert('L'), dtype=np.float32) / 255.0
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

# ノイズレベル (0-255 スケールでの sigma、ここでは 25 を試す)
noise_level = 25
sigma = torch.full((1, 1, 1, 1), noise_level / 255.0).to(device)

with torch.no_grad():
    y = model(x, sigma)

out = y.squeeze().cpu().numpy().clip(0, 1) * 255
Image.fromarray(out.astype(np.uint8)).save('output.png')
```

**評価のコツ**: ノイズレベル sigma を 5, 15, 25, 50, 75 と振って試すと、鉛筆ノイズに対する効きの強さの選択肢が見える。

**所要時間目安**: DnCNN ができていれば追加 5 分

---

### 3. Restormer

Transformer ベース、最大級の品質期待。ただし重い。

**セットアップ**: `setup.py` のインストールは不要。`scripts/run_restormer.py` が `sys.path` で自動的に `models/Restormer` を参照する（basicsr の DCN 依存はコメントアウト済みで import 可能）。

```bash
cd ~/denoiser_eval
git clone https://github.com/swz30/Restormer.git models/Restormer

# 依存パッケージ
pip install einops gdown natsort lpips

# 重みのダウンロード
# Real Denoising
gdown 1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH \
  -O models/Restormer/Denoising/pretrained_models/real_denoising.pth

# Gaussian Denoising（フォルダごと）
gdown --folder 1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0 \
  -O models/Restormer/Denoising/pretrained_models/
# gdown はネストした pretrained_models/pretrained_models/ に展開されるため手動で移動:
mv models/Restormer/Denoising/pretrained_models/pretrained_models/*.pth \
   models/Restormer/Denoising/pretrained_models/
```

**推論**:

```bash
cd ~/denoiser_eval

# Real Denoising（実世界ノイズ、デフォルト）
python scripts/run_restormer.py --input test_inputs/ --output results/Restormer

# Gaussian Gray Denoising（グレースケールブラインド、1ch ネイティブ）
python scripts/run_restormer.py --input test_inputs/ --task Gaussian_Gray_Denoising
```

出力は `results/Restormer/<タスク名>/<元のファイル名>_restormer_<タスク>.png`。
タイルサイズはデフォルト 512 で VRAM を節約済み（1024²: 6〜8GB）。

**所要時間目安**: クローン 5分 + 重みダウンロード 5分 + 動作確認 10分

---

### 4. SCUNet

実世界ブラインドデノイザ。多様な劣化を含む合成データで学習。鉛筆ノイズに最も「なんとなく効きそう」な候補。

**セットアップ**: `setup.py` のインストールは不要。`scripts/run_scunet.py` が `sys.path` で自動的に `models/SCUNet` を参照する。依存: `thop`, `timm`。

```bash
cd ~/denoiser_eval
git clone https://github.com/cszn/SCUNet.git models/SCUNet

pip install thop timm

# 重みのダウンロード
python models/SCUNet/main_download_pretrained_models.py \
  --models "SCUNet" --model_dir models/SCUNet/model_zoo
# model_zoo/ に scunet_color_real_psnr.pth, scunet_color_real_gan.pth,
#   scunet_gray_15/25/50.pth, scunet_color_15/25/50.pth が得られる
```

**モデルの性質**:

| モデル | 特性 |
|---|---|
| `scunet_color_real_psnr` | ピクセル誤差学習。安全だが線がぼやけがち |
| `scunet_color_real_gan` | 敵対学習。シャープだが線のハルシネーションリスクあり |
| `scunet_gray_15/25/50` | グレースケール固定ノイズレベル（1ch ネイティブ） |

本タスクでは **PSNR 版から先に試す** のが妥当（ハルシネーションリスク低）。PSNR より強め・GAN より安全な中間が欲しい場合は gray 3強度を一括比較。

**推論**:

```bash
cd ~/denoiser_eval

# 実世界ノイズ（PSNR版、デフォルト）
python scripts/run_scunet.py --input test_inputs/ --output results/SCUNet

# 実世界ノイズ（GAN版）
python scripts/run_scunet.py --input test_inputs/ --model scunet_color_real_gan

# グレースケール3強度を一括出力
python scripts/run_scunet.py --input test_inputs/ --model scunet_gray_15 scunet_gray_25 scunet_gray_50
```

出力は `results/SCUNet/<元のファイル名>_scunet_<モデル名>.png`。

**所要時間目安**: クローン 5分 + 重みダウンロード 5分 + 動作確認 10分

---

### 5. BSRGAN / BSRNet

×4 超解像モデル。KAIR リポジトリ（DnCNN / FFDNet と共用）に含まれる。

- **BSRGAN**: GAN 学習。実世界の多様な劣化に強いが、ハルシネーションリスクあり。
- **BSRNet**: PSNR 学習。安全だが出力がやや平滑化される。最初に試す場合は BSRNet を推奨。

```bash
cd ~/denoiser_eval/models/KAIR
python main_download_pretrained_models.py --models "BSRGAN"
# model_zoo/ に BSRGAN.pth, BSRNet.pth, BSRGANx2.pth が得られる
```

**推論**:

```bash
cd ~/denoiser_eval

# BSRGAN x4（デフォルト）
# → <basename>_BSRGAN_x4.png（4096²）と <basename>_BSRGAN_lanczos.png（元サイズ）を保存
python scripts/run_esrgan.py --input test_inputs/ --output results/ESRGAN

# BSRNet x4
python scripts/run_esrgan.py --input test_inputs/ --model BSRNet

# 複数モデルを一括実行
python scripts/run_esrgan.py --input test_inputs/ --model BSRGAN BSRNet
```

**タイル推論**: `--tile 512`（デフォルト ON）。1024² 入力 → 4096² 出力のため VRAM 4〜8 GB 使用。  
**ダウンスケール**: `--downscale lanczos`（デフォルト）で元サイズの比較用画像も同時保存。`--downscale none` で省略可。

**所要時間目安**: KAIR クローン済みなら追加 5分（重みダウンロードのみ）

---

## 一括実行

`scripts/run_all.py` で全モデルを一括実行できる。

```bash
python scripts/run_all.py --input test_inputs/
```

---

## トラブルシューティング

### CUDA Out of Memory

- Restormer の場合: `--tile 512`（デフォルト）でタイル推論。さらに厳しければ `--tile 256`
- SCUNet の場合: `--tile 512`（デフォルト）でタイル推論。さらに厳しければ `--tile 256`
- バッチサイズを 1 に固定

### グレースケール 3ch 複製での性能劣化

- Restormer `Real_Denoising` と SCUNet color モデルはカラー3ch入力前提で学習されているが、グレースケールを 3ch 複製した場合でも性能はほぼ保たれる（各チャンネルが同じ情報のため）
- ネイティブ 1ch モデル（DnCNN, FFDNet, Restormer Gaussian Gray, SCUNet gray）との直接比較は条件が異なる点に注意

### データセットなどのダウンロードの失敗

- 公式リポジトリの README にあるリンクが切れていることがある。信頼できるミラーサイトを探す。
