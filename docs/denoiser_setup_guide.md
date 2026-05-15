# 一般画像デノイザ事前学習モデル 評価セットアップガイド

鉛筆スケッチ→インク線画変換プロジェクトのベースライン評価用。
Linux 環境前提、入力 1024×1024 グレースケール想定。

---

## モデル一覧

| モデル | 学習済み即利用 | 推論VRAM (1024²) | 学習VRAM (1024²) | グレー対応 | 推論時間 (RTX 3060) | 第一印象 |
|---|---|---|---|---|---|---|
| DnCNN | ✓ | 1〜2 GB | 4〜6 GB | ネイティブ | 約 0.1 秒 | 軽い・古典・実験用 |
| FFDNet | ✓ | 1〜2 GB | 4〜6 GB | ネイティブ | 約 0.1 秒 | 強度可変・実験用 |
| NAFNet (w32) | ✗ スキップ | — | — | — | — | CUDA 拡張ビルド失敗 |
| Restormer | ✓ | 6〜8 GB | 厳しい (12GB) | Real: 3ch, Gaussian Gray: ネイティブ | 約 1〜2 秒 | 重いが高品質 |
| SCUNet | ✓ | 4〜6 GB | 10〜12 GB | color: 3ch, gray_*: ネイティブ | 約 0.5 秒 | ブラインド設計・本命候補 |

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
│   ├── NAFNet/
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

最も軽量・実装単純。最初に試すのに最適。

**推奨実装**: SaoYan の PyTorch 移植版 (オリジナルは MATLAB/古い PyTorch)

```bash
cd ~/denoiser_eval/models
git clone https://github.com/SaoYan/DnCNN-PyTorch.git DnCNN
cd DnCNN

# 学習済みモデルはリポジトリ内 logs/ に同梱されている
# (DnCNN-S と DnCNN-B、それぞれ固定ノイズレベルとブラインド)
ls logs/
```

別案: KAIR 統合版 (より新しい重み、`dncnn_gray_blind.pth` 等)

```bash
cd ~/denoiser_eval/models
git clone https://github.com/cszn/KAIR.git
cd KAIR
python main_download_pretrained_models.py --models "DnCNN"
# model_zoo/ にダウンロードされる
```

**推論サンプル** (KAIR 版、グレースケール直接対応):

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

### 3. NAFNet

> **⚠ スキップ済み（2026-05-15）**
> NAFNet は独自改変版の basicsr を同梱しており、CUDA 拡張（deform_conv、fused_act）のコンパイルを伴うインストールが必要。
> `python setup.py develop --no_cuda_ext` はモダンな setuptools（PEP 517 経由）では `--no_cuda_ext` フラグが届かず、ソースが存在しない CUDA 拡張のビルドに失敗する。
> 回避策として sys.path 追加も検討したが、インストール不要で動作するか未検証のため、このサイクルの評価対象から除外。

実世界ノイズ (SIDD) で学習された SOTA 級モデル。

```bash
cd ~/denoiser_eval/models
git clone https://github.com/megvii-research/NAFNet.git
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

**事前学習済みモデルのダウンロード**:

公式は Google Drive で配布。リポジトリの README に Drive リンクあり。SIDD 用の `NAFNet-SIDD-width32.pth` と `NAFNet-SIDD-width64.pth` を取得。手動ダウンロードして `experiments/pretrained_models/` に配置。

```bash
mkdir -p experiments/pretrained_models
# ブラウザで Google Drive リンクからダウンロードして配置
# https://github.com/megvii-research/NAFNet#results-and-pre-trained-models
```

**推論サンプル** (公式 demo 利用):

```bash
python basicsr/demo.py \
  -opt options/test/SIDD/NAFNet-width32.yml \
  --input_path ~/denoiser_eval/test_inputs/sketch.png \
  --output_path ~/denoiser_eval/results/NAFNet/sketch_out.png
```

グレースケール画像を投入する場合は事前に 3ch に複製:

```python
from PIL import Image
img = Image.open('sketch_gray.png').convert('L')
img.convert('RGB').save('sketch_rgb.png')  # NAFNet に渡す
```

推論後、出力を再びグレースケール化:

```python
img = Image.open('output_rgb.png').convert('L')
img.save('output_gray.png')
```

**所要時間目安**: 環境構築 20分 + モデルダウンロード 5分 + 動作確認 10分

---

### 4. Restormer

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

### 5. SCUNet

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
| `scunet_color_real_gan` | 敵対学習。シャープだが線の捏造リスクあり |
| `scunet_gray_15/25/50` | グレースケール固定ノイズレベル（1ch ネイティブ） |

本タスクでは **PSNR 版から先に試す** のが妥当（捏造リスク低）。PSNR より強め・GAN より安全な中間が欲しい場合は gray 3強度を一括比較。

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

## 共通の評価スクリプト

各モデルの出力を統一的に比較するための共通スクリプトを推奨。

### 入出力前処理ユーティリティ

`~/denoiser_eval/scripts/io_utils.py`:

```python
import numpy as np
from PIL import Image

def load_gray_as_rgb(path):
    """グレースケール画像を読み込み、3ch RGB に複製"""
    img = Image.open(path).convert('L')
    return img.convert('RGB')

def rgb_to_gray(path_in, path_out):
    """RGB 出力をグレースケールに戻す"""
    img = Image.open(path_in).convert('L')
    img.save(path_out)

def load_gray_as_tensor(path, device='cuda'):
    """グレースケール画像を (1,1,H,W) テンソルに変換"""
    import torch
    img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

def tensor_to_gray(tensor, path):
    """(1,1,H,W) または (1,3,H,W) テンソルをグレースケール画像として保存"""
    arr = tensor.squeeze().cpu().numpy()
    if arr.ndim == 3:  # (3,H,W) -> グレースケール
        arr = arr.mean(axis=0)
    arr = np.clip(arr, 0, 1) * 255
    Image.fromarray(arr.astype(np.uint8)).save(path)
```

### 一括比較スクリプト

`~/denoiser_eval/scripts/compare_all.sh`:

```bash
#!/bin/bash
# すべてのモデルを同じ入力で実行し、results/ 配下に並べる

INPUT_DIR=~/denoiser_eval/test_inputs
RESULTS=~/denoiser_eval/results

# 各モデルの推論を呼び出す (各モデル個別のラッパースクリプトを用意)
python scripts/run_dncnn.py     --input $INPUT_DIR --output $RESULTS/DnCNN
python scripts/run_ffdnet.py    --input $INPUT_DIR --output $RESULTS/FFDNet --sigma 10 15 25
python scripts/run_restormer.py --input $INPUT_DIR --output $RESULTS/Restormer
python scripts/run_scunet.py    --input $INPUT_DIR --output $RESULTS/SCUNet \
  --model scunet_color_real_psnr scunet_gray_15 scunet_gray_25 scunet_gray_50

# 横並び画像を生成
python scripts/make_comparison_grid.py --inputs $INPUT_DIR --results $RESULTS
```

### 結果の可視化

`make_comparison_grid.py` では、各入力画像について「入力 + 5 モデルの出力」を横並びにした PNG を生成すると目視評価がしやすい。1024×1024 を 6 枚並べると 6144×1024 になるので、サムネイル化 (例: 512×512 に縮小) して並べるのが現実的。

---

## 推奨実行順序

**Day 1** (環境構築日):
1. conda 環境を作る
2. DnCNN + FFDNet (KAIR 経由で 2 つ同時取得)
3. 1〜2 枚のテスト画像で動作確認

**Day 2** (重い 2 つ):
1. SCUNet をセットアップして実行（gray 3強度も一括）
2. Restormer をセットアップして実行

**Day 3** (評価):
1. 5〜10 枚の代表的な鉛筆スケッチで全4モデルを実行
2. 横並び比較画像を生成
3. 「鉛筆ノイズがどの程度除去されるか」「線が痩せたり消えたりしないか」を目視評価

**評価の観点**:

- 紙のテクスチャ・粒状ノイズの除去度合い
- 消し跡 (線状ノイズ) の除去度合い
- 主線の保持度合い (太さ・濃淡)
- 線の捏造の有無 (Copainter の欠陥)
- 入り抜きの保持

---

## 期待される結果と次のステップ

おそらく結果は以下のようになります:

- **粒状ノイズはある程度除去される** (どのモデルもガウシアン的なノイズには反応)
- **消し跡 (線状ノイズ) は除去されない** (これは「ノイズ」ではなく「構造」として認識される)
- **主線の細部が失われる** (デノイザは「平滑化」傾向)

ここから先のステップ:

1. **最も筋の良いモデルを 1〜2 つ選定** (おそらく SCUNet か Restormer)
2. **マスク方式にファインチューニング**: 出力層を sigmoid + 1ch に差し替え、手続き的合成データで再学習
3. **損失関数を BCE + Dice + Perceptual (Simo-Serra 特徴) に変更**
4. **本格プロジェクト (Pix2Pix) との比較ベースラインとして位置づけ**

---

## トラブルシューティング

### CUDA Out of Memory

- Restormer の場合: `--tile 512`（デフォルト）でタイル推論。さらに厳しければ `--tile 256`
- SCUNet の場合: architecture 内部で 64px padding を行うため、大きい画像は単純に VRAM が足りなくなる。その場合は手動でタイル分割が必要
- バッチサイズを 1 に固定

### グレースケール 3ch 複製での性能劣化

- Restormer `Real_Denoising` と SCUNet color モデルはカラー3ch入力前提で学習されているが、グレースケールを 3ch 複製した場合でも性能はほぼ保たれる（各チャンネルが同じ情報のため）
- ネイティブ 1ch モデル（DnCNN, FFDNet, Restormer Gaussian Gray, SCUNet gray）との直接比較は条件が異なる点に注意

### Google Drive ダウンロードの失敗

- 公式リポジトリの README にあるリンクが切れていることがある
- 代替: Hugging Face Hub に同モデルがミラーされていることが多い (`huggingface.co/cszn` など)
- リポジトリの Issues セクションで代替リンクが提示されていることもある
