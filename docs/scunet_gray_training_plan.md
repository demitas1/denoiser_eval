# SCUNet グレースケール低強度モデル（sigma ≤ 10）学習計画

公式配布モデルは sigma = 15 / 25 / 50 の 3 種のみ。
sigma = 5〜10 相当の除去強度が必要になった場合の実装・学習手順をまとめる。

作成日: 2026-05-15

---

## 背景と目的

鉛筆スケッチへの適用評価において、`scunet_gray_15` は除去強度がやや強く細い線が減衰する傾向があった。
FFDNet の `sigma=10` と比較しても SCUNet のアーキテクチャ（Swin Transformer + Conv ハイブリッド）で
同等強度のモデルを得たい場合に本プランを実施する。

**前提判断**: 先に FFDNet `--sigma 5 10` および SCUNet gray_15 出力ブレンドで評価し、
それでも不十分と判断した段階で実施すること。

---

## 現状調査サマリ

| 項目 | 状況 |
|---|---|
| SCUNet 公式リポジトリに学習スクリプト | **なし**（推論スクリプトのみ） |
| KAIR に SCUNet 学習オプション JSON | **なし** |
| KAIR の `select_network.py` への SCUNet 登録 | **なし** |
| モデル定義 `network_scunet.py` | `models/SCUNet/models/` に存在 |
| 参照できる最近接の学習コード | `models/KAIR/data/dataset_dncnn.py`、`main_train_dncnn.py` |

**方針**: KAIR フレームワークへの統合は侵襲的なため、
このリポジトリに **自己完結した学習スクリプト** `scripts/train_scunet_gray.py` を新規作成する。

---

## 学習データの準備

### BSD400（推奨）

- Berkeley Segmentation Dataset の学習用 400 枚（BSDS500 の一部）
- 容量: 約 50 MB
- DnCNN・FFDNet・SCUNet gray 公式モデルがすべてこれで学習されている標準データセット
- ライセンス: 研究用途に無償公開

```bash
# UC Berkeley から直接ダウンロード
# https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS500data.tgz
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS500data.tgz
tar xf BSDS500data.tgz
# BSDS500/data/images/train/ の 200 枚 + test/ の 200 枚 = 400 枚を使用
mkdir -p trainsets/trainH_BSD400
cp BSDS500/data/images/train/*.jpg trainsets/trainH_BSD400/
cp BSDS500/data/images/test/*.jpg  trainsets/trainH_BSD400/
```

> **注**: BSDS500 の `val/` 100 枚は BSD68 テストセットと重複するため学習に使わない。

### テストセット（評価用）

```bash
# KAIR リポジトリ内に BSD68 が同梱されている
ls models/KAIR/testsets/bsd68/   # 68 枚の PNG
```

---

## 実装計画

### 新規作成ファイル

```
scripts/train_scunet_gray.py      ← 学習スクリプト本体
options/train_scunet_gray.json    ← ハイパーパラメータ設定
```

### `options/train_scunet_gray.json` の内容

```json
{
  "sigma": 10,
  "sigma_test": 10,
  "n_channels": 1,
  "patch_size": 64,
  "batch_size": 32,
  "lr": 1e-4,
  "lr_milestones": [100000, 200000, 300000],
  "lr_gamma": 0.5,
  "total_iters": 300000,
  "checkpoint_save": 10000,
  "checkpoint_test": 5000,
  "dataroot_train": "trainsets/trainH_BSD400",
  "dataroot_test": "models/KAIR/testsets/bsd68",
  "pretrained": null
}
```

> **fine-tuning 短縮版**: `"pretrained": "models/SCUNet/model_zoo/scunet_gray_15.pth"` を
> 設定し `total_iters` を `100000` に減らすと収束が速い（後述）。

### `scripts/train_scunet_gray.py` の実装構成

#### 1. データローダー

`models/KAIR/data/dataset_dncnn.py` と同じパターンをそのまま流用できる。

```python
class SCUNetGrayDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size, sigma, phase='train'):
        self.paths = glob(os.path.join(data_dir, '*.jpg')) + \
                     glob(os.path.join(data_dir, '*.png'))
        self.patch_size = patch_size
        self.sigma = sigma / 255.0
        self.phase = phase

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.phase == 'train':
            # ランダムクロップ
            h, w = arr.shape
            rh = random.randint(0, h - self.patch_size)
            rw = random.randint(0, w - self.patch_size)
            clean = arr[rh:rh+self.patch_size, rw:rw+self.patch_size]
            # flip/rotate 拡張（8 パターン）
            clean = random_augment(clean)
        else:
            clean = arr

        noisy = clean + np.random.randn(*clean.shape) * self.sigma
        clean_t = torch.from_numpy(clean).unsqueeze(0)   # (1, H, W)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
        return noisy_t, clean_t
```

#### 2. モデルの初期化

```python
import sys
sys.path.insert(0, 'models/SCUNet')
from models.network_scunet import SCUNet

model = SCUNet(in_nc=1, config=[4,4,4,4,4,4,4], dim=64)

# fine-tuning の場合: scunet_gray_15 重みからスタート
if opt['pretrained']:
    model.load_state_dict(torch.load(opt['pretrained']), strict=True)
```

#### 3. 損失関数・オプティマイザ

```python
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=opt['lr_milestones'], gamma=opt['lr_gamma']
)
```

L1 Loss は KAIR の標準設定。L2（MSE）より境界がシャープになりやすい。

#### 4. 学習ループ

```python
for step in range(total_iters):
    noisy, clean = next(train_iter)
    noisy, clean = noisy.to(device), clean.to(device)

    pred = model(noisy)
    loss = criterion(pred, clean)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % checkpoint_save == 0:
        torch.save(model.state_dict(),
                   f'results/scunet_gray_{sigma}_iter{step}.pth')
```

> **注意**: SCUNet の `forward()` は `padding → process → unpad` を内部で行うため、
> パッチサイズが 64 の倍数でなくても動作する。ただし学習時は 64×64 固定パッチが標準。

---

## 学習設定の選択肢

### A. フルトレーニング（sigma=10 特化）

| パラメータ | 値 |
|---|---|
| 初期重み | ランダム初期化 |
| 総イテレーション | 300,000 |
| バッチサイズ | 32 |
| パッチサイズ | 64×64 |
| 推定学習時間 | RTX 3060 で 18〜24 時間 |
| 想定 PSNR (BSD68, σ=10) | 38〜39 dB 程度 |

### B. fine-tuning（scunet_gray_15 から転移）【推奨】

| パラメータ | 値 |
|---|---|
| 初期重み | `scunet_gray_15.pth` |
| 総イテレーション | 100,000（収束が速い） |
| 学習率 | 5e-5（フルの半分） |
| 推定学習時間 | RTX 3060 で 6〜8 時間 |
| 根拠 | sigma=10 は sigma=15 に近く、特徴抽出層はほぼ流用可能 |

fine-tuning の方が少ないデータ・時間で同等以上の品質が得られる見込みが高い。

---

## 検証方法

### 定量評価（BSD68、sigma=10 ノイズ付加）

```bash
python scripts/run_scunet.py \
  --input models/KAIR/testsets/bsd68/ \
  --model scunet_gray_10 \
  --model_zoo results/
# PSNR を計算: FFDNet sigma=10 および scunet_gray_15 と比較
```

PSNR の参考値（sigma=10, BSD68）:

| モデル | PSNR (参考) |
|---|---|
| FFDNet sigma=10 | ~38.7 dB |
| scunet_gray_15（強めの除去） | ～過剰除去になる |
| 目標 scunet_gray_10 | 38〜39 dB 程度 |

### 定性評価（鉛筆スケッチ）

```bash
python scripts/run_scunet.py --input test_inputs/ \
  --model scunet_gray_10 --model_zoo results/
```

評価観点:
1. 紙の粒状ノイズが除去されているか
2. 細い線（髪・入り抜き）が維持されているか
3. scunet_gray_15 より線の保持が良いか
4. FFDNet sigma=10 と比べた質感の差

---

## 作業工数見積もり

| 作業 | 工数 |
|---|---|
| BSD400 ダウンロード・配置 | 30 分 |
| `train_scunet_gray.py` 実装 | 3〜5 時間 |
| fine-tuning 実行（B案） | 6〜8 時間（放置可） |
| 評価・比較 | 1〜2 時間 |
| **合計（実作業）** | **5〜8 時間** |

---

## 注意事項・落とし穴

**1. timm の FutureWarning**
`from timm.models.layers import trunc_normal_, DropPath` が非推奨警告を出す（無害）。
将来 timm が API を削除した場合は `from timm.layers import ...` に変更する。

**2. sigma の単位**
`network_scunet.py` はノイズを受け取るが sigma を引数に取らない。
データローダー側で `sigma / 255.0` のスケール変換を忘れると学習が発散する。

**3. fine-tuning 時の学習率**
scunet_gray_15 から始める場合、デフォルト lr=1e-4 は高すぎる可能性がある。
`5e-5` から始めて損失の推移を確認すること。

**4. 過学習の兆候**
BSD400（400 枚）は小さいデータセットのため、300k イテレーション全体を
回すと過学習気味になることがある。検証 PSNR が下がり始めたら早期終了する。

**5. モデルの保存先と run_scunet.py の連携**
学習済み重みを `models/SCUNet/model_zoo/scunet_gray_10.pth` に置き、
`scripts/run_scunet.py` の `MODEL_CONFIGS` に以下を追加すれば既存の推論パスで動く:

```python
'scunet_gray_10': (1, True),
'scunet_gray_5':  (1, True),
```

---

## 参照ファイル

| ファイル | 用途 |
|---|---|
| `models/SCUNet/models/network_scunet.py` | モデル定義 |
| `models/KAIR/data/dataset_dncnn.py` | データローダーの参考実装 |
| `models/KAIR/options/train_ffdnet.json` | JSON オプション構成の参考 |
| `models/KAIR/main_train_dncnn.py` | 学習ループの参考実装 |
| `models/SCUNet/model_zoo/scunet_gray_15.pth` | fine-tuning の初期重み |
| `models/KAIR/testsets/bsd68/` | 定量評価用テストセット |
