# BSRGAN カスタム学習プラン

## 目的

BSRGAN の汎用劣化パイプライン（idx 0–6）は自然画像を想定して設計されており、
鉛筆スケッチ固有の劣化（消し跡・スメア・紙の粒感・しみ・圧力ムラ）をカバーしない。
本プランでは鉛筆スケッチ → クリーンなインク線画変換を目的として、
これらの劣化を新たなインデックス（idx 7–12）として実装する。

---

## 対象アーキテクチャ

### 主ターゲット: BSRGAN（RRDBNet）

- `scripts/train_bsrgan_gan.py` の `DatasetBlindSR` に本劣化を組み込む
- 既存の idx 0–6 と混合してシャッフル適用することで、鉛筆スケッチ特有のノイズに対応した LR を生成する

### 他アーキテクチャへの適用可能性

本劣化はすべて **データパイプラインレベルの操作**であり、モデルアーキテクチャに依存しない。
以下のモデルのカスタム学習にそのまま流用できる。

| モデル | 用途 | 備考 |
|---|---|---|
| SCUNet | ブラインドデノイズ | `train_scunet_gray.py` のデータローダに組み込み可能 |
| Restormer (Real Denoising) | ブラインドデノイズ | ペア生成パイプラインとして使用 |
| Real-ESRGAN | SR + デノイズ | `degradation_bsrgan` の代替として差し込み可能 |
| SwinIR (Real-world SR) | SR | 同上 |
| NAFNet | デノイズ | ペア生成に使用（SR なし構成） |

**注意**: DnCNN / FFDNet は sigma を明示的に指定する設計のため、
本パイプラインのようなブラインド劣化との相性は悪い。

---

## 操作インデックス一覧

| idx | 操作名 | 概要 |
|---|---|---|
| 0, 1 | ブラー | 既存（等方性 / 非等方性ガウシアン） |
| 2 | 中間ダウンサンプル | 既存 |
| 3 | 最終ダウンサンプル | 既存 |
| 4 | ガウシアンノイズ | 既存 |
| 5 | JPEG 圧縮 | 既存 |
| 6 | ISP カメラノイズ | 既存（no-op） |
| **7** | **消し跡** | パッチを近傍オフセット位置からコピー + 低 alpha 合成 |
| **8** | **汚れ（等方スメア）** | 白地エリアへの低周波等方ブラーノイズ加算 |
| **9** | **汚れ（方向性スメア）** | 白地エリアへのモーションブラーノイズ加算（ランダム角度） |
| **10** | **紙粒感** | テクスチャ画像との Multiply 合成 |
| **11** | **しみ** | ランダム凸多角形 + 境界ぼかし + alpha 合成 |
| **12** | **圧力ムラ** | 黒線エリアへの低周波ブラーノイズ加算 |

---

## 各操作の実装方針

### idx 7: 消し跡

消しゴムで不完全に消した際に黒鉛が薄く残るゴースト型の痕跡。

- ランダムな矩形 or 楕円パッチを選択
- パッチ内容を近傍のランダムオフセット位置（数〜数十 px）からコピー
- 低 alpha（0.2–0.5）で元画像に重ねる
- パッチ境界を GaussianBlur でフェードさせて不自然さを消す

```
パッチ数    : 1–3 個
パッチサイズ : 短辺の 5–20%
オフセット  : 5–30 px（ランダム方向）
```

> スメア（黒鉛が引き伸ばされる効果）は idx 8, 9 で表現するため、
> idx 7 はコピーによるゴースト残像のみを担当する。

---

### idx 8: 汚れ（等方スメア）

手のひらや指が紙に触れてグラファイトが転写された霞。方向性なし。

- `bright_mask = img > 0.7`（白地領域を抽出）
- 一様ノイズに強い GaussianBlur（sigma=20–50）をかけて低周波ノイズマップを生成
- ノイズ値を正方向のみ（暗くする方向）にクランプ
- `bright_mask` 内のピクセルにのみ加算
- 強度スケール: 0.05–0.20

---

### idx 9: 汚れ（方向性スメア）

手が滑った方向に沿ってグラファイトが伸びたスメア。

- ランダム角度（0–180°）のモーションブラーカーネルを生成
- カーネルサイズ: 30–80 px
- 一様ノイズにそのカーネルを適用してノイズマップを生成
- それ以外は idx 8 と同じ（`bright_mask` 内への加算、強度 0.05–0.20）

モーションブラーカーネル生成:
```
angle → 回転行列で対角線上に 1 を持つカーネルを生成 → 端を GaussianBlur で滑らかに
```

---

### idx 10: 紙粒感

紙の物理的なテクスチャが透けて見える現象。ガウシアンノイズと異なり**空間的に相関したテクスチャ**。

**テクスチャ画像あり（推奨）**
- `degradation_inputs/paper_textures/` にグレースケール PNG を複数枚用意
- ランダムに 1 枚選択 → ランダムクロップ → `float32 [0, 1]` に変換
- Multiply 合成: `result = img * texture`

**テクスチャ画像なし（代替）**
- 一様ノイズに中程度の GaussianBlur（sigma=1–3）をかけて疑似テクスチャを生成
- 値域を [0.85, 1.0] にリマップして Multiply 合成

```
テクスチャ推奨サイズ : 512×512 以上
保存場所            : degradation_inputs/paper_textures/*.png（gitignore 対象）
```

---

### idx 11: しみ

水・コーヒー・経年劣化による紙のシミ。

- ランダム凸多角形を生成（`cv2.fillConvexPoly`、頂点 5–8 点）
- 薄いグレー（輝度 0.6–0.9）で塗りつぶし
- `cv2.GaussianBlur` で境界をぼかす（sigma=10–30）
- 元画像と alpha 合成（alpha=0.3–0.7）
- 必要に応じて外縁部を若干暗くしてリング状の乾燥跡を表現

```
個数   : 1–3 個
面積   : パッチ面積の 5–25%
```

---

### idx 12: 圧力ムラ

筆圧変化による同一ストローク内の明度ばらつき。

- `dark_mask = img < 0.4`（黒線領域を抽出）
- 一様ノイズに強い GaussianBlur（sigma=15–40）をかけて低周波「圧力マップ」を生成
- 値域を [-0.15, +0.15] 程度にスケール（正負両方向の変動）
- `dark_mask` 内のピクセルにのみ加算して [0, 1] にクランプ

> idx 8（等方スメア）と構造は同じ。**適用マスクが白地か黒線か**で区別。

---

## 既存パイプラインへの組み込み方針

### visualize_degradation.py への追加（実装済み）

`utils/degradation_custom.py` に dispatcher を実装し、`visualize_degradation.py` を idx 0–12 対応に拡張済み。
idx 7–12 は現在 no-op スタブ。各 issue で実装後、同スクリプトで即座に確認できる。

```bash
python scripts/visualize_degradation.py \
    --input degradation_inputs/example.png \
    --index 7 8 9 10 11 12 --output results/degradation_vis/
```

### train_bsrgan_gan.py への組み込み

`DatasetBlindSR.degradation_bsrgan` の呼び出し部分を差し替え or ラップし、
idx 0–6 と 7–12 を混合したシャッフルリストから劣化を適用する。

混合比（案）:
- 汎用劣化（idx 0–5）: 必ず含む
- 鉛筆特化劣化（idx 7–12）: 各ラウンドで 50% の確率で 1–2 操作を追加

---

## 必要な追加アセット

| アセット | 場所 | 備考 |
|---|---|---|
| 紙テクスチャ画像（複数枚） | `degradation_inputs/paper_textures/` | グレースケール PNG、512px 以上 |
| インク線画 HR 画像 | `trainsets/trainH/custom/` | 4096×4096 PNG 推奨（下記参照） |

---

## インク線画カスタムデータセットの枚数見積もり

### 画像サイズと 320×320 パッチ数

DatasetBlindSR は学習中に各 HR 画像からランダムに 320×320 パッチを切り出す。
非重複パッチ数は画像サイズで決まる。

| 画像サイズ | 非重複パッチ数（理論値） | 実効パッチ数※ |
|---|---|---|
| 1024×1024 | 3×3 = 9 | 〜4 |
| 2048×2048 | 6×6 = 36 | 〜14 |
| 3200×3200 | 10×10 = 100 | 〜40 |
| **4096×4096** | **12×12 = 144** | **〜58** |

※インク画は白地が多く、意味のある線・ディテールを含むパッチは理論値の 30〜40% 程度。

### 4096×4096px での推奨枚数

ファインチューニング（BSRNet.pth 出発点）に必要な実効パッチ数の目安は 1,000〜3,000。

| 枚数 | 全パッチ数 | 実効パッチ数 |
|---|---|---|
| 10枚 | 1,440 | 〜580 |
| **15〜20枚** | **2,160〜2,880** | **〜870〜1,150** ← 推奨 |
| 30枚 | 4,320 | 〜1,730 |

**結論: 4096×4096px のインク画は 15〜20枚が実用的な出発点。**

枚数より**コンテンツの多様性**（作風・線密度・細線/太線/ハッチング混在）の方が効果に直結する。

### Unsplash Lite との混合学習における注意点

DatasetBlindSR は**画像単位**でランダム選択するため、枚数比がそのまま選択確率になる。

| インク画枚数 | Unsplash 2,000 枚と混合時の選択確率 |
|---|---|
| 10枚 | 0.5% — 影響が薄すぎる |
| 20枚 | 1.0% — 同様 |
| 200枚以上 | 〜9% — バランスが取れる |

4096×4096 を 15〜20枚用意した場合、単純混合では効果が出ない。

### 推奨: 2フェーズ学習（GAN フェーズのみ）

公式 BSRNet.pth を PSNR 出発点として使う場合のフロー。

| フェーズ | スクリプト | `--datasets` | iters | 目的 |
|---|---|---|---|---|
| Phase 1 | `train_bsrgan_gan.py` | `unsplash_lite` | 400k | 汎用劣化・超解像能力の獲得 |
| Phase 2 | `train_bsrgan_gan.py` | `custom` | 5k〜20k | インク線画ドメインへの適応 |

Phase 2 の 20k iters・batch_size=4 では 80,000 サンプルを抽出。
15枚・実効パッチ 870 枚として各パッチを約 92 回学習することになり、ファインチューニングとして十分。

---

## BSRNet のゼロからの学習（商用利用時）

### ライセンス上の問題

公式 `BSRNet.pth` の学習データには商用利用不可のデータセットが含まれる。

| データセット | 枚数 | ライセンス |
|---|---|---|
| DIV2K | 800 | 学術利用向け |
| Flickr2K | 2,650 | 学術利用向け |
| WED | 4,744 | **非商用学術利用のみ** |
| FFHQ | 10,000 | **CC BY-NC-SA 4.0（非商用のみ）** |

商用プロダクトへの組み込みには **Unsplash Lite 等の商用 OK データで PSNR フェーズから学習し直す**必要がある。

### 学習フロー（PSNR → GAN）

```
[PSNR フェーズ]  train_bsrgan_psnr.py  → results/train_bsrgan_psnr/best.pth
                      ↓
              BSRNet_unsplash.pth として配置
                      ↓
[GAN フェーズ]   train_bsrgan_gan.py   → results/train_bsrgan_gan/last_E.pth
                      ↓
              BSRGAN_custom.pth として配置
```

### 時間見積もり（RTX 3060）

| フェーズ | iters | 推定時間 | 備考 |
|---|---|---|---|
| PSNR（BSRNet） | 500k | 約 81 時間 | L1 損失のみ、D・VGG なし |
| GAN（BSRGAN） | 400k | 約 110 時間 | D + VGG perceptual + LSGAN |
| **合計** | **900k** | **約 190 時間（8 日）** | |

Phase 2（インク画 fine-tuning）は別途 5k〜20k iters（約 1〜4 時間）。

### コマンド

```bash
# Step 1: PSNR フェーズ（Unsplash Lite のみ）
python scripts/train_bsrgan_psnr.py \
    --config options/train_bsrgan_x4_psnr_unsplash.json \
    --datasets unsplash_lite

# Step 2: 成果物を配置
cp results/train_bsrgan_psnr/best.pth models/KAIR/model_zoo/BSRNet_unsplash.pth

# Step 3: GAN config の pretrained_netG を書き換える
# options/train_bsrgan_x4_gan_finetune.json:
#   "_pretrained_netG_options" キーに切替候補あり
#   "pretrained_netG": "models/KAIR/model_zoo/BSRNet_unsplash.pth"  ← に変更

# Step 4: GAN フェーズ
python scripts/train_bsrgan_gan.py \
    --config options/train_bsrgan_x4_gan_finetune.json \
    --datasets unsplash_lite
```

---

## 学習途中チェックポイントでの推論

`scripts/run_bsrnet_custom.py` で任意の `.pth` を指定して SR 推論を実行できる。
フルチェックポイント（`iter_*.pth`）と state_dict のみ（`best.pth` / `last_E.pth`）の両方に対応。

### 出力

デフォルトではダウンスケール版（元解像度）のみ保存。`--save_upscaled` で x4 版も追加保存。

| ファイル名例 | 内容 |
|---|---|
| `img_bsrnet_step012000_lanczos.png` | SR → LANCZOS ダウンスケール（元サイズ）|
| `img_bsrnet_step012000_x4.png` | x4 アップスケール版（`--save_upscaled` 時のみ）|
| `img_bsrnet_custom_lanczos.png` | `best.pth` / `last_E.pth` 使用時（step 不明）|

### コマンド

```bash
# best.pth（PSNR 最良の EMA 重み）で推論
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input test_inputs/ --output results/bsrnet_custom/

# 特定ステップのフルチェックポイントで推論
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/iter_012000.pth \
    --input test_inputs/ --output results/bsrnet_custom/

# x4 アップスケール版も保存
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input test_inputs/ --save_upscaled

# ダウンスケールアルゴリズムを変更（lanczos / bicubic / bilinear / nearest）
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input test_inputs/ --downscale bicubic
```

---

## 実装優先順位

1. **idx 12（圧力ムラ）** — 実装が最もシンプル、効果が明確
2. **idx 10（紙粒感）** — テクスチャ画像次第だが汎用劣化との差別化度が高い
3. **idx 8, 9（スメア）** — 白地マスク + ノイズ加算、共通構造で実装可能
4. **idx 7（消し跡）** — パッチコピーと境界フェード
5. **idx 11（しみ）** — ポリゴン生成と alpha 合成
