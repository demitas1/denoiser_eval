# 学習ガイド

独自データセット（Unsplash Lite 等）でモデルをカスタム学習する手順です。

公式配布重みの学習データ（WED / BSD400 / FFHQ 等）は非商用ライセンスを含むため、外部配布を検討する場合は Unsplash Lite 等のライセンスが明確なデータで学習し直す必要があります。

> **Unsplash Lite のライセンスについて**: Unsplash Dataset Terms（Section 2.A）により、学習・利用は**内部業務目的に限定**されます（non-sublicensable・non-transferable）。外部配布や商用プロダクトへの組み込みは Unsplash への確認が必要です。CC0 ではありません。

---

## カスタム学習済みモデルの保存と Git LFS 管理

カスタム学習済み重みは `results/trained_models/` に保存し、Git LFS で GitHub に push します。

### 命名規則

```
{arch}_{train_phase}_{dataset}.pth
```

| フィールド | 例 |
|---|---|
| arch | `ffdnet_gray`, `bsrnet_x4`, `bsrgan_x4`, `scunet_gray10` |
| train_phase | `scratch`（ゼロから）, `ft`（fine-tune）, `ganft`（GAN fine-tune） |
| dataset | `unsplash_lite`, `bsd400` など |

### 現在のモデル一覧

| ファイル | 内容 | 状態 |
|---|---|---|
| `ffdnet_gray_scratch_unsplash_lite.pth` | FFDNet gray、500k iters、Best PSNR 33.88 dB @ σ=25 | 完了・push 済み |
| `bsrnet_x4_scratch_unsplash_lite.pth` | BSRNet x4 PSNR phase | 未着手 |
| `bsrgan_x4_ganft_unsplash_lite.pth` | BSRGAN x4 GAN phase | 未着手 |
| `scunet_gray10_ft_unsplash_lite.pth` | SCUNet gray σ=10 fine-tune | 未着手 |

### push 手順

```bash
git add results/trained_models/<name>.pth
git commit -m "Add trained model: <name>"
git push   # LFS オブジェクトも同時に push される
```

### Git LFS セットアップ（新環境）

```bash
git lfs install   # 初回のみ
# .gitattributes に設定済みのため clone 後は自動で LFS ファイルを取得
```

---

## Unsplash Lite データセットの準備

Unsplash Lite は自然風景写真 25,000 枚のデータセットです（Unsplash Dataset Terms — 内部業務目的での機械学習トレーニングに利用可能）。

```bash
# メタデータ zip を公式リポジトリからダウンロードしてプロジェクトルートに置いてから:
unzip unsplash-research-dataset-lite-latest.zip -d trainsets/unsplash_lite_meta/

# 画像をダウンロード（デフォルト: 1024px 以上を最大 2,000 枚）
python scripts/download_unsplash_lite.py

# 枚数を絞る場合
python scripts/download_unsplash_lite.py --max_images 500

# 中断後は再実行するだけで続きから再開（ダウンロード済みをスキップ）
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--max_images` | `2000` | 最大ダウンロード枚数（0 で無制限） |
| `--min_size` | `1024` | 最小解像度（px）フィルタ |
| `--download_width` | `1080` | ダウンロード画像の幅（Unsplash 動的リサイズ） |
| `--delay` | `0.5` | リクエスト間隔（秒） |

### テストセット分割（初回のみ）

```bash
# 対象ファイルの確認（移動は行わない）
python scripts/prepare_unsplash_testset.py --dry_run

# 実際に分割（trainsets/trainH/unsplash_lite/ の末尾100枚を testsets/unsplash_lite_test/ へ移動）
python scripts/prepare_unsplash_testset.py
# → testsets/unsplash_lite_test/: 100枚、trainsets/trainH/unsplash_lite/: 1,899枚
```

> 本環境では 2026-05-21 時点で分割済み。新環境ではこの手順が必要。

---

## FFDNet gray の学習（Unsplash Lite）

ランダム sigma ([0, 75]) でブラインドデノイザをゼロから学習する。

```bash
# 動作確認（200 iters）
python scripts/train_ffdnet_gray.py \
    --config options/train_ffdnet_gray_unsplash.json \
    --max_iters 200 --datasets unsplash_lite

# 本番実行（500k iters、約 3 時間 on RTX 3060）
python scripts/train_ffdnet_gray.py \
    --config options/train_ffdnet_gray_unsplash.json \
    --datasets unsplash_lite

# 公式重みから fine-tuning
python scripts/train_ffdnet_gray.py \
    --config options/train_ffdnet_gray_unsplash.json \
    --datasets unsplash_lite \
    --pretrained models/KAIR/model_zoo/ffdnet_gray.pth

# 中断後の再開
python scripts/train_ffdnet_gray.py \
    --config options/train_ffdnet_gray_unsplash.json \
    --datasets unsplash_lite \
    --resume results/train_ffdnet_gray/iter_005000.pth
```

### 学習完了後

```bash
cp results/train_ffdnet_gray/best.pth results/trained_models/ffdnet_gray_scratch_unsplash_lite.pth

# カスタム重みで推論
python scripts/run_ffdnet.py --input test_inputs/ \
    --model results/trained_models/ffdnet_gray_scratch_unsplash_lite.pth \
    --sigma 10 15 25
```

> 達成済み: 500k iters 完了、Best PSNR **33.88 dB** (σ=25, Unsplash Lite test 100枚)

---

## SCUNet gray の学習（Unsplash Lite）

**準備中** — Unsplash Lite でフルトレーニングする予定。スクリプト（`scripts/train_scunet_gray.py`）と設定ファイル（`options/train_scunet_gray_unsplash.json`）は実装済み。

---

## BSRNet PSNR フェーズの学習（ゼロから）

Unsplash Lite のみで BSRNet 相当モデルをゼロから学習する。完成物を GAN フェーズの出発点として使う。

> **ライセンス注記**: 公式 BSRNet.pth の学習データ（WED: 非商用のみ、FFHQ: CC BY-NC-SA 4.0）は外部配布・商用プロダクトへの組み込みが制限されている。Unsplash Lite で学習し直した場合も Unsplash Dataset Terms（内部業務目的に限定）が適用される。外部配布時はライセンスを配布元で確認すること。

```bash
# 動作確認（100 iters）
python scripts/train_bsrgan_psnr.py \
    --config options/train_bsrgan_x4_psnr_unsplash.json \
    --max_iters 100 \
    --datasets unsplash_lite

# 本番実行（500k iters、約 81 時間 on RTX 3060）
python scripts/train_bsrgan_psnr.py \
    --config options/train_bsrgan_x4_psnr_unsplash.json \
    --datasets unsplash_lite

# 中断後の再開
python scripts/train_bsrgan_psnr.py \
    --config options/train_bsrgan_x4_psnr_unsplash.json \
    --datasets unsplash_lite \
    --resume results/train_bsrgan_psnr/iter_001000.pth
```

### 学習完了後

```bash
cp results/train_bsrgan_psnr/best.pth models/KAIR/model_zoo/BSRNet_unsplash.pth
```

`options/train_bsrgan_x4_gan_finetune.json` の `pretrained_netG` を `BSRNet_unsplash.pth` に書き換えて GAN フェーズへ進む。

---

## BSRGAN GAN フェーズの学習

公式配布の `BSRNet.pth`（PSNR 学習済み）を出発点に、GAN フェーズだけを実行してシャープな超解像モデルを作成する。

- 入力画像は学習中に BSRGAN 劣化パイプライン（ブラー→ダウンサンプリング→ノイズ/JPEG を複数ラウンド）でランダム劣化されるため、LR 画像を事前に用意する必要はない
- GAN では PSNR と知覚品質の相関が低いため `best.pth` は保存せず、EMA 重みを `last_E.pth` として定期保存する

### 1. 追加パッケージのインストール

```bash
pip install matplotlib  # KAIR の utils_image.py が依存
```

### 2. scipy 互換性パッチの適用（scipy ≥ 1.14 の場合）

scipy 1.14 以降で KAIR の劣化パイプラインが使う `interp2d` と `scipy.finfo` が廃止されました。`models/KAIR/` をクローン後、以下を適用してください。

```bash
# interp2d → RectBivariateSpline
sed -i \
  's/from scipy.interpolate import interp2d/from scipy.interpolate import RectBivariateSpline/' \
  models/KAIR/utils/utils_blindsr.py
sed -i \
  's/interp2d(xv, yv, \(.*\))(x1, y1)/RectBivariateSpline(yv, xv, \1)(y1, x1)/g' \
  models/KAIR/utils/utils_blindsr.py

# scipy.finfo → np.finfo
sed -i \
  's/scipy\.finfo(float)/np.finfo(float)/g' \
  models/KAIR/utils/utils_blindsr.py
```

> このパッチは `models/KAIR/` がクローン済みであることを前提とします。再クローン後は再度適用してください。

### 3. HR 画像の配置

Unsplash Lite の準備は上記「Unsplash Lite データセットの準備」セクションを参照。独自データを追加する場合は `trainsets/trainH/custom/` に配置します:

```
trainsets/trainH/
  unsplash_lite/   ← Unsplash Lite（自動取得）
  custom/          ← 独自データ（任意）
```

### 4. 学習実行

```bash
# 動作確認（100 iters）
python scripts/train_bsrgan_gan.py \
    --config options/train_bsrgan_x4_gan_finetune.json \
    --max_iters 100 \
    --datasets unsplash_lite

# 本番 GAN fine-tuning（400k iters）— Unsplash Lite
python scripts/train_bsrgan_gan.py \
    --config options/train_bsrgan_x4_gan_finetune.json \
    --datasets unsplash_lite

# 中断後の再開
python scripts/train_bsrgan_gan.py \
    --config options/train_bsrgan_x4_gan_finetune.json \
    --datasets unsplash_lite \
    --resume results/train_bsrgan_gan/iter_005000.pth
```

### 5. 学習済みモデルの配置と推論

```bash
# EMA 重みを model_zoo に配置
cp results/train_bsrgan_gan/last_E.pth models/KAIR/model_zoo/BSRGAN_custom.pth

# 推論（カスタムモデルを指定）
python scripts/run_esrgan.py \
    --input test_inputs/ \
    --model BSRGAN BSRGAN_custom \
    --model_zoo models/KAIR/model_zoo
```

### HR 画像の要件

| 条件 | 補足 |
|---|---|
| 解像度 ≥ 1024px（両辺） | 設定の `H_size=320` パッチが切り出せれば動作するが、多様なクロップを得るには 1024px 以上を推奨 |
| ノイズ・ブラー・圧縮アーティファクトなし | HR は正解画像として学習されるため、汚れがあると「汚れを出力するモデル」になる |
| PNG 保存（ロスレス） | JPEG を HR に使うと圧縮ブロックノイズを正解として学習してしまう |
| 内容の多様性 | テクスチャ・輝度・構造が偏ると汎化性能が落ちる |

### 鉛筆スケッチ向けの独自データセット作成指針

BSRNet.pth は自然画像で学習済みのため、clean な ink line art を少量追加するだけでドメイン適応が期待できます。

**推奨構成（混合戦略）**: Unsplash Lite を汎用ベースとし、独自線画 200〜500 枚を `trainsets/trainH/custom/` に追加して `--datasets unsplash_lite custom` のように混合するとドメイン品質の向上が期待できます。

- スキャン解像度 600 dpi 以上、PNG で保存。紙のテクスチャが残っていないクリーンな状態が望ましい
- 線の太さ・密度にバリエーションを持たせる
- グレースケール線画の場合は RGB に変換してから配置する（BSRGAN は 3ch 入力）

```bash
# ImageMagick でグレースケール PNG を RGB に一括変換
mogrify -colorspace sRGB -type TrueColor trainsets/trainH/custom/*.png
```

---

## 劣化パイプラインの可視化

`scripts/visualize_degradation.py` は BSRGAN の on-the-fly 劣化パイプラインで適用される各操作を個別またはランダムな組み合わせで視覚的に確認するためのスクリプトです。

`degradation_inputs/` ディレクトリに可視化用の入力画像を配置します（gitignore 対象）。付属の `degradation_inputs/example.png`（320×320 モノクロ グリッド画像）でブラーのぼけ、ダウンサンプルのブロック状ピクセル化、JPEG のリンギングを確認できます。

### 操作インデックス

| idx | 操作 | 内容 |
|---|---|---|
| 0, 1 | ブラー | 等方性 / 非等方性ガウシアンカーネルをランダム適用 |
| 2 | 中間ダウンサンプル | ランダム倍率（1〜2×sf）でリサイズ → nearest で元サイズに拡大 |
| 3 | 最終ダウンサンプル | ×1/sf にリサイズ → nearest で元サイズに拡大 |
| 4 | ガウシアンノイズ | カラー / グレー / 相関ノイズをランダム選択 |
| 5 | JPEG 圧縮 | 品質 30〜95 でランダム圧縮 |
| 6 | ISP カメラノイズ | no-op（モデルなし） |

### 使い方

```bash
# 付属のグリッド画像で全操作を個別確認
python scripts/visualize_degradation.py --input degradation_inputs/example.png \
    --index 0 1 2 3 4 5 --output results/degradation_vis/

# シャッフルモード 5 サンプル
python scripts/visualize_degradation.py --input degradation_inputs/example.png \
    --index 0 1 2 3 4 5 --shuffle --num_samples 5 --output results/degradation_vis/

# 再現性が必要な場合はシードを固定
python scripts/visualize_degradation.py --input degradation_inputs/example.png \
    --index 0 1 4 --seed 42
```

### オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像パス |
| `--index` | （必須） | 操作インデックス（0〜6、複数指定可） |
| `--output` | 入力画像と同じディレクトリ | 出力先ディレクトリ |
| `--shuffle` | off | 指定インデックスをランダムな順序で連続適用 |
| `--num_samples` | `1` | シャッフル時の生成サンプル数 |
| `--sf` | `4` | スケール因子（idx 2, 3 のダウンサンプル操作に使用） |
| `--seed` | なし | 乱数シード（再現性が必要な場合） |
| `--patch_size` | `320` | 入力パッチサイズ（中央クロップ後のサイズ） |
