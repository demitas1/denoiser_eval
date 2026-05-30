# その他ツールガイド

その他のユーティリティのリファレンス。

---

## visualize_degradation.py — 劣化パイプライン可視化

スクリプト: `scripts/visualize_degradation.py`

BSRGAN 劣化パイプライン（idx 0–6）および鉛筆スケッチ向けカスタム劣化（idx 7–12）を
単独または組み合わせで画像に適用し、結果を PNG として保存する。

### 操作インデックス

| idx | 名前 | 実装状況 |
|---|---|---|
| 0, 1 | blur_A / blur_B | ✅ |
| 2 | downsample_mid | ✅ |
| 3 | downsample_final | ✅ |
| 4 | gaussian_noise | ✅ |
| 5 | jpeg_noise | ✅ |
| 6 | isp_noise | no-op（モデルなし） |
| 7 | eraser_trace（消し跡） | no-op（issue #6 で実装予定） |
| 8 | isotropic_smear（等方スメア） | no-op（issue #4 で実装予定） |
| 9 | directional_smear（方向性スメア） | no-op（issue #5 で実装予定） |
| 10 | paper_grain（紙粒感） | no-op（issue #3 で実装予定） |
| 11 | stain（しみ） | no-op（issue #7 で実装予定） |
| 12 | pressure_variation（圧力ムラ） | no-op（issue #2 で実装予定） |

### 基本的な使い方

```bash
# 単一操作
python scripts/visualize_degradation.py --input degradation_inputs/example.png --index 4

# 複数操作を個別に保存（ファイルを各 idx ごとに出力）
python scripts/visualize_degradation.py \
    --input degradation_inputs/example.png \
    --index 0 1 4 5 --output results/degradation_vis/

# シャッフルモード（ランダムな順序で連続適用、3サンプル）
python scripts/visualize_degradation.py \
    --input degradation_inputs/example.png \
    --index 0 1 2 3 4 5 --shuffle --num_samples 3

# カスタム劣化 idx 7–12（現在は no-op）
python scripts/visualize_degradation.py \
    --input degradation_inputs/example.png \
    --index 7 8 9 10 11 12 --output results/degradation_vis/

# seed 固定で再現
python scripts/visualize_degradation.py \
    --input degradation_inputs/example.png \
    --index 0 1 4 --seed 42
```

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像パス |
| `--index` | （必須） | 適用する操作インデックス（複数指定可、0–12） |
| `--output` | 入力と同ディレクトリ | 出力先ディレクトリ |
| `--shuffle` | OFF | 指定インデックスをランダム順で連続適用 |
| `--num_samples` | 1 | シャッフルモードで生成するサンプル数 |
| `--sf` | 4 | ダウンサンプル操作のスケール倍率 |
| `--patch_size` | 320 | 入力画像から切り出すパッチサイズ |
| `--seed` | なし | 再現性のための乱数シード |

### 出力ファイル名

- 通常モード: `{stem}_{idx}.png`（例: `example_4.png`）
- シャッフルモード: `{stem}_{idx0}_{idx1}_..._{idxN}.png`（複数サンプル時は末尾に `_sN`）

---

## utils/degradation_custom.py — カスタム劣化モジュール

モジュール: `utils/degradation_custom.py`

鉛筆スケッチ固有の劣化（idx 7–12）を提供する関数モジュール。
`visualize_degradation.py` と `DatasetBlindSR`（issue #9）の両方から共用する。

### インターフェース

```python
from utils.degradation_custom import apply_custom_op, CUSTOM_OPS

# idx に対応する劣化を適用
result = apply_custom_op(idx, img)   # img: np.ndarray float32 [0,1], (H,W) or (H,W,C)

# 有効な idx 一覧
print(sorted(CUSTOM_OPS.keys()))     # [7, 8, 9, 10, 11, 12]
```

### 各フィルターの実装時の追加方法

各フィルターは `utils/degradation_custom.py` 内の対応関数（`apply_eraser_trace` など）の
`return img.copy()` を実際の処理に置き換えるだけで有効になる。
`visualize_degradation.py` や `DatasetBlindSR` 側は変更不要。

---

## scan_image_quality.py — JPEG アーティファクト検出

スクリプト: `scripts/scan_image_quality.py`

テストセットや学習データから JPEG 圧縮アーティファクトの強い画像を検出し、除外候補をリストアップする。

### 2 つの指標

| 指標 | 対象 | 内容 |
|---|---|---|
| `blocking` | PNG / JPEG | 8px ブロック境界の差分と内部差分の比率。1.0 = アーティファクトなし |
| `jpeg_q` | **元 JPEG のみ** | 量子化テーブル係数の平均値。数値が小さいほど高品質 |

**blocking スコアの目安**

| スコア | 状態 |
|---|---|
| 1.0〜1.2 | 良好（アーティファクトほぼなし） |
| 1.2〜1.5 | 軽微（多くの Pexels 画像はこの範囲） |
| 1.5〜1.8 | 目立つ（要確認） |
| 1.8〜   | 強い（除外推奨） |

**jpeg_q の目安**

| jpeg_q | 相当する JPEG 品質 |
|---|---|
| 2〜5 | quality 95 以上 |
| 8〜12 | quality 85 前後 |
| 15〜25 | quality 75 前後 |
| 38〜   | quality 60〜70（Pexels の標準ダウンロード品質） |

### 基本的な使い方

```bash
# ディレクトリ全体をスキャン（blocking 降順）
python scripts/scan_image_quality.py testsets/custom_natural/pexels-cc0-100-1/original/

# 上位 20 件のみ表示
python scripts/scan_image_quality.py testsets/custom_natural/pexels-cc0-100-1/original/ --top 20

# 閾値フィルター（blocking >= 1.8 のみ）
python scripts/scan_image_quality.py testsets/custom_natural/pexels-cc0-100-1/original/ \
    --min-blocking 1.8

# JSON 出力
python scripts/scan_image_quality.py testsets/custom_natural/pexels-cc0-100-1/original/ \
    --format json
```

### パイプ処理との連携

```bash
# blocking >= 1.8 のファイルパスだけ抽出
python scripts/scan_image_quality.py dir/original/ \
    --min-blocking 1.8 --no-header | cut -f1

# 該当ファイルを別ディレクトリに移動
mkdir -p dir/original/blocking-1_8
python scripts/scan_image_quality.py dir/original/ \
    --min-blocking 1.8 --no-header | cut -f1 | \
    xargs -I{} mv {} dir/original/blocking-1_8/

# ファイル名リストをテキスト保存
python scripts/scan_image_quality.py dir/original/ \
    --min-blocking 1.8 --no-header | cut -f1 > bad_images.txt
```

### PNG 変換済み画像の注意

768×768 PNG（LANCZOS リサイズ後）に対してスキャンすると blocking スコアが 1.01〜1.03 程度に
収束し、弁別力がほぼなくなる。これはリサイズ時に 8px ブロック境界の位置がずれるため。

**除外候補の選定は `original/` の元 JPEG に対して実行すること。**

### 全オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `input` | （必須） | スキャン対象のファイルまたはディレクトリ |
| `-r`, `--recursive` | OFF | サブディレクトリも再帰的にスキャン |
| `--ext EXT [EXT ...]` | `jpg jpeg png` | 対象拡張子 |
| `--sort blocking\|jpeg_q\|path` | `blocking` | ソートキー（blocking/jpeg_q は降順、path は昇順） |
| `--top N` | 全件 | 上位 N 件のみ出力 |
| `--min-blocking SCORE` | なし | blocking >= SCORE のみ出力 |
| `--min-jpeg-q SCORE` | なし | jpeg_q >= SCORE のみ出力（JPEG ファイル限定） |
| `--format tsv\|csv\|json` | `tsv` | 出力形式 |
| `--no-header` | OFF | ヘッダー行を省略（パイプ処理向け） |
| `--nan STR` | `""` | N/A 値のプレースホルダー文字列 |
