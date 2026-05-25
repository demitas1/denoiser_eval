# その他ツールガイド

その他のユーティリティのリファレンス。

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
