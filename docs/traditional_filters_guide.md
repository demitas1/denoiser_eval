# 従来型フィルターガイド

機械学習を使わない古典的フィルターで処理した画像を生成し、ML ベースのデノイザや超解像モデルとの比較基準として使います。

スクリプト: `scripts/run_traditional.py`

---

## フィルター一覧

| フィルター | 手法 | 主な用途 |
|---|---|---|
| `gaussian` | ガウシアンブラー | 線形平滑化・弱いノイズ除去 |
| `median` | メディアンフィルター | 塩胡椒ノイズ・点状ノイズに強い |
| `bilateral` | バイラテラルフィルター | エッジを保持しつつ平滑化 |
| `nlmeans` | NL-Means | パッチ類似度による非局所平均（古典手法の上限） |
| `upscale` | 補間アップスケール→Lanczos 縮小 | 超解像モデルとの比較基準 |
| `combine` | ノイズ除去→アップスケール の2段階 | デノイズ＋超解像の組み合わせ |

---

## 基本的な使い方

```bash
# デフォルト4フィルター（gaussian / median / bilateral / nlmeans）をまとめて実行
python scripts/run_traditional.py --input test_inputs/ --output results/traditional/

# フィルターを絞って実行
python scripts/run_traditional.py --input test_inputs/ --filters gaussian bilateral

# upscale（bicubic→Lanczos、lanczos→Lanczos の2種）
python scripts/run_traditional.py --input test_inputs/ --filters upscale

# combine（ノイズ除去→アップスケール）
python scripts/run_traditional.py --input test_inputs/ --filters combine \
    --combine_denoise gaussian bilateral \
    --combine_upscale bicubic
```

---

## デフォルトパラメータ

| フィルター | デフォルト値 |
|---|---|
| `gaussian` | σ = 1.0 |
| `median` | k = 3 |
| `bilateral` | σ ∈ {10, 20, 30, 40} |
| `nlmeans` | h ∈ {5, 10, 15, 20} |
| `upscale` | 倍率 ×4、手法: bicubic / lanczos |
| `combine` | デノイズ: 全4種、アップスケール: bicubic / lanczos、倍率 ×4 |

パラメータはコマンドラインで上書き可能です:

```bash
python scripts/run_traditional.py --input test_inputs/ \
    --filters gaussian bilateral \
    --gaussian_sigma 0.5 1.0 \
    --bilateral_sigma 20 40
```

---

## 出力ファイル命名規則

| フィルター | 出力例 |
|---|---|
| `gaussian` | `sketch_gaussian_s1.0.png` |
| `median` | `sketch_median_k3.png` |
| `bilateral` | `sketch_bilateral_s25.png` |
| `nlmeans` | `sketch_nlmeans_h10.png` |
| `upscale` | `sketch_upscale_x4_bicubic.png` |
| `combine` | `sketch_combine_bilateral_s25_bicubic_x4.png` |

---

## オプション

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--input` | （必須） | 入力画像ファイルまたはディレクトリ |
| `--output` | `results/traditional` | 出力ディレクトリ |
| `--filters` | `gaussian median bilateral nlmeans` | 実行するフィルター（複数指定可） |
| `--gaussian_sigma` | `1.0` | Gaussian σ（複数指定でスイープ） |
| `--median_ksize` | `3` | Median カーネルサイズ（奇数）（複数指定でスイープ） |
| `--bilateral_sigma` | `10 20 30 40` | Bilateral σ\_color = σ\_space（複数指定でスイープ） |
| `--nlmeans_h` | `5 10 15 20` | NL-Means フィルター強度 h（複数指定でスイープ） |
| `--upscale_factor` | `4` | アップスケール倍率（`upscale` / `combine` 共通） |
| `--upscale_method` | `bicubic lanczos` | `upscale` モードの補間方法 |
| `--combine_denoise` | `gaussian median bilateral nlmeans` | `combine` モードの前段フィルター |
| `--combine_upscale` | `bicubic lanczos` | `combine` モードの後段補間方法 |

> **NL-Means の処理時間**: 1024² 画像 1 枚あたり 30〜60 秒かかります。時間が惜しい場合は `--filters gaussian bilateral` のように絞って実行してください。
