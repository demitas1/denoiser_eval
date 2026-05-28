# BSRGAN / BSRNet 再現実験 参照文献

BSRNet（PSNR フェーズ）および BSRGAN（GAN フェーズ）の再現実験・ベースライン比較に使用する論文・リソースのまとめ。

---

## 主論文

### BSRGAN / BSRNet

> Zhang, K., Liang, J., Van Gool, L., & Timofte, R.
> "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
> ICCV 2021
> arXiv: https://arxiv.org/abs/2103.14006
> GitHub（公式実装）: https://github.com/cszn/BSRGAN

本プロジェクトで再現対象とする論文。以下を定義している：

- **BSRNet**: PSNR 損失のみで学習した超解像モデル（RRDBNet バックボーン）
- **BSRGAN**: BSRNet を出発点に GAN フェーズで fine-tuning したモデル
- **劣化パイプライン**: ブラー（等方性/異方性ガウシアン・sinc）→ダウンサンプル（nearest/bilinear/bicubic/area/Lanczos）→ノイズ（ガウシアン・ポアソン・JPEG 圧縮）をランダム順で 2〜3 ラウンド適用

論文中の比較数値は Table（PSNR/SSIM、×4 SR）を参照。評価プロトコルは著者の劣化パイプラインで生成した LR 画像を入力とする。

---

## バックボーン・ネットワーク

### RRDB / ESRGAN

> Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Loy, C. C., Qiao, Y., & Tang, X.
> "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
> ECCV 2018 Workshop
> arXiv: https://arxiv.org/abs/1809.00219

BSRNet/BSRGAN が使用するネットワーク構造 `RRDBNet`（Residual-in-Residual Dense Block × 23）の出典。本プロジェクトの実装は `models/KAIR/models/network_rrdbnet.py`（`nf=64, gc=32, nb=23`）。

---

## 関連研究（比較対象）

### Real-ESRGAN

> Wang, X., Xie, L., Dong, C., & Shan, Y.
> "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
> ICCV 2021 Workshop
> arXiv: https://arxiv.org/abs/2107.10833
> GitHub: https://github.com/xinntao/Real-ESRGAN

BSRGAN と同じ ICCV 2021 に発表された競合研究。劣化パイプラインの設計思想が異なる（高次劣化・U-Net Discriminator）。BSRGAN 論文の比較表に登場するため、設計選択の差異を理解するために参照する。

---

## 評価ベンチマーク

### Urban100

> Huang, J.-B., Singh, A., & Ahuja, N.
> "Single Image Super-Resolution From Transformed Self-Exemplars"
> CVPR 2015
> GitHub: https://github.com/jbhuang0604/SelfExSR

PSNR 比較に多くの論文で使用されているテストセット（100 枚）。

**ライセンス**: Flickr CC-BY-4.0 収集。商用利用可。ただし各画像の撮影者帰属情報は配布版に含まれていない（Attribution 情報は未整備）。

**入手先**: Hugging Face `eugenesiow/Urban100` または `jbhuang0604/SelfExSR` リポジトリ

---

## 公式比較数値の参照方法

論文 arXiv:2103.14006 の Table を直接参照すること。比較条件：

- スケール: ×4
- モデル: BSRNet（PSNR 行）/ BSRGAN（GAN 行）
- テストセット: Set5 / Set14 / BSD100 / **Urban100** / Manga109

### 注意：直接比較の前提条件

本プロジェクトでは主にUnsplash LiteとPexelsのCC0画像をトレーニングと評価に用いて論文モデルでの実測値により相対的な比較を行っている。

本プロジェクトの学習済みモデルと論文値を比較する際は、以下を揃える必要がある：

| 条件 | 論文 | 本プロジェクト |
|---|---|---|
| 学習データ | DIV2K + Flickr2K + WED + FFHQ | Unsplash Lite のみ |
| テストセット | Set5 / Set14 / BSD100 / Urban100 / Manga109 | Urban100（比較用） |
| 評価用 LR 生成 | 論文の劣化パイプライン | 同スクリプト（`utils_blindsr.py`）|

**公式 BSRNet.pth と自学習モデルを同一テストセット・同一劣化条件で評価することが再現比較の前提。**

```bash
# Urban100 HR 画像を配置後:
# 1. 公式モデルで評価
python scripts/run_esrgan.py \
    --input testsets/urban100/HR/ \
    --model BSRNet --output results/urban100_official/

# 2. 自学習モデルで評価
python scripts/run_bsrnet_custom.py \
    --checkpoint results/train_bsrgan_psnr/best.pth \
    --input testsets/urban100/HR/ \
    --output results/urban100_custom/

# 3. PSNR 計算（スクリプト未作成、別途必要）
```

> PSNR 計算スクリプトは未作成。必要に応じて作成する。
