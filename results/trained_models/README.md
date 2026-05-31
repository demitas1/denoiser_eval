# trained_models — カスタム学習済みモデル

このディレクトリにはプロジェクトで学習したモデルの重みを保存する。  
Git LFS で管理（`.gitattributes` に設定済み）。

命名規則: `{arch}_{train_phase}_{dataset}[_{variant}].pth`

---

## モデル一覧

### ffdnet_gray_scratch_unsplash_lite.pth

| 項目 | 値 |
|---|---|
| アーキテクチャ | FFDNet gray (`in_nc=1, nc=64, nb=15`) |
| 学習フェーズ | ゼロから（scratch） |
| 学習データ | Unsplash Lite（1,899 枚 JPEG、1080px幅） |
| テストデータ | Unsplash Lite test（100 枚、訓練セットから分割） |
| 学習設定 | `options/train_ffdnet_gray_unsplash.json` |
| イテレーション | 500k |
| 学習時間 | (記録なし) |
| Best PSNR | **33.88 dB**（σ=25、Unsplash test 100 枚） |
| PSNR | **28.83 dB**（σ=25、pexels-cc0-100-2 100 枚、seed=0） |
| 保存内容 | `state_dict` のみ（Best PSNR 時点） |
| 完了日 | 2026-05-22 |

```bash
# 推論（sigma スイープ）
python scripts/run_ffdnet_custom.py \
    --checkpoint results/trained_models/ffdnet_gray_scratch_unsplash_lite.pth \
    --input test_inputs/ --sigma 5 10 15 20 25
```

---

### bsrnet_x4_scratch_unsplash_lite_v1_best.pth

| 項目 | 値 |
|---|---|
| アーキテクチャ | RRDBNet x4 (`nf=64, gc=32, nb=23`) |
| 学習フェーズ | ゼロから（scratch）— PSNR フェーズ（L1 損失のみ） |
| 学習データ | Unsplash Lite（1,899 枚 JPEG、1080px幅） |
| テストデータ | pexels-cc0（41 枚 PNG、768×768）※ v1 時点の旧テストセット |
| 学習設定 | `options/train_bsrgan_x4_psnr_unsplash.json` |
| イテレーション | 500k（Best は step 254k） |
| 学習時間 | (記録なし) |
| LR スケジュール | 初期 1e-4 → 200k で 5e-5 → 400k で 2.5e-5 |
| 劣化パイプライン | BSRGAN degradation（`lq_patchsize=72, H_size=320, sf=4`） |
| EMA decay | 0.999 |
| Best PSNR（学習中） | **23.21 dB**（旧テストセット 41 枚、seed なし） |
| PSNR | **20.76 dB**（pexels-cc0-100-2 100 枚、seed=0） |
| 公式 BSRNet との差 | −0.10 dB（公式 20.86 dB、評価ばらつき以内） |
| 保存内容 | EMA 重み（`netE`）の `state_dict` のみ |
| 完了日 | 2026-05-26 |

```bash
# 推論（ダウンスケール版を保存）
python scripts/run_bsrnet_custom.py \
    --checkpoint results/trained_models/bsrnet_x4_scratch_unsplash_lite_v1_best.pth \
    --input test_inputs/ --output results/bsrnet_v1/

# PSNR 評価
python scripts/eval_bsrnet_psnr.py \
    --checkpoint results/trained_models/bsrnet_x4_scratch_unsplash_lite_v1_best.pth \
    --testset testsets/custom_natural/pexels-cc0-100-2/
```

---

### bsrnet_x4_scratch_unsplash_lite_v1_last.pth

| 項目 | 値 |
|---|---|
| アーキテクチャ | RRDBNet x4 (`nf=64, gc=32, nb=23`) |
| 学習フェーズ | ゼロから（scratch）— PSNR フェーズ（L1 損失のみ） |
| 学習データ | Unsplash Lite（1,899 枚 JPEG、1080px幅） |
| その他条件 | v1_best と同じ |
| イテレーション | step 500k（最終 EMA 重み） |
| PSNR | **20.62 dB**（pexels-cc0-100-2 100 枚、seed=0） |
| 保存内容 | EMA 重み（`netE`）の `state_dict` のみ |
| 備考 | GAN フェーズの出発点候補（best より知覚品質が高い場合あり） |

```bash
# GAN フェーズの pretrained_netG として使用する場合
cp results/trained_models/bsrnet_x4_scratch_unsplash_lite_v1_last.pth \
   models/KAIR/model_zoo/BSRNet_unsplash_v1.pth
# → options/train_bsrgan_x4_gan_finetune.json の pretrained_netG を上記パスに変更
```

---

### bsrnet_x4_psnr_scratch_unsplash_lite_v2_best.pth

| 項目 | 値 |
|---|---|
| アーキテクチャ | RRDBNet x4 (`nf=64, gc=32, nb=23`) |
| 学習フェーズ | ゼロから（scratch）— PSNR フェーズ（L1 損失のみ） |
| 学習データ | Unsplash Lite（3,511 枚 JPEG、1080px幅） |
| テストデータ（学習中モニタリング） | pexels-cc0-100-1（100 枚 PNG、768×768） |
| テストデータ（seed 固定評価） | pexels-cc0-100-2（100 枚 PNG、768×768） |
| 学習設定 | `options/train_bsrgan_x4_psnr_unsplash.json` |
| イテレーション | 500k（Best は step 470k） |
| GPU 学習時間 | 約 79h（RTX 3060、10 セッション合計） |
| LR スケジュール | 初期 1e-4 → 200k で 5e-5 → 400k で 2.5e-5 |
| 劣化パイプライン | BSRGAN degradation（`lq_patchsize=72, H_size=320, sf=4`） |
| EMA decay | 0.999 |
| Best PSNR（seed=0） | **21.14 dB**（pexels-cc0-100-2 100 枚） |
| Best SSIM（seed=0） | **0.4616**（同上） |
| 保存内容 | EMA 重み（`netE` @ step 470k）の `state_dict` のみ |
| 評価方法 | `np.random` + `random` 両方 seed=0 固定（quality_curve2.tsv） |
| 完了日 | 2026-05-30 |

```bash
# 推論（ダウンスケール版を保存）
python scripts/run_bsrnet_custom.py \
    --checkpoint results/trained_models/bsrnet_x4_psnr_scratch_unsplash_lite_v2_best.pth \
    --input test_inputs/ --output results/bsrnet_v2/

# PSNR+SSIM 評価
python scripts/eval_bsrnet_psnr.py \
    --checkpoint results/trained_models/bsrnet_x4_psnr_scratch_unsplash_lite_v2_best.pth \
    --testset testsets/custom_natural/pexels-cc0-100-2/
```

---

### bsrnet_x4_psnr_scratch_unsplash_lite_v2_last.pth

| 項目 | 値 |
|---|---|
| アーキテクチャ | RRDBNet x4 (`nf=64, gc=32, nb=23`) |
| 学習フェーズ | ゼロから（scratch）— PSNR フェーズ（L1 損失のみ） |
| その他条件 | v2_best と同じ |
| イテレーション | step 500k（最終 EMA 重み） |
| 保存内容 | EMA 重み（`last_E`）の `state_dict` のみ |
| 備考 | GAN フェーズの pretrained_netG 候補（best との比較用） |

```bash
# GAN フェーズの pretrained_netG として使用する場合
cp results/trained_models/bsrnet_x4_psnr_scratch_unsplash_lite_v2_best.pth \
   models/KAIR/model_zoo/BSRNet_unsplash_v2.pth
# → options/train_bsrgan_x4_gan_finetune.json の pretrained_netG を上記パスに変更
```

---

## v1 → v2 の変更点

| 項目 | v1 | v2 |
|---|---|---|
| 学習データ | Unsplash Lite ~1,900 枚 | Unsplash Lite 3,511 枚 |
| テストデータ（学習中） | pexels-cc0（41 枚）| pexels-cc0-100-1（100 枚） |
| 評価方式 | seed なし | seed=0 固定 |
| Best PSNR | 23.21 dB（旧テストセット）/ 20.76 dB（pexels-cc0-100-2） | 21.54 dB（pexels-cc0-100-2） |
| 出力ディレクトリ | `results/train_bsrgan_psnr/` | `results/train_bsrgan_psnr_v2/` |
