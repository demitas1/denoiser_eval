# testsets/

評価用テストセットの置き場。**画像ファイルは gitignore 対象**（README / .gitkeep のみ追跡）。

## ディレクトリ構成

```
testsets/
  unsplash_lite_test/   — FFDNet / SCUNet 検証用（Unsplash Lite から分割、100枚 JPEG）
  custom_natural/       — BSRNet / BSRGAN 検証用（手動収集 CC0 画像、20〜30枚 PNG）
```

## テストセットの選択方法

各 `options/*.json` の `dataroot_test` を書き換えるか、スクリプトに `--testset` 引数を渡す
（`train_bsrgan_psnr.py` / `train_bsrgan_gan.py` は `dataroot_test` で制御）。

```json
// options/train_bsrgan_x4_psnr_unsplash.json
"dataroot_test": "testsets/custom_natural"
```

## テストセット一覧

| ディレクトリ | 対象モデル | 枚数 | 備考 |
|---|---|---|---|
| `unsplash_lite_test/` | FFDNet, SCUNet | 100 | prepare_unsplash_testset.py で自動生成 |
| `custom_natural/` | BSRNet, BSRGAN | 20〜30 | 手動収集（収集指針は README 参照） |
