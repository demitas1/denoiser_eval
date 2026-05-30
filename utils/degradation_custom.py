"""
カスタム劣化パイプライン（idx 7–12）。

鉛筆スケッチ固有の劣化を追加実装する。
BSRGAN 既存劣化（idx 0–6, utils_blindsr.py）の拡張として設計され、
visualize_degradation.py と DatasetBlindSR の両方から使用できる。

インターフェース:
    入力: np.ndarray float32 [0, 1]、shape (H, W) または (H, W, C)
    出力: 同型・同値域

idx 割り当て:
    7  : 消し跡フィルター     (eraser_trace)       — issue #6
    8  : 等方スメアフィルター  (isotropic_smear)    — issue #4
    9  : 方向性スメアフィルター(directional_smear)  — issue #5
    10 : 紙粒感フィルター      (paper_grain)        — issue #3
    11 : しみフィルター        (stain)              — issue #7
    12 : 圧力ムラフィルター    (pressure_variation) — issue #2
"""

import numpy as np


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _to_hwc(img: np.ndarray):
    """(H, W) → (H, W, 1) に正規化。(H, W, C) はそのまま返す。"""
    if img.ndim == 2:
        return img[:, :, np.newaxis], True
    return img, False


def _from_hwc(img: np.ndarray, was_2d: bool) -> np.ndarray:
    """_to_hwc で 2D だった場合、squeeze して (H, W) に戻す。"""
    if was_2d:
        return img[:, :, 0]
    return img


# ---------------------------------------------------------------------------
# idx 7–12 の各フィルター（現在はすべて no-op スタブ）
# ---------------------------------------------------------------------------

def apply_eraser_trace(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 7: 消し跡フィルター（issue #6）— 未実装。"""
    print('[degradation_custom] idx 7 (eraser_trace) is not yet implemented, returning input.')
    return img.copy()


def apply_isotropic_smear(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 8: 等方スメアフィルター（issue #4）— 未実装。"""
    print('[degradation_custom] idx 8 (isotropic_smear) is not yet implemented, returning input.')
    return img.copy()


def apply_directional_smear(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 9: 方向性スメアフィルター（issue #5）— 未実装。"""
    print('[degradation_custom] idx 9 (directional_smear) is not yet implemented, returning input.')
    return img.copy()


def apply_paper_grain(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 10: 紙粒感フィルター（issue #3）— 未実装。"""
    print('[degradation_custom] idx 10 (paper_grain) is not yet implemented, returning input.')
    return img.copy()


def apply_stain(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 11: しみフィルター（issue #7）— 未実装。"""
    print('[degradation_custom] idx 11 (stain) is not yet implemented, returning input.')
    return img.copy()


def apply_pressure_variation(img: np.ndarray, **kwargs) -> np.ndarray:
    """idx 12: 圧力ムラフィルター（issue #2）— 未実装。"""
    print('[degradation_custom] idx 12 (pressure_variation) is not yet implemented, returning input.')
    return img.copy()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CUSTOM_OPS = {
    7:  apply_eraser_trace,
    8:  apply_isotropic_smear,
    9:  apply_directional_smear,
    10: apply_paper_grain,
    11: apply_stain,
    12: apply_pressure_variation,
}


def apply_custom_op(idx: int, img: np.ndarray, **params) -> np.ndarray:
    """idx に対応するカスタム劣化を適用する。"""
    if idx not in CUSTOM_OPS:
        raise ValueError(f'Unknown custom degradation index: {idx}. Valid: {sorted(CUSTOM_OPS)}')
    return CUSTOM_OPS[idx](img, **params)
