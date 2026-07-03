from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(reference, clip01(estimate), data_range=1.0))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(structural_similarity(reference, clip01(estimate), data_range=1.0))

