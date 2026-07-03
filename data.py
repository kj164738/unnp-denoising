from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from skimage import color, data, img_as_float32, transform


@dataclass(frozen=True)
class TestImage:
    name: str
    clean: np.ndarray


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    image = img_as_float32(image)
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return np.asarray(image, dtype=np.float32)


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top : top + side, left : left + side]


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    image = _center_crop_square(image)
    resized = transform.resize(
        image,
        (size, size),
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(resized.astype(np.float32), 0.0, 1.0)


def load_standard_images(size: int = 64, names: Iterable[str] | None = None) -> list[TestImage]:
    sources = {
        "camera": data.camera(),
        "astronaut": data.astronaut(),
        "coffee": data.coffee(),
        "coins": data.coins(),
        "moon": data.moon(),
    }
    selected_names = list(names) if names is not None else list(sources)
    images: list[TestImage] = []
    for name in selected_names:
        if name not in sources:
            raise ValueError(f"Unknown image {name!r}. Available: {', '.join(sources)}")
        clean = _resize(_to_gray_float(sources[name]), size)
        images.append(TestImage(name=name, clean=clean))
    return images


def add_gaussian_noise(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma / 255.0, size=clean.shape).astype(np.float32)
    return np.clip(clean + noise, 0.0, 1.0).astype(np.float32)

