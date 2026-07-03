from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "loss", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(history)


def plot_psnr_curve(path: Path, history: list[dict[str, float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [item["step"] for item in history]
    values = [item["psnr"] for item in history]
    best_index = int(np.argmax(values))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.5)
    plt.scatter([steps[best_index]], [values[best_index]], color="crimson", zorder=3)
    plt.annotate(
        f"best: step {int(steps[best_index])}, {values[best_index]:.2f} dB",
        xy=(steps[best_index], values[best_index]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
        va="top",
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    plt.xlabel("Iteration")
    plt.ylabel("PSNR vs clean image (dB)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_comparison_grid(
    path: Path,
    clean: np.ndarray,
    noisy: np.ndarray,
    unnp: np.ndarray,
    bm3d: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [noisy, unnp, bm3d, clean]
    labels = ["Noisy", "UNNP / DIP", "BM3D", "Clean"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for axis, image, label in zip(axes, images, labels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 0.92))
    plt.savefig(path, dpi=180)
    plt.close()
