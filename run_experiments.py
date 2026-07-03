from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", message=".*TripleDES.*")
warnings.filterwarnings("ignore", message=".*Blowfish.*")

import numpy as np

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unnp_denoise.baseline import denoise_bm3d
from unnp_denoise.data import add_gaussian_noise, load_standard_images
from unnp_denoise.metrics import psnr, ssim
from unnp_denoise.visualize import (
    make_comparison_grid,
    plot_psnr_curve,
    save_history_csv,
    save_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNNP / Deep Image Prior denoising experiments.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[15.0, 25.0, 50.0])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--demo-steps", type=int, default=1600)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input-channels", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def package_version(package_name: str) -> str:
    vendor = ROOT / "vendor"
    added_vendor = False
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.append(str(vendor))
        added_vendor = True
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not found"
    finally:
        if added_vendor:
            sys.path.remove(str(vendor))


def result_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "clean": output_dir / "images" / "clean",
        "noisy": output_dir / "images" / "noisy",
        "unnp": output_dir / "images" / "unnp",
        "bm3d": output_dir / "images" / "bm3d",
        "curves": output_dir / "curves",
        "comparisons": output_dir / "comparisons",
        "tables": output_dir / "tables",
        "histories": output_dir / "histories",
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    paths = result_paths(output_dir)
    images = load_standard_images(size=args.image_size)[: args.images]
    if not images:
        raise RuntimeError("No test images selected.")

    import torch

    from unnp_denoise.train import TrainConfig, has_early_stop_decline, train_dip

    torch.set_num_threads(max(1, min(args.cpu_threads, torch.get_num_threads())))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    comparison_count = 0
    demo_info: dict[str, object] | None = None

    print(f"Device: {device}")
    print(f"Images: {[image.name for image in images]}")
    print(f"Sigmas: {args.sigmas}")

    for image_index, image in enumerate(images):
        save_image(paths["clean"] / f"{image.name}.png", image.clean)
        for sigma_index, sigma in enumerate(args.sigmas):
            run_seed = args.seed + image_index * 100 + int(sigma * 10)
            noisy = add_gaussian_noise(image.clean, sigma=sigma, seed=run_seed)
            noisy_psnr = psnr(image.clean, noisy)
            noisy_ssim = ssim(image.clean, noisy)
            save_image(paths["noisy"] / f"{image.name}_sigma{int(sigma)}.png", noisy)

            bm3d_output = denoise_bm3d(noisy, sigma=sigma, scratch_dir=output_dir / "_bm3d_tmp")
            bm3d_psnr = psnr(image.clean, bm3d_output)
            bm3d_ssim = ssim(image.clean, bm3d_output)
            save_image(paths["bm3d"] / f"{image.name}_sigma{int(sigma)}.png", bm3d_output)

            is_demo_run = image_index == 0 and int(sigma) == 25
            steps = max(args.steps, args.demo_steps) if is_demo_run else args.steps
            config = TrainConfig(
                steps=steps,
                eval_interval=args.eval_interval,
                lr=args.lr,
                input_channels=args.input_channels,
                base_channels=args.base_channels,
                input_noise_std=args.input_noise_std,
                seed=run_seed,
                device=device,
            )
            print(f"Training UNNP: image={image.name} sigma={sigma:g} steps={steps}")
            result = train_dip(image.clean, noisy, config)

            base_name = f"{image.name}_sigma{int(sigma)}"
            save_image(paths["unnp"] / f"{base_name}_best.png", result.best_output)
            save_history_csv(paths["histories"] / f"{base_name}.csv", result.history)

            if is_demo_run:
                curve_path = paths["curves"] / f"{base_name}_early_stopping.png"
                plot_psnr_curve(
                    curve_path,
                    result.history,
                    title=f"Early stopping curve: {image.name}, sigma={int(sigma)}",
                )
                demo_info = {
                    "image": image.name,
                    "sigma": sigma,
                    "best_step": result.best_step,
                    "best_psnr": result.best_psnr,
                    "best_ssim": result.best_ssim,
                    "curve": str(curve_path.relative_to(ROOT)),
                    "has_decline": has_early_stop_decline(result.history),
                }

            if comparison_count < 3 and sigma in {25.0, 50.0}:
                make_comparison_grid(
                    paths["comparisons"] / f"{base_name}_comparison.png",
                    clean=image.clean,
                    noisy=noisy,
                    unnp=result.best_output,
                    bm3d=bm3d_output,
                    title=f"{image.name}, gaussian noise sigma={int(sigma)}",
                    subtitle=(
                        f"Noisy {noisy_psnr:.2f} dB | "
                        f"UNNP {result.best_psnr:.2f} dB at step {result.best_step} | "
                        f"BM3D {bm3d_psnr:.2f} dB"
                    ),
                )
                comparison_count += 1

            rows.append(
                {
                    "image": image.name,
                    "sigma": int(sigma),
                    "noisy_psnr": round(noisy_psnr, 4),
                    "noisy_ssim": round(noisy_ssim, 4),
                    "unnp_psnr": round(result.best_psnr, 4),
                    "unnp_ssim": round(result.best_ssim, 4),
                    "unnp_best_step": result.best_step,
                    "unnp_final_psnr": round(result.final_psnr, 4),
                    "bm3d_psnr": round(bm3d_psnr, 4),
                    "bm3d_ssim": round(bm3d_ssim, 4),
                }
            )
            long_rows.extend(
                [
                    {
                        "image": image.name,
                        "sigma": int(sigma),
                        "method": "noisy",
                        "psnr": round(noisy_psnr, 4),
                        "ssim": round(noisy_ssim, 4),
                        "best_step": "",
                    },
                    {
                        "image": image.name,
                        "sigma": int(sigma),
                        "method": "UNNP",
                        "psnr": round(result.best_psnr, 4),
                        "ssim": round(result.best_ssim, 4),
                        "best_step": result.best_step,
                    },
                    {
                        "image": image.name,
                        "sigma": int(sigma),
                        "method": "BM3D",
                        "psnr": round(bm3d_psnr, 4),
                        "ssim": round(bm3d_ssim, 4),
                        "best_step": "",
                    },
                ]
            )

    wide_table = paths["tables"] / "results_wide.csv"
    long_table = paths["tables"] / "results_long.csv"
    write_rows(wide_table, rows)
    write_rows(long_table, long_rows)

    summary = {
        "device": device,
        "torch_version": torch.__version__,
        "bm3d_version": package_version("bm3d"),
        "image_size": args.image_size,
        "images": [image.name for image in images],
        "sigmas": args.sigmas,
        "steps": args.steps,
        "demo_steps": args.demo_steps,
        "eval_interval": args.eval_interval,
        "demo": demo_info,
        "tables": {
            "wide": str(wide_table.relative_to(ROOT)),
            "long": str(long_table.relative_to(ROOT)),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Results table: {wide_table}")
    if demo_info is not None:
        print(
            "Early stopping demo: "
            f"step={demo_info['best_step']} psnr={demo_info['best_psnr']:.2f} "
            f"decline_seen={demo_info['has_decline']}"
        )


if __name__ == "__main__":
    main()
