from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="BM3D worker isolated from PyTorch imports.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("sigma", type=float)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    vendor = project_root / "vendor"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.append(str(vendor))

    from bm3d import bm3d

    noisy = np.load(args.input_path).astype(np.float32)
    denoised = bm3d(noisy, sigma_psd=args.sigma / 255.0)
    np.save(args.output_path, np.clip(denoised, 0.0, 1.0).astype(np.float32))


if __name__ == "__main__":
    main()
