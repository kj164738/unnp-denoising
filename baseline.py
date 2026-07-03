from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from .metrics import clip01


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_bm3d_worker(input_path: Path, output_path: Path, sigma: float) -> None:
    project_root = _project_root()
    src = project_root / "src"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unnp_denoise.bm3d_worker",
            str(input_path),
            str(output_path),
            str(sigma),
        ],
        cwd=project_root,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def denoise_bm3d(noisy: np.ndarray, sigma: float, scratch_dir: Path | None = None) -> np.ndarray:
    project_root = Path(__file__).resolve().parents[2]
    scratch = scratch_dir if scratch_dir is not None else project_root / "results" / "_bm3d_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    input_path = scratch / f"{run_id}_input.npy"
    output_path = scratch / f"{run_id}_output.npy"
    np.save(input_path, noisy.astype(np.float32))
    try:
        _run_bm3d_worker(input_path, output_path, sigma)
        denoised = np.load(output_path)
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
    return clip01(denoised)

