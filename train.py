from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import psnr, ssim
from .model import DIPUNet


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 600
    eval_interval: int = 25
    lr: float = 0.01
    input_channels: int = 8
    base_channels: int = 24
    input_noise_std: float = 0.03
    seed: int = 123
    device: str = "cpu"


@dataclass
class TrainResult:
    best_output: np.ndarray
    final_output: np.ndarray
    best_step: int
    best_psnr: float
    best_ssim: float
    final_psnr: float
    final_ssim: float
    history: list[dict[str, float]]


def _to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32))[None, None].to(device)


def train_dip(clean: np.ndarray, noisy: np.ndarray, config: TrainConfig) -> TrainResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = config.device
    target = _to_tensor(noisy, device)
    net = DIPUNet(
        input_channels=config.input_channels,
        base_channels=config.base_channels,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    net_input = torch.rand(
        1,
        config.input_channels,
        clean.shape[0],
        clean.shape[1],
        device=device,
    )

    history: list[dict[str, float]] = []
    best_output: np.ndarray | None = None
    final_output: np.ndarray | None = None
    best_step = 0
    best_psnr = -float("inf")
    best_ssim = -float("inf")

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        model_input = net_input
        if config.input_noise_std > 0:
            model_input = net_input + torch.randn_like(net_input) * config.input_noise_std
        output = net(model_input)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        should_eval = step == 1 or step % config.eval_interval == 0 or step == config.steps
        if should_eval:
            with torch.no_grad():
                eval_output = net(net_input).detach().cpu().numpy()[0, 0]
            eval_output = np.clip(eval_output.astype(np.float32), 0.0, 1.0)
            current_psnr = psnr(clean, eval_output)
            current_ssim = ssim(clean, eval_output)
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "psnr": current_psnr,
                    "ssim": current_ssim,
                }
            )
            final_output = eval_output
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_ssim = current_ssim
                best_step = step
                best_output = eval_output.copy()

    if best_output is None or final_output is None:
        raise RuntimeError("Training produced no evaluation output.")

    return TrainResult(
        best_output=best_output,
        final_output=final_output,
        best_step=best_step,
        best_psnr=best_psnr,
        best_ssim=best_ssim,
        final_psnr=psnr(clean, final_output),
        final_ssim=ssim(clean, final_output),
        history=history,
    )


def has_early_stop_decline(history: list[dict[str, float]], min_drop_db: float = 0.05) -> bool:
    if len(history) < 3:
        return False
    psnr_values = [item["psnr"] for item in history]
    peak_index = int(np.argmax(psnr_values))
    if peak_index >= len(psnr_values) - 1:
        return False
    tail_min = min(psnr_values[peak_index + 1 :])
    return psnr_values[peak_index] - tail_min >= min_drop_db

