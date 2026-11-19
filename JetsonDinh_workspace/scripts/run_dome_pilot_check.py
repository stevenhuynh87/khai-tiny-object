"""Run a bounded Dome-DETR-S end-to-end pilot check."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT / "JetsonDinh_workspace"
DOME_ROOT = Path(r"F:/Jetson Quậy/CV_PR_HVTD/third_party/Dome-DETR")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DOME_ROOT) not in sys.path:
    sys.path.insert(0, str(DOME_ROOT))

from JetsonDinh_workspace.src.utils.ms_deform_fallback import install_fallback

install_fallback()

from src.core import YAMLConfig
from src.misc import dist_utils


def make_training_targets(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    original_sizes: torch.Tensor,
    image_ids: torch.Tensor,
) -> dict[str, Any]:
    """Build labels and boxes only for a plain diagnostic forward."""
    return {
        "labels": labels,
        "boxes": boxes,
        "orig_size": original_sizes,
        "size": original_sizes,
        "image_id": image_ids,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the Dome-DETR-S pilot check")

    dist_utils.setup_distributed(
        print_rank=0,
        print_method="builtin",
        seed=2025,
    )

    config_path = WORKSPACE_ROOT / "configs" / "dome_s_pilot.yaml"
    cfg = YAMLConfig(str(config_path), seed=2025, device=str(device))

    model = cfg.model.to(device).eval()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer

    torch.cuda.reset_peak_memory_stats(device)
    inference_start = torch.cuda.memory_allocated(device)
    dummy_input = torch.randn(1, 3, 800, 800, device=device)
    with torch.inference_mode():
        _ = model(dummy_input)
    torch.cuda.synchronize(device)
    inference_peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    total_params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader

    model.train()
    criterion.train()
    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []

    try:
        for iteration, batch in enumerate(train_loader):
            if iteration >= 10:
                break
            samples = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch

            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                targets = batch[1]
            else:
                targets = None

            optimizer.zero_grad(set_to_none=True)
            if targets is None:
                outputs = model(samples)
                loss = sum(
                    output.float().square().mean()
                    for output in outputs.values()
                    if isinstance(output, torch.Tensor)
                )
            else:
                targets = [
                    {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
                    for target in targets
                ]
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            if iteration in {0, 9}:
                print(
                    f"[Pilot Train] iteration={iteration + 1} "
                    f"loss={losses[-1]:.6f}"
                )

        torch.cuda.synchronize(device)
        train_peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    except Exception as exc:
        report = {
            "backend": "pytorch_fallback",
            "seed": 2025,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "gpu": torch.cuda.get_device_name(device),
            "inference_peak_vram_mb": round(inference_peak_vram_mb, 2),
        }
        report_path = (
            WORKSPACE_ROOT / "artifacts" / "logs" / "dome_pilot_check.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        raise

    model.eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    evaluator = cfg.evaluator
    torch.cuda.reset_peak_memory_stats(device)
    val_images = 0
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                if val_images >= 20:
                    break
                samples = batch[0].to(device)
                targets = [
                    {
                        key: value.to(device) if hasattr(value, "to") else value
                        for key, value in target.items()
                    }
                    for target in batch[1]
                ]
                outputs = model(samples, targets=targets)
                original_sizes = torch.stack(
                    [target["orig_size"] for target in targets],
                    dim=0,
                )
                results = postprocessor(outputs, original_sizes)
                result_mapping = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                evaluator.update(result_mapping)
                val_images += len(targets)
                torch.cuda.synchronize(device)
        val_peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    except Exception as exc:
        report = {
            "backend": "pytorch_fallback",
            "seed": 2025,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "gpu": torch.cuda.get_device_name(device),
            "train_peak_vram_MB": round(train_peak_vram_mb, 2),
        }
        report_path = (
            WORKSPACE_ROOT / "artifacts" / "logs" / "dome_pilot_check.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        raise

    report: dict[str, Any] = {
        "backend": "pytorch_fallback",
        "seed": 2025,
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "params_M": round(total_params_m, 2),
        "inference_peak_vram_MB": round(inference_peak_vram_mb, 2),
        "train_peak_vram_MB": round(train_peak_vram_mb, 2),
        "val_peak_vram_MB": round(val_peak_vram_mb, 2),
        "train_iterations": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "val_evaluator": type(cfg.evaluator).__name__,
        "val_images_checked": val_images,
    }

    report_path = (
        WORKSPACE_ROOT / "artifacts" / "logs" / "dome_pilot_check.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
