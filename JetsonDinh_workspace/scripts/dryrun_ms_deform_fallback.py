"""Verify the pure-PyTorch MSDeformAttn fallback on Windows."""

import os
import sys
import json
from pathlib import Path

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

from src.zoo.dome.ops.modules.ms_deform_attn import MSDeformAttn


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this dry-run")

    device = torch.device("cuda")
    attention = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4).to(device)

    batch_size = 1
    num_queries = 300
    num_heads = 8
    num_levels = 4
    num_points = 4
    head_dim = 32

    query = torch.randn(batch_size, num_queries, 256, device=device, requires_grad=True)
    reference_points = torch.rand(
        batch_size,
        num_queries,
        num_levels,
        2,
        device=device,
    )
    spatial_shapes = torch.tensor(
        [[200, 200], [100, 100], [50, 50], [25, 25]],
        dtype=torch.long,
        device=device,
    )
    level_start_index = torch.tensor([0, 40000, 50000, 52500], device=device)
    input_flatten = torch.randn(
        batch_size,
        53125,
        256,
        device=device,
        requires_grad=True,
    )
    losses: list[float] = []
    forward_shapes: list[tuple[int, ...]] = []

    for iteration in range(50):
        output = attention(
            query,
            reference_points,
            input_flatten,
            spatial_shapes,
            level_start_index,
        )
        loss = output.float().square().mean()
        losses.append(float(loss.item()))
        forward_shapes.append(tuple(output.shape))
        loss.backward()
        attention.zero_grad(set_to_none=True)
        query.grad = None
        input_flatten.grad = None
        if iteration in {0, 49}:
            print(
                f"[Fallback Dry Run] iteration={iteration + 1} "
                f"loss={loss.item():.6f} shape={tuple(output.shape)}"
            )

    report = {
        "kernel": "ms_deform_attn_core_pytorch",
        "iterations": 50,
        "input_shape": [1, 53125, 256],
        "query_shape": [1, num_queries, 256],
        "output_shape": [1, num_queries, 256],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "device": torch.cuda.get_device_name(0),
    }
    report_path = (
        WORKSPACE_ROOT / "artifacts" / "logs" / "ms_deform_fallback_dryrun.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print("[Fallback Dry Run] 50 iterations completed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
