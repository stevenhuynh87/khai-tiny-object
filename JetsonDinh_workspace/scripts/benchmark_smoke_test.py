"""Run a fixed-shape Conv2d smoke benchmark."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT / "JetsonDinh_workspace"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from JetsonDinh_workspace.src.utils.benchmark import (  # noqa: E402
    measure_edge_metrics,
    save_benchmark_report,
)


def main() -> None:
    model = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    results = measure_edge_metrics(
        model,
        input_shape=(1, 3, 800, 800),
        warmup_iterations=100,
        measured_iterations=1000,
        device="cuda",
    )
    output_path = (
        WORKSPACE_ROOT / "artifacts" / "logs" / "conv2d_smoke_benchmark.json"
    )
    save_benchmark_report(output_path, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
