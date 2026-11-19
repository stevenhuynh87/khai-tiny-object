"""Measure edge inference cost and latency for a PyTorch model."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def measure_edge_metrics(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 800, 800),
    warmup_iterations: int = 100,
    measured_iterations: int = 1000,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure parameters, GFLOPs, peak VRAM, latency percentiles, and FPS."""
    if warmup_iterations < 100:
        raise ValueError("warmup_iterations must be at least 100")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; edge benchmark requires a CUDA device")

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)
    total_params = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    gflops = -1.0
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = float(macs * 2.0 / 1e9)
    except ImportError:
        print("[Warning] thop is not installed; GFLOPs is set to -1.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        _ = model(dummy_input)
    torch.cuda.synchronize(device)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize(device)

    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(measured_iterations):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    mean_latency = float(np.mean(latency_array))
    return {
        "input_shape": list(input_shape),
        "params_M": round(total_params, 2),
        "gflops": round(gflops, 2),
        "peak_vram_MB": round(peak_vram_mb, 2),
        "latency_mean_ms": round(mean_latency, 2),
        "latency_p50_ms": round(float(np.percentile(latency_array, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latency_array, 95)), 2),
        "fps": round(1000.0 / mean_latency, 2),
        "device": torch.cuda.get_device_name(device),
    }


def save_benchmark_report(output_path: Path, results: dict[str, Any]) -> None:
    """Save benchmark results as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    sample_model = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    sample_results = measure_edge_metrics(
        sample_model,
        warmup_iterations=100,
        measured_iterations=1000,
    )
    print(json.dumps(sample_results, indent=2))
