"""Run Dome-DETR-S baseline training with the workspace fallback backend."""

import argparse
import os
import runpy
import sys
from pathlib import Path

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

def parse_args() -> argparse.Namespace:
    """Parse workspace-specific training options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Install the fallback and delegate to the upstream training entry point."""
    args = parse_args()
    config_path = WORKSPACE_ROOT / "configs" / "dome_s_pilot.yaml"
    output_dir = args.output_dir or (
        WORKSPACE_ROOT / "artifacts" / "logs" / f"dome_s_baseline_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        __import__("MultiScaleDeformableAttention")
    except ModuleNotFoundError:
        from JetsonDinh_workspace.src.utils.ms_deform_fallback import install_fallback

        install_fallback()

    train_entrypoint = DOME_ROOT / "train.py"
    sys.argv = [
        str(train_entrypoint),
        "--config",
        str(config_path),
        "--seed",
        str(args.seed),
        "--update",
        f"output_dir={output_dir}",
    ]
    if args.epochs is not None:
        sys.argv.append(f"epoches={args.epochs}")

    runpy.run_path(str(train_entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
