"""Validate the Dome-DETR-S AI-TOD configuration before training."""

import os
import sys
from pathlib import Path

import yaml

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT / "JetsonDinh_workspace"
DOME_ROOT = Path(r"F:/Jetson Quậy/CV_PR_HVTD/third_party/Dome-DETR")
DATASET_ROOT = Path(r"F:/Jetson Quậy/CV_PR_HVTD/data/AI-TOD-v2")


def load_yaml(path: Path) -> dict:
    """Load a YAML file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    dome_config = WORKSPACE_ROOT / "configs/dome_s_pilot.yaml"
    dataset_config = DOME_ROOT / "configs/dataset/aitod_detection.yml"
    dataloader_config = DOME_ROOT / "configs/dome/include/dataloader.yml"
    hgnet_config = DOME_ROOT / "configs/dome/include/dome_hgnetv2.yml"
    pilot_values = load_yaml(dome_config)
    train_dataset = pilot_values["train_dataloader"]["dataset"]
    val_dataset = pilot_values["val_dataloader"]["dataset"]

    values = {
        "config_exists": dome_config.exists(),
        "dataset_config_exists": dataset_config.exists(),
        "train_annotation_exists": (
            Path(train_dataset["ann_file"])
        ).exists(),
        "val_annotation_exists": (
            Path(val_dataset["ann_file"])
        ).exists(),
        "train_images_exists": Path(train_dataset["img_folder"]).exists(),
        "val_images_exists": Path(val_dataset["img_folder"]).exists(),
    }

    dataset_values = load_yaml(dataset_config)
    dataloader_values = load_yaml(dataloader_config)
    hgnet_values = load_yaml(hgnet_config)

    values.update(
        {
            "train_batch_size": pilot_values["train_dataloader"]["total_batch_size"],
            "val_batch_size": pilot_values["val_dataloader"]["total_batch_size"],
            "eval_spatial_size": hgnet_values["eval_spatial_size"],
            "train_input_size": dataloader_values["train_dataloader"]["dataset"]["transforms"]["ops"][5][
                "size"
            ],
            "num_classes": dataset_values["num_classes"],
            "evaluator": dataset_values["evaluator"]["type"],
        }
    )

    missing = [name for name, exists in values.items() if name.endswith("exists") and not exists]
    if missing:
        raise RuntimeError(f"Missing required Dome-DETR paths: {missing}")

    report_path = (
        WORKSPACE_ROOT / "artifacts" / "env_reports" / "dome_s_config_validation.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        import json

        json.dump(values, file, indent=2)

    print(f"[Dome-S] Configuration validation completed: {report_path}")
    for key, value in values.items():
        print(f"[Dome-S] {key}: {value}")


if __name__ == "__main__":
    main()
