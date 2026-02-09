from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    metrics_path: Path
    dissection_dir: Path
    tracking_dir: Path
    analysis_dir: Path
    plots_dir: Path
    config_path: Path


@dataclass
class Config:
    environment: str = "dev"
    run_name: str = "places365_resnet18"
    data: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    tracking: Dict[str, Any] = field(default_factory=dict)
    migration: Dict[str, Any] = field(default_factory=dict)
    functional_death: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Any] = field(default_factory=dict)
    netdissect: Dict[str, Any] = field(default_factory=dict)


def default_config(environment: str = "dev") -> Config:
    env = environment.lower()
    if env not in {"dev", "test", "prod"}:
        env = "dev"

    if env == "dev":
        max_train_samples = 20000
        epochs = 5
        batch_size = 32
    elif env == "test":
        max_train_samples = 2000
        epochs = 2
        batch_size = 16
    else:
        max_train_samples = None
        epochs = 90
        batch_size = 128

    return Config(
        environment=env,
        data={
            "dataset_root": "data/places365",
            "train_split": "train-standard",
            "val_split": "val",
            "small": True,
            "num_workers": 4,
            "max_train_samples": max_train_samples,
            "seed": 42,
        },
        training={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "step_size": 30,
            "gamma": 0.1,
            "pretrained": True,
            "device": "auto",
            "save_every": 1,
        },
        tracking={
            "tau": 0.04,
            "k": 3,
            "min_alive_epochs": 1,
            "smoothing_window": 1,
            "layers": ["layer4"],
        },
        migration={
            "window": 3,
            "min_delta": 0.01,
        },
        functional_death={
            "enabled": False,
            "max_samples": 128,
            "batch_size": 16,
            "threshold": 1e-4,
        },
        plots={
            "top_concepts": 20,
            "dpi": 160,
        },
        netdissect={
            "root": "ND-ON-SD/NetDissect-Lite",
            "settings_module": "settings",
            "python": "python",
        },
    )


def load_config(path: Path) -> Config:
    if not path.exists():
        return default_config()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    base = default_config(raw.get("environment", "dev"))
    for key in ["environment", "run_name"]:
        if key in raw:
            setattr(base, key, raw[key])
    for section in [
        "data",
        "training",
        "tracking",
        "migration",
        "functional_death",
        "plots",
        "netdissect",
    ]:
        if section in raw and isinstance(raw[section], dict):
            getattr(base, section).update(raw[section])
    return base


def save_config(config: Config, path: Path) -> None:
    payload = {
        "environment": config.environment,
        "run_name": config.run_name,
        "data": config.data,
        "training": config.training,
        "tracking": config.tracking,
        "migration": config.migration,
        "functional_death": config.functional_death,
        "plots": config.plots,
        "netdissect": config.netdissect,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_run_paths(root: Path, run_name: str) -> RunPaths:
    run_dir = root / run_name
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        metrics_path=run_dir / "metrics.csv",
        dissection_dir=run_dir / "dissection",
        tracking_dir=run_dir / "tracking",
        analysis_dir=run_dir / "analysis",
        plots_dir=run_dir / "plots",
        config_path=run_dir / "config.json",
    )
