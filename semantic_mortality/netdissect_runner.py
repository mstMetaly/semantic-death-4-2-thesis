from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from typing import Iterable

from .config import Config, RunPaths


def _checkpoint_epochs(checkpoints_dir: Path) -> list[int]:
    epochs = []
    for ckpt in checkpoints_dir.glob("epoch_*.pt"):
        try:
            epoch = int(ckpt.stem.split("_")[-1])
        except ValueError:
            continue
        epochs.append(epoch)
    return sorted(epochs)


def _write_run_settings(
    settings_path: Path,
    *,
    model_file: Path,
    output_folder: Path,
    dataset: str,
) -> None:
    content = (
        "from settings import *\n"
        f"MODEL_FILE = r\"{model_file.as_posix()}\"\n"
        "MODEL_PARALLEL = False\n"
        f"DATASET = \"{dataset}\"\n"
        f"OUTPUT_FOLDER = r\"{output_folder.as_posix()}\"\n"
        "TEST_MODE = False\n"
        "GPU = True\n"
        "CLEAN = False\n"
    )
    settings_path.write_text(content, encoding="utf-8")


def run_netdissect_per_checkpoint(
    config: Config,
    paths: RunPaths,
    epochs: Iterable[int] | None = None,
) -> None:
    net_cfg = config.netdissect
    net_root = Path(net_cfg["root"])

    run_settings = net_root / "settings_run.py"
    if epochs is None:
        epochs = _checkpoint_epochs(paths.checkpoints_dir)

    for epoch in epochs:
        ckpt_path = paths.checkpoints_dir / f"epoch_{epoch}.pt"
        if not ckpt_path.exists():
            continue
        output_dir = paths.dissection_dir / f"epoch_{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)

        _write_run_settings(
            run_settings,
            model_file=ckpt_path,
            output_folder=output_dir,
            dataset="places365",
        )

        if str(net_root) not in sys.path:
            sys.path.insert(0, str(net_root))

        sys.modules.pop("settings", None)
        spec = importlib.util.spec_from_file_location("settings_run", run_settings)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load NetDissect settings override.")
        settings_override = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings_override)
        sys.modules["settings"] = settings_override

        runpy.run_path(str(net_root / "main.py"), run_name="__main__")
