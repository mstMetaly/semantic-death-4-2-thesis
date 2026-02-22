from __future__ import annotations

import importlib.util
import runpy
import sys
import subprocess
from pathlib import Path
from typing import Iterable

from .config import Config, RunPaths


def _list_checkpoints(checkpoints_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted list of (label, path) for all checkpoints."""
    results = []
    for ckpt in sorted(checkpoints_dir.glob("epoch_*.pt")):
        label = ckpt.stem  # e.g. "epoch_0_iter_300" or "epoch_0"
        results.append((label, ckpt))
    return results


def _write_run_settings(
    settings_path: Path,
    *,
    model_file: Path,
    output_folder: Path,
    dataset: str,
    data_directory: Path,
    index_file: str,
) -> None:
    model_file = model_file.resolve()
    output_folder = output_folder.resolve()
    data_directory = data_directory.resolve()
    content = (
        "from settings import *\n"
        f"MODEL_FILE = r\"{model_file.as_posix()}\"\n"
        "MODEL_PARALLEL = False\n"
        f"DATASET = \"{dataset}\"\n"
        "NUM_CLASSES = 365\n"
        f"DATA_DIRECTORY = r\"{data_directory.as_posix()}\"\n"
        f"INDEX_FILE = \"{index_file}\"\n"
        f"OUTPUT_FOLDER = r\"{output_folder.as_posix()}\"\n"
        "TEST_MODE = False\n"
        "GPU = True\n"
        "CLEAN = False\n"
    )
    settings_path.write_text(content, encoding="utf-8")


def run_netdissect_per_checkpoint(
    config: Config,
    paths: RunPaths,
    checkpoint_labels: Iterable[str] | None = None,
) -> None:
    net_cfg = config.netdissect
    net_root = Path(net_cfg["root"]).expanduser()
    if not net_root.is_absolute():
        net_root = (Path.cwd() / net_root).resolve()

    run_settings = net_root / "settings_run.py"

    all_ckpts = _list_checkpoints(paths.checkpoints_dir)
    if checkpoint_labels is not None:
        labels_set = set(checkpoint_labels)
        all_ckpts = [(label, path) for label, path in all_ckpts if label in labels_set]

    for label, ckpt_path in all_ckpts:
        output_dir = paths.dissection_dir / label
        output_dir.mkdir(parents=True, exist_ok=True)

        broden_dir = net_root / "dataset" / "broden1_224"
        _write_run_settings(
            run_settings,
            model_file=ckpt_path,
            output_folder=output_dir,
            dataset="places365",
            data_directory=broden_dir,
            index_file="index.csv",
        )
        print(f"Running NetDissect for {label}")

        python_exec = sys.executable
        code = (
            "import importlib.util, runpy, sys\n"
            "spec = importlib.util.spec_from_file_location('settings_run', 'settings_run.py')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "sys.modules['settings'] = mod\n"
            "runpy.run_path('main.py', run_name='__main__')\n"
        )
        subprocess.run(
            [python_exec, "-c", code],
            cwd=str(net_root),
            check=True,
        )
