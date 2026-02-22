from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class TallyRow:
    unit: int
    category: str
    label: str
    score: float


def _parse_iter(name: str) -> int | None:
    """Extract global iteration from folder name like 'epoch_3_iter_1200'."""
    match = re.search(r"iter[_-](\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"epoch[_-](\d+)", name)
    if match:
        return int(match.group(1))
    return None


def _read_tally_csv(path: Path) -> List[TallyRow]:
    rows: List[TallyRow] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                TallyRow(
                    unit=int(row["unit"]),
                    category=row["category"],
                    label=row["label"],
                    score=float(row["score"]),
                )
            )
    return rows


def scan_dissection_dirs(dissection_root: Path) -> List[Tuple[int, Path]]:
    checkpoints: Dict[int, Path] = {}
    for child in dissection_root.iterdir():
        if not child.is_dir():
            continue
        step = _parse_iter(child.name)
        if step is not None:
            checkpoints[step] = child
    return sorted(checkpoints.items(), key=lambda item: item[0])


def build_trajectories(
    dissection_root: Path,
    output_path: Path,
    layers: Iterable[str],
) -> None:
    epochs = scan_dissection_dirs(dissection_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "checkpoint", "layer", "unit", "label", "score", "category"]
        )
        writer.writeheader()

        for step, ckpt_dir in checkpoints:
            for layer in layers:
                tally_path = ckpt_dir / layer / "tally.csv"
                if not tally_path.exists():
                    tally_path = ckpt_dir / "tally.csv"
                if not tally_path.exists():
                    continue
                rows = _read_tally_csv(tally_path)
                for row in rows:
                    writer.writerow(
                        {
                            "step": step,
                            "checkpoint": ckpt_dir.name,
                            "layer": layer,
                            "unit": row.unit,
                            "label": row.label,
                            "score": row.score,
                            "category": row.category,
                        }
                    )
