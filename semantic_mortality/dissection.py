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


def _parse_epoch(name: str) -> int | None:
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
    epochs: Dict[int, Path] = {}
    for child in dissection_root.iterdir():
        if not child.is_dir():
            continue
        epoch = _parse_epoch(child.name)
        if epoch is not None:
            epochs[epoch] = child
    return sorted(epochs.items(), key=lambda item: item[0])


def build_trajectories(
    dissection_root: Path,
    output_path: Path,
    layers: Iterable[str],
) -> None:
    epochs = scan_dissection_dirs(dissection_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "layer", "unit", "label", "score", "category"]
        )
        writer.writeheader()

        for epoch, epoch_dir in epochs:
            for layer in layers:
                tally_path = epoch_dir / layer / "tally.csv"
                if not tally_path.exists():
                    tally_path = epoch_dir / "tally.csv"
                if not tally_path.exists():
                    continue
                rows = _read_tally_csv(tally_path)
                for row in rows:
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "layer": layer,
                            "unit": row.unit,
                            "label": row.label,
                            "score": row.score,
                            "category": row.category,
                        }
                    )
