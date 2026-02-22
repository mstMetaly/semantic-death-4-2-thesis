from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .mortality import MortalityEvent


def write_summary(events: List[MortalityEvent], output_path: Path) -> None:
    by_step = Counter(event.step_death for event in events)
    by_layer = Counter(event.layer for event in events)
    by_concept = Counter(event.label for event in events)
    migrated = sum(1 for event in events if event.migrated)
    functional = sum(1 for event in events if event.functional_dead)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_events", len(events)])
        writer.writerow(["migrated_events", migrated])
        writer.writerow(["functional_dead_events", functional])

        for step, count in sorted(by_step.items()):
            writer.writerow([f"step_{step}_deaths", count])

        for layer, count in sorted(by_layer.items()):
            writer.writerow([f"layer_{layer}_deaths", count])

    concept_path = output_path.parent / "concept_death_counts.csv"
    with concept_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count"])
        for label, count in by_concept.most_common():
            writer.writerow([label, count])


def load_events(events_path: Path) -> List[MortalityEvent]:
    events: List[MortalityEvent] = []
    with events_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                MortalityEvent(
                    unit=int(row["unit"]),
                    layer=row["layer"],
                    label=row["label"],
                    step_birth=int(row["step_birth"]),
                    step_death=int(row["step_death"]),
                    score_birth=float(row["score_birth"]),
                    score_death=float(row["score_death"]),
                    migrated=row["migrated"].lower() == "true",
                    functional_dead=row["functional_dead"].lower() == "true",
                )
            )
    return events


def mortality_counts_by_step(events: List[MortalityEvent]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for event in events:
        counts[event.step_death] += 1
    return dict(sorted(counts.items()))
