from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from .analysis import mortality_counts_by_step
from .mortality import MortalityEvent


def plot_mortality_by_step(events: list[MortalityEvent], output_path: Path, dpi: int) -> None:
    counts = mortality_counts_by_step(events)
    if not counts:
        return
    steps = list(counts.keys())
    values = [counts[s] for s in steps]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, values, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Deaths")
    plt.title("Semantic Mortality by Training Step")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_cumulative_mortality(events: list[MortalityEvent], output_path: Path, dpi: int) -> None:
    counts = mortality_counts_by_step(events)
    if not counts:
        return
    steps = list(counts.keys())
    cumulative = []
    total = 0
    for step in steps:
        total += counts[step]
        cumulative.append(total)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, cumulative, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Cumulative Deaths")
    plt.title("Cumulative Semantic Mortality")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_top_concepts(events: list[MortalityEvent], output_path: Path, dpi: int, top_n: int) -> None:
    counts: Dict[str, int] = {}
    for event in events:
        counts[event.label] = counts.get(event.label, 0) + 1
    if not counts:
        return
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    plt.figure(figsize=(8, 5))
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel("Deaths")
    plt.title("Top Concepts by Mortality")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
