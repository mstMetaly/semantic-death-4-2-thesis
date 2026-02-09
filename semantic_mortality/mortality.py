from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from .config import Config, RunPaths


@dataclass
class MortalityEvent:
    unit: int
    layer: str
    label: str
    epoch_birth: int
    epoch_death: int
    score_birth: float
    score_death: float
    migrated: bool = False
    functional_dead: bool = False


def _load_trajectories(path: Path) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    series: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["layer"], int(row["unit"]))
            series.setdefault(key, []).append(row)
    for key in series:
        series[key].sort(key=lambda r: int(r["epoch"]))
    return series


def _smooth(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_vals = values[start : idx + 1]
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed


def detect_semantic_mortality(
    trajectories_path: Path,
    tau: float,
    k: int,
    min_alive_epochs: int,
    smoothing_window: int,
) -> List[MortalityEvent]:
    series = _load_trajectories(trajectories_path)
    events: List[MortalityEvent] = []

    for (layer, unit), rows in series.items():
        epochs = [int(r["epoch"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        labels = [r["label"] for r in rows]

        scores = _smooth(scores, smoothing_window)
        alive_epochs = [i for i, s in enumerate(scores) if s >= tau]
        if len(alive_epochs) < min_alive_epochs:
            continue

        for idx in alive_epochs:
            if idx + k >= len(scores):
                continue
            window = scores[idx : idx + k + 1]
            if not all(window[i] >= window[i + 1] for i in range(len(window) - 1)):
                continue
            death_idx = None
            for j in range(idx + 1, len(scores)):
                if scores[j] < tau:
                    death_idx = j
                    break
            if death_idx is None:
                continue
            if any(scores[j] >= tau for j in range(death_idx + 1, len(scores))):
                continue

            events.append(
                MortalityEvent(
                    unit=unit,
                    layer=layer,
                    label=labels[death_idx - 1],
                    epoch_birth=epochs[idx],
                    epoch_death=epochs[death_idx],
                    score_birth=scores[idx],
                    score_death=scores[death_idx],
                )
            )
            break
    return events


def detect_migration(
    events: List[MortalityEvent],
    trajectories_path: Path,
    tau: float,
    window: int,
    min_delta: float = 0.0,
) -> None:
    series = _load_trajectories(trajectories_path)
    concept_index: Dict[str, List[Tuple[int, int, float]]] = {}

    for rows in series.values():
        for row in rows:
            concept_index.setdefault(row["label"], []).append(
                (int(row["epoch"]), int(row["unit"]), float(row["score"]))
            )

    for event in events:
        candidates = concept_index.get(event.label, [])
        before_scores = [
            score
            for epoch, unit, score in candidates
            if event.epoch_death - window <= epoch < event.epoch_death and score >= tau
        ]
        after_scores = [
            score
            for epoch, unit, score in candidates
            if event.epoch_death < epoch <= event.epoch_death + window and score >= tau
        ]
        before_avg = sum(before_scores) / max(len(before_scores), 1)
        after_avg = sum(after_scores) / max(len(after_scores), 1)
        event.migrated = len(after_scores) > len(before_scores) and (after_avg - before_avg) >= min_delta


def _build_model(pretrained: bool) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 365)
    return model


def _layer_from_model(model: nn.Module, layer_name: str) -> nn.Module:
    if layer_name == "layer4":
        return model.layer4
    raise ValueError(f"Unsupported layer: {layer_name}")


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt.state_dict())


def compute_functional_death(
    events: List[MortalityEvent],
    config: Config,
    paths: RunPaths,
    dataloader: DataLoader,
) -> None:
    if not config.functional_death.get("enabled", False):
        return

    captum_spec = importlib.util.find_spec("captum")
    if captum_spec is None:
        return

    from captum.attr import LayerConductance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = config.functional_death

    for event in events:
        before_epoch = max(event.epoch_birth, event.epoch_death - 1)
        after_epoch = event.epoch_death + 1
        before_path = paths.checkpoints_dir / f"epoch_{before_epoch}.pt"
        after_path = paths.checkpoints_dir / f"epoch_{after_epoch}.pt"
        if not before_path.exists() or not after_path.exists():
            continue

        def avg_conductance(checkpoint: Path) -> float:
            model = _build_model(pretrained=False).to(device)
            _load_checkpoint(model, checkpoint)
            model.eval()
            layer = _layer_from_model(model, event.layer)
            cond = LayerConductance(model, layer)

            channel_index = event.unit - 1
            total = 0.0
            count = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                attributions = cond.attribute(images, target=labels)
                channel_attr = attributions[:, channel_index]
                total += channel_attr.abs().mean().item()
                count += 1
                if count * images.size(0) >= thresholds["max_samples"]:
                    break
            return total / max(count, 1)

        before_score = avg_conductance(before_path)
        after_score = avg_conductance(after_path)
        event.functional_dead = after_score < thresholds["threshold"] and after_score < before_score


def write_events(events: List[MortalityEvent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "unit",
                "layer",
                "label",
                "epoch_birth",
                "epoch_death",
                "score_birth",
                "score_death",
                "migrated",
                "functional_dead",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "unit": event.unit,
                    "layer": event.layer,
                    "label": event.label,
                    "epoch_birth": event.epoch_birth,
                    "epoch_death": event.epoch_death,
                    "score_birth": event.score_birth,
                    "score_death": event.score_death,
                    "migrated": event.migrated,
                    "functional_dead": event.functional_dead,
                }
            )
