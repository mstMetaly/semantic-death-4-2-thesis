from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import load_events, write_summary
from .config import Config, default_config, load_config, resolve_run_paths
from .dissection import build_trajectories
from .mortality import (
    detect_migration,
    detect_semantic_mortality,
    compute_functional_death,
    write_events,
)
from .netdissect_runner import run_netdissect_per_checkpoint
from .plots import (
    plot_cumulative_mortality,
    plot_mortality_by_epoch,
    plot_top_concepts,
)
from .train import train_and_checkpoint, _build_dataloaders


def _load_or_default(path: Path | None) -> Config:
    if path is None:
        return default_config()
    return load_config(path)


def run_train(config: Config, paths_root: Path) -> None:
    paths = resolve_run_paths(paths_root, config.run_name)
    train_and_checkpoint(config, paths)


def run_tracking(config: Config, paths_root: Path) -> Path:
    paths = resolve_run_paths(paths_root, config.run_name)
    trajectories_path = paths.tracking_dir / "trajectories.csv"
    build_trajectories(paths.dissection_dir, trajectories_path, config.tracking["layers"])
    return trajectories_path


def run_dissection(config: Config, paths_root: Path) -> None:
    paths = resolve_run_paths(paths_root, config.run_name)
    run_netdissect_per_checkpoint(config, paths)


def run_analysis(config: Config, paths_root: Path) -> Path:
    paths = resolve_run_paths(paths_root, config.run_name)
    trajectories_path = paths.tracking_dir / "trajectories.csv"
    events = detect_semantic_mortality(
        trajectories_path=trajectories_path,
        tau=config.tracking["tau"],
        k=config.tracking["k"],
        min_alive_epochs=config.tracking["min_alive_epochs"],
        smoothing_window=config.tracking["smoothing_window"],
    )
    detect_migration(
        events=events,
        trajectories_path=trajectories_path,
        tau=config.tracking["tau"],
        window=config.migration["window"],
        min_delta=config.migration["min_delta"],
    )

    if config.functional_death.get("enabled", False):
        paths = resolve_run_paths(paths_root, config.run_name)
        _, val_loader = _build_dataloaders(
            config, batch_size_override=config.functional_death["batch_size"]
        )
        compute_functional_death(events, config, paths, val_loader)

    events_path = paths.analysis_dir / "mortality_events.csv"
    write_events(events, events_path)
    write_summary(events, paths.analysis_dir / "summary.csv")
    return events_path


def run_plots(config: Config, paths_root: Path) -> None:
    paths = resolve_run_paths(paths_root, config.run_name)
    events_path = paths.analysis_dir / "mortality_events.csv"
    events = load_events(events_path)

    plot_mortality_by_epoch(
        events,
        paths.plots_dir / "mortality_by_epoch.png",
        dpi=config.plots["dpi"],
    )
    plot_cumulative_mortality(
        events,
        paths.plots_dir / "cumulative_mortality.png",
        dpi=config.plots["dpi"],
    )
    plot_top_concepts(
        events,
        paths.plots_dir / "top_concepts.png",
        dpi=config.plots["dpi"],
        top_n=config.plots["top_concepts"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Mortality Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs_subset",
        help="Root directory for outputs",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["train", "dissect", "track", "analyze", "plot", "all"],
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config = _load_or_default(config_path)
    run_root = Path(args.run_root)

    if args.stage in {"train", "all"}:
        run_train(config, run_root)
    if args.stage in {"dissect", "all"}:
        run_dissection(config, run_root)
    if args.stage in {"track", "all"}:
        run_tracking(config, run_root)
    if args.stage in {"analyze", "all"}:
        run_analysis(config, run_root)
    if args.stage in {"plot", "all"}:
        run_plots(config, run_root)


if __name__ == "__main__":
    main()
