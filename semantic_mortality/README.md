# Semantic Mortality Pipeline

This folder contains the training + checkpointing pipeline and the temporal
tracking/analysis utilities for semantic mortality experiments.

## Quick Start

1) Train with checkpointing:

```
python -m semantic_mortality.pipeline --stage train --config configs/places365_resnet18.json
```

2) Run NetDissect-Lite per checkpoint and place outputs under:
`runs/places365_resnet18/dissection/epoch_<N>/tally.csv`

Or run it automatically:

```
python -m semantic_mortality.pipeline --stage dissect --config configs/places365_resnet18.json
```

3) Build trajectories and analyze mortality:

```
python -m semantic_mortality.pipeline --stage track --config configs/places365_resnet18.json
python -m semantic_mortality.pipeline --stage analyze --config configs/places365_resnet18.json
```

4) Generate plots:

```
python -m semantic_mortality.pipeline --stage plot --config configs/places365_resnet18.json
```

## Notes
- The pipeline expects `tally.csv` from NetDissect-Lite for each epoch.
- Functional death uses Captum; enable it in config and install `captum`.
- Colab section appended in `creating-checkpoint-notebook/semantic_death_create_checkpoint.ipynb`.

## Colab Pipeline: What Happens Per Cell
This brief describes the **PIPELINE** cells in
`creating-checkpoint-notebook/semantic_death_create_checkpoint.ipynb`.

1) **PIPELINE MODE**
   - Sets `RUN_MODE` to `quick` (sanity run) or `full` (thesis run).

2) **REPO SETUP**
   - Clones/pulls the GitHub repo.
   - Installs dependencies (`torch`, `torchvision`, `scikit-image`, `imageio`, etc.).

3) **PLACES365 DATASET**
   - Downloads Places365 **val** set and metadata.
   - Full train download is optional (needed for full run).

4) **TRAIN + CHECKPOINT**
   - Loads config and updates it for Colab paths.
   - Trains ResNet-18 and saves checkpoints per epoch.
   - Output: `runs/places365_resnet18/checkpoints/epoch_<N>.pt`
   - Output: `runs/places365_resnet18/metrics.csv`

5) **NETDISSECT DATA (BRODEN)**
   - Downloads Broden and verifies `dataset/broden1_224/index.csv`.

6) **NETDISSECT PER CHECKPOINT**
   - Runs NetDissect-Lite for each checkpoint.
   - Output: `runs/places365_resnet18/dissection/epoch_<N>/tally.csv`

7) **TRACKING + ANALYSIS + PLOTS**
   - Builds trajectories across epochs.
   - Detects semantic mortality and migration events.
   - Generates plots and summary tables.
   - Output: `runs/places365_resnet18/analysis/mortality_events.csv`
   - Output: `runs/places365_resnet18/analysis/summary.csv`
   - Output: `runs/places365_resnet18/analysis/concept_death_counts.csv`
   - Output: `runs/places365_resnet18/plots/*.png`

## Expected Outputs (What They Look Like)
- `metrics.csv`: epoch-wise train/val loss + accuracy.
- `tally.csv`: per-unit top concept + IoU score.
- `trajectories.csv`: per unit, per epoch concept/score.
- `mortality_events.csv`: unit, concept, birth epoch, death epoch, migration flag.
- Plots: mortality per epoch, cumulative mortality, top concepts.

## Is This Thesis-Worthy?
Yes—**if** you run the full experiment (full Places365 train split, enough epochs,
and consistent NetDissect runs per checkpoint). The pipeline produces:
- a longitudinal dataset of neuron-concept trajectories,
- a measurable definition of semantic mortality,
- interpretable plots/tables for analysis chapters.

For a full thesis result, use **RUN_MODE = "full"** and report results across
multiple seeds or at least one ablation (e.g., threshold `tau` or architecture).
