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
