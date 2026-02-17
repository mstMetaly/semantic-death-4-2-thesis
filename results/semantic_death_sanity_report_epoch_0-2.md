# Semantic Death Sanity Report (Epoch 0–2)

Source file: `ND-ON-SD/results/trajectories_epoch_0-2.csv`  
Goal: identify early signs of semantic death (concept alignment drops below tau).

## Summary
This 3‑epoch slice shows **clear early decay** and **label drift** for multiple units.
That is a *good sign* for semantic death, but **not sufficient** to confirm permanent loss
because we do not observe later recovery.

## Assumptions
- Threshold (tau) = 0.04 (matches pipeline default).
- “Semantic death” requires sustained drop and no recovery (cannot be confirmed with only 3 epochs).

## Likely candidates (early death / drift)
These units show strong alignment at epoch 0 and fall below tau by epoch 2:

### Unit 222 (layer4)
- Epoch 0: **grass** 0.1497 (object)
- Epoch 1: **grass** 0.0847 (object)
- Epoch 2: **cracked** 0.0201 (texture, below tau)
**Interpretation:** strong early concept, then large drop + label shift → strong candidate.

### Unit 134 (layer4)
- Epoch 0: **water** 0.1034 (object)
- Epoch 1: **water** 0.0600 (object)
- Epoch 2: **water** 0.0339 (object, below tau)
**Interpretation:** consistent label but decays below tau → candidate for semantic death.

### Unit 228 (layer4)
- Epoch 0: **skyscraper-s** 0.0932 (scene)
- Epoch 1: **grooved** 0.0423 (texture)
- Epoch 2: **grooved** 0.0339 (texture, below tau)
**Interpretation:** label drift + drop below tau → candidate for semantic death or migration.

## Possibly stable (not dead yet)
### Unit 420 (layer4)
- Epoch 0: **grooved** 0.1177 (texture)
- Epoch 1: **grooved** 0.0649 (texture)
- Epoch 2: **grooved** 0.0447 (texture, still above tau)
**Interpretation:** decaying but still above tau; not dead yet.

## Is this enough to claim semantic death?
Not yet. With only 3 epochs, you **cannot prove permanence** (no recovery observed).
However, you *do* have strong early signals (drop below tau + label drift).

## Recommendation
Continue with more checkpoints (full run). With 10+ epochs, you can confirm:
- sustained decline
- no recovery
- migration vs true death

If later epochs show recovery for these units, the effect is *not* semantic death.
If they stay below tau without recovery, you have direct evidence.
