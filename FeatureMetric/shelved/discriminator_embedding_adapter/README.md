# Shelved: embedding-adapter discriminator (Aug 2026)

Snapshot of an experiment that did not pan out and was shelved in favour of the
plain discriminator. Kept here for reference only — it is **not wired in** and will
not run against the current `Discriminator/` without a merge.

## Idea

Instead of feeding raw fields to the SqueezeNet discriminator, pass them first
through a **frozen self-supervised encoder** (MAE, I-JEPA, or SFNO) and train the
discriminator on the resulting latent tokens. Before that, a "Phase-A" probe
checked whether frozen tokens separate ERA5 from forecasts at all
(`FeatureMetric/eval/eval_embedding_probe.py`). Logistic-probe AUC on frozen tokens:

| Encoder | Pangu | GraphCast |
|---|---|---|
| SFNO (best) | 0.845 | 0.856 |
| I-JEPA | 2nd best | |
| MAE | weakest | |
| Spectral baseline (no encoder) | 0.865 | 0.795 |

The encoders did not clearly beat a simple spectral baseline, which is why the
direction was shelved.

## Contents

| File | Original location | Purpose |
|---|---|---|
| `embedding_adapter.py` | `Discriminator/scripts/` | Wraps a frozen FeatureMetric/SFNO encoder as a discriminator input stage |
| `embedding_config.yaml` | `Discriminator/conf/` | Hydra config for the embedding-discriminator variant |
| `eval_unseen_models.py` | `Discriminator/scripts/` | Scores forecast models held out of training |
| `APPROACH_EXPLAINED.md` | `Discriminator/` | Long-form write-up of the discriminator approach vs. the embedding direction |
| `discriminator_wiring.patch` | — | Changes to existing Discriminator scripts (config, training, eval, plotting) that hooked the adapter in |

The probe eval lives in `FeatureMetric/eval/eval_embedding_probe.py` and uses the
`extract_patch_tokens` methods on the MAE/I-JEPA models in `FeatureMetric/utils/models.py`.
The SFNO encoder wrapper is `FeatureMetric/utils/sfno_embedding.py` (needs the separate
`SFNO-Embedding` repo).

## Reviving

Copy the three code/config files back to their original locations and apply
`discriminator_wiring.patch` from the repo root (`git apply -3`). The patch was taken
against `Discriminator/` as of 2026-08-19; `config.yaml`, `train_discriminator.py`,
`evaluate_discriminator.py` and `plot_logits_vs_lead_time_temporal_holdout.py` have
changed since, so expect conflicts.
