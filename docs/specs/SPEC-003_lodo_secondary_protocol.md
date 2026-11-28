# SPEC-003 — Leave-one-dataset-out secondary protocol, and a separate GAN evaluation track

**Status:** APPROVED (in principle, owner, 2026-11-28; §13 decisions remain open before implementation)
**Change level:** C (`RESEARCH_PROTOCOL.md` §3: dataset composition, evaluation protocol, invariant 1)
**Related task IDs:** new `P6-T7` (LODO), new `P6-T8` (GAN track); touches `P2`, `P3-T4`, `P6-T1`
**Author:** Claude   **Date:** 2026-11-28
**Handoff record:** `docs/handoffs/SPEC-003.md`

**Owner direction (2026-11-28, chat):** "Yes, I agree" to keeping FF++ → all as the primary
protocol and adding train-on-combinations as a second experiment. The GAN track stays "separate from
main work but along with that secondary combinations part".

## 1. Objective and hypothesis

The primary protocol (train on FF++ c23 only, test on everything else) stays exactly as it is.
It is what makes our numbers comparable with the literature and with DeepfakeBench.

This SPEC adds a **secondary** experiment that asks a broader question: *does reliability-conditioned
fusion generalise better than a strong baseline regardless of which dataset(s) it is trained on?*

**Hypothesis H-LODO:** for every training combination in §6, the final model's AUC on the held-out
dataset(s) is higher than the reference baseline's, averaged over 3 seeds.
**Refuted if** the final model loses on the majority of the five new combinations, or if every gain
lies inside seed variance. A refutation is reported, not hidden: it bounds the paper's claim to the
FF++ protocol.

**Fact:** single-source FF++ training is the primary protocol, so its runs are reused, not repeated.

## 2. Literature basis

Leave-one-domain-out evaluation is standard practice in domain-generalisation research. No specific
citation is relied on here. Any citation used to justify this design in the paper must first be
verified in `C-02` (invariant 9).

## 3. Architecture

n/a. No model changes. The same final model (Phase 5) and the reference baseline are trained with
their **primary-protocol hyperparameters, unchanged**. Only the training data changes.

## 4. Mathematical formulation

Sources S = {FF (FF++ c23), DF (DFDC), CD (Celeb-DF v2)}. A run trains on a non-empty T ⊂ S,
T ≠ S, and is tested on every dataset in S \ T, plus the auxiliary sets allowed by §6.
There are exactly six such subsets, which are the six the owner listed:

| Run | Train on (T) | Held-out test sets |
|---|---|---|
| L1 | FF | DF, CD (**= the primary protocol; reuse those runs**) |
| L2 | DF | FF, CD |
| L3 | CD | FF, DF |
| L4 | FF + DF | CD |
| L5 | FF + CD | DF |
| L6 | DF + CD | FF |

Summary statistic per model: mean held-out AUC over L1–L6 (each run's AUC is itself the mean over
3 seeds), plus every cell individually. No new parameters; invariant 7 is unaffected.

## 5. Input/output tensor contracts

Unchanged from the primary pipeline (`IMPLEMENTATION.md` §P4, §P5).

## 6. Data and preprocessing

- **Same preprocessing for every dataset:** the Phase 3 pipeline (detector, 32 frames, 320 px,
  1.3× margin) read from `configs/preprocess.yaml`. No per-dataset tuning.
- **Per-dataset splits** built by the same `P3-T4` split builder: identity-disjoint
  train / val / test within each dataset. Official test lists stay test-only: Celeb-DF v2's official
  518-video test list, and DFDC's official public test set.
- **Validation comes only from the run's own training sources** (e.g. L4 validates on FF++ val +
  DFDC val). Checkpoint selection, early stopping and thresholds use it. A held-out dataset is never
  touched before final testing (invariant 2).
- **Near-duplicate domains:** DFDCP is a preview subset of DFDC. When DF is in T, DFDCP is not
  reported. DFD comes from the FF++ authors' pipeline. When FF is in T, DFD is reported but marked
  "near-domain".
- **Class balance:** Celeb-DF v2 has about 590 real vs about 5,600 fake videos. Every run uses the
  class-balanced sampling and `pos_weight` computed from **that run's training split**, and logs
  per-source real/fake counts.
- **Source balance in pairs:** each source is sampled equally per epoch (no dataset dominates by
  size). This is a design choice, not a tunable.
- **SBI (`P5-T4`)** applies to LODO runs exactly as in the primary protocol. It is training-time
  synthetic data, not a test set.

## 7. Training methodology

- **Models (chosen before any LODO result exists):** the final Phase 5 model, and the reference
  baseline. The reference is the DeepfakeBench baseline (`P4-T3`) with the highest mean **FF++
  validation** AUC. It is never chosen by a test score.
- **Seeds:** 42, 1337, 2024 (invariant 6).
- **Hyperparameters:** frozen from the primary protocol. No per-combination re-tuning. This keeps
  the budget fixed and removes a route for indirect test-set selection.
- **Budget:** 5 new combinations × 3 seeds × 2 models = **30 training runs**. That is about 5× the
  primary training cost for those two models. DFDC preprocessing (≈470 GB raw) dominates. See §13.

## 8. Evaluation methodology

Video-level AUC primary, with AP / EER / accuracy (validation-selected threshold), and a video-level
bootstrap CI (invariant 8). Every run writes per-video scores (invariant 5b), with
`train_set` = `+`-joined sources (e.g. `ffpp+dfdc`).

- [x] same seeds, splits, budget across arms
- [x] no test-set selection (models, hyperparameters, thresholds all fixed beforehand)
- [ ] oracle/upper bound: n/a

**Reporting:** a separate table and figure (`C-30`), always labelled "secondary protocol". The
headline claims and abstract numbers come from the primary protocol only.

## 9. Acceptance tests

- `verify_splits --protocol lodo --run <L2..L6>` passes for every run. It asserts zero ID overlap
  between the run's train/val and **every** held-out test set, identity-disjointness inside each
  source, and no official-test-list video in any train/val split.
- A unit test proves a deliberately contaminated LODO split (one held-out ID in train) fails.
- Each results row carries the correct `train_set`, and per-video score files exist for every
  (run, seed, test set).
- L1 numbers are byte-identical to the primary `P6-T1` rows (reused, not re-run).

## 10. Expected artifacts

`C-30_lodo_matrix.csv` (long format) + `.md`, and `C-30_lodo_heatmap.png/.pdf/_data.csv` (train
combination × test set, cell = AUC ± CI) via `save_figure`. Results rows with `task_id=P6-T7`.
`P6-T8` (GAN track) writes `C-31_gan_track.*`.

## 11. Permitted engineering optimisations (Level A)

Embedding caching across runs that share a source (the backbone is frozen), dataloader workers,
and sharding DFDC preprocessing across sessions.

## 12. Restricted scientific changes (need the owner)

Adding or removing a training combination, re-tuning hyperparameters per combination, changing the
reference baseline after seeing LODO results, merging LODO numbers into the primary table, and
using any held-out set for any decision.

### Invariant 1, rewritten (applies once this SPEC is accepted)

> **1. Train only on the declared training sources; test sets never touch training.** The primary
> protocol trains on FF++ only. The secondary leave-one-dataset-out protocol (SPEC-003, `P6-T7`)
> trains on the declared combinations of FF++, DFDC and Celeb-DF v2. In every run, that run's test
> datasets must never appear in its training split, validation split, hyperparameter selection,
> early stopping or threshold tuning. DFDCP, DFD, GAN-track sets and FF++ c40 are evaluation-only,
> always.

## 13. Unresolved research decisions

1. **DFDC training budget.** DFDC is ≈470 GB in 50 parts. **Recommendation:** train on a fixed,
   seeded, identity-disjoint subset of parts, matched in size to FF++ (≈1,000 real + ≈4,000 fake
   videos). Test on the official DFDC test set. Options: full DFDC (weeks of preprocessing), or the
   recommended subset.
2. **Licences.** Training on DFDC and Celeb-DF v2 must be allowed by their terms. Check this while
   requesting access (Phase 2, `C-09`). If a licence forbids training, that source leaves this
   protocol.
3. **GAN track (`P6-T8`): which datasets.** Our model reads **videos** (32 frames), so the
   preferred sets are GAN-based *video* forgeries. All candidates below are **unverified**: names
   from memory, to be checked in `C-02` before use.
   - *Video, GAN-based (preferred):* FakeAVCeleb (includes FSGAN face swaps), DeeperForensics-1.0
     (its own DF-VAE swapping method), ForgeryNet.
   - *Images (only if we add a frame-level variant):* whole-image GAN sets such as StyleGAN-generated
     faces (e.g. the ForenSynths / CNNDetection collection). These are whole synthetic images, not
     face swaps, and have no temporal stream. They would need a "temporal stream masked" variant of
     the model, which is an architecture question. **Recommendation:** start with one video set;
     add images only if the owner wants the frame-level variant.
   GAN-track results are evaluation-only, reported in their own table (`C-31`), and never mixed into
   the primary or LODO tables.
4. **Where LODO sits in the timeline.** **Recommendation:** after `P6-T1` (the primary matrix),
   so the primary paper never waits on it. It can be cut if time runs out, without touching the core
   claim.
