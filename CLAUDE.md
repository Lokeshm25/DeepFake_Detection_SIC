# CLAUDE.md

Guidance for AI coding agents working in this repository.

## What this project is

A deepfake detection research project being upgraded from a working capstone into a journal
submission. **The research protocol matters more than the code.** A clean implementation that
produces a contaminated number is worthless here; a plain implementation that produces an honest
number is the deliverable.

Read in this order before writing code:

| Document | Contains |
|---|---|
| `RESEARCH_ANALYSIS.md` | Why this project exists, why this mechanism, literature position |
| `ROADMAP.md` | What to do, in what order, with gates and capture points |
| `IMPLEMENTATION.md` | How to do it — file layout, signatures, algorithms, exact commands |

Tasks carry IDs (`P3-T1`). `ROADMAP.md` says what and why; `IMPLEMENTATION.md` has a section with
the same ID saying how. **Always open both.** Implement one ID at a time, run its acceptance test,
and only then move on.

---

## Invariants — never violate these

**1. Train on FF++ only.** Celeb-DF, DFDC, DFDCP, and DFD are evaluation-only, permanently. They
must never appear in a training split, a validation split, hyperparameter selection, early stopping,
or threshold tuning.

> This project's original results were invalidated by exactly this: `split_dataset.ipynb:180`
> carried the comment *"we include in entries but you'll not use them for training"*, then appended
> Celeb-DF to `entries`, which was shuffled into every split. 905 of 5000 training videos were
> Celeb-DF, and the published "external" AUC was measured largely on trained-on data. **If you are
> ever tempted to add an eval dataset to a training list "just to check something," don't.**

**2. Never tune on a test set.** Use the FF++ validation split for every decision. If you need a
number from a test set to make a choice, the choice is wrong.

**3. Verify splits before training.** `python -m tools.verify_splits` must pass. It is not a
formality — it is the check that would have caught invariant 1.

**4. Log every number.** Any script producing a metric appends to `paper_assets/results_log.csv` via
`log_result()` (`IMPLEMENTATION.md` §P0-T5), including the git commit. A number with no logged
provenance cannot go in the paper.

**5. Never delete anything in `paper_assets/`.** Superseded results demonstrate progress and defend
against "did you tune on the test set?" The directory is git-ignored, not disposable.

**5b. Every eval run writes per-video scores.** `paper_assets/per_video_scores/<model>_<test_set>_seed<n>.csv`
with `video_id,dataset,true_label,pred_prob`, every time, no exceptions. Every figure downstream —
ROC curves, calibration bins, failure galleries, bootstrap CIs, any re-slicing by manipulation method
or compression level — is recomputable from these files without a checkpoint. Skipping them turns a
later question into a retrain.

**6. Three seeds minimum** (`42, 1337, 2024`) for any claimed improvement, reported as mean ± std.
A single-seed gain inside seed variance is not a result.

**7. Zero-init all conditioning parameters.** `W_rel`, FiLM `gamma`s, adapter up-projections. This
makes ablations exact nested comparisons — without it, "conditioned vs unconditioned" confounds the
mechanism with initialisation and the central ablation becomes uninterpretable.

**8. Bootstrap over videos, never frames.** Frames within a video are correlated; frame-level
bootstrapping produces confidence intervals that are far too narrow.

**9. Every citation verified.** Nothing enters a document unless it is marked verified in
`paper_assets/C-02_citation_audit.md`. Some source material for this project contains fabricated or
non-research references (see `RESEARCH_ANALYSIS.md` §3.1) — a fabricated citation found in review is
a desk-reject.

**10. Gates are gates.** A failed gate means diagnose and fix, not proceed and hope. Each gate exists
because a failure there is cheap to fix and expensive to discover later.

---

## Conventions

**Config-driven.** Every script takes `--config configs/<name>.yaml`. No hardcoded paths, no
hardcoded hyperparameters. If you need a new parameter, add it to the YAML with a comment.

**Paths.** Relative to repo root, forward slashes, in split files and configs. The legacy
`data/train.txt` contains dead absolute paths (`c:\Users\...\OneDrive\...`) — do not imitate it.

**Reuse, don't reimplement.** This repo already has working code you must not duplicate:

| Existing | Location | Use for |
|---|---|---|
| ffmpeg CRF re-encode | `src/process_data/augmentations.py` | the compression sweep (currently uncalled) |
| Embedding cache pattern | `embeddings/*.npy`, `build_features.py` | frozen-backbone caching |
| OOF stacking (leak-free) | `train_ensemble_final.py:70-82` | correct reference implementation |
| Split/NaN verification | `notebooks/check_embeddings.ipynb` | the pattern for `verify_splits` |

**Legacy directories.** `src/ground_truth/` and `src/process_data/` are the capstone pipeline,
preserved for the Phase 4 baseline comparison. Fix bugs there only as specified in `P1-T3`; do not
refactor, restructure, or "clean up." That code needs to keep producing comparable numbers.

**Style.** Match the surrounding file. Type hints on new public functions. Docstrings stating shapes
and units for anything touching tensors. Comments only where the reasoning is non-obvious — most of
the *why* belongs in these three documents, not scattered through the source.

**Determinism.** Seed `random`, `numpy`, and `torch`; set `torch.backends.cudnn.deterministic`. The
legacy code already does this correctly — follow it.

---

## Hardware

RTX 5070 Ti Laptop, **12 GB VRAM**, Blackwell **sm_120**. Core Ultra 9 275HX (24 cores), 64 GB RAM.

- Budget for 12 GB. If something OOMs, prefer gradient accumulation or a frozen backbone over
  reducing frames below 32 — 32 frames is a research requirement, not a tunable.
- AMP is available and should be on. It was previously disabled only because of the `-1e9` mask bug
  fixed in `P1-T3`.
- sm_120 is new: `mamba-ssm` and `causal-conv1d` may not build. `configs/fusion_ssm.yaml:operator`
  records the outcome of that gate, and **only** that key should branch on it.
- 8–12 dataloader workers. PNG decode is the bottleneck at these batch sizes, and 64 GB of RAM is
  there to be used.

---

## The capture protocol

A teammate who was not present for the experiments is writing the paper. Anything not captured when
it existed is lost, because later phases overwrite splits, checkpoints, and metrics.

When a task's roadmap entry declares a `CAPTURE C-nn`, produce it as part of that task — not later.
Files go in `paper_assets/`, named `C-nn_description.ext`, each with a sibling `.md` note giving the
date, the exact command, the git commit, and two or three sentences of plain-English meaning.

**The note is the part that matters.** A PNG with no explanation is unusable to someone who wasn't
there. Write the interpretation while you still remember it.

Each capture declares a **kind** in `ROADMAP.md` and in `paper_assets/captures.yaml`:

| Kind | An agent's job |
|---|---|
| `[AUTO]` | Produce every file. This is the default and covers 22 of the 29 captures. |
| `[HUMAN-IN-LOOP]` | Produce every file, then **report which judgement the user still owes** (a spot-check count, an interpretation line, a case selection). Never invent it, never leave it blank without saying so. |
| `[MANUAL]` | Produce nothing but the `STATUS: PENDING` stub note, then tell the user exactly which screenshot to take. C-09 (licence approvals) is the only wholly manual capture, but **C-02 contains a manual folder**: `C-02_citation_audit/` holds one dated screenshot per suspect arXiv ID, which only the user can take. Treat that folder as `[MANUAL]` even though C-02 itself is `[HUMAN-IN-LOOP]`. |

**Output format is fixed** (`IMPLEMENTATION.md` §P0-T6):

- **Figures** — always PNG (300 dpi) **and** PDF (vector) **and** `_data.csv` with every plotted
  value. Write them through `save_figure()`; do not call `savefig` directly. A figure whose numbers
  are not saved can only be regenerated from a checkpoint, and checkpoints get overwritten.
- **Tables** — CSV in long format (one row per model × test set × seed × metric). A rendered `.md` or
  `.tex` is optional and never the only copy.
- **Console output** — redirected to `.txt` via `save_console()`.
- Distinguish plotted series by **marker and linestyle**, not colour alone; these figures get printed
  in greyscale.

**Never ask the user to screenshot something code can save**, and never screenshot it yourself: not
plots, not tables, not terminal output, not diffs, not configs. Screenshots exist in this project
only for things inside someone else's UI — email approvals, licence pages, arXiv pages. If saving a
figure or table seems to require a screenshot, the script is missing an output path; add it.

**Never fabricate a manual capture.** Do not generate a placeholder image, a mock-up of an email, or
a stand-in for a licence page to make `capture_check` pass. A missing manual capture is a to-do for
the user; a fabricated one is a fabricated provenance record.

### STOP-AND-WAIT — Non-negotiable for MANUAL and HUMAN-IN-LOOP

**If you reach a `[MANUAL]` or `[HUMAN-IN-LOOP]` capture during any task:**

1. Complete any auto-generated files for that capture
2. **STOP execution of the current task**
3. Report to the user:
   - The capture ID and kind (e.g., "C-10 [HUMAN-IN-LOOP]")
   - Exactly what the user needs to do (screenshot, judgement, interpretation)
   - The exact file path where the user's output belongs
   - Any context they need from the auto-generated artifacts
4. **WAIT for explicit user confirmation** before continuing

This applies to EVERY agent working on this project — Claude Code, Antigravity, or any other.
A `[HUMAN-IN-LOOP]` capture with an agent-invented judgement is worse than no capture at all,
because it passes `capture_check` with false content.

---

## When to stop and ask

- A spec in `IMPLEMENTATION.md` is ambiguous or contradicts the code. Do not guess.
- A gate fails and the fix isn't obvious from the diagnostic notes.
- You would need to violate an invariant to make something work.
- A result looks too good. An unexpectedly high cross-dataset AUC usually means leakage, not
  success — this project has already produced one such number. Check the splits first.
- You find a defect not listed in `RESEARCH_ANALYSIS.md` §1.2. Report it rather than silently fixing
  it; it may affect results already captured.
- **You encounter a `[MANUAL]` or `[HUMAN-IN-LOOP]` capture.** Stop, report what the user needs to
  do, and wait for their confirmation. See the STOP-AND-WAIT rule above.

---

## Known defects and their status

| Defect | Location | Status |
|---|---|---|
| Celeb-DF leaked into train/val/test | `split_dataset.ipynb:180` | Fixed in `P3-T4` |
| `-1e9` mask overflows fp16, blocks AMP | `models.py:85` | Fixed in `P1-T3` |
| `test_internal` fallback mislabels test set as "External" | `build_features.py:218`, `train_ensemble_final.py:50-54` | Fixed in `P1-T3` |
| Hardcoded unsupported "99.2%" accuracy | `app.py:866` | Fixed in `P1-T3` |
| Stale absolute OneDrive paths | `data/*.txt` | Fixed in `P3-T4` |
| 8 frames per video (cannot support temporal claims) | `extract_frames.ipynb:34` | Fixed in `P3-T3` (→32) |
| EfficientNet-B3 run at 224px, native is 300 | `models.py:31` | Addressed in `P4-T2` |
| ffmpeg CRF helpers never called | `augmentations.py` | Used in `P6-T2` |
| No face detection in pipeline; `face_crop.py` is dead code | `src/process_data/face_crop.py` | Replaced in `P3-T1`/`T3` |

Do not quote `checkpoints/ensemble_final/results.txt` "Test (External) AUC: 0.9695" as an external
result anywhere. It is the internal test set, mislabelled by the fallback above.
