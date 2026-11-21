# SPEC-<nnn> — <title>

**Status:** DRAFT  <!-- DRAFT → APPROVED → IMPLEMENTED → REVIEWED → ACCEPTED | REJECTED. Only the owner sets APPROVED/ACCEPTED. -->
**Change level:** <A | B | C>  (`RESEARCH_PROTOCOL.md` §3)
**Related task IDs:** <e.g. P5-T3>   **Author:** <Claude | Codex>   **Date:** <YYYY-MM-DD>
**Handoff record:** `docs/handoffs/SPEC-<nnn>.md`

Use only the sections that apply. A Level B SPEC usually needs 1, 7, 8, 9 and 12. Write "n/a" rather
than padding. Label claims as **Fact** (with source), **Hypothesis**, or **Assumption**.

## 1. Objective and hypothesis
What question this answers, and the falsifiable hypothesis. What result would refute it.

## 2. Literature basis
Only citations marked verified in `paper_assets/C-02_citation_audit.md` (invariant 9). Unverified
sources are listed as such and carry no weight.

## 3. Architecture
Components, where they sit in the existing pipeline, and what stays unchanged.

## 4. Mathematical formulation
Equations with every symbol defined. Initialisation of every new parameter (invariant 7).

## 5. Input/output tensor contracts
| Module / function | Input (shape, dtype, range) | Output (shape, dtype, range) |
|---|---|---|

## 6. Data and preprocessing
Datasets and roles (invariant 1), splits, preprocessing config keys. Training/inference parity.

## 7. Training methodology
Config file and keys, seeds (invariant 6), compute budget against 12 GB VRAM.
For Level B: the controlled comparison — baseline config, the single variable changed, everything
held fixed.

## 8. Evaluation methodology
Metrics (video-level primary), test sets, threshold source (FF++ val), bootstrap over videos, and the
ablations that isolate the claim:
- [ ] nested comparison possible (zero-init)? [ ] static/non-adaptive competitor?
- [ ] oracle/upper bound? [ ] same seeds, splits, budget across arms?

## 9. Acceptance tests
Commands and pass criteria. Include the unit tests from `IMPLEMENTATION.md` §Acceptance tests that apply.

## 10. Expected artifacts
Files written, including any `CAPTURE C-nn`, `results_log.csv` rows, per-video scores.

## 11. Permitted engineering optimisations (Level A — implementer's call)
E.g. vectorisation, caching, dataloader workers, kernel choice — anything that leaves outputs unchanged.

## 12. Restricted scientific changes (need the owner)
Explicit list of things the implementer must not change without returning to the owner.

## 13. Unresolved research decisions
Open questions, each with options and a recommendation. The SPEC cannot be APPROVED while a decision
here blocks implementation.
