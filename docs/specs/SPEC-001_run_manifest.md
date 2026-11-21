# SPEC-001 — Run identifier and per-run manifest

**Status:** APPROVED — owner, 2026-11-22
**Change level:** C (changes the `IMPLEMENTATION.md` §P0-T5 results-log schema)
**Related task IDs:** P0-T5 (and every training/eval script that calls `log_result`)
**Author:** Claude   **Date:** 2026-11-21
**Handoff record:** `docs/handoffs/SPEC-001.md`

## 1. Objective and hypothesis
Make every logged number traceable to the exact split, environment and checkpoint that produced it.
Today `results_log.csv` records commit, config path and seed (**Fact**, §P0-T5) but not: a run ID
that links the rows of one run, the research question, which split files were used, package
versions, or which checkpoint was evaluated. If splits are regenerated (P3-T4) or a checkpoint is
overwritten, an existing row cannot be tied back to its inputs. Not a hypothesis test — a provenance
requirement. Cost is lowest now, because P0-T5 is not yet implemented.

## 3. Design
1. **`run_id` column** — appended as the **last** column of `results_log.csv`, so the §P0-T5 column
   order is otherwise unchanged. Format: `<YYYYMMDD-HHMMSS>_<task_id>_<model_name>_seed<n>`.
2. **Per-run manifest** — `paper_assets/runs/<run_id>/manifest.yaml`, written at run start and
   updated at run end (invariant 5: never deleted):

```yaml
run_id: 20261201-213000_P4-T1_baseline_b3_seed42
task_id: P4-T1
research_question: "Honest FF++-only baseline: cross-dataset AUC of the legacy architecture"
git_commit: abc1234-dirty          # same rule as results_log
config_path: configs/baseline_b3.yaml
resolved_config: paper_assets/C-28_hyperparameters/baseline_b3_seed42.yaml
seed: 42
splits:                            # sha256 of each split file actually read
  data/splits/train.txt: <sha256>
  data/splits/val.txt: <sha256>
environment:
  python: "3.11.x"
  torch: "2.x"
  cuda: "12.8"
  gpu: "RTX 5070 Ti Laptop (sm_120)"
  pip_freeze: paper_assets/runs/<run_id>/pip_freeze.txt
checkpoint:                        # checkpoint this run PRODUCED; null for eval-only runs
  path: checkpoints/baseline_b3/seed42/best.pt
  sha256: <sha256>
  selected_by: "max val AUC (FF++ val)"   # invariant 2
evaluated_checkpoint:              # checkpoint this run LOADED; null for training runs
  path: checkpoints/baseline_b3/seed42/best.pt
  sha256: <sha256>                 # hashed at load time
  training_run_id: 20261201-213000_P4-T1_baseline_b3_seed42   # the run whose checkpoint.sha256 matches
started: 2026-12-01T21:30:00
finished: 2026-12-02T03:10:00
status: completed                  # running | completed | failed
```

3. `log_result()` gains a `run_id: str` parameter (required). A helper `start_run(task_id,
   model_name, config_path, seed, research_question) -> str` writes the manifest and returns the ID.

## 9. Acceptance tests
- `test_results_log_schema` checks the new last column.
- `test_run_manifest_written`: a dry run creates the manifest with all keys, split hashes match
  `sha256sum` of the files, and a dirty tree yields a `-dirty` commit.
- A `log_result` call without `run_id` raises.
- `test_eval_run_links_checkpoint`: an eval-only dry run gets its own `run_id`, has
  `checkpoint: null`, and its `evaluated_checkpoint.sha256` equals the `checkpoint.sha256` in the
  manifest named by `evaluated_checkpoint.training_run_id`.

## 12. Restricted scientific changes
None — this changes provenance only. No model, data, or metric behaviour changes.

## 13. Unresolved research decisions
None open.

**Decided (owner, 2026-11-22):** eval-only runs (P4-T4, P6-T1) get their **own** `run_id`
(one run = one process invocation), linked back through `evaluated_checkpoint` (path, sha256,
`training_run_id`) as in §3.

**After acceptance:** fold §3 and §9 into `IMPLEMENTATION.md` §P0-T5 and `docs/API_CONTRACTS.md`
(results_log schema), per `RESEARCH_PROTOCOL.md` §5.
