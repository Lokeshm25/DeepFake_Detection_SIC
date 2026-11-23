# Handoff — SPEC-001

**SPEC:** `docs/specs/SPEC-001_run_manifest.md` (APPROVED by the owner, 2026-11-22)
**Status:** ACCEPTED

**Owner decisions carried into the SPEC:** eval-only runs get their own `run_id`; they link to the
checkpoints they loaded through `evaluated_checkpoints` (list of path, sha256, `training_run_id`;
the ID may be null only for checkpoints that predate SPEC-001; amendment 1, 2026-11-22).

**Scope note:** SPEC-001 changes the §P0-T5 results-log schema, so implement it together with
P0-T5 (`log_result`), not as a retrofit afterwards. Fold §3 and §9 into `IMPLEMENTATION.md` §P0-T5
and `docs/API_CONTRACTS.md` only after the owner accepts.

## 1. Implementation plan (Codex)

**Claude review response — amendment 1, round 1:**

1. Preserve Claude's §4 findings, the owner-approved amended SPEC, existing assets, and all
   unrelated files. Follow the approved list-valued checkpoint schema; no document folding or commits.
2. Update `start_run()` to accept `evaluated_checkpoints` inputs containing `path` and
   `training_run_id`; require `predates_spec=True` when that ID is null. This is an explicit caller
   declaration for legacy artifacts, since file modification dates cannot establish checkpoint age.
   Always compute the actual SHA-256 and validate every non-null ID against its training manifest.
3. Update the manifest, eval detection/config snapshots, and `finish_run()` to use the list.
   Allow completed runs without checkpoints; retain both `selected_by` guards. Require non-empty
   `split_paths` when no evaluated checkpoints are supplied. Reject unknown keys in `metrics`.
4. Update acceptance tests for the new schema and add `test_eval_run_legacy_checkpoints`, missing
   split/legacy-declaration checks, checkpoint-free completion, and unknown-metric tests. Run the
   changed cases first and confirm failures before production changes; then run `python -m pytest`,
   `python -m ruff check src/eval tests`, `python -m ruff format --check src/eval tests`,
   `python -m mypy src/eval`, and `git diff --check`.
5. Declare start-time Git provenance and the legacy input declaration as Level A in §2. Paste
   fresh output in §3, respond to all five Claude findings in §5, and retain IN REVIEW for re-review.

Implement the approved P0-T5 + SPEC-001 together, in this workspace, with no commits or pushes.
The SPEC is the design and this record is the execution plan. Baseline:
`e0f9fc87c8fd3f9c85960cb1f6e84ce8c2ccd675`, initially clean; 31 protected files hashed before edits.

**Files and interfaces:**

- Create `src/eval/__init__.py` and `src/eval/metrics.py`: `initialize_results_log()` creates or
  validates the exact P0-T5 header plus final `run_id`; `log_result()` preserves P0-T5 arguments
  and requires keyword-only `run_id`. Re-export `start_run()` and `finish_run()`.
- Create `src/eval/run_manifest.py`: `start_run(task_id, model_name, config_path, seed,
  research_question, *, split_paths=(), resolved_config=None, evaluated_checkpoints=None,
  repo_root=None) -> str`. Capture actual supplied split hashes, config
  snapshot, git/dirty state, environment, and pip freeze. Every non-null training ID is linked by
  checkpoint hash; null IDs require a legacy attestation. `finish_run(run_id, *, checkpoint_path=None, selected_by=None,
  status="completed", repo_root=None) -> None` records produced checkpoints and end status/time.
- Create `src/eval/_locking.py` for shared process/thread file locking. Publish startup YAML only
  after a complete flushed/fsynced staged write; preserve interrupted staging files. Training
  config captures retain C-28 names; eval configuration snapshots live in their unique run directory.
- Create `tests/conftest.py`, `tests/test_results_log.py`, and `tests/test_run_manifest.py`.
  Use temporary local Git clones (no commits/pushes) and synthetic config/split/checkpoint fixtures;
  exercise real file writes, hashing, Git provenance, and YAML/CSV reads outside repository assets.
- Create only the missing `paper_assets/results_log.csv` header in the real project after tests.
  No synthetic metric rows or dry-run manifests go into the project's `paper_assets/`.

**Execution and verification:**

1. Write acceptance tests first: exact schema/order, absent metric blanks vs real zero, mandatory
   `run_id`, start manifest keys/hashes/dirty suffix, lifecycle updates, and
   `test_eval_run_links_checkpoint`. Run
   `python -m pytest tests/test_results_log.py tests/test_run_manifest.py -v`; record the missing
   implementation failures before production code.
2. Implement the CSV and manifest helpers using standard-library CSV/hashlib/subprocess/pathlib
   and installed PyYAML 6.0.3. Preserve old assets on schema mismatch, ID collisions, checkpoint
   hash mismatch, and conflicting resolved configs. Test append concurrency and failed updates.
3. Run the targeted tests, `python -m pytest`, `python -m ruff check src/eval tests`,
   `python -m ruff format --check src/eval tests`, `python -m mypy src/eval`, and `git diff --check`.
   Investigate actual failures; record fresh output in §3.
4. Obtain one independent engineering review of these new modules/tests while final checks run;
   reconcile findings. Claude's SPEC review remains §4. Verify protected-file hashes, unchanged
   HEAD, and the real CSV's empty schema; set this handoff to IN REVIEW.

**Engineering choices (Level A):** optional keyword inputs carry actual split/checkpoint/resolved
config paths because current configs do not declare split files. `repo_root` supports independent
workspaces and dry-run tests. `finish_run()` implements the SPEC's required run-end update.
Run IDs keep the approved second-resolution format; a collision fails without overwriting.

## 2. Deviations (Codex)
| What differs from the SPEC | Why | Level (A/B/C) | Owner decision |
|---|---|---|---|
| Additional keyword inputs and `finish_run()` / `initialize_results_log()` helpers | Carry actual provenance inputs, support run-end updates and initialize the required CSV without fake results. The five specified `start_run` positional arguments and P0-T5 logger arguments remain intact. | A | Within approved implementation scope; declared for Claude review. |
| Eval `resolved_config` points to `runs/<run_id>/resolved_config.yaml` | An eval config can differ from the training config for the same model/seed; preserve the existing C-28 training capture while recording actual eval settings. Manifest keys and scientific behaviour are unchanged. | A | Engineering provenance layout; declared for Claude review. |
| CSV rows reuse the manifest's start-time `git_commit` instead of querying at log time | Keeps every row tied to the code loaded for its run. Scientific behaviour is unchanged. | A | Declared per Claude finding 3; fold wording into P0-T5 only after acceptance. |
| Checkpoint inputs require `predates_spec=True` for a null training ID | Explicit caller attestation enforces amendment 1's legacy-only exception without guessing from file timestamps or fabricating a training manifest. The flag is not a manifest field. | A | Engineering input validation within the approved amendment; declared for Claude review. |
Level B/C deviations stop work until the owner decides.

Amendment 1's list-valued checkpoint schema is an explicitly owner-approved change (2026-11-22),
not an unapproved deviation. The completed-run checkpoint restriction was removed per finding 2.

## 3. Verification (Codex)

**RED, before production code:**

Command: `python -m pytest tests/test_results_log.py tests/test_run_manifest.py -v`.
The first sandbox run failed in local-clone fixture setup with Git's MSYS
`CreateFileMapping ... Win32 error 5`. The owner-approved elevated run reached the feature tests:

```text
collected 14 items
ModuleNotFoundError: No module named 'src.eval'
============================= 14 failed in 32.18s =============================
```

All 14 failures were the missing implementation; no production code existed before this run.
**Initial GREEN:** the same acceptance command returned exit 0:

```text
============================= 14 passed in 41.00s =============================
```

Initial quality checks identified three Ruff style fixes (UTC alias and literal header fixture)
and four Windows mypy errors for the inactive POSIX `fcntl` branch. The branch now uses
`sys.platform == "win32"`, which mypy recognises; final checks will verify this correction.
Review diff baseline:
`e0f9fc87c8fd3f9c85960cb1f6e84ce8c2ccd675` versus the working tree.

### Independent engineering review and responses

One read-only `code-reviewer` reviewed the new files. Its two material findings were reproduced,
then fixed with regression tests. The reviewer independently verified 80 concurrent Windows
process appends with Unicode/multiline notes, one correct header, and no lost rows.

| Item | Finding / decision | Response and evidence |
|---|---|---|
| E1 | Missing CSV terminators can merge records; quoted notes can be incomplete even after a newline. | Fixed: reject unterminated or malformed CSV and validate existing row widths before appending. Header, metric-row, and unfinished-multiline-note preservation tests pass. |
| E2 | Interrupted startup YAML can poison the shared config capture or expose a partial manifest. | Fixed: stage, flush/fsync, then atomically publish YAML; serialize shared config publication. Both partial-write preservation tests pass; failed staging is retained. |
| C1 | Completed training could omit checkpoint provenance. | The initial implementation added a checkpoint requirement. Claude finding 2 identified this as too strict; the requirement was removed in the current response while preserving `selected_by` validation. The old verification below is historical. |
| Self1 | Eval settings can differ from the training config for the same model/seed. | Fixed: the eval manifest references its own resolved config while the C-28 training capture stays unchanged. The regression test passes; the path choice is declared Level A in §2. |

Before fixes, the six new regression cases failed (`6 failed in 18.97s`). The additional quoted-note
case also failed (`1 failed in 4.42s`) before its fix. The final suite below covers all of them.

**Review scope:** the reviewer declined to judge future caller integration (actual split
consumption, checkpoint-load timing, and validation-only selection). This task provides explicit
provenance inputs; those callers and their integration tests belong to their later task IDs.

### Initial implementation verification — before amendment 1

**`python -m pytest` — exit 0:**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\lkmah\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: E:\VS Code\DeepFake_Detection_SIC
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.0, cov-7.1.0
collecting ... collected 21 items

tests/test_results_log.py::test_results_log_schema PASSED                [  4%]
tests/test_results_log.py::test_log_result_appends_rows_preserving_zero_and_missing_metrics PASSED [  9%]
tests/test_results_log.py::test_log_result_requires_run_id PASSED        [ 14%]
tests/test_results_log.py::test_existing_incompatible_results_log_is_not_modified PASSED [ 19%]
tests/test_results_log.py::test_log_result_rejects_unknown_or_mismatched_run PASSED [ 23%]
tests/test_results_log.py::test_concurrent_log_appends_keep_one_header_and_all_rows PASSED [ 28%]
tests/test_results_log.py::test_incomplete_results_log_is_preserved[header] PASSED [ 33%]
tests/test_results_log.py::test_incomplete_results_log_is_preserved[metric-row] PASSED [ 38%]
tests/test_results_log.py::test_unfinished_multiline_note_is_preserved PASSED [ 42%]
tests/test_run_manifest.py::test_run_manifest_written PASSED             [ 47%]
tests/test_run_manifest.py::test_finish_run_records_produced_checkpoint PASSED [ 52%]
tests/test_run_manifest.py::test_eval_run_links_checkpoint PASSED        [ 57%]
tests/test_run_manifest.py::test_eval_rejects_changed_checkpoint_without_creating_run PASSED [ 61%]
tests/test_run_manifest.py::test_run_id_collision_preserves_existing_manifest PASSED [ 66%]
tests/test_run_manifest.py::test_failed_run_is_finalized_once PASSED     [ 71%]
tests/test_run_manifest.py::test_conflicting_resolved_config_is_preserved PASSED [ 76%]
tests/test_run_manifest.py::test_unsafe_run_identifier_is_rejected PASSED [ 80%]
tests/test_run_manifest.py::test_eval_run_keeps_its_own_config_without_changing_training_snapshot PASSED [ 85%]
tests/test_run_manifest.py::test_failed_config_write_is_not_published PASSED [ 90%]
tests/test_run_manifest.py::test_failed_start_manifest_write_is_not_published PASSED [ 95%]
tests/test_run_manifest.py::test_completed_training_requires_checkpoint_provenance PASSED [100%]

======================== 21 passed in 64.69s (0:01:04) ========================
```

**`python -m ruff check src/eval tests` — exit 0:**

```text
All checks passed!
```

**`python -m ruff format --check src/eval tests` — exit 0:**

```text
7 files already formatted
```

**`python -m mypy src/eval` — exit 0:**

```text
Success: no issues found in 4 source files
```

**Real output initialization:** called `initialize_results_log()` in the repository and read the
CSV back. Actual output:

```text
{"path": "E:/VS Code/DeepFake_Detection_SIC/paper_assets/results_log.csv", "columns": 20, "metric_rows": 0}
```

**Preservation check:** compared SHA-256 hashes against the pre-edit snapshot and enumerated
actual project assets. Actual output:

```text
PASS: 31/31 protected file hashes unchanged
PASS: HEAD unchanged (e0f9fc87c8fd3f9c85960cb1f6e84ce8c2ccd675)
PASS: paper_assets/ contains only the new results_log.csv header
```

Only this handoff and the new `src/eval/` / `tests/` files are changed in Git. The new CSV is
ignored as configured. No existing paper assets, config values, legacy code, shared research
documents, or API contracts changed. No training, inference, experimental metrics, commit, or
push was performed.

**`git diff --check` — exit 0:**

```text
warning: in the working copy of 'docs/handoffs/SPEC-001.md', LF will be replaced by CRLF the next time Git touches it
```

### Amendment 1 / Claude review response — fresh verification, 2026-11-23

The changed contract cases were run before production edits:

```text
python -m pytest tests/test_run_manifest.py::test_eval_run_links_checkpoint tests/test_run_manifest.py::test_eval_run_legacy_checkpoints tests/test_run_manifest.py::test_null_training_id_requires_legacy_declaration tests/test_run_manifest.py::test_completed_run_can_omit_checkpoint tests/test_run_manifest.py::test_training_run_requires_split_paths tests/test_results_log.py::test_log_result_rejects_unknown_metric_keys -v
============================= 8 failed in 31.16s ==============================
```

These failures were the old single-checkpoint API, the completion restriction, and missing
split/metric validation. All five changes are covered by the final suite.

**`python -m pytest` — exit 0:**

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\lkmah\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: E:\VS Code\DeepFake_Detection_SIC
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.0, cov-7.1.0
collecting ... collected 28 items

tests/test_results_log.py::test_results_log_schema PASSED                [  3%]
tests/test_results_log.py::test_log_result_appends_rows_preserving_zero_and_missing_metrics PASSED [  7%]
tests/test_results_log.py::test_log_result_requires_run_id PASSED        [ 10%]
tests/test_results_log.py::test_existing_incompatible_results_log_is_not_modified PASSED [ 14%]
tests/test_results_log.py::test_log_result_rejects_unknown_or_mismatched_run PASSED [ 17%]
tests/test_results_log.py::test_concurrent_log_appends_keep_one_header_and_all_rows PASSED [ 21%]
tests/test_results_log.py::test_incomplete_results_log_is_preserved[header] PASSED [ 25%]
tests/test_results_log.py::test_incomplete_results_log_is_preserved[metric-row] PASSED [ 28%]
tests/test_results_log.py::test_unfinished_multiline_note_is_preserved PASSED [ 32%]
tests/test_results_log.py::test_log_result_rejects_unknown_metric_keys[metrics0] PASSED [ 35%]
tests/test_results_log.py::test_log_result_rejects_unknown_metric_keys[metrics1] PASSED [ 39%]
tests/test_run_manifest.py::test_run_manifest_written PASSED             [ 42%]
tests/test_run_manifest.py::test_finish_run_records_produced_checkpoint PASSED [ 46%]
tests/test_run_manifest.py::test_eval_run_links_checkpoint PASSED        [ 50%]
tests/test_run_manifest.py::test_eval_rejects_changed_checkpoint_without_creating_run PASSED [ 53%]
tests/test_run_manifest.py::test_run_id_collision_preserves_existing_manifest PASSED [ 57%]
tests/test_run_manifest.py::test_failed_run_is_finalized_once PASSED     [ 60%]
tests/test_run_manifest.py::test_conflicting_resolved_config_is_preserved PASSED [ 64%]
tests/test_run_manifest.py::test_unsafe_run_identifier_is_rejected PASSED [ 67%]
tests/test_run_manifest.py::test_eval_run_keeps_its_own_config_without_changing_training_snapshot PASSED [ 71%]
tests/test_run_manifest.py::test_failed_config_write_is_not_published PASSED [ 75%]
tests/test_run_manifest.py::test_failed_start_manifest_write_is_not_published PASSED [ 78%]
tests/test_run_manifest.py::test_completed_run_can_omit_checkpoint PASSED [ 82%]
tests/test_run_manifest.py::test_eval_run_legacy_checkpoints PASSED      [ 85%]
tests/test_run_manifest.py::test_null_training_id_requires_legacy_declaration PASSED [ 89%]
tests/test_run_manifest.py::test_training_run_requires_split_paths[omitted] PASSED [ 92%]
tests/test_run_manifest.py::test_training_run_requires_split_paths[empty] PASSED [ 96%]
tests/test_run_manifest.py::test_selected_by_still_requires_produced_checkpoint PASSED [100%]

======================== 28 passed in 80.20s (0:01:20) ========================
```

**`python -m ruff check src/eval tests` — exit 0:**

```text
All checks passed!
```

**`python -m ruff format --check src/eval tests` — exit 0:**

```text
7 files already formatted
```

**`python -m mypy src/eval` — exit 0:**

```text
Success: no issues found in 4 source files
```

**`git diff --check` — exit 0:**

```text
warning: in the working copy of 'docs/handoffs/SPEC-001.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/specs/SPEC-001_run_manifest.md', LF will be replaced by CRLF the next time Git touches it
```

**Preservation verification:** SHA-256 and before/after review-text comparisons against this
response's pre-edit snapshot. Actual output:

```text
PASS: 35/35 protected file hashes unchanged, including existing paper_assets content
PASS: Claude review and owner-decision text preserved
PASS: HEAD unchanged (e0f9fc87c8fd3f9c85960cb1f6e84ce8c2ccd675)
PASS: existing Git status entries preserved; no new paths
```

The already-approved amended SPEC was preserved. Existing paper assets, configs, legacy code,
`IMPLEMENTATION.md`, and API contracts remain unchanged. No training, inference, experimental
results, commits, or pushes were produced. Tests ran in temporary local Git workspaces using
the previously approved pytest permission.

## 4. Review (Claude) — round 1
**Diff for review:** working tree versus `e0f9fc87c8fd3f9c85960cb1f6e84ce8c2ccd675`; new
`src/eval/`, `tests/`, and the empty `paper_assets/results_log.csv`.
**Reviewed:** 2026-11-22. **Verdict:** CONFORMANT WITH FINDINGS. The schema, the required
`run_id`, manifest keys, split/checkpoint hashing, the `-dirty` rule, eval-run linking and the
locking are correct. Fresh run by Claude: `python -m pytest tests -q`, result `21 passed in 59.65s`.

| # | Severity | Category | Location | Finding (Fact/Hypothesis) | Evidence | Suggested fix |
|---|---|---|---|---|---|---|
| 1 | major | spec-drift (SPEC gap, Claude's) | `run_manifest.py:167-177`; SPEC-001 §3 | **Fact:** a run that evaluates checkpoints with no SPEC-001 training manifest cannot be recorded as an eval run. A run that loads several checkpoints cannot be recorded either. | `IMPLEMENTATION.md:274-281` (P1-T2) loads the legacy frozen `spatial/best.pt`, `temporal/best.pt` and `ensemble_final/model.pkl`. No training manifest exists for any of them, and `evaluated_checkpoint` holds only one. **Owner approved amendment 1 (2026-11-22); SPEC-001 §3, §9, §13 updated.** `evaluated_checkpoint` becomes `evaluated_checkpoints`, a list of `{path, sha256, training_run_id}`. `training_run_id: null` is allowed only for checkpoints that predate SPEC-001; the hash is still recorded. Codex implements once approved. |
| 2 | major | spec-drift | `run_manifest.py:253-258` | **Fact:** a run can be marked completed only if it produced a checkpoint or loaded one. SPEC-001 does not require this. | Runs that log numbers without a checkpoint can only be closed as `failed`, which falsifies status. Examples: P0-T1 operator gate, P5-T2 reliability gate, and analysis-only scripts. | Remove this rule. Keep the rule that `selected_by` requires a produced checkpoint (`:242-247`). |
| 3 | minor | spec-drift (undeclared, Level A) | `metrics.py:110` | **Fact:** `git_commit` is taken from the manifest at run start. The §P0-T5 docstring says to run `git rev-parse` at log time. | The start-time commit is the code that was actually loaded, so this is the better choice. It is not listed in §2. | Declare it in §2 as Level A. It will be folded into §P0-T5 at acceptance. |
| 4 | minor | engineering | `metrics.py:117-119` | **Fact:** unknown metric keys are silently dropped. | `metrics={"AUC": 0.81}` or `{"auc_ci_lo": ...}` writes a blank cell, which reads as "not computed". | Raise `ValueError` on any key outside the nine metric columns. Add one test. |
| 5 | minor | invariant (3) | `run_manifest.py:134`, `:162-165` | **Hypothesis:** callers will forget `split_paths`, and the manifest will record `splits: {}`. | Split hashing is the main purpose of the manifest (it protects against splits regenerated in P3-T4). The empty default is silent. | Require a non-empty `split_paths` for training runs (no `evaluated_checkpoints`). Eval runs keep it optional. Add one test. |

## 5. Response (Codex) — round 1
| # | Fixed / Disputed / Deferred | Note or evidence |
|---|---|---|
| 1 | Fixed | **Fact:** `run_manifest.py:144`, `:180`, and `:235` use `evaluated_checkpoints` for every loaded artifact. Each SHA-256 is computed from the file; non-null IDs must match the training manifest. Null IDs require caller `predates_spec=True`, declared Level A in §2. `test_eval_run_links_checkpoint`, `test_eval_run_legacy_checkpoints`, and the undeclared-legacy rejection test pass. |
| 2 | Fixed | **Fact:** `finish_run()` can mark checkpoint-free runs completed. Both `selected_by` guards remain (`run_manifest.py:262`). `test_completed_run_can_omit_checkpoint` and `test_selected_by_still_requires_produced_checkpoint` pass. The superseded initial restriction is annotated in §3. |
| 3 | Fixed | **Fact:** §2 now declares start-time Git provenance as Level A. CSV rows continue using the manifest's captured commit (`metrics.py:115`); no scientific behaviour or additional code change was introduced for this finding. |
| 4 | Fixed | **Fact:** `metrics.py:99` rejects keys outside the nine metric columns with `ValueError` before any append. Both uppercase/typo cases in `test_log_result_rejects_unknown_metric_keys` pass and preserve the CSV bytes. |
| 5 | Fixed | **Fact:** `run_manifest.py:162` requires non-empty `split_paths` when no evaluated checkpoints are supplied; omitted and explicitly empty inputs both fail before artifact creation. Eval splits remain optional. Both training-input cases and legacy eval acceptance pass. |

Fresh amendment verification is in §3: all 28 tests, lint, formatting, mypy, and diff checks passed.
No findings are disputed or deferred. Status remains IN REVIEW for Claude's re-review; owner
acceptance remains pending in §6. No document folding or commits were performed.

## 4b. Review (Claude) — round 2 (2026-11-23)

| # | Verdict | Evidence |
|---|---|---|
| 1 | Resolved | **Fact:** `run_manifest.py:176-202` hashes every loaded file and checks non-null IDs against the training manifest. A null ID requires `predates_spec=True`, which is not stored in the manifest; I accept this as Level A (§2). This matches amendment 1. |
| 2 | Resolved | **Fact:** `finish_run` no longer requires a checkpoint. Both `selected_by` guards remain (`:262-267`). |
| 3 | Resolved | **Fact:** declared Level A in §2. |
| 4 | Resolved | **Fact:** `metrics.py:99-102` raises before anything is appended. |
| 5 | Resolved | **Fact:** `run_manifest.py:162-163`. Note: a run with no checkpoint and no splits can no longer use `start_run`. No currently specified metric-logging task falls into that case. Revisit only if one does. |

Fresh run by Claude: `python -m pytest tests -q` gave `28 passed in 68.54s`. `ruff check` was clean.
`mypy src/eval` reported `Success: no issues found in 4 source files`. The SPEC file contains only
Claude's amendment 1. No new findings.

**Verdict: CONFORMANT.** Ready for owner acceptance (§6). At acceptance, Claude folds SPEC-001 §3
and §9 into `IMPLEMENTATION.md` §P0-T5 and `docs/API_CONTRACTS.md`, including Codex's two declared
Level A choices. SPEC-001 then moves to ACCEPTED.

## 6. Owner decision

**Accepted** by the owner, 2026-11-24. Commit and push authorised. SPEC-001 folded into
`IMPLEMENTATION.md` §P0-T5 and `docs/API_CONTRACTS.md` §5 by Claude.
