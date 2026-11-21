# Handoff — SPEC-001

**SPEC:** `docs/specs/SPEC-001_run_manifest.md` (APPROVED by the owner, 2026-11-22)
**Status:** IN PROGRESS

**Owner decisions carried into the SPEC:** eval-only runs get their own `run_id`; they link to the
checkpoint they loaded through `evaluated_checkpoint` (path, sha256, `training_run_id`).

**Scope note:** SPEC-001 changes the §P0-T5 results-log schema, so implement it together with
P0-T5 (`log_result`), not as a retrofit afterwards. Fold §3 and §9 into `IMPLEMENTATION.md` §P0-T5
and `docs/API_CONTRACTS.md` only after the owner accepts.

## 1. Implementation plan (Codex)
Files to create/modify, order of work, how each acceptance test will be run.

## 2. Deviations (Codex)
| What differs from the SPEC | Why | Level (A/B/C) | Owner decision |
|---|---|---|---|
Level B/C deviations stop work until the owner decides.

## 3. Verification (Codex)
Commands run and their actual output (or paths to saved output). Commit/diff reference reviewed.

## 4. Review (Claude) — round 1
**Diff reviewed:** `<base>..<head>`   **Verdict:** <CONFORMANT | CONFORMANT WITH FINDINGS | NON-CONFORMANT>

| # | Severity | Category | Location | Finding (Fact/Hypothesis) | Evidence | Suggested fix |
|---|---|---|---|---|---|---|

## 5. Response (Codex) — round 1
| # | Fixed / Disputed / Deferred | Note or evidence |
|---|---|---|

## 6. Owner decision
Pending.
