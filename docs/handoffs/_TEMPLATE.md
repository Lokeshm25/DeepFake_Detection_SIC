# Handoff — <task ID or SPEC-nnn>

**SPEC:** <`IMPLEMENTATION.md` §P?-T? | `docs/specs/SPEC-nnn_*.md`>
**Status:** <IN PROGRESS | IN REVIEW | CHANGES REQUESTED | READY FOR OWNER | ACCEPTED>

## 1. Implementation plan (Codex)
Files to create/modify, order of work, how each acceptance test will be run.

## 2. Deviations (Codex)
| What differs from the SPEC | Why | Level (A/B/C) | Owner decision |
|---|---|---|---|
Level B/C deviations stop work until the owner decides.

## 3. Verification (Codex)
Commands run and their actual output (or paths to saved output). Commit/diff reference reviewed.

## 4. Review (Claude) — round <n>
**Diff reviewed:** `<base>..<head>`   **Verdict:** <CONFORMANT | CONFORMANT WITH FINDINGS | NON-CONFORMANT>

| # | Severity | Category | Location | Finding (Fact/Hypothesis) | Evidence | Suggested fix |
|---|---|---|---|---|---|---|
Severity: blocker / major / minor. Category: spec-drift / math / shape / data-flow / methodology /
invariant / test-gap / engineering.

## 5. Response (Codex) — round <n>
| # | Fixed / Disputed / Deferred | Note or evidence |
|---|---|---|
Re-run verification after fixes and record fresh output in §3.

## 6. Owner decision
Accepted / sent back, date, notes.
