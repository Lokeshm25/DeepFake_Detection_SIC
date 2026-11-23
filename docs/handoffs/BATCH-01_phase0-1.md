# Batch brief — BATCH-01: finish Phase 0 and run Phase 1 (overnight)

**Issued by:** Claude, for the owner, 2026-11-24
**Authorised by:** the owner, by sending this brief to Codex
**Binding rules:** `RESEARCH_PROTOCOL.md`. Where this brief and the protocol disagree, the protocol wins.

## Goal

Finish all the Phase 0 tasks that need no human input, then do the whole of Phase 1. Phase 1
archives the old capstone and measures how much the Celeb-DF leak inflated its result. That
measurement becomes **Figure 1 of the paper**. It can only be taken *before* the splits are rebuilt,
so it is the most time-sensitive work in the project.

Each task's `IMPLEMENTATION.md` section (or its `ROADMAP.md` entry where marked "no spec needed")
is its SPEC. Do the tasks **in this order**, because later ones use earlier ones:

| # | Task | SPEC | Produces | Notes |
|---|---|---|---|---|
| 1 | **P0-T6** figure/table helpers | `IMPLEMENTATION.md` §P0-T6 | `src/eval/figures.py` + tests | Every later figure depends on it. |
| 2 | **P0-T4** capture manifest + checker | §P0-T4 | `paper_assets/captures.yaml`, `tools/capture_check.py` + tests | |
| 3 | **P0-T3** environment freeze | `ROADMAP.md` Gate 0-C (no spec needed) | `environment.yml`, C-03 | Freeze the env as it is **before** P0-T1 installs anything. Re-export after P0-T1 if packages were added, and say so in the C-03 note. |
| 4 | **P0-T1** Blackwell build gate | §P0-T1 | C-01; writes `configs/fusion_ssm.yaml:operator` | Writing `operator` is spec-written (protocol §3); it is not a config change. Try attempt 1 (native Windows, then once with `TORCH_CUDA_ARCH_LIST="12.0"`). Attempt 2 (WSL2): the owner confirms WSL2 is installed. Use the existing distro (check with `wsl -l -v`). Installing the CUDA toolkit and Python packages *inside* that distro is allowed. **Do not install or replace a distro.** If it isn't Ubuntu 22.04-compatible or has no GPU passthrough (`nvidia-smi` inside WSL fails), record why in C-01 and go to attempt 3 (pure-PyTorch scan with its 1e-4 correctness test). Docker is not part of the §P0-T1 SPEC, so don't use it. If an install would break the existing environment, stop and report instead of forcing it. |
| 5 | **P1-T1** archive legacy state | `ROADMAP.md` §P1-T1 (no spec needed) | C-04 | Copy, never move. Create the git tag **locally only**; never push it. Contents of the archive stay untouched. |
| 6 | **P1-T2** quantify the leak | §P1-T2 | C-05, C-06, `results_log.csv` rows | Use `start_run(..., evaluated_checkpoints=[... predates_spec=True ...])` for the three legacy checkpoints. Per-video scores (invariant 5b). Bootstrap over videos (invariant 8). |
| 7 | **P1-T3** the four fixes | §P1-T3 | C-07 | Only after P1-T1 and P1-T2 are done: P1-T2 reads the old `data/*.txt`, and P1-T3 rewrites them. Legacy-code edits are allowed **only** as §P1-T3 specifies. When a defect is fixed, update its row in protocol §9 (a pure status correction; report it). |
| 8 | **Phase 1 exit gate** | `ROADMAP.md` Phase 1 exit gate | `python -m tools.capture_check --phase 1` output | |

## Expected results that are *not* a stop signal

- In P1-T2, the **seen** Celeb-DF subset will probably score a very high AUC, maybe above 0.95.
  The model trained on those videos, so that's the leak being measured, not a new problem. Record it
  and explain it in the C-05 note. The "too good, stop" rule (protocol §4) covers *unexplained*
  high numbers. **Unexplained** high numbers anywhere else still stop work.
- `0.9695` in `checkpoints/ensemble_final/results.txt` is the mislabelled internal test set. Never
  present it as external (C-04 note).

## Stop and leave a note instead of guessing (protocol §8)

Stop the affected task and write the blocker in its handoff record when:

- a spec is ambiguous or contradicts the code;
- a fix would need a change outside the task's SPEC (Level B or C);
- something needed is missing from disk, such as a checkpoint, embeddings, or Celeb-DF crops;
- you reach a `[MANUAL]` or `[HUMAN-IN-LOOP]` capture. None is planned in this batch; C-02 and C-09
  are **out of scope**.

When a task is blocked, **skip to the next task that doesn't depend on it**. Don't stop the whole
batch. Tasks 1–4 are independent of tasks 5–7, except that task 6 needs task 1.

## How to record the work

- One handoff record per task: `docs/handoffs/<task ID>.md`, from `_TEMPLATE.md`, with the plan,
  any deviations, and verification with fresh command output. Set each record to IN REVIEW when done.
- **Local commits: one per task, on `main`, only after that task's tests pass. Never push.** This
  keeps each task as a separate diff for Claude's review. The owner authorises these local commits
  by issuing this brief; that authorisation covers only this batch.
- Never delete or overwrite anything in `paper_assets/`. Don't touch other configs, the datasets or
  `checkpoints/`, apart from reading them.
- At the end, write a summary at the bottom of this file: each task's status (done / blocked and
  why), the commit hashes, the headline numbers from P1-T2 (all / seen / unseen AUC with CIs and
  n), the P0-T1 operator outcome, and anything the owner must decide.

## Summary (Codex, at the end)

_Not started._
