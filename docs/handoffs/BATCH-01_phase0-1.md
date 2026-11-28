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

**Execution date:** 2026-11-24. **Branch:** main. **No pushes.** Six completed tasks have one local
commit each. Their handoffs are IN REVIEW; Claude review and owner acceptance remain separate.

| Task | Status / evidence | Local commit |
|---|---|---|
| P0-T6 | DONE / IN REVIEW — immutable figure, table and console helpers; regression tests pass | `918eb7d56200d25229ddf109b22da239e8a499c1` |
| P0-T4 | DONE / IN REVIEW — all 29 capture declarations and checker | `573084ed25cc933bf860e5d61b8165c86d3a04d9` |
| P0-T3 | DONE / IN REVIEW — owner-designated aiml Conda export/lock, C-03; original environment freezes preserved | `667d3643bd61cdbfef68212628b89818bf7f54d7` |
| P0-T1 | DONE / IN REVIEW — C-01 native attempts, verified fallback, spec-written operator; post-install environment supplement | `b9a0efa1fc9a6b4e8947498b1aefc34492100378` |
| P1-T1 | DONE / IN REVIEW — C-04 verified archive: 52 files, 6,789,342,956 bytes; source/archive hashes, sizes and original source mtimes matched | `6d65be0c563af6ada0dab5d748dda52bb0c0a84e` |
| P1-T2 | DONE / IN REVIEW — owner-approved SPEC-002 diagnostic, completed 1168-video run, C-05/C-06, scores, three metric rows and hash receipt | `faddbf30ff43334339ea26c771f6cf98f470bfd6` |
| P1-T3 | BLOCKED / source fixes in review — safe mask, explicit internal-test requirement and unsupported Accuracy removal implemented; trainer AMP re-enable/real C-07 curves blocked. Owner deferred Fix4 to P3-T4 | No commit: required task acceptance incomplete; tested source diff/handoff preserved on main |
| Phase1 exit | FAILED — actual capture_check exit 1: C-04/C-05/C-06 pass; C-07 loss PNG/PDF/data missing | No successful-gate commit; evidence in PHASE-1-EXIT.md |

**P0-T1 operator outcome:** `mamba_torch`. The existing aiml Python3.11.14 / Torch2.8.0+cu128
supports CUDA12.8 on the RTX5070Ti Laptop (sm_120). Native extension builds lacked the MSVC
compiler. The only installed WSL distro was docker-desktop, which did not meet the authorized
Ubuntu-compatible environment requirement, so no Docker/new distro was used. The PyTorch fallback
passed the prescribed recurrence/correctness and CUDA backward checks. Original 208 aiml
distributions were preserved; nine verification-tool additions and the post-build export are
explicitly documented in C-03/P0-T1.

**P1-T2 measured diagnostics — Fact, not an exposure claim:**

| Nominal current-list cohort | n (real/fake) | Video AUC | 95% video-bootstrap CI |
|---|---:|---:|---|
| All | 1168 (588/580) | 0.651864884 | [0.620418467, 0.683496422] |
| Seen (train or val text-list membership) | 1064 (535/529) | 0.658025193 | [0.626065525, 0.689617071] |
| Unseen (remaining current-list membership) | 104 (53/51) | 0.597854236 | [0.492411282, 0.705430638] |

Run ID: `20261124-130232_P1-T2_legacy_ensemble_nominal_seed42`. Bootstrap uses 1000 IID video
resamples, seed42; authenticated historical producer seed is unknown. Membership counts are
905 train /159 val /69 internal test. All1168 labels and independently recomputed frozen temporal
probabilities align exactly with the archived matrix (maximum error0). All1180 consumed input
hashes and three checkpoint links passed verification; the artifact/code receipt is
`docs/handoffs/P1-T2_artifacts.csv`. Three metric rows carry the run_id and `6d65be0-dirty` start
revision. The exact all-AUC recomputes the saved notebook statistic without establishing training
reproducibility. The 104-video nominal-unseen estimate has a wide interval.

**Interpretation required by approved SPEC-002:** retained execution consumed4066 training cases,
not the later5000-entry lists. Historical checkpoint-linked training IDs were not recovered.
The legacy spatial-loader prefix mismatch loaded zero saved head tensors; preserved four spatial
cache columns were not reconstructed and their producer RNG/checkpoint linkage is unverified.
Consequently neither the gap nor the unseen AUC identifies causal leak inflation or clean
external generalization. C-05/C-06 notes state these limits. No ensemble was chosen using a test
score; the source-defined saved-evaluation trio was used unchanged.

**Fresh final verification:** `191 passed in 140.47s`; Ruff passes, all21 new-runtime/test files
are formatted, mypy passes11 source files, and diff --check passes. All four modified legacy/app
files parse. New P1-T3 regressions were observed failing on the original fp16 overflow/fallback/
mislabel, then passing; CPU and CUDA forward/backward and exact FP32 parity pass. Independent
engineering reviews found no remaining material issue. No real training or speedup is claimed.
Full outputs and exact commands are in P1-T2.md, P1-T3.md and PHASE-1-EXIT.md; logs are retained
under `.deepfake/batch01/`.

**Owner/Claude resolution still needed:**

1. Identify the actual target for the prescribed temporal AMP guard edit: train_temporal.py is
   absent; both retained clean/debugging notebooks contain the disabled-AMP comment. No notebook
   cells were executed or altered, and no substitute trainer was invented.
2. Supply/approve the FF++-only split lineage and split-verification prerequisite needed for
   C-07. `python -m tools.verify_splits` exits1: module absent; existing contaminated lists are
   not silently repaired or used for training. Fix4 stays deferred to P3-T4 per the owner's
   explicit decision. Any early split repair/new training scope needs a SPEC/owner decision.
3. Claude reviews the completed task commits and approved SPEC-002 implementation; owner decides
   acceptance. SPEC-002 has not been folded into IMPLEMENTATION/ROADMAP/protocol.
4. Track the separately reported spatial-loader mismatch and notebook global-label pos_weight
   issue; neither is silently fixed under P1-T3 or used to expand the task.

No dataset/checkpoint or existing paper asset was overwritten/deleted. Only authorized captures,
run records and metric appends were added. C-02/C-09 human captures remain outside the batch.
Phase1 is explicitly incomplete, and no downstream training gate was bypassed.

Final read-only preservation verification passed: all52 C-04/source hashes, sizes and source
mtimes still match, and every P1-T2 run/capture/runtime receipt matches except the explicitly
authorized later model-source mask change. Full fresh output is in PHASE-1-EXIT.md and
`.deepfake/batch01/BATCH-01_preservation_verify.txt`.
