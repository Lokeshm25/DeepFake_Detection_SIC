# AGENTS.md — Agent Orchestration Guide

## Purpose

Codex-specific instructions for collaborating with the research owner and Claude on this project.
**`RESEARCH_PROTOCOL.md` is binding and wins on conflict.** Report conflicts to the research owner.
It contains the shared invariants, change control, experimental integrity, capture rules, and
handoff loop. `CLAUDE.md` contains Claude-specific instructions; `ROADMAP.md` and
`IMPLEMENTATION.md` contain the task sequence and technical specifications.

---

## Project Identity

**Project:** Reliability-Conditioned Selective Fusion for Generalizable Deepfake Detection
**Goal:** Publish a genuine, novel research paper in the deepfake detection domain
**Secondary goal:** Deploy as a working, production-grade project
**Hardware:** RTX 5070 Ti Laptop (Blackwell, sm_120, 12GB VRAM)

---

## Agent Roles

Preferred specialisms from `RESEARCH_PROTOCOL.md` §1; neither model may redefine the research
unilaterally.

| Who | Default role |
|---|---|
| Research owner | Owns objectives, approves Level C changes, resolves disagreements, and accepts results and final implementations. |
| Claude | Literature review and synthesis, research gaps, hypotheses and methodology, mathematical formulation, architecture proposals, SPEC writing, and independent code and methodology review. |
| Codex | Implements approved specifications: Python/PyTorch/CV modules, preprocessing, training and evaluation pipelines. Runs tests and experiments, debugs, optimises engineering and compute performance, and proposes alternative implementations. |
| Antigravity | Inactive in the default workflow; follows the binding protocol if the owner uses it. |

Label claims in proposals, reviews, and handoffs as **Fact** (with verified source), **Hypothesis**
(testable but untested), or **Assumption** (with the reason for assuming it). Record disagreements
with each side's evidence. Contested Level B/C points stop and go to the owner.

### Reusable Codex specialists

| Agent | Scope |
|---|---|
| `python-ml-engineer` | Implementation of approved Python/ML specifications. |
| `test-engineer` | SPEC acceptance tests and applicable unit tests from `IMPLEMENTATION.md` §Acceptance tests. |
| `code-reviewer` | Independent engineering review; supplements Claude's SPEC and methodology review. |
| `architect` | Level C proposals only; output goes to the owner as a SPEC draft in `docs/specs/`, before implementation. |

Use specialists only for genuinely independent work that benefits from delegation, with at most
three concurrent subagents. Prefer read-only investigation; give precise scope, file ownership,
and expected deliverables. Do not overlap edits or revert another agent's work. Codex consolidates
findings and independently verifies the final changes. Reuse existing agents; do not reconfigure them.

---

## Mandatory First Steps for ANY Agent

Before doing any work on this project, every agent must:

1. **Read `RESEARCH_PROTOCOL.md`** first — the binding shared rules.
2. **Read the SPEC and matching `ROADMAP.md` entry and `IMPLEMENTATION.md` section.** Implement
   one task ID at a time. For IDs explicitly listed as "no spec needed," the roadmap is the SPEC.
   Consult `RESEARCH_ANALYSIS.md` for research context. Do not implement a DRAFT SPEC.
3. **Check Git status, repository structure, and applicable instructions.** Preserve existing work.
4. **Check task state** in `docs/specs/` and the per-task records in `docs/handoffs/`.
   Do not infer completed gates, captures, review, or acceptance from a roadmap listing.
5. **For implementation tasks, open `docs/handoffs/<ID>.md`** from `docs/handoffs/_TEMPLATE.md`
   and write the implementation plan before changes, including files, order of work, and
   acceptance-test commands.

The approved SPEC is the research plan of record. Keep the short execution plan in the handoff;
general-purpose skills must not create a competing research plan or override approved decisions.

---

## Scientific Change Control

**Classify every change as A, B, or C before changing anything** (`RESEARCH_PROTOCOL.md` §3).
When uncertain, use the higher level. A Level B/C change or deviation stops implementation and
goes to the owner; an agent does not decide adoption.

| Level | Covers | Action |
|---|---|---|
| A — Engineering | Refactoring, logging, error handling, type hints, tests, and performance work with identical scientific behaviour: same data, order, and maths. | Implement within the authorized task; verify behaviour is unchanged. |
| B — Experiment-affecting | Batch size, precision, augmentation, sampling, resolution, optimiser/scheduler, DataLoader order, early stopping, or setting a provisional config value, unless the value is already fixed (below). | Stop and propose a short SPEC. Run only after the owner OKs the run, as a controlled comparison against the unchanged baseline: same seeds/splits, one variable changed, selected on FF++ validation. Owner decides adoption afterwards. |
| C — Methodology | Architecture, losses, datasets, splits, labels, evaluation, metric definitions, hypotheses, gates, captures, or changing any fixed value (below), even one that otherwise looks like Level B. | Stop; write a SPEC draft from `docs/specs/_TEMPLATE.md`. Require explicit owner approval before implementation. |

**Fixed, provisional, and spec-written config values** (`RESEARCH_PROTOCOL.md` §3):

- **Fixed** — specified in `IMPLEMENTATION.md`, or adopted by the owner after a Level B comparison.
  Changing it is Level C.
- **Provisional** — present only in a config marked `STUB` and specified nowhere in
  `IMPLEMENTATION.md`. Setting it is Level B; it becomes fixed once the owner adopts the result.
- **Spec-written** — a value an approved task is instructed to write, such as P0-T1's
  `fusion_ssm.yaml:operator` or P3-T1's `preprocess.yaml:detector`. Writing it executes that approved
  task; changing it afterwards is Level C.

Preserve established baselines: config, code path, and logged results are never edited in place.
An approved variant uses a new config or specified ablation switch and its own `model_name`.
For an approved SPEC, §11 defines permitted engineering optimisations; §12's restricted changes
require returning to the owner. All invariants and fixed values still apply.

Editing `ROADMAP.md`, `IMPLEMENTATION.md`, `RESEARCH_ANALYSIS.md`, or `RESEARCH_PROTOCOL.md`
is Level C unless it is a pure correction or alignment with an already-approved decision; report
even such corrections. Change Claude-specific instructions only when explicitly authorized.

---

## Safety Rails for Research Integrity

### Never Do These

1. **Never train on anything but FF++ (c23)** — invariant 1
2. **Never tune hyperparameters on a test set** — invariant 2
3. **Never fabricate a citation** — invariant 9
4. **Never invent a manual capture or human judgement** — `RESEARCH_PROTOCOL.md` §7
5. **Never quote invalidated metrics** — 0.9695 is a mislabelled internal result, and the legacy
   Celeb-DF evaluation is contaminated; neither establishes external generalization
6. **Never skip a gate** — invariant 10

### Always Do These

1. **Log every metric** via `log_result()` to `paper_assets/results_log.csv` with git provenance.
2. **Require `python -m tools.verify_splits` to pass before any training** — invariant 3;
   run the task's specified split checks before evaluation.
3. **Select thresholds, checkpoints, hyperparameters, early stopping, and calibrators on FF++
   validation only.** External datasets never enter training, validation, or model selection.
4. **Use three seeds** (`42, 1337, 2024`) for any claimed improvement; report mean ± std.
5. **Write per-video scores on every evaluation:** `video_id,dataset,true_label,pred_prob` in
   `paper_assets/per_video_scores/<model>_<test_set>_seed<n>.csv` — invariant 5b.
6. **Bootstrap over videos**, never frames — invariant 8. Video probability is the mean of frame
   probabilities; video-level metrics are primary.
7. **Zero-init all conditioning parameters**, including `W_rel`, FiLM gammas, and adapter
   up-projections — invariant 7.
8. **Save figures as PNG + PDF + `_data.csv` plus the explanatory note** via `save_figure()`.
   Use the specified helpers for tables and console output; never substitute screenshots.
9. **Stop on a failed gate or cross-dataset AUC above 0.95.** Re-run split verification and
   explain a suspicious result before continuing (`RESEARCH_PROTOCOL.md` §§4, 8).

Never fabricate citations, results, verification output, provenance, or acceptance. Preserve all
existing `paper_assets/` content; only produce artifacts required by the authorized task.

---

## Handoff Protocol Between Agents

Follow `RESEARCH_PROTOCOL.md` §5, using one `docs/handoffs/<ID>.md` record per task or SPEC:

1. Read the SPEC and write the implementation plan in the handoff record. A matching
   `IMPLEMENTATION.md` section is the approved SPEC; Level A work needs no new SPEC.
2. Implement within the approved scope and permitted optimisations. Record **every deviation**
   with its A/B/C level and rationale. Level B/C deviations stop and go to the owner.
3. Run **every applicable acceptance test** and record actual fresh output, exact commands, and
   the commit/diff reviewed. Do not claim an unexecuted check passed.
4. Hand over for **Claude's review** against the SPEC. Claude records severity, location,
   evidence, claim label, suggested fix, and verdict in the same record.
5. Answer **each finding** as **Fixed**, **Disputed** (with evidence), or **Deferred** (with reason).
   Re-run verification after fixes and paste fresh output into the record.
6. Update task/review state accurately. **Only the owner approves SPECs and accepts final work.**
   Status follows `DRAFT → APPROVED → IMPLEMENTED → REVIEWED → ACCEPTED` (or `REJECTED`).

**Commit or push only when the owner authorizes it for that task.** A handoff is not authorization
to commit. Preserve uncommitted work and research artifacts while awaiting review or acceptance.
Save requirements, decisions, and findings meant for another agent in the repository handoff
record; do not rely on that agent having access to another conversation.

---

## STOP-AND-WAIT Protocol

See `RESEARCH_PROTOCOL.md` §7 and the matching capture in `ROADMAP.md`.

Summary: When an agent encounters a `[MANUAL]` or `[HUMAN-IN-LOOP]` capture, it MUST:

1. Complete the capture's auto-generated files; for a wholly manual capture, write only the
   `STATUS: PENDING` stub note.
2. **STOP execution of the current task.**
3. Tell the owner the capture ID/kind, the exact action or judgement needed, the exact output
   path, and the context from generated artifacts.
4. **WAIT for explicit user confirmation** before continuing. Never invent the judgement or
   fabricate screenshots, including the manual folder inside C-02.

Also stop for ambiguous or conflicting specs, unlisted defects, invariant violations, and contested
Level B/C findings (`RESEARCH_PROTOCOL.md` §8). Report the evidence and needed owner decision.

---

## File Ownership

| Directory | Owner | Notes |
|---|---|---|
| `src/ground_truth/` | Legacy — do not modify except as specified in P1-T3 |  |
| `src/process_data/` | Legacy — do not modify except as specified in P1-T3; `augmentations.py` is reused, never rewritten |  |
| `src/data/` | Codex implementation — P3 | Approved detector, preprocessing, split, and dataset specs |
| `src/eval/` | Codex implementation — P0-T6 onward | Approved metrics, output helpers, and evaluation specs |
| `src/models/`, `src/train/` | Codex implementation — P4/P5 | Approved model and training specs |
| `configs/` | Shared, under change control | Fixed changes: Level C; provisional STUB-only settings: Level B; spec-written values: approved task execution (§3) |
| `docs/specs/` | Claude by default; Codex may propose | Drafts require owner approval before implementation |
| `docs/handoffs/` | Codex plan/verification/responses; Claude review; owner decision | One record per ID |
| Other `docs/` | Within authorized engineering scope | Preserve scientific decisions |
| `tests/` | Codex / `test-engineer` | SPEC acceptance tests and `IMPLEMENTATION.md` unit tests |
| `tools/` | Codex implementation | Approved verification and capture tools |
| `paper_assets/` | Authorized research artifacts | Captures, metrics, provenance, and per-video scores; never delete |
| `deploy/` | Codex within authorized deployment scope | When deployment begins |

Leave global `~/.codex/AGENTS.md`, agent definitions, plugins, Playwright MCP, and Claude-only
`.claude/skills/` unchanged. Do not load or copy Claude-only skills into Codex; shared rules live
in `RESEARCH_PROTOCOL.md`. Collaboration setup does not authorize changes to configuration values,
legacy code, or `paper_assets/`. `SPEC-001_run_manifest.md` was approved on 2026-11-22; implement it
through `docs/handoffs/SPEC-001.md`.

---

## Config Hierarchy

```
configs/
├── preprocess.yaml      # P3 — face detection, cropping, frames
├── baseline_b3.yaml     # P4 — honest baseline retrain
├── backbone_clip.yaml   # P5-T1 — frozen backbone + adapters
├── reliability.yaml     # P5-T2 — reliability estimator
├── fusion_ssm.yaml      # P5-T3/T4 + P6-T4 ablations — THE core mechanism config
├── eval_matrix.yaml     # P6 — final evaluation matrix
└── *.local.yaml         # git-ignored personal overrides
```

Every downstream script reads from these configs. **No hardcoded paths in code.**
The `operator` key in `fusion_ssm.yaml` records P0-T1 (Gate 0-A) and is read by everything else.
P0-T1's `operator` write and P3-T1's `detector` write are spec-written outputs of those approved
tasks; changing them afterwards is Level C. Other fixed-value changes are also Level C;
provisional STUB-only settings are Level B, with owner OK before the comparison and adoption
decided afterwards. Use the Scientific Change Control definitions above.
