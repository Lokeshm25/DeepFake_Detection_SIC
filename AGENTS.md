# AGENTS.md — Agent Orchestration Guide

## Purpose

This document defines how AI coding agents (Claude Code, Antigravity, or any future agent) should
interact with this research project. It complements `CLAUDE.md` (invariants and conventions) and
`ROADMAP.md` (what to do and when).

---

## Project Identity

**Project:** Reliability-Conditioned Selective Fusion for Generalizable Deepfake Detection
**Goal:** Publish a genuine, novel research paper in the deepfake detection domain
**Secondary goal:** Deploy as a working, production-grade project
**Hardware:** RTX 5070 Ti Laptop (Blackwell, sm_120, 12GB VRAM)

---

## Agent Roles

### Research Agent (Primary)
- Executes the research pipeline defined in `ROADMAP.md` phases 0–7
- Follows `IMPLEMENTATION.md` for code specifications
- Produces captures per the protocol
- **Must read** `CLAUDE.md` before any work — it contains 10 invariants that are non-negotiable

### Engineering Agent (Secondary)
- Builds deployment infrastructure (FastAPI, Docker, CI/CD)
- Creates and maintains `docs/` documentation
- Manages project configuration (pyproject.toml, configs/)
- Does NOT modify research pipeline code without explicit approval

---

## Mandatory First Steps for ANY Agent

Before doing any work on this project, every agent must:

1. **Read `CLAUDE.md`** — contains 10 invariants that cannot be violated
2. **Read `ROADMAP.md`** — contains the phase structure and capture protocol
3. **Check `IMPLEMENTATION.md`** — for the specific task's code specification
4. **Check the task tracker** — know what's been done, what's in progress

---

## Task Delegation Rules

| Task Type | Primary Agent | Approval Needed? |
|---|---|---|
| Research pipeline code (src/) | Research Agent | No — follow IMPLEMENTATION.md |
| Config files (configs/) | Either | No |
| Documentation (docs/) | Engineering Agent | No |
| Evaluation code (src/eval/) | Research Agent | No |
| Deployment code (deploy/) | Engineering Agent | No |
| Modifying CLAUDE.md | NEVER by agent alone | **User approval required** |
| Modifying ROADMAP.md | NEVER by agent alone | **User approval required** |
| Modifying IMPLEMENTATION.md | NEVER by agent alone | **User approval required** |

---

## Safety Rails for Research Integrity

### Never Do These
1. **Never train on anything but FF++ (c23)** — invariant 1
2. **Never tune hyperparameters on a test set** — invariant 2
3. **Never fabricate a citation** — invariant 9
4. **Never invent a manual capture** — see STOP-AND-WAIT in CLAUDE.md
5. **Never quote invalidated metrics** — the old 0.9695 "external" AUC is data leakage
6. **Never skip a gate** — invariant 10

### Always Do These
1. **Log every result** to `paper_assets/results_log.csv` with git commit hash
2. **Run `verify_splits`** before any evaluation
3. **Use three seeds** (42, 1337, 2024) for any claimed improvement
4. **Bootstrap over videos**, never frames — invariant 8
5. **Save figures as PNG + PDF + `_data.csv`** — invariant via P0-T6

---

## Handoff Protocol Between Agents

When one agent completes a task and another continues:

1. **Update the task tracker** — mark completed items
2. **Commit all changes** with a descriptive message referencing the task ID
3. **Write a handoff note** in the commit message or task tracker:
   - What was done
   - What's next
   - Any decisions made
   - Any issues discovered
4. **Never leave uncommitted research results** — they can be lost

---

## STOP-AND-WAIT Protocol

See `CLAUDE.md` § "STOP-AND-WAIT" and `ROADMAP.md` § "MANDATORY AGENT BEHAVIOR".

Summary: When an agent encounters a `[MANUAL]` or `[HUMAN-IN-LOOP]` capture, it MUST:
1. Complete auto-generated files
2. **STOP**
3. Tell the user exactly what they need to do
4. **WAIT** for user confirmation before continuing

---

## File Ownership

| Directory | Owner | Notes |
|---|---|---|
| `src/ground_truth/` | Legacy — do not modify except as specified in P1-T3 |  |
| `src/process_data/` | Research Agent — rebuilt in P3 |  |
| `src/eval/` | Research Agent — new in P0-T6 |  |
| `src/models/` | Research Agent — new in P5 |  |
| `configs/` | Shared — stubs created, values filled per phase |  |
| `docs/` | Engineering Agent |  |
| `tests/` | Research Agent — unit tests from IMPLEMENTATION.md |  |
| `tools/` | Research Agent — verify_splits, capture_check |  |
| `paper_assets/` | Research Agent — captures only |  |
| `deploy/` | Engineering Agent — when deployment begins |  |

---

## Config Hierarchy

```
configs/
├── preprocess.yaml      # P3 — face detection, cropping, frames
├── baseline_b3.yaml     # P4 — honest baseline retrain
├── fusion_ssm.yaml      # P5 — THE core mechanism config
├── eval_matrix.yaml     # P6 — final evaluation matrix
└── *.local.yaml         # git-ignored personal overrides
```

Every downstream script reads from these configs. **No hardcoded paths in code.**
The `operator` key in `fusion_ssm.yaml` is written by P0-T1 (Gate 0-A) and read by everything else.
