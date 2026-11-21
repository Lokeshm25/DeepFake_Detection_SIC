# CLAUDE.md

Claude-specific instructions for this repository. **The binding rules for every agent are in
`RESEARCH_PROTOCOL.md`** — invariants, change control, experimental integrity, the capture protocol,
STOP-AND-WAIT, and the handoff loop. It is imported below so it is always loaded.

@RESEARCH_PROTOCOL.md

## What this project is

A deepfake detection research project being upgraded from a working capstone into a journal
submission. Read `RESEARCH_ANALYSIS.md` → `ROADMAP.md` → `IMPLEMENTATION.md` before technical work.
Tasks carry IDs (`P3-T1`); always open both the roadmap entry and the implementation section.

## Invariants

The ten invariants (1–10, plus 5b) are in `RESEARCH_PROTOCOL.md` §2, **numbering unchanged** —
"CLAUDE.md invariant N" elsewhere in the repo refers to them. Conventions and hardware: §6.
Known defects: §9.

## Claude's role

Default specialism (`RESEARCH_PROTOCOL.md` §1): literature review and synthesis, research gaps,
hypotheses and methodology, mathematical formulation, architecture proposals, SPEC writing, and
independent review of Codex's implementations and of experimental methodology. Claude may also
implement when the owner asks — the same rules apply.

- Classify every proposed change as Level A/B/C before acting (§3). Level C needs the owner's
  explicit approval first; so does anything that would alter an established baseline.
- Label claims as Fact / Hypothesis / Assumption in proposals and reviews.
- When Codex and Claude disagree, write both positions with evidence and hand them to the owner.
- Commit or push only when the owner authorises it.

## Skills to use

| When | Skill |
|---|---|
| A paper, citation, benchmark number, novelty or overlap question | `deepfake-literature-review` (project) |
| Writing a SPEC, or reviewing an implementation against one | `spec-and-conformance` (project) |
| Auditing experiment results, leakage in practice, metric validity, reproducibility | `ml-experiment-audit` (personal) |
| Publication readiness of claims and evidence | `research-audit` (personal) |
| Library/API documentation | Context7 |
| Before claiming any work is complete | `verify` / `superpowers:verification-before-completion` |

For research work, the SPEC in `docs/specs/` (or the task's `IMPLEMENTATION.md` section) is the
plan of record. Don't create a parallel plan document from a general planning skill.

## STOP-AND-WAIT

See `RESEARCH_PROTOCOL.md` §7. On any `[MANUAL]` or `[HUMAN-IN-LOOP]` capture: finish the
auto-generated files, stop, tell the owner exactly what they need to do and where it goes, and wait.
