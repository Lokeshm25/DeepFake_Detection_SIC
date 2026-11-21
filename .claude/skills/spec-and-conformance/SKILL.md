---
name: spec-and-conformance
description: Write implementation SPECs for this deepfake project (methodology, maths, tensor contracts, ablation design, acceptance tests, permitted vs restricted changes) and review Codex's implementation diffs for conformance to the approved SPEC — mathematical fidelity, tensor shapes, data flow, invariants, and scientific-methodology drift. Use before a Level B/C change or new research direction, and whenever asked to review an implementation against its spec.
---

# SPEC writing and conformance review

Binding rules: `RESEARCH_PROTOCOL.md` — §3 change levels, §5 SPEC and handoff loop, §2 invariants.
Technical source of truth: `IMPLEMENTATION.md`. Template: `docs/specs/_TEMPLATE.md`.
Handoff record: `docs/handoffs/<ID>.md` from `docs/handoffs/_TEMPLATE.md`.

## Mode 1 — Write a SPEC

**First decide whether one is needed** (`RESEARCH_PROTOCOL.md` §5): an existing task ID's
`IMPLEMENTATION.md` section already *is* the SPEC. Write a new one only for Level B proposals,
Level C proposals, new directions, or a gap/contradiction in `IMPLEMENTATION.md`. Don't write a SPEC
for Level A work.

1. Read the relevant `ROADMAP.md` and `IMPLEMENTATION.md` sections and the code they touch. A SPEC
   that contradicts existing code without saying so is a defect.
2. Copy the template to `docs/specs/SPEC-<next nnn>_<slug>.md`, status `DRAFT`.
3. Fill only the sections that apply. Make it implementable without guessing:
   - every equation symbol defined, every new parameter's initialisation stated (invariant 7);
   - every tensor contract as shape + dtype + value range, including batch and time axes;
   - config keys named exactly as they will appear in YAML;
   - acceptance tests as runnable commands with pass criteria.
4. **Ablation design** (for any claimed mechanism): nested comparison via zero-init, a static
   non-adaptive competitor, an oracle/upper bound where one exists, and identical seeds, splits and
   compute budget across arms. Say what result would refute the hypothesis.
5. Split §11 (permitted, Level A) from §12 (restricted, needs the owner) explicitly — that list is
   what lets Codex optimise freely without drifting the science.
6. Put open questions in §13 with options and a recommendation. Ask the owner; don't decide Level C
   points yourself. Only the owner sets `APPROVED`.

## Mode 2 — Conformance review of an implementation

Inputs: the SPEC (or `IMPLEMENTATION.md` section), the handoff record, and the diff
(`git diff <base>..<head>` — ask for the range if not given). Read the code the diff touches, not
only the hunks.

Check, in this order:
1. **Deviations declared?** Compare the diff with the handoff record's §2. An undeclared Level B/C
   deviation is a blocker regardless of whether it helps.
2. **Maths.** Each equation in the SPEC against the code, term by term: signs, which tensor is
   conditioned, softplus/sigmoid placement, reductions (mean vs sum, over which axis).
3. **Initialisation.** Every conditioning parameter zero-init (invariant 7); the
   `gate: none` model is an exact nested model.
4. **Shapes and data flow.** Trace one batch from input to loss with concrete shapes. Where possible,
   run a shape check or the SPEC's unit tests rather than reasoning only.
5. **Train/inference parity.** Same preprocessing, normalisation and frame sampling, read from config.
6. **Invariants touched.** Splits (1, 3), threshold/selection on FF++ val only (2), logging and
   per-video scores (4, 5b), seeds (6), bootstrap unit (8).
7. **Config.** Keys match the SPEC; no hardcoded paths or hyperparameters; baselines not edited in
   place.
8. **Tests.** The SPEC's acceptance tests exist and were run with fresh output in the handoff
   record. "Passes" without output is not evidence.
9. **Classify every difference** as Level A (fine, note it), B (needs a controlled comparison and the
   owner), or C (needs the owner before merge).

For experiment *results* (leakage in practice, metric validity, calibration, reproducibility of
logged numbers) use the `ml-experiment-audit` skill alongside this one.

**Write findings** into `docs/handoffs/<ID>.md` §4, using its table: severity (blocker / major /
minor), category, `file:line`, the finding labelled Fact or Hypothesis, the evidence, and a suggested
fix. End with a verdict: CONFORMANT / CONFORMANT WITH FINDINGS / NON-CONFORMANT. Don't fix the code
yourself unless the owner asks. The implementer addresses findings, and the owner accepts.
