# Handoff — SPEC-002

**SPEC:** `docs/specs/SPEC-002_legacy_audit_provenance.md`
**Status:** IN REVIEW — owner-approved amendment implemented; verification in P1-T2
**Related task:** P1-T2; source evidence in `docs/handoffs/P1-T2.md`.

## 1. Implementation plan (Codex)

Implement the owner-approved bounded nominal-cohort audit, because historical training exposure
is unverified. The pre-code plan, file ownership, acceptance commands and Level A review fix plan
are in P1-T2 §1. Preserve all decisions outside this explicit amendment; do not fold the SPEC until
the owner accepts it.

## 2. Deviations (Codex)

Owner-approved Level C amendment: qualify C-05 as nominal cohort statistics, replace the forced
AUC-ordering gate with completeness, and qualify the Phase1 baseline claim. Approved 2026-11-24.
Engineering deviations and verification are declared in P1-T2 §2; no additional unapproved B/C
change is adopted. Replacing stale DRAFT wording here and in the SPEC is a pure alignment with
the recorded approval, reported as Level A documentation work.

## 3. Verification (Codex)

P1-T2 §3 records 185 passing tests, lint/format/type checks and the completed diagnostic run
`20261124-130232_P1-T2_legacy_ensemble_nominal_seed42`. All/seen/unseen AUCs and 95% video-bootstrap
CIs are 0.651864884 [0.620418467,0.683496422] (1168), 0.658025193 [0.626065525,0.689617071]
(1064), 0.597854236 [0.492411282,0.705430638] (104). C-05/C-06, 1180 input hashes, three legacy
checkpoint links, three metric rows and saved-score recomputation passed acceptance verification.
Historical exposure/spatial cache producer linkage remain unverified and disclosed. No clean
external or causal leak-inflation claim is made. Claude review and owner final acceptance remain.

## 4. Review (Claude) — round 1

Pending Claude's independent scientific and implementation review; scope is owner-approved.

## 5. Response (Codex) — round 1

No findings received.

## 6. Owner decision

Owner decision, 2026-11-24: "Approve SPEC-002 diagnostic audit (Recommended)". This is explicit
approval for the diagnostic amendment, not acceptance of an implementation. Record execution,
verification and Claude review in P1-T2's handoff. Do not fold the SPEC before owner acceptance.
