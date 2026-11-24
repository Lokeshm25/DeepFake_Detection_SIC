# SPEC-002 — Provenance-aware legacy cohort audit

**Status:** IMPLEMENTED — IN REVIEW; owner approved diagnostic audit in chat, 2026-11-24
**Change level:** C (`RESEARCH_PROTOCOL.md` §3: hypothesis/capture/gate interpretation)
**Related task IDs:** P1-T2, dependent P1-T3 and Phase1 exit gate
**Author:** Codex   **Date:** 2026-11-24
**Handoff record:** `docs/handoffs/SPEC-002.md`; evidence in `docs/handoffs/P1-T2.md`.

## 1. Objective and hypothesis

Approved amendment: evaluate the frozen legacy classifier on the existing1168-video Celeb-DF
subset and report all/current-list-seen/current-list-unseen AUCs as **diagnostic cohort statistics**.
Do not attribute the gap to training exposure or call it leak inflation without historical IDs.

**Fact:** saved training/preprocessing counts4066/762/255 differ from contaminated text-list
counts5000/937/314/200. **Hypothesis:** training used an earlier Celeb-DF-excluded split; the count
arithmetic matches exactly. A verified historical checkpoint-to-input-ID association would test
that hypothesis. Current text membership alone does not.

This amendment concerns the legacy motivation figure only. The reliability-conditioned fusion
thesis and every future FF++-only training invariant remain in force.

## 2. Literature basis

n/a. This is a source/provenance correction, not a literature claim. Primary notebook/script
evidence and its limits are recorded in P1-T2's handoff; no new scholarly citations are introduced.

## 3. Architecture

Preserve the source-defined frozen trio used by the executed Celeb-DF notebook:

- `checkpoints/spatial/spatial_best_valAUC.pth`
- `checkpoints/temporal/temporal_best_valAUC.pth`
- `checkpoints/ensemble_final/ensemble_artifacts.joblib`

**Fact:** `test_CelebDF.ipynb` cells6–9 name/load this trio; the ensemble producer saves the same
artifact (`train_ensemble_final.py:161`). The app's different joblib is not proved equivalent.
No notebook execution, model retraining, legacy source edit or checkpoint modification is allowed.

Reuse the five-feature base pipeline, followed by one-column meta/calibration on `base_prob`.
Do not adopt the wholly unexecuted DeepfakevFinal draft's incompatible direct-five-feature call.

## 4. Mathematical formulation

For video i, preserve the existing legacy probability p_i. Define nominal membership
S = {i: stem_i occurs in the current archived train/val text lists}; U = V minus S.
Report video-level ROC AUC and video-bootstrap95% CIs for V, S and U.

No new parameters or initializations are introduced. AUC(S)−AUC(U) is a cohort difference, not an
identified causal estimate of leakage. An observed ordering is a result, not a value to force.

## 5. Input/output tensor contracts

| Component | Input | Output |
|---|---|---|
| Legacy feature construction | Stored frames/embeddings using the actual producing code's geometry/sampling/normalization | Five features in source order: spatial sigmoid mean/max/std/top-three mean, sigmoid temporal probability |
| `base_full` | float array (n_videos,5) | fake-class probability (n_videos,1) |
| meta/calibrator | one-column base probability | final fake-class probability (n_videos,), finite in[0,1] |
| cohort/metric computation | Unique video IDs, labels0/1, one probability per video | all/nominal-seen/nominal-unseen counts, AUC, video-bootstrap CIs |

Legacy tensor dimensions come from the producing code/checkpoint; no new32-frame/320px Phase3
preprocessing is applied to this frozen audit. This preserves the task's existing legacy inference.

## 6. Data and preprocessing

Use only the available1168-video legacy Celeb-DF evaluation subset, its existing stored frames and
embeddings, and authoritative label mapping. Keep all datasets/checkpoints/legacy lists unchanged.
Archive/source hashes from C-04 bind the current inputs; they do not authenticate historical IDs.

Nominal cohorts remain exactly the original SPEC's stem-membership definition:1064 seen,104 unseen
in the current inventory. The all aggregate is1168; seen+unseen=all. Never infer actual exposure
from these names. If required input files/labels/alignments are missing, stop rather than drop cases.

## 7. Training methodology

n/a: no training. Preserve the frozen legacy producer's settings and use `aiml`; no package or
config value replacement. A companion audit run YAML outside the protected configs directory may
record source-derived paths/legacy geometry and reference unchanged evaluation bootstrap settings.

Record the known producer seed (or unknown if not established), separately from the bootstrap
seed. No improvement claim or three-seed robustness claim is made by a single frozen-model audit.

## 8. Evaluation methodology

Video-level AUC/CI only for this diagnostic comparison. Preserve the legacy video-ensemble
aggregation, feature schema and calibration; do not choose a threshold/calibrator on Celeb-DF.
Bootstrap independent rows at the video unit using the approved evaluation settings; never frames.

Approved gate amendment: replace P1-T2's compulsory `unseen_auc < all_auc` pass condition with
data/probability/provenance completeness. Report either ordering and uncertainty faithfully.
The causal leakage hypothesis remains unverified unless historical exposure is established.

Keep the too-good/unexplained-AUC stop rule. Nominal "seen" does not automatically explain AUC>0.95.
Phase1 exit must explicitly qualify a diagnostic legacy AUC as historical-training-provenance
unverified; it cannot claim clean independent external generalization from these artifacts.
The owner approved this interpretation/gate amendment in §13 before implementation.

## 9. Acceptance tests

Required criteria; fresh execution evidence is recorded in the P1-T2 handoff:

- Stem parsing handles legacy Windows and relative paths without changing lists.
- Nominal seen/unseen are disjoint and cover1168 labeled unique videos; counts match all=1168,
  seen=1064, unseen=104, with the actual observed class counts.
- Base/meta schema uses five features then one base-probability column; reject incompatible
  artifact schemas and misaligned labels/IDs instead of guessing.
- Every video has one finite probability in[0,1]. `C-05_per_video_scores.csv` retains the approved
  `video_id,group,true_label,pred_prob` exception schema; all is derivable from the partition rows.
- Bootstrap resamples videos; intervals and figures are recomputable from saved per-video scores.
- `start_run()` records all three legacy SHA-256s with null training IDs and `predates_spec=True`;
  it does not invent historical run IDs. `log_result()` writes actual AUC/CI rows with run_id.
- Captions/data/notes explicitly identify nominal grouping and unknown historical exposure.
  Either AUC ordering is acceptable; no result is altered to match an expectation.
- Run targeted/full pytest, Ruff/format/mypy, capture checks and `git diff --check` in `aiml`.

## 10. Expected artifacts

Keep C-05/C-06's existing filenames and machine-readable schemas, PNG/PDF/raw-data/note sets,
results-log rows and C-05's documented per-video exception. Add the evidence/limitations to notes.
Use approved helpers and preserve all previous assets. Record produced artifacts and their
verification in the P1-T2 handoff; never fabricate an output or acceptance result.

## 11. Permitted engineering optimisations (Level A)

Read-only batching, vectorized metrics, typed validation and caching with explicit video IDs and
checkpoint/input provenance. Current ID-less NPZs are not sufficient cache-lineage evidence on
their own. Performance changes must leave the frozen inference/aggregation and selected data intact.

## 12. Restricted scientific changes (need the owner)

No retraining, checkpoint selection by test score, cohort redefinition, test threshold tuning,
preprocessing/model/loss changes, fabricated exposure labels or forced AUC relation. Do not change
configs/legacy code/checkpoints/datasets. The newly reported notebook global-label `pos_weight`
issue is separate; this proposal does not authorize fixing it or enlarging P1-T3's four fixes.
Do not fold this SPEC into IMPLEMENTATION/ROADMAP/protocol before owner acceptance.

## 13. Owner decision

**Owner decision recorded (2026-11-24):** "Approve SPEC-002 diagnostic audit (Recommended)".
Option1 below is selected; no implementation-blocking decision remains in this SPEC.

1. **Recommended: authorize the diagnostic amendment above**, with nominal cohort labels,
   complete provenance disclosure and completeness-based P1-T2 gate. Historical leak-inflation
   claims and clean-external claims remain unsupported; Claude revises the paper motivation.
2. **Alternative: require historical input-ID recovery first.** Keep the original causal P1-T2
   task blocked until checkpoint-linked training/validation IDs and exposure are established.
   If those establish no frame exposure, revise/retire the original leak-inflation capture anyway.

The owner approved the diagnostic amendment, including nominal cohort labels, the completeness
gate and qualified Phase1 baseline claim. Codex records that explicit approval; no causal leakage
claim or global-label loss-weight fix is authorized. P1-T2 implementation may resume; the SPEC
is not folded into the main project documents before owner acceptance.
