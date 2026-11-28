# Implementation Specification

Technical companion to `ROADMAP.md`. Section IDs match roadmap task IDs.

**For coding agents — read this first:**

1. Read `CLAUDE.md` before writing any code. It contains invariants you must not violate.
2. Implement **one task ID at a time.** Run its acceptance test. Do not start the next until it passes.
3. Every script takes `--config configs/<name>.yaml`. No hardcoded paths, ever.
4. Every script that produces a number appends to `paper_assets/results_log.csv` (schema §P0-T5).
5. If a spec here is ambiguous or contradicts the code, **stop and ask.** Do not guess and proceed.
6. Reuse existing utilities. This repo already has working code for embedding caching, split
   verification, and ffmpeg re-encoding — do not write parallel versions.

---

## Coverage — which task IDs have a spec here

`CLAUDE.md` tells you to open both documents for every task. Some roadmap tasks are not code, so
they have no section here **by design**. If a task appears in the right-hand column, `ROADMAP.md`
alone is the complete instruction — that is not an omission, and it is not a reason to stop and ask.

| Has a spec in this file | No spec needed — roadmap is sufficient |
|---|---|
| `P0-T1`, `P0-T4`, `P0-T5`, `P0-T6` | `P0-T2` (open arXiv pages and record what you see), `P0-T3` (conda env + lockfile) |
| `P1-T2`, `P1-T3` | `P1-T1` (git tag + archive the current state) |
| — | `P2-T1`–`P2-T5` (downloads, licences, disk loop) |
| `P3-T1`, `P3-T2`, `P3-T3`, `P3-T4` | — |
| `P4-T1`/`T2`/`T4` (one shared section), `P4-T3` | — |
| `P5-T1`, `P5-T2`, `P5-T3`, `P5-T4` | — |
| `P6-T1`, `P6-T2`, `P6-T3`, `P6-T4`, `P6-T5`, `P6-T6` | — |
| — | `P7-T1`–`P7-T4` (assembly, writing, bibliography, submission) |

Every ID in the left column is a `##` header you can jump to. `P4-T1`, `P4-T2` and `P4-T4` share one
section because they share a config and a code path.

---

## Target repository layout

```
DeepFake_Detection_SIC/
├── configs/                     # YAML configs — all parameters live here
│   ├── preprocess.yaml
│   ├── baseline_b3.yaml
│   ├── backbone_clip.yaml
│   ├── reliability.yaml
│   ├── fusion_ssm.yaml          # also carries the P6-T4 `ablation:` block
│   └── eval_matrix.yaml         # P6-T1  model roster x test sets x seeds
├── src/
│   ├── data/
│   │   ├── detectors.py         # P3-T1  face-detector wrappers, one interface
│   │   ├── preprocess.py        # P3-T3  video → 32 PNG crops
│   │   ├── splits.py            # P3-T4  identity-disjoint split builder
│   │   └── datasets.py          # P4     torch Dataset over crop dirs
│   ├── models/
│   │   ├── backbones.py         # P5-T1  frozen CLIP/DINOv2 + adapters
│   │   ├── reliability.py       # P5-T2  degradation estimator
│   │   ├── fusion_ssm.py        # P5-T3  reliability-conditioned SSM
│   │   ├── fusion_attn.py       # P5-T3  FiLM cross-attention fallback
│   │   └── heads.py
│   ├── train/
│   │   ├── train_baseline.py
│   │   ├── train_reliability.py
│   │   └── train_fusion.py
│   ├── eval/
│   │   ├── metrics.py           # AUC/AP/EER/ECE/MCE/Brier + bootstrap CIs
│   │   ├── figures.py           # P0-T6  save_figure/save_table — the only way figures are written
│   │   ├── evaluate.py          # P6-T1  evaluate_model — the harness T2/T3/T6 all call
│   │   ├── sweep_compression.py # P6-T2
│   │   ├── calibration.py       # P6-T3  calibration_report
│   │   ├── run_ablations.py     # P6-T4  config-grid runner
│   │   ├── profile.py           # P6-T5  profile_model
│   │   └── sweep_robustness.py  # P6-T6
│   ├── ground_truth/            # LEGACY — preserved for Phase 4 comparison
│   └── process_data/            # LEGACY — augmentations.py is reused, do not rewrite
├── tools/
│   ├── capture_check.py         # P0-T4
│   ├── verify_splits.py         # P3-T4
│   ├── quantify_leak.py         # P1-T2
│   └── benchmark_detectors.py   # P3-T1
├── paper_assets/                # git-ignored, never deleted
│   ├── captures.yaml            # P0-T4  declared files per capture — the checker's contract
│   ├── results_log.csv          # P0-T5
│   └── per_video_scores/        # every eval run, always — see §P4
└── data/
    ├── raw/                     # downloaded originals (deleted after processing)
    ├── crops/                   # 320x320 PNG, permanent
    └── splits/                  # train.txt, val.txt, test_<dataset>.txt
```

---

## P0-T1 — Blackwell build gate

Run in order; stop at first success. Record every attempt's output verbatim for `C-01`.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# Expect: 2.x, 12.8+, (12, 0)   <- (12,0) confirms Blackwell sm_120
```

**Attempt 1 — native Windows.**
```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
python -c "from mamba_ssm import Mamba2; import torch; m=Mamba2(d_model=256).cuda(); print(m(torch.randn(2,32,256).cuda()).shape)"
```
Expect `torch.Size([2, 32, 256])`. If compilation fails, capture the full error — the arch list is
usually the cause, and `TORCH_CUDA_ARCH_LIST="12.0"` before install sometimes fixes it. Try that once.

**Attempt 2 — WSL2.** Ubuntu 22.04, CUDA toolkit ≥12.8, repeat attempt 1.

**Attempt 3 — pure PyTorch fallback.** Implement a selective-scan SSM with no custom CUDA. Use a
sequential scan over T (T=32 is short, so the O(T) Python loop is acceptable — profile before
optimising). Correctness test: with Δ, B, C constant, output must match an explicit linear recurrence
to within 1e-4.

**Write the result to `configs/fusion_ssm.yaml` as `operator: mamba_cuda | mamba_torch | film_attn`.**
Everything downstream reads that key. No other file branches on the gate outcome.

---

## P0-T4 — Capture manifest and checker

Every capture's expected files are declared once, in `paper_assets/captures.yaml`. The checker reads
that manifest; nothing hardcodes filenames in Python.

```yaml
# paper_assets/captures.yaml — the contract capture_check enforces
C-05:
  phase: 1
  kind: auto              # auto | human_in_loop | manual
  title: Leak quantification (seen/unseen/all)
  files:
    - C-05_leak_quantification.png
    - C-05_leak_quantification.pdf
    - C-05_leak_quantification_data.csv
    - C-05_leak_quantification.md
C-09:
  phase: 2
  kind: manual
  title: EULA approvals and licence pages
  files:
    - C-09_licences/ffpp_eula_email.png
    - C-09_licences/celebdf_eula_email.png
    - C-09_licences/dfdc_terms_page.png
    - C-09_licences/dfdcp_terms_page.png
    - C-09_licences/dfd_terms_page.png
    - C-09_licences.md
```

`tools/capture_check.py`:

```python
def check_phase(phase: int, manifest: Path, assets: Path) -> int:
    """Verify every declared capture for `phase`. Returns the number of failures.

    Per capture, in order:
      1. every file in `files` exists and is non-empty (a 0-byte PNG is a failed savefig)
      2. the sibling .md note exists
      3. the note does NOT begin with 'STATUS: PENDING'
      4. kind == 'manual'  -> report as a TODO for the human, never as something to generate
         kind == 'human_in_loop' -> warn if the note is under 200 chars, which in practice
                                    means the interpretation paragraph was never written

    Print failures grouped by kind, manual first — those are the ones with a deadline that
    is not under the agent's control. Exit non-zero on any failure.
    """
```

**Kinds are instructions to an agent, not decoration.** A `manual` capture must never be marked
complete by generating a placeholder image. When `capture_check` reports a missing manual capture,
the correct agent behaviour is to write the `STATUS: PENDING` stub note (if absent) and tell the
user which screenshot to take — nothing else.

```bash
python -m tools.capture_check --phase 2
# CAPTURE CHECK — phase 2
#   [MANUAL]  C-09  MISSING 2 files — you must take these:
#               C-09_licences/celebdf_eula_email.png   (Celeb-DF v2 approval email)
#               C-09_licences/dfd_terms_page.png       (DFD terms page, dated)
#   [AUTO]    C-08  ok
# 1 capture incomplete.
```

---

## P0-T5 — Results log schema

`paper_assets/results_log.csv`, append-only, created once with this header:

```csv
timestamp,git_commit,task_id,model_name,config_path,train_set,test_set,seed,auc,auc_ci_low,auc_ci_high,ap,eer,acc,ece,mce,brier,n_videos,notes,run_id
```

Helper every training/eval script must call — implement once in `src/eval/metrics.py`:

```python
def log_result(task_id: str, model_name: str, config_path: str, train_set: str,
               test_set: str, seed: int, metrics: dict, n_videos: int, notes: str = "",
               *, run_id: str) -> None:
    """Append one row to paper_assets/results_log.csv.
    run_id is required and must come from start_run() for this task/model/seed/config.
    git_commit is the run's manifest commit (`git rev-parse --short HEAD` at start_run, with
    '-dirty' if the tree was dirty) - the code that was actually loaded.
    Missing metrics are written as empty strings, never as 0.0 (0.0 is a valid AUC).
    Unknown metric keys raise ValueError.
    """
```

The dirty-tree suffix matters: a number logged from uncommitted code is not reproducible, and you
want that visible in the CSV rather than discovered at writing time.

### Run identifier and per-run manifest (from SPEC-001, accepted 2026-11-24)

Every run that logs a number first calls `start_run()` (`src/eval/run_manifest.py`), which returns
`run_id = <YYYYMMDD-HHMMSS>_<task_id>_<model_name>_seed<n>` and writes
`paper_assets/runs/<run_id>/manifest.yaml` (never deleted, invariant 5). One run = one process
invocation; eval-only runs get their own `run_id`.

```python
def start_run(task_id, model_name, config_path, seed, research_question, *,
              split_paths=(), resolved_config=None, evaluated_checkpoints=None) -> str: ...
def finish_run(run_id, *, checkpoint_path=None, selected_by=None,
               status="completed") -> None: ...   # status: completed | failed
```

The manifest records: `run_id`, `task_id`, `research_question`, `git_commit`, `config_path`,
`resolved_config` (training: the C-28 file; eval: `runs/<run_id>/resolved_config.yaml`), `seed`,
`splits` (sha256 of every split file read), `environment` (python, torch, cuda, gpu, `pip_freeze.txt`),
`checkpoint` (the checkpoint this run **produced**: path, sha256, `selected_by` on FF++ val; null for
eval runs), `evaluated_checkpoints` (list of checkpoints this run **loaded**: path, sha256,
`training_run_id`; null for training runs), `started`, `finished`, `status`.

Rules: training runs must pass non-empty `split_paths`. A non-null `training_run_id` must match that
run's `checkpoint.sha256`. `training_run_id` may be null only for checkpoints that predate SPEC-001
(the legacy capstone checkpoints in P1-T2); the caller declares this with `predates_spec=True`, and
the sha256 is still recorded.

**Acceptance tests:** `python -m pytest tests/test_results_log.py tests/test_run_manifest.py -v`
(schema and column order, blank-vs-zero metrics, required `run_id`, manifest keys and split hashes,
`-dirty` suffix, eval-run checkpoint linking including legacy checkpoints).

---

## P0-T6 — Figure and table output helpers

Implement once in `src/eval/figures.py`. **Every figure in this project is written by this helper**,
so the four-file rule cannot be forgotten one plot at a time.

```python
def save_figure(fig, capture_id: str, name: str, data: pd.DataFrame,
                caption: str, assets: Path = Path("paper_assets")) -> None:
    """Write the four artifacts a paper figure requires:
        {capture_id}_{name}.png       300 dpi, bbox_inches='tight'
        {capture_id}_{name}.pdf       vector — what Elsevier wants
        {capture_id}_{name}_data.csv  every value plotted, one row per point
        {capture_id}_{name}.md        stub note: date, command (sys.argv), git commit, caption

    `data` is not optional. A figure whose numbers cannot be recovered is a figure that must
    be regenerated from a checkpoint that may no longer exist.
    """

def save_table(df: pd.DataFrame, capture_id: str, name: str, caption: str,
               tex: bool = False) -> None:
    """Write {capture_id}_{name}.csv (long format), .md (rendered), and optionally .tex
    (booktabs). Long format — one row per model x test_set x seed x metric — because a wide
    table has to be rewritten every time a model or test set is added."""

def save_console(text: str, capture_id: str, name: str, caption: str) -> None:
    """Write captured stdout/stderr to {capture_id}_{name}.txt with the note.
    Use this rather than asking anyone to screenshot a terminal."""

def write_pending_stub(capture_id: str, name: str, what_to_capture: str) -> None:
    """Create {capture_id}_{name}.md beginning with 'STATUS: PENDING' plus instructions
    for the human. capture_check fails while any stub is pending, so a manual capture
    cannot silently slip. This is the ONLY thing an agent may do for a manual capture."""
```

**Plot style, set once in `figures.py` and never per-script:** matplotlib's default cycle is fine, but
fix figure width to a single-column 3.5 in (double-column 7.16 in), font size 8–9 pt, and always call
`ax.set_xlabel`/`set_ylabel` with units. Figures rebuilt late for a column width are figures rebuilt
from checkpoints you no longer have.

**Colour must not carry information alone.** Distinguish series by marker and linestyle as well —
these figures get printed in greyscale, and some readers have a colour vision deficiency. This matters
most in `C-24`, where the whole claim is that one line's slope differs from the others'.

---

## P1-T2 — Quantify the leak

> **Executed as SPEC-002 (provenance-aware diagnostic).** The "seen/unseen" groups below are
> *list membership*, not training exposure: verification showed no model was trained on the lists
> (`RESEARCH_ANALYSIS.md` §1.2(a)), and the `unseen_auc < all_auc` acceptance was replaced by a
> data/provenance-completeness gate. Result: all 0.652 [0.620, 0.683] n=1168; listed 0.658 n=1064;
> unlisted 0.598 [0.492, 0.705] n=104. The original spec text below is kept as history.

`tools/quantify_leak.py`. **Run before Phase 3 regenerates splits.** After that, this measurement is
permanently unavailable.

```python
def load_split_stems(split_path: Path) -> set[str]:
    """Return video stems from a split file. Handles the legacy absolute Windows paths in
    data/train.txt (c:\\Users\\...\\OneDrive\\...) by taking Path(line).stem only."""

def partition_celebdf(celebdf_manifest: Path, train_stems: set[str],
                      val_stems: set[str]) -> tuple[list, list]:
    """Split Celeb-DF videos into (seen, unseen). seen = stem in train_stems | val_stems.
    Expected: ~1064 of 1168 seen (905 train + 159 val), ~104 unseen."""
```

Then load the **existing frozen** `checkpoints/ensemble_final/` and evaluate on each group.

```bash
python -m tools.quantify_leak \
  --spatial checkpoints/spatial/best.pt \
  --temporal checkpoints/temporal/best.pt \
  --ensemble checkpoints/ensemble_final/model.pkl \
  --out paper_assets/C-05_leak_quantification
```

**Emit** — all figures via `save_figure` (§P0-T6), so PNG + PDF + `_data.csv` + note come as a set:

| File | Contents |
|---|---|
| `C-05_leak_quantification.png` / `.pdf` | 3-bar chart, AUC per group, **n printed on each bar** |
| `C-05_leak_quantification_data.csv` | `group,n_videos,n_real,n_fake,auc,ci_low,ci_high` |
| `C-05_leak_quantification.md` | the numbers, the contamination counts, and the one-sentence mechanism |
| `C-06_roc_seen_vs_unseen.png` / `.pdf` | both ROC curves on one axis, AUC in the legend |
| `C-06_roc_seen_vs_unseen_data.csv` | `group,fpr,tpr,threshold` — full curves, not just the AUC |
| `C-06_roc_seen_vs_unseen.md` | note |

Plus rows in `results_log.csv` with `task_id=P1-T2`.

Also dump `C-05_per_video_scores.csv` (`video_id,group,true_label,pred_prob`). It costs nothing now,
and it is the only way to recompute this figure differently after Phase 3 regenerates the splits —
at which point the model that produced these scores can no longer be pointed at the same data.

**This one file is the documented exception to invariant 5b's naming.** It lives beside the capture
rather than in `paper_assets/per_video_scores/`, and its second column is `group`
(`seen`/`unseen`/`all`) rather than `dataset`, because the partition being measured is contamination
status, not provenance. Do not "correct" it to the 5b schema — that would discard the only axis this
measurement is about. Every *other* eval run follows 5b exactly.

**Acceptance:** three AUCs reported with sample counts summing to 1168. `unseen_auc < all_auc`.

**If `unseen` has fewer than ~80 videos**, the estimate is noisy — report a bootstrap CI and say so
explicitly in the note. A wide CI honestly stated is fine; a point estimate implying precision it
lacks is not.

---

## P1-T3 — The four fixes

**Fix 1 — AMP-blocking mask.** `src/ground_truth/models.py:85`:

```python
# BEFORE — overflows fp16 (max 65504) to -inf, softmax then yields NaN
scores = scores.masked_fill(mask, float("-1e9"))
# AFTER
scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
```
The AMP re-enable and its FP32-vs-AMP loss-curve check (`C-07`) moved to `P4-T1` (owner decision
2026-11-28): the legacy trainer is `src/process_data/train_temporal.py`, which has no AMP path and
its own copy of the `-1e9` mask, and the only valid data to train it on are the Phase 3 splits.
`P4-T1` applies Fix 1 to that file too, adds AMP, and captures both curves for `C-07`.

**Fix 2 — test-split mislabel.** In `src/ground_truth/build_features.py:218`, the list includes
`"test_internal"` but `ensemble_features_final/` never receives it, so
`train_ensemble_final.py:50-54` falls back to `test.npz` and prints "Test (External)". Either build
the missing `.npz` or drop it from the list — then make the fallback loud:

```python
if X_test is None:
    raise FileNotFoundError(f"Expected {name}.npz; refusing to silently substitute another split.")
```
Silent fallback caused a mislabelled published number once already. Make it impossible to repeat.

**Fix 3 — `app.py:866`.** Replace the hardcoded `"99.2%"` with a value read from
`results_log.csv`, or delete the metric.

**Fix 4 — splits with relative paths.** Deferred to `P3-T4`, which rewrites them anyway.

---

## P3-T1 — Detector benchmark

`src/data/detectors.py` — one interface, so the rest of the pipeline never branches per detector:

```python
class FaceDetector(ABC):
    name: str
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Box]:
        """frame: HxWx3 uint8 RGB. Returns boxes sorted by confidence descending.
        Box = namedtuple('Box', 'x1 y1 x2 y2 score')  (absolute pixel coords)
        Returns [] when no face is found. Must never raise on a valid frame."""
```

Implement `RetinaFaceDetector`, `MTCNNDetector`, `S3FDDetector`, `YOLOv8FaceDetector`. Skip any that
takes over an hour to install and record the skip in `C-10` — this is a means, not the research.

`tools/benchmark_detectors.py`:

```python
def build_benchmark_set(seed: int = 42) -> list[VideoRef]:
    """200 videos: 50 FF++ c23, 50 FF++ c40, 50 Celeb-DF v2, 50 DFDC.
    Stratified 50/50 real/fake. Fixed seed — the same set for every detector, or the
    comparison is meaningless."""

def evaluate_detector(det: FaceDetector, videos: list[VideoRef],
                      frames_per_video: int = 10) -> dict:
    """Returns:
      detection_rate      frames with >=1 face / total frames
      mean_iou_adjacent   mean IoU of primary box between consecutive sampled frames,
                          computed WITHIN a video only
      jump_rate           fraction of adjacent pairs with IoU < 0.5
      area_cv             std(box area)/mean(box area) within a video, averaged
      sec_per_frame       wall-clock throughput on GPU
      per_dataset         all of the above, broken down by source dataset
    """
```

**Interpretation — read this before deciding.** `detection_rate` alone is misleading: a detector
firing on shadows and hands scores highest while poisoning the training set. Read it together with
`jump_rate`. High rate + high jump_rate = false positives, which is worse than a miss, because a
miss is a dropped frame while a false positive is a training sample of a non-face labelled "fake".

`mean_iou_adjacent` is the discriminating metric. Your temporal stream reads a *sequence* of crops,
so an unstable detector injects spurious motion the model can learn as a deepfake cue — a
generalization failure that looks like a modelling problem.

**Decision rule, in order:**
1. Eliminate `detection_rate < 0.95` on FF++ c23 (fails the easy case).
2. Among survivors, highest `mean_iou_adjacent`.
3. Tie → manual contact-sheet inspection (`--contact-sheet`, 50 random crops per detector; count
   non-faces, half-faces, wrong-person-in-multi-face).
4. Still tied → `sec_per_frame`.
5. Genuinely close on 1–4 → run `P3-T2`.

Also flag any detector whose `detection_rate` varies more than 5 points **across datasets** — that
asymmetry manufactures a domain gap that will masquerade as a generalization result.

```bash
python -m tools.benchmark_detectors --out paper_assets/C-10_detector_bakeoff --contact-sheet
```

**Contact sheets are composed images, not screenshots.** Build them with PIL: a 10×5 grid of 50
crops, each captioned with `video_id:frame_idx` and the detection confidence, saved as
`C-10_detector_bakeoff/contact_sheet_<detector>.png`. The caption is what lets you re-find a bad crop
in the source video after you spot it.

For `C-11`, save the same frame cropped by every detector as one panel via `save_figure`, with each
detector's returned box coordinates in the `_data.csv` so the panel can be rebuilt without re-running
detection.

**Acceptance:** `C-10_detector_bakeoff.csv` holds all five metrics × all candidates × 4 datasets;
contact sheets written for every candidate; `C-10_detector_bakeoff.md` states the choice and the
reason in 2–3 sentences **and carries the human spot-check counts** (non-faces, half-faces,
wrong-person) filled in per detector. Those counts come from a person looking at the sheets — an
agent finishing this task writes every file and then reports that the Metric-4 column awaits the
user, rather than leaving it blank silently or inventing values.

---

## P3-T2 — Downstream tie-break

Only if `P3-T1` steps 1–4 leave two candidates genuinely close. Preprocess the same 500 FF++ videos
with each finalist, train the identical small model (EfficientNet-B0, 10 epochs, fixed seed) on each,
compare validation AUC. Same seed, same hyperparameters, same videos — the **only** difference is
the detector. Log both with `task_id=P3-T2`.

---

## P3-T3 — Preprocessing pipeline

`configs/preprocess.yaml`:

```yaml
detector: null                # REQUIRED — set from the P3-T1 result. Deliberately unset:
                              # preprocess.py must raise if this is null. Shipping a default
                              # here would let an agent skip the bake-off by accepting it.
frames_per_video: 32
sampling: uniform             # uniform over full duration
crop_size: 320
crop_margin: 1.3              # multiplier on max(box_w, box_h) -> square crop
output_format: png
png_compress_level: 3         # 0-9; 3 balances size and write speed
min_face_size: 64             # skip detections smaller than this (px)
max_workers: 12               # 24-core CPU; leave headroom for the GPU feed
on_missing_face: nearest      # nearest | skip | black
```

`src/data/preprocess.py`:

```python
def extract_crops(video_path: Path, out_dir: Path, cfg: PreprocessConfig) -> CropManifest:
    """1. Probe frame count with cv2/ffprobe.
       2. Choose `frames_per_video` uniformly spaced indices over the FULL duration.
       3. Decode only those frames (seek, do not decode all — 10x faster).
       4. Detect faces; take the highest-confidence box.
       5. Square the box: side = max(w,h) * crop_margin, centred on the box centre.
       6. Clamp to frame bounds; if clamping would distort the aspect, pad with edge
          replication rather than resizing non-uniformly.
       7. Resize to crop_size x crop_size (cv2.INTER_AREA when downscaling,
          INTER_CUBIC when upscaling).
       8. Write {out_dir}/{idx:03d}.png
       9. Return manifest: per-frame box coords, confidence, whether a face was found.

    on_missing_face='nearest': reuse the nearest successful frame's box. Record the
    substitution in the manifest — it must be visible in the data, not hidden.
    """
```

**Square-crop rationale:** non-square crops resized to square distort facial geometry differently
per video, and that distortion becomes a dataset-specific signature the model can learn instead of
the forgery. Square-then-resize keeps geometry consistent.

**Manifest is mandatory.** Write `{out_dir}/manifest.json` per video with box coordinates,
confidence, and substitution flags. Two reasons: it lets you re-derive different crop margins
without re-detecting, and the fraction of substituted frames is a data-quality number a reviewer may
ask for.

**Storage estimate:** ~200–250 KB per PNG × 32 × ~120k videos ≈ 100 GB. Verify on the first 100
videos and extrapolate before launching the full run.

DFDC part loop:
```bash
for part in $(seq -w 0 49); do
  python -m tools.download_dfdc --part $part --out data/raw/dfdc_$part
  python -m src.data.preprocess --config configs/preprocess.yaml \
      --input data/raw/dfdc_$part --output data/crops/dfdc --manifest-only-on-exists
  python -m tools.verify_crops --dir data/crops/dfdc --expect-from data/raw/dfdc_$part
  rm -rf data/raw/dfdc_$part     # only after verify_crops passes
done
```
**`verify_crops` must pass before the `rm`.** Validate the whole loop on part 00 first — the
delete is irreversible and re-downloading 9.4 GB to recover from a bad assumption is avoidable.

---

## P3-T4 — Identity-disjoint splits

`src/data/splits.py`:

```python
def parse_identity(video_path: Path, dataset: str) -> str:
    """FF++ original:  'youtube/123.mp4'          -> '123'
       FF++ manipulated: 'Deepfakes/123_456.mp4'  -> frozenset({'123','456'})  BOTH identities
       Celeb-DF: 'id0_id1_0001.mp4'               -> frozenset({'id0','id1'})
       DFDC: from metadata.json 'original' field
    A manipulated video contains TWO identities. Both must land in the same split or the
    source identity leaks across the boundary — this is the subtle version of the bug that
    produced the 905-video contamination."""

def build_splits(ff_root: Path, ratios=(0.7, 0.15, 0.15), seed=42) -> dict:
    """Group FF++ by connected components of the identity co-occurrence graph, then assign
    whole components to splits. Greedy: sort components by size descending, assign each to
    whichever split is furthest below its target ratio.
    Stratify so all four manipulation methods appear in every split.
    Write RELATIVE paths, forward slashes, from the repo root."""
```

**Why connected components:** if video A pairs identities {1,2} and video B pairs {2,3}, then
1, 2 and 3 must all share a split. Assigning per-video, or even per-identity without transitivity,
leaks. This is the same class of error as the original leak, one level deeper.

`tools/verify_splits.py` — must be run before every training job:

```python
ASSERTIONS = [
    "no video path appears in more than one split",
    "no identity appears in more than one FF++ split",
    "train contains ONLY FF++ videos (zero celeb/dfdc/dfd)",
    "all four FF++ manipulation methods present in each split",
    "every listed path exists on disk",
    "every crop dir has exactly frames_per_video PNGs",
]
```
Exit non-zero on any failure, printing the offending entries (capped at 20).

```bash
python -m tools.verify_splits --splits data/splits --config configs/preprocess.yaml
grep -ci celeb data/splits/train.txt    # MUST print 0
```

---

## P4-T1 · P4-T2 · P4-T4 — Baseline retrain and full-matrix eval

One section for three roadmap tasks because they share a config and a code path: `P4-T1` is the
retrain on clean splits, `P4-T2` is the same run at 300px as well as 224, `P4-T4` is the evaluation
across the full test matrix. `P4-T3` has its own section below.

`configs/baseline_b3.yaml`:

```yaml
backbone: efficientnet_b3
pretrained: true
img_size: 300                 # B3 native; also run 224 for the comparison
batch_size: 24
amp: true                     # enabled by the P1-T3 fix
epochs: 30
optimizer: adamw
lr: 1.0e-4
weight_decay: 1.0e-2
scheduler: cosine
warmup_epochs: 2
label_smoothing: 0.05
early_stop_patience: 7
seeds: [42, 1337, 2024]
```

Keep the legacy architecture otherwise **unchanged** — this run isolates the effect of the protocol
fix from the effect of the architecture change. A reviewer will require that separation, and
conflating them forfeits contribution 5.

`src/eval/metrics.py`:

```python
def bootstrap_auc(y_true, y_score, n_boot=1000, seed=42) -> tuple[float, float, float]:
    """Video-level bootstrap. Resample VIDEOS, not frames — frames within a video are
    correlated, so frame-level bootstrapping gives CIs that are far too narrow.
    Returns (auc, ci_low, ci_high) at 95%."""

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float: ...
def maximum_calibration_error(y_true, y_prob, n_bins=10) -> float: ...
def brier_score(y_true, y_prob) -> float: ...
def eer(y_true, y_score) -> float: ...
```

**Always report ECE with MCE and Brier.** ECE alone is gameable — a model predicting the base rate
for every input achieves near-zero ECE while being useless. Reviewers in this area know it.

**Video-level aggregation:** average the 32 frame probabilities per video (mean, not max — max is
sensitive to a single bad crop). Report video-level AUC as primary; frame-level as secondary.

### What every training and eval run must write

Not optional, and not something to add later — the figures in Phases 4–6 are all built from these:

| Artifact | Written by | Why it cannot wait |
|---|---|---|
| `paper_assets/C-16_training_curves.csv` row **per epoch** (`model,seed,epoch,train_loss,val_loss,val_auc,lr,epoch_seconds,peak_vram_gb`) | trainer, at each epoch end | A crashed run still leaves its curve. Scraping console scrollback afterwards does not scale to 3 seeds × every model. |
| `paper_assets/C-28_hyperparameters/<model>_seed<n>.yaml` — the **resolved** config | trainer, at run start | A hand-kept table drifts from what actually ran, invisibly. |
| `paper_assets/per_video_scores/<model>_<test_set>_seed<n>.csv` (`video_id,dataset,true_label,pred_prob`) | evaluator, always | Every later figure — ROC, calibration bins, failure gallery, qualitative panel, bootstrap CIs — is recomputable from these without reloading a checkpoint. This one file class is what stops "can we see that split by manipulation method?" from meaning a retrain. |
| `results_log.csv` row | evaluator, always | Invariant 4. |

Peak VRAM via `torch.cuda.max_memory_allocated()`, reset at epoch start. Epoch wall-clock from a
monotonic timer, not from timestamps in the log.

---

## P4-T3 — DeepfakeBench baselines

Clone DeepfakeBench, point its config at **your** splits, run Xception and EfficientNet-B4 (add SBI
if it installs cleanly).

**Do not use its pretrained weights on its own splits.** That measures its protocol, not yours, and
the numbers will not be comparable to any row in your `results_log.csv`. Retrain inside its codebase
on your split files, so the only difference between its rows and yours is the model.

Log with `task_id=P4-T3` and `model_name=dfb_xception` / `dfb_efficientnet_b4`, so the provenance of
a reproduced number is never ambiguous against your own implementation of the same architecture.

**Acceptance:** both baselines produce in-domain AUC within a few points of their published FF++
numbers. A large shortfall means the split wiring is wrong, not that the published number was
inflated — check that its loader is reading `data/splits/` before concluding anything.

---

## P5-T1 — Frozen backbone + adapters

`src/models/backbones.py`:

```python
class FrozenBackboneWithAdapters(nn.Module):
    """CLIP ViT-B/16 or DINOv2 ViT-B/14, fully frozen, with trainable bottleneck adapters.

    Adapter, inserted after each transformer block:
        h = h + up(gelu(down(ln(h))))     down: D->D/8, up: D/8->D
        up initialised to ZERO so the adapted model starts identical to the frozen one.
    Trainable: adapters + final norm + head. ~3M of ~86M params.
    """
    def __init__(self, name="clip_vit_b16", adapter_dim_ratio=8, n_frames=32): ...
    def forward(self, x):  # x: (B, T, 3, H, W)
        """Returns (B, T, D) per-frame tokens. Reshape to (B*T, ...) for the backbone,
        then unflatten — do not loop over T in Python."""
```

**Zero-init the up-projection.** With it, training begins exactly at the frozen backbone's behaviour
and can only improve from there; random init throws away the pretrained representation in the first
few steps and often trains worse than frozen.

**Cache embeddings.** With the backbone frozen, extract once and reuse — the same pattern the repo
already uses for `embeddings/*.npy`:

```bash
python -m src.train.extract_embeddings --config configs/backbone_clip.yaml \
  --split all --out embeddings/clip_vit_b16/
```
This is what makes the Phase 6 ablation grid affordable: each fusion experiment then trains on
cached tensors in minutes rather than hours. **Caveat: cached embeddings are only valid while
adapters are frozen.** When adapters train, either re-extract per epoch or train end-to-end. Do the
adapter-training run once end-to-end, then freeze and cache for the ablation grid.

**Acceptance:** cross-dataset AUC exceeds the Phase 4 legacy baseline. If not, stop and diagnose
before building anything on top.

---

## P5-T2 — Reliability estimator

**The insight that makes this free:** you apply the degradation yourself, so the ground-truth label
requires no annotation.

Training-data generation:
```python
def make_degradation_pairs(video_paths, crf_values=(18, 23, 28, 32, 36, 40, 44)):
    """For each video, re-encode at each CRF using the EXISTING helpers in
    src/process_data/augmentations.py (reencode_video). Do not rewrite them.
    Also apply: downscale-then-upscale (0.5x, 0.75x), gaussian blur (sigma 0.5-2.0),
    JPEG (quality 40-95), additive gaussian noise (sigma 2-10).
    Label = the applied degradation parameters. Free supervision."""
```

`src/models/reliability.py`:

```python
class ReliabilityEstimator(nn.Module):
    """Small CNN, <1M params, predicting an INTERPRETABLE 4-D descriptor from a crop:
        r[0] blockiness   8x8 DCT block-boundary energy   (JPEG/H.264 artifacts)
        r[1] hf_ratio     high-freq / total spectral energy (detail loss)
        r[2] entropy      local Shannon entropy            (information content)
        r[3] blur         variance of Laplacian            (sharpness)

    Architecture: 4 conv blocks (16,32,64,128), stride 2, BN, GELU
                  -> global avg pool -> Linear(128, 4) -> sigmoid
    Input 320x320 downsampled to 160x160 (artifact statistics survive 2x downsampling
    and it is 4x cheaper).

    Output r in [0,1]^4 per frame, where HIGHER = MORE RELIABLE evidence.
    """
```

**Targets must be computed analytically, not learned from labels directly.** Compute each of the four
statistics with a closed-form function on the degraded crop; the CNN learns to predict them. Two
reasons this beats regressing CRF directly: the descriptor stays interpretable (so `C-20` is
meaningful), and it generalizes to degradations not in your sweep, because it measures the *effect*
rather than recognising the *cause*.

```python
def compute_blockiness(img: np.ndarray) -> float:
    """Mean abs gradient across 8x8 block boundaries, divided by mean abs gradient at
    non-boundary positions. ~1.0 for clean, grows with block artifacts. Normalise to
    [0,1] with 1 - tanh(max(0, ratio - 1)), so 1.0 = reliable."""
```

Loss: MSE on the 4-D descriptor. 20 epochs, AdamW, lr 3e-4, batch 128. Trains in under an hour.

**GATE — falsifiable, and everything downstream depends on it.** On held-out videos, Spearman
correlation between predicted `r[0]` (blockiness) and true CRF must satisfy |ρ| > 0.8, with the sign
negative (higher CRF → lower reliability). Emit the scatter plot for `C-19`.

If it fails: check the target normalisation first (a saturated `tanh` kills the gradient), then
whether your CRF range actually produces measurable blockiness differences on 320px crops. **Do not
proceed past this gate** — the fusion mechanism has nothing to condition on if reliability is noise.

---

## P5-T3 — Reliability-conditioned selective fusion

### Token construction

Three streams per frame `t`:
- **spatial** — backbone token `s_t ∈ R^D`
- **frequency** — DCT/FFT magnitude map through a small CNN → `f_t ∈ R^D`
- **temporal** — frame difference or optical-flow features → `m_t ∈ R^D`

**Interleave**, do not concatenate:
```
[s_1, f_1, m_1, s_2, f_2, m_2, ..., s_32, f_32, m_32]     length 96
```
Add a learned 3-way **stream-type embedding** plus a temporal position embedding. Interleaving lets
the SSM's state carry cross-stream evidence *within* each timestep as it scans, which
separate-then-fuse cannot. `P6-T4` ablates exactly this choice.

### The conditioning — the paper's core mechanism

In a selective SSM, `Δ_t` controls how strongly token `t` is written into the hidden state.
Standard Mamba: `Δ_t = softplus(W_Δ x_t + b_Δ)`.

**Condition it on reliability:**

```python
delta_t = F.softplus(self.W_delta(x_t) + self.b_delta + self.W_rel[k](r_t))
```
where `r_t ∈ R^4` is the reliability descriptor of the **frame** token `t` comes from (one
estimator, one descriptor per frame, §P5-T2), and `k ∈ {spatial, frequency, temporal}` is the
token's stream. All three tokens of a frame share `r_t`; each stream has its **own** `W_rel[k]`.
Degradation is a property of the frame; how much each stream should trust it is learned. A
single shared `W_rel` would add the same term to all three streams and could not express
"blockiness hurts the frequency stream but not the temporal one", which is the per-stream claim.
The temporal token `m_t` (built from frames t-1 and t) uses `r_t`. *(Decided 2026-11-21.)*

```python
class ReliabilityConditionedSSM(nn.Module):
    def __init__(self, d_model, d_state=16, rel_dim=4, operator="mamba_cuda"):
        """operator from configs/fusion_ssm.yaml (set by P0-T1).
        W_rel: one Linear(rel_dim, d_model) PER STREAM (3 total), ALL ZERO-INITIALISED
        -> training starts as standard Mamba, so any gain is attributable to
        conditioning rather than to a different initialisation. This is what makes
        the ablation honest.
        Also condition B and C projections via FiLM, also per stream:
            B_t = B_t * (1 + gamma_B[k](r_t)),  gamma_B[k] zero-init (likewise C)
        """
```

**Why Δ and not a weighted sum.** Δ is the SSM's own decision about what to remember. Low
reliability → small Δ → the token barely enters the state → the stream is effectively ignored *for
that input*. A post-hoc weighted sum cannot express this: it rescales a contribution that has already
been computed and mixed, and its weights are learned as an average over the training distribution.
This is the sentence that answers "why not just concatenate?" and it is the paper's central claim —
state it in these terms in the methods section.

**Zero-init everything reliability-related.** It makes the no-gate ablation an exact nested model,
so the comparison isolates the mechanism rather than confounding it with initialisation.

### Fallback (`operator: film_attn`)

If `P0-T1` reached attempt 3 and even that is too slow, use cross-attention with FiLM-conditioned
K/V:
```python
K = K * (1 + gamma_k[k](r_t));  V = V * (1 + gamma_v[k](r_t))   # per stream k, zero-init
# optionally: attn_logits = attn_logits + beta(r)
```
Same thesis, same experiments, same claims. Only the operator differs — and `RESEARCH_ANALYSIS.md`
§6.4 explains why that was designed to be true.

**GATE:** beats the unconditioned equivalent on c40. If conditioning does not help where evidence is
actually degraded, the thesis is wrong — and finding that out here is the point of gating it.

---

## P5-T4 — Stream-specific SBI and calibration

```python
def spatial_sbi(img):    """Blend a distorted copy along the landmark hull. Standard SBI."""
def temporal_sbi(clip):  """Perturb inter-frame alignment: small per-frame affine jitter on the
                            blended region only, creating temporal inconsistency without
                            spatial artifacts. Forces the temporal stream to learn dynamics
                            rather than re-learning spatial cues."""
def frequency_sbi(img):  """Inject spectral artifacts: upsample a downsampled copy inside the
                            hull (checkerboard/aliasing), leaving the spatial appearance nearly
                            unchanged."""
```
Each stream sees its own variant, so the three learn non-redundant evidence rather than three views
of one signal. This is also the direct answer to the FSBI overlap in `RESEARCH_ANALYSIS.md` §3.2.

Calibration term:
```python
# Mean reliability across streams modulates logit confidence.
# Low reliability -> logits pulled toward 0 -> predictions toward 0.5.
logit = raw_logit * torch.sigmoid(self.conf_scale(r_mean))
loss = F.binary_cross_entropy_with_logits(logit, y) + lam * calibration_penalty
```
Sweep `lam ∈ {0, 0.05, 0.1, 0.2}` and report the trade-off curve. **Gate: ECE improves under shift
with no AUC regression.** If ECE improves only because AUC dropped, that is not a result — report it
as the trade-off it is.

---

## P6-T1 — Final cross-dataset matrix

The evaluation harness every Phase 6 task calls. Write it once; `P6-T2`/`T3`/`T6` differ only in
which transform they apply to the test videos before it runs.

```python
def evaluate_model(ckpt: Path, split_file: Path, cfg: EvalConfig,
                   transform: Callable | None = None) -> pd.DataFrame:
    """Score every video in `split_file`. Returns one row per VIDEO:
        video_id, dataset, true_label, pred_prob, n_frames_scored

    `transform` is applied per frame before the model sees it (None for the clean matrix,
    a CRF re-encode for P6-T2, noise/resize/etc. for P6-T6). Keeping it a parameter is
    what stops four near-identical eval scripts from drifting apart.

    Video-level probability = MEAN over the 32 frame probabilities, not max — max is
    hostage to a single bad crop. Frame-level scores stay available in the returned
    frame table for the secondary metric.
    """
```

Roster and grid — every model × every test set × three seeds:

| Models | Test sets |
|---|---|
| `baseline_b3` (P4-T1), `dfb_xception`, `dfb_efficientnet_b4` (P4-T3), `clip_adapters` (P5-T1), `fusion_nogate`, `fusion_static`, `fusion_conditioned` (P5-T3) | FF++ c23 (in-domain), FF++ c40, CDF-v2, DFDC, DFDCP, DFD |

```bash
python -m src.eval.evaluate --config configs/eval_matrix.yaml \
  --models baseline_b3 dfb_xception dfb_efficientnet_b4 clip_adapters \
           fusion_nogate fusion_static fusion_conditioned \
  --test-sets ffpp_c23 ffpp_c40 cdf2 dfdc dfdcp dfd \
  --seeds 42 1337 2024 \
  --out paper_assets/C-23_final_matrix
```

Metrics per cell: AUC (primary), AP, EER, accuracy at the **validation-selected** threshold. That
threshold comes from the FF++ validation split and nowhere else — picking it per test set is
invariant 2 violated in its most tempting form, because it looks like fair per-dataset tuning.

Aggregate across seeds as mean ± std. **Write both files**: `C-23_final_matrix.csv` keeps one row
per seed, `C-23_final_matrix_aggregated.csv` holds the mean ± std. A std cannot be recovered from a
mean, and the teammate writing the paper will need the per-seed rows to answer a reviewer asking
about variance.

Bootstrap CIs via `bootstrap_auc` (§P4) — **over videos**, invariant 8.

**Acceptance:** 7 models × 6 test sets × 3 seeds = 126 rows in the per-seed CSV, no gaps.
`per_video_scores/` holds 126 files. Every row has a matching `results_log.csv` entry with
`task_id=P6-T1`.

**If any cross-dataset AUC exceeds ~0.95, stop and re-run `verify_splits` before recording it.**
This project has already produced one such number, and it was leakage rather than success.

---

## P6-T2 — Compression sweep

```bash
python -m src.eval.sweep_compression --config configs/fusion_ssm.yaml \
  --crf 23 28 32 36 40 44 --social-media \
  --models baseline_b3 clip_adapters fusion_nogate fusion_static fusion_conditioned \
  --out paper_assets/C-24_compression_sweep
```

Reuse `src/process_data/augmentations.py` — it already implements the CRF re-encode and nothing calls
it. `--social-media`: 720p downscale → CRF 28 → 2-pass re-encode, approximating platform transcoding.

**Read the sweep for slope, not intercept.** The claim is not "we are better at c23," it is "we
degrade more slowly." Report the AUC drop from c23→c40 per model as an explicit column; **that delta
is the contribution.** If your slope matches the baselines' while your intercept is higher, you have
a better backbone and not a robustness mechanism — say so honestly, and the paper becomes a weaker
but still true claim.

**Emit (via `save_figure`/`save_table`):**

| File | Contents |
|---|---|
| `C-24_compression_sweep.png` / `.pdf` | AUC vs CRF, one line per model, mean across seeds with ±std shaded. Distinct **marker and linestyle** per model, not colour alone — the slope comparison must survive greyscale printing. |
| `C-24_compression_sweep_data.csv` | `model,seed,crf,auc,ci_low,ci_high,n_videos` plus a `crf=social_media` row per model |
| `C-24_slope_table.csv` | `model,auc_c23,auc_c40,delta,fitted_slope` — **the delta column is the contribution**, so it is a table in its own right |
| `C-24_compression_sweep.md` | note, stating explicitly whether the gap widens |

Write the per-video scores at every CRF into `paper_assets/per_video_scores/` as usual. Re-encoding
the eval sets at six CRFs is the expensive part of this task; keeping the scores means any later
question about the sweep is answered by re-reading a CSV rather than re-encoding.

---

## P6-T3 — Calibration under shift

**This refreshes `C-22`; it does not create a new capture.** `P5-T4` produced C-22 for three arms
during development. Here you regenerate the same files across the full Phase 6 roster and all six
test sets. Same capture ID, same filenames, more rows — the capture index stays at 29.

Arms, and why each is present:

| Arm | Answers |
|---|---|
| `baseline_b3` raw | how badly an uncalibrated detector overconfides off-distribution |
| `baseline_b3` + Platt | does cheap post-hoc calibration already solve it? |
| `baseline_b3` + temperature | the strongest post-hoc baseline, one parameter |
| `fusion_nogate` | is any gain just the better backbone? |
| `fusion_conditioned` | the claim: calibration as a consequence of the mechanism |

**Fit every post-hoc calibrator on the FF++ validation split only.** Fitting a temperature on the
shifted test set measures nothing — it is invariant 2 violated in the exact place the paper makes its
claim, and a reviewer checking one thing will check this.

```python
def calibration_report(scores: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """From per-video scores -> one row per (model, test_set, seed):
        ece, mce, brier, nll, and the per-bin table separately.
    Equal-width bins over [0,1]. Also emit equal-mass bins as a robustness check:
    equal-width ECE is sensitive to bin count when predictions cluster at the extremes,
    which they do off-distribution."""
```

Read from `paper_assets/per_video_scores/` written by `P6-T1` — do not re-run inference. That is the
whole reason invariant 5b exists.

**Emit** into the existing C-22 filenames: `C-22_reliability_diagrams.png`/`.pdf` (panel grid, model
× test set), `C-22_reliability_diagrams_data.csv` (model, test_set, seed, bin_index, bin_lo, bin_hi,
n_in_bin, mean_predicted, observed_frequency), `C-22_calibration_metrics.csv` (model × test_set ×
seed × ECE/MCE/Brier/NLL), `C-22_reliability_diagrams.md`.

**Keep `n_in_bin`.** A bin holding four videos looks wildly miscalibrated when it is merely empty,
and that is the first thing someone reading the figure will misread.

**GATE:** ECE improves under shift with **no AUC regression** versus `fusion_nogate`. If ECE only
improved because AUC fell, report it as the trade-off it is — a model that hedges everything is
well-calibrated and useless. Report all four metrics together; ECE alone is gameable by a
non-discriminative predictor and reviewers in this area know it.

---

## P6-T4 — Ablations

Where reviewers look hardest. Every ablation is a **config switch**, never a separate model file —
a forked training script diverges from the main one and the ablation silently stops measuring what
it names.

```yaml
# configs/fusion_ssm.yaml — the ablation surface
ablation:
  gate: conditioned      # none | static | conditioned | oracle
  operator: null         # null = use the top-level `operator` (Gate 0-A result).
                         # Set `transformer` ONLY for the SSM-vs-Transformer row.
  tokens: interleaved    # interleaved | separate
  backbone: adapted      # frozen | adapted
  sbi: stream_specific   # shared | stream_specific
  reliability: learned   # learned | true_crf
```

| Row | Switch | Question |
|---|---|---|
| no-gate | `gate: none` | is conditioning doing anything? |
| static-gate | `gate: static` | is it the *input-dependence*, or just per-stream weights? |
| learned-gate | `gate: conditioned` | the full model |
| **oracle-gate** | `gate: oracle` | upper bound — feeds the *true* degradation parameters |
| SSM vs Transformer | `ablation.operator: transformer` | the operator, or just capacity? (if Gate 0-A ended on `film_attn`, this row is FiLM-attention vs plain Transformer — rename it in the paper) |
| interleaved vs separate | `tokens: separate` | does interleaving matter? |
| frozen vs adapted | `backbone: frozen` | how much is the backbone alone? |
| shared vs stream-specific SBI | `sbi: shared` | does non-redundant supervision help? |
| true CRF vs learned reliability | `reliability: true_crf` | what does estimation error cost? |

**`gate: none` must be an exact nested model.** Because `W_rel` and every FiLM `gamma` are zero-init
(invariant 7), the conditioned model at initialisation *is* the unconditioned model. So `none`
differs from `conditioned` by the mechanism and nothing else — not by initialisation, not by
parameter count in any way that matters. `test_zero_init_conditioning` guards this; if it fails, the
central ablation is uninterpretable and no number from this task is worth reporting.

**`gate: static`** learns one weight per stream, shared across all inputs — the standard multi-stream
fusion this paper argues against. It is the honest competitor, not a strawman. If `conditioned` does
not beat `static`, the thesis is that input-dependence matters and the data says it does not.

**`gate: oracle`** substitutes the reliability descriptor computed analytically from the *known*
applied degradation instead of the estimator's prediction. Only available on the compression sweep,
where you applied the CRF yourself. It bounds the headroom: `oracle − conditioned` is what better
reliability estimation could still buy, and `conditioned − none` is what you actually captured.
Quoting the second without the first invites "how much is left on the table?" with no answer.

**Matched parameters for the Transformer arm.** Tune the Transformer's width/depth so total trainable
parameter count is within 5 % of the SSM's, and say so in the note. An operator comparison at
mismatched capacity measures capacity.

Three seeds per row. `task_id=P6-T4`, per-video scores as always.

```bash
python -m src.eval.run_ablations --config configs/fusion_ssm.yaml \
  --grid ablation.gate=none,static,conditioned,oracle \
  --seeds 42 1337 2024 --out paper_assets/C-25_ablations
```

**Acceptance:** every row present at 3 seeds with deltas vs the full model and CIs.
`C-25_ablations.md` carries **your** one-line interpretation per row — an agent finishing this task
writes every file and then reports that the interpretation column awaits you, rather than inventing
it. Write those lines while the run is fresh; reconstructing them months later is guesswork.

---

## P6-T5 — Efficiency

```python
def profile_model(model: nn.Module, input_shape=(1, 32, 3, 224, 224)) -> dict:
    """Returns:
      params_total, params_trainable      count directly, do not trust a printed summary
      gflops                              fvcore.nn.FlopCountAnalysis, one video
      latency_ms_bs1, latency_ms_bs8      median of 100 timed runs after 20 warmup runs,
                                          torch.cuda.synchronize() around each
      peak_vram_gb                        torch.cuda.max_memory_allocated(), reset first
    """
```

**Warmup and synchronize are not optional.** Without warmup you time cuDNN autotuning; without
`synchronize()` you time kernel *launches* and get numbers that are impossibly fast. Both mistakes
produce a table that looks fine and is wrong.

Report **median, not mean** — one scheduler hiccup skews a mean and nothing else in the table moves.

Measure every model in the `P6-T1` roster on identical hardware in one session, with nothing else on
the GPU. Record the hardware string from `C-03` in the CSV: a latency number with no hardware
attached is unusable to a reader.

**Emit** `C-26_efficiency.csv` (model, params_total, params_trainable, gflops, latency_ms_bs1,
latency_ms_bs8, peak_vram_gb, hardware) and `C-26_efficiency.md`.

**Expect to lose on some rows and say so.** The frozen backbone means few trainable parameters but
full inference cost, so `baseline_b3` will likely win on latency. State that plainly — a robustness
paper that pretends to also be the fastest reads as overclaiming, and the honest row costs nothing
because speed is not the contribution.

---

## P6-T6 — Robustness beyond compression

Same harness as `P6-T1`, different `transform`. Reuse `src/process_data/augmentations.py` wherever it
already implements the perturbation.

| Perturbation | Levels | Probes |
|---|---|---|
| Downscale→upscale | 0.75×, 0.5×, 0.25× | resolution loss, independent of codec |
| Gaussian noise | σ = 2, 5, 10, 20 | sensor/transmission noise |
| Frame dropping | 10 %, 25 %, 50 % of the 32 frames | the temporal stream specifically |
| Brightness/contrast | ±20 %, ±40 % | photometric shift, no artifact change |
| Gaussian blur | σ = 0.5, 1.0, 2.0 | detail loss without blocking |

**Frame dropping is the discriminating one.** It degrades temporal evidence while leaving spatial and
frequency evidence intact — the reverse of compression. If reliability conditioning is a general
mechanism rather than a compression detector, the *temporal* reliability should fall here while the
others hold. That is a second, independent test of the thesis, and it is nearly free.

**Brightness/contrast is the control.** It changes appearance without destroying forensic evidence,
so reliability should stay roughly flat. If it drops, the estimator is tracking image statistics
generally rather than evidence quality, and `C-20`'s interpretation weakens. Report it either way.

**Outputs go under `C-24`** — this is the same robustness claim as the compression sweep, so it
extends that capture rather than adding a thirtieth. Files: `C-24_robustness_sweep.png` + `.pdf` +
`_data.csv` (model, seed, perturbation, level, auc, ci_low, ci_high, n_videos), and
`C-24_robustness_reliability.csv` (perturbation, level, mean of each of the 4 descriptor components, mean per-stream trust as defined for `C-21`) for the
frame-dropping argument above. Add the files to `captures.yaml` under `C-24`; do not create a new ID.

Per-video scores at every perturbation level, as always. Re-encoding the eval sets is the expensive
part; the scores make any later question a CSV read rather than a re-run.

---

## Acceptance tests

Run at every phase boundary:

```bash
python -m tools.verify_splits --splits data/splits --config configs/preprocess.yaml   # zero leakage
grep -ci celeb data/splits/train.txt                        # 0
python -m tools.capture_check --phase N                     # all captures present
python -m pytest tests/ -v                                  # unit tests
```

Minimum unit tests — write these, they catch the errors that silently corrupt results:

| Test | Guards against |
|---|---|
| `test_attention_pool_fp16_no_nan` | the `-1e9` regression returning |
| `test_identity_parser` | split leakage via mis-parsed identities |
| `test_split_disjointness` | the original 905-video bug |
| `test_crop_geometry` | non-square crops distorting geometry |
| `test_bootstrap_resamples_videos` | over-narrow CIs from frame-level bootstrap |
| `test_zero_init_conditioning` | conditioned model ≠ unconditioned at init (breaks ablations) |
| `test_reliability_monotonic` | estimator not tracking degradation |
| `test_save_figure_writes_four_files` | a figure landing without its PDF, `_data.csv`, or note |
| `test_capture_check_rejects_empty_png` | a 0-byte `savefig` counting as a completed capture |
| `test_capture_check_rejects_pending_stub` | a `STATUS: PENDING` note passing as done |
