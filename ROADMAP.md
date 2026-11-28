# Research Roadmap

**Reliability-Conditioned Selective Fusion for Robust and Calibrated Deepfake Detection**

Companion documents:

| File | Audience | Contains |
|---|---|---|
| `RESEARCH_ANALYSIS.md` | you + supervisor | Why this project, why this mechanism, literature position |
| **`ROADMAP.md`** (this) | you | **What** to do, in **what order**, with gates and capture points |
| `IMPLEMENTATION.md` | you or a coding agent | **How** to do it — files, signatures, algorithms, commands |
| `CLAUDE.md` | AI agents | Repo conventions and invariants an agent must not violate |

Every task below carries an ID like `P3-T2`. `IMPLEMENTATION.md` has a section with the same ID
holding the technical spec. **Never implement from this file alone — always open the matching
`IMPLEMENTATION.md` section.**

---

## How to use the capture protocol

Your teammate writes the paper. She was not present for the experiments. Anything not captured at
the moment it existed is lost, because later phases overwrite splits, checkpoints, and metrics.

Capture markers appear inline as:

> **CAPTURE C-nn** `[KIND]` — what to save, and why the paper needs it.
> **Files:** the exact filenames to produce.

### Who produces it — the three kinds

| Kind | Meaning | Can a coding agent finish it? |
|---|---|---|
| `[AUTO]` | A script writes every file. Figures are plotted and `savefig`'d, tables are written as CSV, logs are redirected stdout. | **Yes, end to end.** |
| `[HUMAN-IN-LOOP]` | A script produces the artifact, but **a person must look at it** and write the counts, judgement, or interpretation into the note. | It makes the files; you write the note. |
| `[MANUAL]` | Only you can produce it — screenshots of email, licence portals, or dataset pages. An agent has no eyes on your inbox or your browser. | **No.** It writes the stub note and stops. |

Every `[MANUAL]` and `[HUMAN-IN-LOOP]` item also appears in the
[Manual capture checklist](#manual-capture-checklist) with the moment it must be taken. Several stop
being obtainable later: a licence approval email is trivial to screenshot the week it arrives and
awkward to reconstruct six months on.

### MANDATORY AGENT BEHAVIOR for `[MANUAL]` and `[HUMAN-IN-LOOP]` captures

**ANY agent (Claude Code, Antigravity, or any future agent) MUST follow this protocol:**

**When you encounter a `[MANUAL]` capture:**
1. Write the `STATUS: PENDING` stub note to `paper_assets/C-nn_description.md`
2. **STOP ALL WORK on that task**
3. Tell the user:
   - **What** to capture (exact artifact description)
   - **How** to capture it (full-window screenshot, URL bar visible, etc.)
   - **Where** to save it (exact file path in `paper_assets/`)
   - **Naming** convention (e.g., `C-09_licences/ffpp_eula_email.png`)
4. **WAIT** for the user to confirm: "I have done this, proceed ahead"
5. Only then continue with subsequent tasks

**When you encounter a `[HUMAN-IN-LOOP]` capture:**
1. Produce ALL auto-generated files (figures, CSVs, data)
2. **STOP ALL WORK on that task**
3. Tell the user:
   - **What judgement** they need to make (counts, interpretation, case selection)
   - **Which file** to write it in (the `.md` note)
   - **What format** the judgement should take (see the specific task’s instructions)
   - Show them the generated artifacts so they can make informed judgements
4. **WAIT** for the user to confirm: "I have written my judgement, proceed ahead"
5. Only then continue with subsequent tasks

**NEVER:**
- Invent a judgement for a `[HUMAN-IN-LOOP]` capture
- Generate a placeholder image for a `[MANUAL]` capture
- Skip a capture and continue to the next phase
- Mark a `PENDING` capture as complete without user confirmation

### File formats — fixed, so your teammate never has to ask

| Capture kind | Produce |
|---|---|
| **Figure** | `C-nn_name.png` at **300 dpi** *and* `C-nn_name.pdf` (vector — what Elsevier actually wants) *and* `C-nn_name_data.csv` (the numbers behind every plotted point) |
| **Table** | `C-nn_name.csv` (raw, machine-readable). A rendered `.md` copy is optional and never the only copy. |
| **Terminal output / log** | `C-nn_name.txt` — real redirected stdout, never a photo of a terminal |
| **Archive** | `C-nn_name/` directory, contents untouched |
| **Manual** | `C-nn_name/` directory of `.png` / `.pdf` screenshots, one per file, each named for what it shows |

Always with `C-nn_name.md` beside it.

**The `_data.csv` beside every figure is not optional.** Journals ask for source data, reviewers ask
for exact values, and your teammate will need to re-plot at least one figure to fit a column width.
A PNG she cannot re-plot is a PNG she has to ask you to regenerate months later — by which time the
checkpoint may be gone.

**Never screenshot what a script can save.** A screenshot of a plot is lower resolution than the
plot, a screenshot of a table is not machine-readable, and a screenshot of a terminal cannot be
grepped or diffed. Screenshots exist for one purpose here: things that live only inside someone
else's UI — email, licence portals, dataset download pages. If an agent ever asks you to screenshot
a figure or a console, it is doing it wrong.

**Rules:**

1. Everything goes in `paper_assets/` (git-ignored — it will be large).
2. Filename **must** start with the capture ID: `C-04_baseline_leaked_vs_clean.png`.
3. Every capture needs a sibling `.md` note with the same stem, containing: date, the exact command
   run, the git commit hash, and two or three sentences of plain-English meaning. **The note is
   the part your teammate actually needs.** A PNG with no explanation is unusable to her.
4. Numbers go in `paper_assets/results_log.csv` (schema in `IMPLEMENTATION.md` §P0-T5) as well as
   in any figure — she needs the raw values to typeset tables.
5. **Never delete a capture, even when a later run supersedes it.** Superseded results are how you
   show progress, and how you defend against "did you tune on the test set?"
6. A `[MANUAL]` capture you have not taken yet still gets its note written **immediately**, with
   `STATUS: PENDING` as the first line. `capture_check` fails while any stub is pending, so a
   missing screenshot cannot quietly become "I'll do that later."

At the end of every phase, run `python -m tools.capture_check --phase N`. It reads the capture
manifest (`paper_assets/captures.yaml`, §P0-T4), checks that every declared file for that phase
exists and that no note is still `PENDING`, and prints the outstanding manual items as a to-do list.
Do not start the next phase until it passes.

---

## Phase 0 — Gates and environment

**Objective:** resolve the three unknowns that change the plan, before spending time on anything
that depends on them. Nothing here takes more than a day.

### Gate 0-A — Does `mamba-ssm` build on sm_120?

`P0-T1`. Your RTX 5070 Ti Laptop is Blackwell (compute capability 12.0). `mamba-ssm` and
`causal-conv1d` ship CUDA extensions that must compile for that architecture, and Windows plus a
new arch is the hardest case for those packages.

Try in order, stopping at the first success: (1) native Windows pip install; (2) WSL2 with CUDA
passthrough; (3) pure-PyTorch associative-scan SSM (always works, slower, no compilation).

**Outcome decides the fusion operator only — never the thesis.**

| Result | Operator | Section |
|---|---|---|
| 1 or 2 succeeds | Reliability-conditioned selective SSM | `RESEARCH_ANALYSIS.md` §6.2 |
| Only 3 works | Same, on the slower scan — acceptable, note the cost | §6.2 |
| All fail | Reliability-conditioned cross-attention (FiLM) | §6.4 |

> **CAPTURE C-01** `[AUTO]` — terminal output of the successful install **and of every failed
> attempt**, plus `nvidia-smi`, `nvcc --version`, and `python -c "import torch;
> print(torch.__version__, torch.cuda.get_device_capability())"`. The paper's reproducibility
> section needs exact versions, and if you end up on the fallback operator you must be able to state
> why.
> **Files:** `C-01_env_gate.txt` (all of it, redirected — `... 2>&1 | tee`, not a screenshot),
> `C-01_env_gate.md`.

### Gate 0-B — Citation verification

`P0-T2`. **PARTIALLY VERIFIED (September 2026).** All five arXiv IDs confirmed as real papers.
Two author attributions corrected. Aletheia overlap question **still open** pending full PDF.

**C-02 has NOT been produced yet.** `paper_assets/` does not exist. Gate 0-B remains open
until C-02 is properly generated via P0-T4 infrastructure and the user confirms each verdict.

Verified IDs (arXiv pages fetched, titles match):

| arXiv ID | Title Match | Authors (ACTUAL) | Notes |
|---|---|---|---|
| `2604.16486` | ✅ PhyLAA-X / Aletheia | Devendra Ghori, 13 Apr 2026 | ✅ Correct |
| `2605.04405` | ✅ Detecting Deepfakes via Hamiltonian Dynamics | Harry Cheng, Ming-Hui Liu, Tianyi Wang, Weili Guan, Liqiang Nie, Mohan Kankanhalli, 6 May 2026 | ⚠️ **NOT Miyachi** — Miyachi is `2607.13234` |
| `2605.00630` | ✅ CMTA | Hang Wang, Chao Shen, Chenhao Lin, Minghui Yang, Lei Zhang, Cong Wang, 1 May 2026 | ⚠️ **NOT Hwang** — first author is Hang Wang |
| `2607.13234` | ✅ BMF / BitMind Forensics | Ken Jon Miyachi, Dylan Uys, 14 Jul 2026 | ✅ Correct |
| `2607.17441` | ✅ MSCA-FFT | Pamela Kirui, Cho Hyuk, Qingzhong Liu, Haodi Jiang, 19 Jul 2026 | ✅ Correct. ⚠️ Image-level, not video |

**Aletheia fusion mechanism — OPEN, needs full PDF:**
The abstract describes TWO mechanisms: (1) three physics feature volumes (optical-flow curl,
specular-reflectance skewness, rPPG power spectra) fed "directly into LAA-X attention via
cross-attention gating" — this IS input-derived conditioning of an attention mechanism, and
is closer to our approach than initially assessed; (2) "uncertainty-aware adaptive weighting"
for a three-branch ensemble — this is prediction-confidence-based.

Settling this requires the full PDF or the code, not just the abstract. Until then:
- **Do not claim "No conflict" in any document**
- **Do not adopt 0.966 DFDC AUC as the bar to beat** (see `RESEARCH_ANALYSIS.md` §3.1)
- The thesis remains distinct (SSM Δ-conditioning on self-supervised degradation descriptors
  is not cross-attention gating on physics feature volumes), but the differentiation table
  cannot be finalized until the full mechanism is understood

Aletheia repo note: `github.com/devghori1264/Aletheia` (3 stars, 4 commits, 29 open PRs)
describes a Django/NLP system for "audio-linguistic misalignments" — not the optical-flow-curl
/ specular / rPPG architecture in the paper. Cite cautiously.

> **CAPTURE C-02** `[HUMAN-IN-LOOP]` — a table of every cited work with arXiv ID / DOI, verified
> yes/no, date checked, and the venue if published. **This file is the bibliography's source of
> truth.** Nothing enters the manuscript that is not in it. An agent can fetch and fill the table;
> **you** confirm each verdict, because "the fetch 404'd" and "this paper does not exist" are not
> the same claim and only one of them is safe to write down.
> **Files:** `C-02_citation_audit.md` (the table itself — this one is a document, not a note),
> `C-02_citation_audit.csv` (same rows, machine-readable, for the bibliography pass in `P7-T3`),
> plus `C-02_citation_audit/` holding one screenshot per **suspect** ID — the abstract page as it
> renders today, or the 404. `[MANUAL]` for that folder: a fabricated citation challenged in review
> is answered by a dated screenshot, and arXiv pages change.

### Gate 0-C — Environment freeze

`P0-T3`. Create the conda environment, pin every version, export a lockfile.
`P0-T4`. Set up `paper_assets/`, `.gitignore` entries, the capture manifest (`captures.yaml`), and
the capture checker.
`P0-T5`. Create `paper_assets/results_log.csv` with the fixed schema.
`P0-T6`. Write `src/eval/figures.py` — `save_figure` / `save_table` / `save_console` /
`write_pending_stub`. **Do this before the first figure exists, not after.** It is what makes
"PNG + PDF + `_data.csv` + note" automatic rather than something to remember 29 times.

> **CAPTURE C-03** `[AUTO]` — `environment.yml` plus `pip freeze` output, and a one-paragraph
> hardware description (CPU, RAM, GPU, VRAM, OS, CUDA). Reviewers ask for this; write it once,
> reuse it.
> **Files:** `C-03_environment.yml`, `C-03_pip_freeze.txt`, `C-03_hardware.md`.

---

## Phase 1 — Freeze the old project and audit its provenance

**Do this before touching anything else. It is not optional and it is not cleanup.**

Two reasons. First, once Phase 3 regenerates the splits, the current results become permanently
irreproducible — and those results are what your capstone was graded on, so they must be archived
intact. Second, and more valuable:

> **You can measure how much the Celeb-DF leak inflated your reported number using the model you
> already have, without retraining anything.** That measurement is a paper figure. It stops being
> obtainable the moment the splits change.

> **Correction (2026-11-28, done and verified):** the split lists `data/*.txt` do contain 905 / 159 / 69 Celeb-DF entries, but **no trained model consumed them**. The lists were regenerated *after* frame extraction (`extract_frames.ipynb`'s saved output shows it read 4066 / 762 / 255 entries); every legacy trainer reads the directories `preprocessed/frames/<split>/` or `embeddings/<split>/`, never the lists; and those directories contain 0 Celeb-DF videos (`train_temporal_clean.ipynb` output: "Train videos: 4066"; `ensemble_features_final/train.npz` has 4066 rows). The capstone trained on FF++ + DFDC, so its 0.65 Celeb-DF AUC is a genuine cross-dataset number. `P1-T2` therefore became a
> provenance audit (SPEC-002), not a leak-inflation measurement, and C-05/C-06 are provenance
> evidence rather than Figure 1. The paper's motivating figure is C-15 (Phase 4).

### `P1-T1` — Archive the current state

Tag the commit, copy `checkpoints/`, `data/*.txt`, `ensemble_features_final/`, and
`checkpoints/ensemble_final/results.txt` into `paper_assets/C-04_legacy_baseline/` untouched.

> **CAPTURE C-04** `[AUTO]` — the whole archive, plus the git tag name and commit hash. Note in the
> sibling `.md` that `results.txt` line "Test (External) AUC: 0.9695" is **mislabelled** and is
> actually the *internal* test set (cause: `build_features.py:218` lists a `test_internal` split
> that `ensemble_features_final/` never received, so `train_ensemble_final.py:50-54` silently falls
> back to `test.npz`). Your teammate must not quote 0.9695 as an external result.
> **Files:** `C-04_legacy_baseline/` (the archive, contents untouched),
> `C-04_legacy_baseline.md` (tag, commit, and the mislabel warning).

### `P1-T2` — Quantify the leak

Partition the 1168 Celeb-DF videos into **seen** (appearing in `train.txt` or `val.txt`) and
**unseen**. Evaluate the existing frozen ensemble separately on each.

Expected shape of the result: seen-subset AUC well above unseen-subset AUC, with the published 0.65
sitting between them. **The unseen number is your project's true starting point**, and it will be
lower than 0.65.

> **CAPTURE C-05** `[AUTO]` — a three-bar chart (all / seen / unseen) with the exact AUCs, sample
> counts printed on each bar, and the 905 / 159 / 69 split-contamination counts. **This is Figure 1
> of the paper's motivation section.** The sibling note should state the mechanism in one sentence:
> `split_dataset.ipynb:180` comments that Celeb-DF is excluded from training, then appends it to
> `entries`, which is shuffled into every split.
> **Files:** `C-05_leak_quantification.png` + `.pdf` + `_data.csv` (one row per group: group, n_videos,
> n_real, n_fake, auc, ci_low, ci_high), `C-05_leak_quantification.md`.

> **CAPTURE C-06** `[AUTO]` — ROC curves for seen vs unseen on the same axes. Visually striking, and
> it makes the point faster than a table.
> **Files:** `C-06_roc_seen_vs_unseen.png` + `.pdf` + `_data.csv` (fpr, tpr, thresholds per group —
> she may want to redraw this at a different aspect ratio), `C-06_roc_seen_vs_unseen.md`.

### `P1-T3` — Fix the four cheap defects

Two affect results, two are hygiene. Do them now so no later measurement inherits them.

| Fix | Location | Effect |
|---|---|---|
| `-1e9` → `torch.finfo(scores.dtype).min` | `models.py:85` | fp16 overflow → NaN; **re-enables AMP**, roughly halves temporal training time |
| Build `test_internal.npz` or remove the split from the list | `build_features.py:218` | ends the "Test (External)" mislabel |
| Remove or correct the hardcoded `"99.2%"` | `app.py:866` | unsupported by any measurement in the repo |
| Regenerate splits with **relative** paths | `data/*.txt` | current files hold dead `c:\Users\...\OneDrive\...` paths |

> **CAPTURE C-07** `[AUTO]` — **moved to Phase 4 (`P4-T1`), owner decision 2026-11-28; the diff is
> already captured.** A before/after `git diff` of the `-1e9` fix, plus a short loss-curve
> comparison showing FP32 vs AMP training now agree. If you ever claim a speed improvement in the
> paper, this is the evidence.
> **Files:** `C-07_amp_fix.diff` (`git diff > ...`, not a screenshot of the editor),
> `C-07_amp_loss_curves.png` + `.pdf` + `_data.csv` (epoch, fp32_loss, amp_loss, fp32_val_auc,
> amp_val_auc, and the wall-clock seconds per epoch for each — the speed claim needs the timing
> column, not just the agreement), `C-07_amp_fix.md`.

### Phase 1 exit gate

`python -m tools.capture_check --phase 1` passes (C-04–C-06), and you can state your
pre-modification cross-dataset AUC in one sentence: *trained on FF++ + DFDC, 0.65 [0.62, 0.68] on
Celeb-DF*.

---

## Phase 2 — Dataset acquisition

**Objective:** original sources on disk, with provenance documented. No preprocessing yet.

`P2-T1`. FF++ base, **c23 and c40** (12 GB) — the training set.
`P2-T2`. FF++ DeepFakeDetection extension, c23 + c40 (28.4 GB) — eval only.
`P2-T3`. Celeb-DF v2 (10 GB) — eval only. Needs the EULA via your institutional mail.
`P2-T4`. DFDCP preview (~5 GB) — **fetch this first.** It unblocks all method development so you
are never waiting on the 470 GB pull.
`P2-T5`. DFDC full, 50 parts (~470 GB) via the download → process → delete → next loop.

> **Screenshot each approval and licence page as it arrives — this is the `[MANUAL]` capture C-09,
> and it is the one thing in this phase you cannot automate or redo later.** Thirty seconds per
> dataset, at the moment the email lands.

> **Validate the loop end-to-end on part 00 before committing to all 50.** One bad assumption
> repeated 50 times costs a week.

> **CAPTURE C-08** `[AUTO]` — per dataset, the exact source URL, version, download date, compression
> levels obtained, real/fake video counts **as counted on your disk** (not as quoted from the
> paper), and total bytes. This table goes in the paper almost verbatim.
> **Files:** `C-08_datasets.csv` (one row per dataset × compression level, counted by script),
> `C-08_datasets.md` (the same table rendered, plus any discrepancy against the official figures
> written out in words).

> **CAPTURE C-09** `[MANUAL]` — **you must take these yourself; no agent can reach your inbox.** The
> EULA approval emails for FF++ and Celeb-DF v2, plus the licence/terms page of every dataset as it
> reads on the day you download. Some journals require evidence of licensed access at submission.
> **Files:** `C-09_licences/` containing, one file each:
> `ffpp_eula_email.png`, `celebdf_eula_email.png`, `dfdc_terms_page.png`, `dfdcp_terms_page.png`,
> `dfd_terms_page.png`, plus `C-09_licences.md` listing for each: who granted it, the date on the
> email, and the address it was sent to.
>
> **Take these the week each approval lands.** Mailbox retention policies, expiring institutional
> accounts, and reorganised dataset websites all destroy this evidence, and none of it is
> reconstructable at submission time. Full-window screenshots including the sender, date, and URL
> bar — a cropped body with no header proves nothing. Redact nothing except your own signature
> block; a reviewer needs to see the sender is genuine.

### Phase 2 exit gate

Counted video totals match the official figures, and `C-08` is complete. Any discrepancy gets
written down rather than silently accepted — a missing 3 % of DFDC changes your denominators.

---

## Phase 3 — Preprocessing, with a detector bake-off

### `P3-T1` — How to choose a face detector when you have no ground truth

You asked how to benchmark detectors and said you don't know how to judge which is better. Here is
the protocol. **The intuitive approach — "which detector finds the most faces" — is actively
misleading**, because a detector that fires on shadows and hands scores highest on that metric while
poisoning your training set. Detection *rate* is not detection *quality*.

There is no bounding-box ground truth in these datasets. So instead of measuring accuracy directly,
measure four proxies that together are hard to fake, then let the **downstream task** break ties.

Candidates: **RetinaFace**, **MTCNN**, **S3FD**, **YOLOv8-face**, **DSFD** (drop any that won't
install in under an hour — this is a means, not the research).

**Benchmark set: 200 videos**, stratified so it stresses the failure modes that matter —
50 FF++ c23, 50 FF++ **c40** (heavy compression is where cheap detectors collapse),
50 Celeb-DF v2, 50 DFDC (the hardest: in-the-wild pose, lighting, occlusion, motion blur).
Sample 10 frames per video → 2000 frames. Runs in minutes per detector.

**Metric 1 — Detection rate.** Fraction of frames yielding ≥1 face. Necessary, not sufficient.
Read it *with* Metric 4; high rate plus poor stability means false positives.

**Metric 2 — Temporal stability (the discriminating metric).** Faces move smoothly through real
video, so within one video, consecutive boxes should move smoothly. Compute the frame-to-frame
IoU of the primary box and report the mean, plus the count of "jumps" (IoU < 0.5 between adjacent
frames). Low stability means the detector is switching targets or hallucinating — **and because
your temporal stream reads a sequence of crops, an unstable detector injects fake motion that the
model may learn as a deepfake cue.** This is the metric most likely to change your choice.

**Metric 3 — Box consistency.** Standard deviation of box area within a video, normalised by mean
area. A face doesn't triple in size between adjacent frames; if the box does, the detector is
unreliable.

**Metric 4 — Manual spot check.** Sample 50 crops per detector into a contact sheet and look at
them. Count obvious failures: non-faces, half-faces, wrong person in multi-face frames. **Do not
skip this.** Fifteen minutes of looking catches failure modes no aggregate reveals — and you will
be asked in review whether you inspected your data.

**Metric 5 — Throughput.** Seconds per frame on your GPU. At 5,000 videos × 32 frames, a 3×
difference is days. Use it only to break a tie between detectors that pass 1–4.

**Decision rule, in order:** eliminate anything with detection rate < 95 % on FF++ c23 (it fails the
easy case); among survivors take the best temporal stability; break ties on manual inspection, then
throughput. **If two remain genuinely close, run `P3-T2`.**

### `P3-T2` — Downstream tie-break (only if needed)

Preprocess 500 videos with each finalist, train the *same* small model on each, compare validation
AUC. This is the only measurement that answers the question you actually care about — which
detector produces data that trains a better detector. It costs a few hours, so use it only as a
tie-break, not as the primary method.

> **CAPTURE C-10** `[HUMAN-IN-LOOP]` — the full detector comparison table (all five metrics, all
> candidates), the contact sheets, and **a written justification of the final choice in two or three
> sentences**. Reviewers ask "why this detector?" and "the tutorial used it" is not an answer. This
> is a methods-section paragraph and a supplementary table.
> **Files:** `C-10_detector_bakeoff.csv` (detector × dataset × the five metrics),
> `C-10_detector_bakeoff/contact_sheet_<detector>.png` (a **script-composed** grid of 50 crops with
> the source video ID printed under each — assembled with PIL, not screenshotted from a viewer),
> `C-10_detector_bakeoff.md` (the table, your Metric-4 counts, and the justification).
>
> **Metric 4 is yours and cannot be delegated.** A script builds the contact sheets; you open them
> and count non-faces, half-faces, and wrong-person-in-multi-face crops per detector, then type
> those counts into the note. Fifteen minutes of looking, and it is the one number in this table an
> agent cannot produce.

> **CAPTURE C-11** `[AUTO]` — a side-by-side figure: the same frame cropped by each detector. One
> image that shows margin and centering differences better than any table.
> **Files:** `C-11_detector_crops.png` + `.pdf` (script-composed panel, detector name and box
> coordinates labelled on each tile), `C-11_detector_crops_data.csv` (video ID, frame index, and the
> box each detector returned, so the panel can be rebuilt), `C-11_detector_crops.md`.

### `P3-T3` — Build the preprocessing pipeline

Fixed parameters, and why each is what it is:

| Parameter | Value | Reason |
|---|---|---|
| Frames per video | **32** (was 8) | 8 samples span hundreds of ms — micro-expression and flicker dynamics are aliased away. 8 cannot support a temporal claim. |
| Stored crop | **320×320** | Larger than any training resolution (224 ViT / 300 B3 / 299 Xception), so input size can change without re-extraction. Re-extraction is the expensive step. |
| Crop margin | **1.3×** detected box | Blending boundaries live at the face *edge*. Tight crops delete the artifact being detected, and SBI needs the hull plus surrounding context to blend. |
| Format | **PNG** | JPEG adds its own 8×8 DCT quantization on top of H.264's — a second uncontrolled artifact layered over the one you measure. Fatal for frequency claims. |
| Detector | one, from `P3-T1` | Mismatched detectors between train and test create a domain gap that *looks* like a generalization result. |
| Compression | train **c23**, eval **c40** | Community standard, and the axis reviewers check. |

**Note on your "which resolution" question:** it resolves differently than expected — FF++'s
variants (raw/c23/c40) are *compression* levels, not resolutions. The resolution decision is the
crop size, which you control, and the answer is "store larger than you train."

> **CAPTURE C-12** `[AUTO]` — the crop-margin diagnostic. Crop the same 200 videos at 1.0× and 1.3×,
> train the same model on each, compare cross-dataset AUC. I suspect your third-party crops are
> tight, and that this alone explains part of the weak generalization. **If margin alone moves the
> number meaningfully, that is a finding worth a paragraph** — and it costs one extra run.
> **Files:** `C-12_crop_margin.csv` (margin × test set × AUC with CIs),
> `C-12_crop_margin.png` + `.pdf` + `_data.csv` (grouped bars, one group per test set), and
> `C-12_crop_margin_examples.png` — the same face at 1.0× and 1.3× side by side, so the reader sees
> that the blending boundary is inside one crop and outside the other. That example panel is what
> makes the paragraph land. `C-12_crop_margin.md`.

### `P3-T4` — Splits, rebuilt correctly

This is the fix for the leak, and the rule is absolute:

- **Train: FF++ only.** Celeb-DF, DFDC, DFD are evaluation-only and are never seen in training or
  model selection.
- **Identity-disjoint** within FF++: the same source identity must never appear in two splits.
  Identity lives in the original filename (`id0_id1_0001.mp4`) and directory structure — which is
  the fourth reason the third-party crops had to go.
- **Manipulation-aware**: all four FF++ methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
  represented in each split.

**Automated check, must pass before any training:** `python -m tools.verify_splits`, asserting zero
identity overlap, zero non-FF++ videos in train, and no path in two splits.

> **CAPTURE C-13** `[AUTO]` — `verify_splits` output showing all assertions pass, alongside the *old*
> contamination counts (905 / 159 / 69) for contrast. **This is the evidence that the new protocol
> is clean** — the single most important credibility artifact in the project.
> **Files:** `C-13_verify_splits.txt` (redirected stdout of the passing run, every assertion line
> visible), `C-13_split_composition.csv` (split × dataset × manipulation method × n_videos ×
> n_identities), `C-13_contamination_before_after.png` + `.pdf` + `_data.csv` (old vs new non-FF++
> counts per split — the 905/159/69 bars going to zero), `C-13_verify_splits.md`.

### Phase 3 exit gate

`grep -ci celeb data/splits/train.txt` returns **0**. `verify_splits` passes. Captures C-10 through
C-13 complete.

> Check `data/splits/train.txt`, **not** `data/train.txt`. The legacy file is the archived
> pre-rebuild split and still contains its 905 Celeb-DF lines by design — `P1-T1` preserves it as
> evidence. Grepping the old path makes this gate unpassable on a correctly rebuilt project.

---

## Phase 4 — The honest baseline

**Objective:** the number the paper's motivation rests on: the legacy architecture under the
standard FF++-only protocol, comparable with published work.

`P4-T1`. Retrain the **legacy architecture** (EfficientNet-B3 + Bi-LSTM + LR stacking) on the clean
FF++-only splits, unchanged except for the fixes from `P1-T3`. Before training, also fix the spatial-head loader
(`build_features.py`, weights never applied in the capstone) and the duplicate `-1e9` mask in
`src/process_data/train_temporal.py`, then re-enable AMP and capture **C-07** (moved here from Phase 1). This isolates the protocol change
from the architecture change — a reviewer will insist on that separation.

`P4-T2`. Run B3 at its **native 300×300** as well as 224. Your current pipeline runs a B3 at
roughly B0's input scale, paying B3's parameter cost without its resolution benefit. Report both.

`P4-T3`. Reproduce **two published baselines through DeepfakeBench** (recommend Xception and
EfficientNet-B4, or SBI if it installs cleanly) on your splits. This is what makes your numbers
comparable rather than merely internal, and it is a common reason for rejection when missing.

`P4-T4`. Evaluate everything on the full matrix: FF++ c23 (in-domain), FF++ c40, CDF-v2, DFDC,
DFDCP, DFD.

> **CAPTURE C-14** `[AUTO]` — the full baseline results matrix: rows = models, columns = test sets,
> cells = AUC / AP / EER, plus 95 % confidence intervals via bootstrap over videos. **This is Table 1
> of the paper.** Give your teammate the raw CSV, not only a rendered table.
> **Files:** `C-14_baseline_matrix.csv` (long format — one row per model × test set × seed × metric,
> with `ci_low`/`ci_high`; long format survives adding a model, a wide table does not),
> `C-14_baseline_matrix.md` (rendered for reading), `C-14_baseline_matrix.tex` (booktabs, ready to
> paste). No figure — this one is a table everywhere it appears.

> **CAPTURE C-15** `[AUTO]` — the capstone-vs-FF++-only comparison (renamed 2026-11-28; the old
> "honest vs inflated" framing assumed a leak that never reached training): the capstone's Celeb-DF
> AUC (0.65, FF++ + DFDC training, from C-05), this phase's FF++-only legacy retrain, and the
> DeepfakeBench baselines, each with in-domain vs Celeb-DF bars, on one axis with a one-paragraph
> explanation. **This is the paper's motivating figure: high in-domain AUC collapses on unseen data.**
> **Files:** `C-15_capstone_vs_ffpp_only.png` + `.pdf` + `_data.csv` (label, auc, ci_low, ci_high,
> n_videos, and a `provenance` column naming the run each bar came from — three numbers from three
> different protocols on one axis is exactly where a reviewer asks "measured how?"),
> `C-15_capstone_vs_ffpp_only.md`.

> **CAPTURE C-16** `[AUTO]` — training curves (loss and val AUC vs epoch) for every model, plus
> wall-clock time, epochs to best, and peak VRAM. Needed for the reproducibility section, and it
> lets you state hardware cost honestly.
> **Files:** `C-16_training_curves.png` + `.pdf` (a panel per model), `C-16_training_curves.csv`
> (model, seed, epoch, train_loss, val_loss, val_auc, lr, epoch_seconds, peak_vram_gb — logged **as
> training runs**, not scraped from console scrollback afterwards), `C-16_training_cost.csv` (one
> row per model: total wall-clock, epochs to best, peak VRAM), `C-16_training_curves.md`.
>
> Have the trainer write its CSV row every epoch. A crashed run then still leaves its curve behind,
> and you never find yourself re-running thirty epochs to recover a figure.

> **CAPTURE C-17** `[HUMAN-IN-LOOP]` — a failure-case gallery: 20 videos the baseline gets
> confidently wrong, with predicted probability and true label. Qualitative figures like this are
> consistently what reviewers remember, and they motivate the mechanism.
> **Files:** `C-17_failure_gallery.png` + `.pdf` (script-composed grid: one representative crop per
> video, captioned with true label, predicted probability, dataset, and video ID),
> `C-17_failure_gallery/` (the full 32-frame crop sequence per selected video, so a specific case
> can be enlarged for the paper without re-running anything),
> `C-17_failure_gallery.csv` (video ID, dataset, true label, predicted prob, manipulation method),
> `C-17_failure_gallery.md`.
>
> **The selection is automatic; the reading is not.** A script picks the 20 highest-confidence
> errors and builds the sheet. You then look at them and write one or two sentences in the note on
> what they have in common — heavy compression, extreme pose, small faces, one manipulation method
> dominating. That observation is what connects this figure to the mechanism, and it is the reason
> the figure is in the paper rather than merely in the appendix.

### Phase 4 exit gate

You can state, with a confidence interval: *"Trained on FF++ c23 only, the legacy architecture
reaches X AUC in-domain and Y AUC on Celeb-DF v2."* Y may be below 0.65, since FF++ alone is less
diverse than the capstone's FF++ + DFDC. **That gap is
the problem the paper solves — do not be discouraged by it, it is the deliverable of this phase.**

---

## Phase 5 — The mechanism

Build in **four independently testable increments**. Each has a gate. If an increment fails its
gate, you learn it in isolation rather than debugging a fully assembled system.

### `P5-T1` — Backbone swap (component, not contribution)

Frozen CLIP ViT-B/16 (or DINOv2) + lightweight trainable adapters. Extract embeddings **once** and
cache them, exactly as your current `embeddings/*.npy` pattern does. This turns each later fusion
experiment from hours into minutes and is what makes the Phase 6 ablation grid affordable.

**Gate:** frozen-backbone-plus-adapters beats the Phase 4 legacy baseline cross-dataset. If not,
stop and diagnose before adding anything.

> **CAPTURE C-18** `[AUTO]` — backbone comparison (B3 / CLIP ViT-B/16 / DINOv2) on the full eval
> matrix, with parameter counts, trainable-parameter counts, and extraction wall-clock.
> **Files:** `C-18_backbone_comparison.csv` (long format, as C-14), `C-18_backbone_comparison.md`.

### `P5-T2` — Reliability estimator

A small CNN (< 1 M params) predicting an interpretable degradation descriptor: 8×8 blockiness
energy, high-frequency spectral ratio, local entropy, blur. **Trained self-supervised on your own
CRF sweep** — you apply the degradation, so ground truth is free. No annotation, no extra data.

**Gate — this is the falsifiable one.** Predicted reliability must correlate monotonically with
held-out CRF (Spearman |ρ| > 0.8). If it does not, the estimator is not measuring degradation and
the mechanism has no signal to condition on. **Fix this before proceeding — everything downstream
depends on it.**

> **CAPTURE C-19** `[AUTO]` — scatter plot: predicted reliability vs true CRF, with Spearman ρ. This
> is the estimator's validation figure.
> **Files:** `C-19_reliability_vs_crf.png` + `.pdf` + `_data.csv` (video ID, frame index, true CRF,
> all four predicted descriptor components — save all four even though the gate is on `r[0]`; the
> other three cost nothing now and are unrecoverable once the checkpoint is overwritten),
> `C-19_reliability_vs_crf.md` stating ρ per component and whether the gate passed.

### `P5-T3` — Reliability-conditioned fusion

The hinge. Condition the SSM's selection parameter **Δ** (and the B/C projections) on the estimated
frame-level reliability descriptor — with a learned, per-stream response to it — over an interleaved multi-stream token sequence. Δ governs how
strongly the current token writes into state, so it is the natural home for "trust this stream less,
right now" — which is why gating and sequence modelling are one mechanism here, not two stacked
contributions.

If Gate 0-A failed, this is FiLM-conditioned cross-attention instead. **Identical thesis, identical
experiments, different operator.**

**Gate:** beats the unconditioned equivalent on the c40 compression sweep. If conditioning doesn't
help where evidence is actually degraded, the thesis is wrong — and you want to know that here, at
week ~14, not at review.

> **CAPTURE C-20** `[AUTO]` — the **mechanism verification figure**, and the most important figure in
> the paper: learned frequency-stream reliability plotted against measured CRF. If it declines
> monotonically as compression rises, the mechanism visibly does what the thesis claims. Save the
> raw per-video values too — your teammate may want to re-plot it.
> **Files:** `C-20_mechanism_verification.png` + `.pdf` + `_data.csv` (video ID, dataset, CRF, and
> mean reliability **per stream** — spatial, frequency, temporal — plus the mean Δ per stream, so the
> figure can be redrawn as either reliability-vs-CRF or Δ-vs-CRF), `C-20_mechanism_verification.md`.
>
> **This figure will be re-plotted.** It is the one a reviewer engages with hardest, and it is the
> one most likely to need a different framing (per-stream panels, log axis, CRF as a categorical).
> Save the per-video values with more columns than the current plot uses, and it stays re-plottable
> after the checkpoint is gone.

> **CAPTURE C-21** `[AUTO]` — a per-stream trust heatmap across datasets, showing which
> evidence the model learns to distrust where. Per-stream **trust** = mean over that stream's tokens
> of `Δ_conditioned / Δ_unconditioned` (the same Δ with the `W_rel[k]` term removed); below 1 means
> the stream is being written into state less than it would be without conditioning. The 4-D
> descriptor itself is per frame and shared by all streams, so it cannot show this on its own. Highly interpretable, and it is the figure that makes
> the contribution feel real rather than mechanical.
> **Files:** `C-21_reliability_heatmap.png` + `.pdf` + `_data.csv` (dataset × stream × mean
> trust × std × n_videos, plus the mean 4-D descriptor per dataset). Print the cell values inside the cells — a heatmap without numbers
> is unreadable in greyscale print, and some reviewers still print. `C-21_reliability_heatmap.md`.

### `P5-T4` — Stream-specific pseudo-fakes and calibration

Each stream sees a different SBI variant (spatial blends texture boundaries, temporal perturbs
inter-frame alignment, frequency injects spectral artifacts), so the streams learn non-redundant
evidence. Then add the reliability-weighted confidence term, making calibration a *predicted
consequence* of the mechanism rather than a post-hoc layer.

**Gate:** ECE improves under distribution shift *without* AUC regression.

> **CAPTURE C-22** `[AUTO]` — reliability diagrams (predicted vs observed frequency, 10 bins) for
> baseline, Platt-scaled baseline, and the full model, on each cross-dataset test set. Report ECE
> **with MCE and Brier** — ECE alone is gameable by a non-discriminative model, and a good reviewer
> knows it.
> **Files:** `C-22_reliability_diagrams.png` + `.pdf` (panel grid: model × test set),
> `C-22_reliability_diagrams_data.csv` (model, test set, bin index, bin range, n_in_bin,
> mean_predicted, observed_frequency — the per-bin counts matter, since a bin holding four videos
> looks miscalibrated when it is merely empty), `C-22_calibration_metrics.csv` (model × test set ×
> ECE / MCE / Brier / NLL), `C-22_reliability_diagrams.md`.

---

## Phase 6 — Full evaluation

`P6-T1`. Cross-dataset matrix, final: train FF++ c23 only → CDF-v2, DFDC, DFDCP, DFD, FF++ c40.
Every number with bootstrap confidence intervals over videos, and **three seeds minimum** — a
single-seed improvement inside seed variance is not a result.

`P6-T2`. **Compression sweep — the money figure.** c23 → c28 → c32 → c36 → c40 → simulated
social-media re-encode. Use the ffmpeg CRF helpers already sitting unused in
`src/process_data/augmentations.py`. Plot AUC vs CRF for your model and every baseline. **Baselines
should degrade faster; that widening gap *is* the contribution.**

`P6-T3`. Calibration under shift: ECE / MCE / Brier / NLL across all test sets. **This refreshes
`C-22` with the full Phase 6 roster** rather than declaring a new capture — same filenames, more
rows. Post-hoc calibrators are fit on the FF++ validation split only; fitting a temperature on the
shifted test set measures nothing and is invariant 2 violated where the paper makes its claim.

`P6-T4`. Ablations — this is where reviewers look hardest:

| Ablation | Question it answers |
|---|---|
| no-gate / static-gate / learned-gate / **oracle-gate** | How much of the headroom does conditioning capture? Oracle is the upper bound. |
| SSM vs Transformer at **matched parameters** | Is the gain the operator, or just capacity? |
| Interleaved vs separate-then-fuse tokens | Does the interleaving matter? |
| Frozen vs adapted backbone | How much comes from the backbone alone? |
| Shared vs stream-specific SBI | Does non-redundant supervision help? |
| Reliability from real CRF vs learned | Does estimation error cost much? |

`P6-T5`. Efficiency: parameters, FLOPs, inference latency, peak VRAM, versus every baseline.

`P6-T7` / `P6-T8` (**secondary, planned — `docs/specs/SPEC-003_lodo_secondary_protocol.md`,
approved in principle 2026-11-28**). Leave-one-dataset-out training over FF++, DFDC and Celeb-DF v2
(6 combinations, the FF++-only one reused from `P6-T1`) for the final model and the reference
baseline, plus a separate GAN-forgery evaluation track. Reported in their own tables (`C-30`,
`C-31`), never mixed into the primary results. They run after `P6-T1` and can be cut without
touching the core claim.

`P6-T6`. Robustness beyond compression: resize, additive noise, frame dropping, brightness/contrast
shifts, blur. Cheap to add, and it broadens the robustness claim. **Outputs extend `C-24`** — the
same robustness claim, so new files under that ID rather than a thirtieth capture.

Two of these levels earn their place beyond breadth. **Frame dropping** degrades temporal evidence
while leaving spatial and frequency evidence intact — the reverse of compression — so if the
mechanism is general, *temporal* reliability should fall while the others hold. **Brightness /
contrast** is the control: appearance changes, forensic evidence does not, so reliability should stay
flat. If it drops there, the estimator is tracking image statistics rather than evidence quality and
`C-20`'s interpretation weakens. Report both outcomes either way.

> **CAPTURE C-23** `[AUTO]` — final results matrix, all models × all test sets × 3 seeds, mean ± std.
> **Table 2.**
> **Files:** `C-23_final_matrix.csv` (long format, **one row per seed** — never pre-aggregated only;
> she cannot recompute a std from a mean), `C-23_final_matrix_aggregated.csv` (mean ± std, for
> convenience), `C-23_final_matrix.md`, `C-23_final_matrix.tex`.

> **CAPTURE C-24** `[AUTO]` — compression sweep plot, AUC vs CRF, all models. **The paper's key
> figure.**
> **Files:** `C-24_compression_sweep.png` + `.pdf` + `_data.csv` (model, seed, CRF, AUC, ci_low,
> ci_high, n_videos — plus a `social_media` row for the simulated re-encode),
> `C-24_slope_table.csv` (per model: AUC at c23, AUC at c40, the delta, and the fitted slope —
> **the delta column is the contribution**, so it is a table in its own right, not just a figure
> annotation), `C-24_compression_sweep.md`.
>
> Plot mean across seeds with a shaded ±std band. A sweep drawn from one seed invites the question
> of whether the widening gap is real, which is the entire claim.

> **CAPTURE C-25** `[HUMAN-IN-LOOP]` — the ablation table with a one-line interpretation per row.
> Write the interpretations *as you go*; reconstructing them months later is guesswork.
> **Files:** `C-25_ablations.csv` (ablation, test set, seed, AUC, delta vs full model, with CIs),
> `C-25_ablations.md` (the table plus **your** one-line interpretation per row — this is the column
> only you can write), `C-25_ablation_deltas.png` + `.pdf` + `_data.csv` (a horizontal bar chart of
> delta-vs-full per ablation, which makes "which component matters" readable at a glance).

> **CAPTURE C-26** `[AUTO]` — efficiency table.
> **Files:** `C-26_efficiency.csv` (model, total params, trainable params, GFLOPs, inference ms/video
> at batch 1 and batch 8, peak VRAM, and the hardware string from C-03 — a latency number with no
> hardware attached is unusable), `C-26_efficiency.md`.

> **CAPTURE C-27** `[HUMAN-IN-LOOP]` — qualitative comparison: cases the baseline fails and your
> model gets right, **with the reliability values shown**, so the figure explains *why*.
> **Files:** `C-27_qualitative_wins.png` + `.pdf` (per case: the crop, both models' predicted
> probabilities, the true label, the 4-D reliability descriptor and the per-stream trust values printed as small bars
> beside the crop), `C-27_qualitative_wins/` (full frame sequences for the chosen cases),
> `C-27_qualitative_wins.csv` (video ID, dataset, CRF if known, true label, baseline prob, ours prob,
> 4-D descriptor, per-stream trust), `C-27_qualitative_wins.md`.
>
> **Pick these cases by hand from a script-generated shortlist.** Have the script rank every video
> where the baseline is confidently wrong and your model confidently right; then choose the handful
> that actually illustrate the mechanism — ideally where low frequency-stream reliability coincides
> with heavy compression. The automatic top-ranked case is often just an easy video, and this figure
> is only worth its column inches if it shows the mechanism working.

> **CAPTURE C-28** `[AUTO]` — every hyperparameter, in one file, for every reported model.
> Non-negotiable for reproducibility, and painful to reconstruct.
> **Files:** `C-28_hyperparameters/` containing a **copy of every resolved config YAML** actually
> used (dumped by the training script at run start, not hand-transcribed afterwards), each named
> `<model>_seed<n>.yaml`, plus `C-28_hyperparameters.csv` flattening the lot into one comparable
> table, and `C-28_hyperparameters.md`.
>
> Dump the config from inside the training script. A hand-maintained table drifts from what was run,
> and the drift is invisible until someone tries to reproduce a number.

### Phase 6 exit gate

Every number in the paper traceable to a `results_log.csv` row and a git commit.

---

## Phase 7 — Writing handoff

`P7-T1`. Assemble `paper_assets/` into the DOCX your teammate will work from — every capture in ID
order, each with its explanatory note. **Order by capture ID, not by importance**; the IDs follow
the narrative because the phases do.

`P7-T2`. Write the methods section yourself while the details are fresh, even roughly. You know why
each choice was made; she does not, and reconstructing intent from artifacts is where papers lose
accuracy.

`P7-T3`. Final bibliography pass against `C-02`. **Nothing enters the manuscript that is not marked
verified in that file.** A fabricated citation found in review is a desk-reject, and the source
documents for this project demonstrably contain some.

`P7-T4`. Submit to **Information Fusion** (Elsevier). Fallbacks in `RESEARCH_ANALYSIS.md` §6.7.

> **CAPTURE C-29** `[HUMAN-IN-LOOP]` — the handoff document itself, plus a one-page project summary: the thesis
> in a paragraph, the mechanism in a paragraph, the three headline results. **Your teammate reads
> this first.** Write it last, when you know what the story turned out to be.
> **Files:** `C-29_handoff/` containing `summary.md` (the one-pager) and `captures_index.md`
> (every capture ID linking to its note, grouped by phase, so she knows what exists and where to
> find it), plus `methods_draft.md` (your rough methods section from `P7-T2` — she will edit it, but
> the first-pass decisions while details are fresh are irreplaceable), `C-29_handoff.md`.

---

## Capture index

`Kind`: `[A]` auto — a script writes it all. `[H]` human-in-loop — script makes the files, you write
the judgement. `[M]` manual — only you can produce it.

| ID | Phase | Kind | Artifact | Primary file types | Paper role |
|---|---|---|---|---|---|
| C-01 | 0 | A | Environment + build gate output | txt | Reproducibility |
| C-02 | 0 | H | **Citation audit table** | md + csv + png | **Bibliography source of truth** |
| C-03 | 0 | A | Environment freeze + hardware spec | yml + txt + md | Reproducibility |
| C-04 | 1 | A | Legacy state archive | dir | Provenance |
| C-05 | 1 | A | Legacy Celeb-DF audit (listed/unlisted/all) | png/pdf + csv | Provenance; source of the 0.65 |
| C-06 | 1 | A | Listed vs unlisted ROC | png/pdf + csv | Provenance (supplementary) |
| C-07 | 4 | A | AMP fix diff + curves (moved from 1) | diff + png/pdf + csv | Methods note |
| C-08 | 2 | A | Dataset provenance table | csv + md | Methods, near-verbatim |
| C-09 | 2 | **M** | **EULA approvals + licence pages** | **png screenshots** | Submission requirement |
| C-10 | 3 | H | **Detector bake-off** | csv + png sheets | Methods + supplementary |
| C-11 | 3 | A | Detector crop comparison | png/pdf + csv | Methods figure |
| C-12 | 3 | A | Crop-margin diagnostic | csv + png/pdf | Possible standalone finding |
| C-13 | 3 | A | **Clean-split verification** | txt + csv + png/pdf | **Credibility anchor** |
| C-14 | 4 | A | Baseline matrix | csv + md + tex | Table 1 |
| C-15 | 4 | A | **Capstone vs FF++-only** | png/pdf + csv | **Motivating figure** |
| C-16 | 4 | A | Training curves + cost | png/pdf + csv | Reproducibility |
| C-17 | 4 | H | Baseline failure gallery | png/pdf + dir + csv | Qualitative motivation |
| C-18 | 5 | A | Backbone comparison | csv + md | Ablation |
| C-19 | 5 | A | Reliability vs CRF scatter | png/pdf + csv | Estimator validation |
| C-20 | 5 | A | **Mechanism verification** | png/pdf + csv | **Most important figure** |
| C-21 | 5 | A | Per-stream reliability heatmap | png/pdf + csv | Interpretability |
| C-22 | 5 | A | Reliability diagrams | png/pdf + csv | Calibration result |
| C-23 | 6 | A | Final matrix, 3 seeds | csv + md + tex | Table 2 |
| C-24 | 6 | A | **Compression sweep** | png/pdf + csv | **Key figure** |
| C-25 | 6 | H | Ablation table | csv + md + png/pdf | Table 3 |
| C-26 | 6 | A | Efficiency table | csv + md | Table 4 |
| C-27 | 6 | H | Qualitative wins + reliability | png/pdf + dir + csv | Qualitative figure |
| C-28 | 6 | A | All hyperparameters | yaml dir + csv | Reproducibility |
| C-29 | 7 | H | Handoff doc + summary | md dir | Teammate onboarding |

**Totals: 22 auto, 6 human-in-loop, 1 manual.** The manual one is the only capture that can become
permanently unobtainable through delay rather than through overwriting — which is why it has its own
checklist below.

---

## Manual capture checklist

Everything a coding agent **cannot** finish for you, with the moment to do it. Print this, or keep
it open — the items are individually small and collectively the ones that get skipped.

### Screenshots — only you can take these

| When | What | Goes in |
|---|---|---|
| The week the FF++ EULA approval arrives | Full-window screenshot of the approval email: sender, date, recipient, subject all visible | `C-09_licences/ffpp_eula_email.png` |
| The week the Celeb-DF v2 approval arrives | Same, for Celeb-DF v2 | `C-09_licences/celebdf_eula_email.png` |
| The day you download each of DFDC, DFDCP, DFD | The terms/licence page as it reads that day, URL bar visible | `C-09_licences/<dataset>_terms_page.png` |
| During `P0-T2`, only for IDs that fail to verify | The arXiv page or the 404, dated | `C-02_citation_audit/<arxiv_id>.png` |

Rules for all of them: capture the **whole window** including sender, date, and URL bar — a cropped
message body proves nothing. Redact only your own signature block. Name the file for what it shows,
not `Screenshot 2026-08-09 141233.png`.

**Take each one when it happens.** Mailboxes get purged, institutional accounts expire, and dataset
sites reorganise. Every other capture in this project can be regenerated by re-running something; a
licence approval email cannot.

### Judgements — a script prepares them, you decide

| Task | What you must do | Goes in |
|---|---|---|
| `P0-T2` / C-02 | Confirm each citation verdict yourself. A failed fetch is not proof of a fabricated paper — and the reverse error is a desk-reject | `C-02_citation_audit.md` |
| `P3-T1` / C-10 | Open every contact sheet. Count non-faces, half-faces, wrong-person crops per detector. Write the counts and the final choice with its reason | `C-10_detector_bakeoff.md` |
| `P4-T4` / C-17 | Look at the 20 failure cases. Write what they have in common | `C-17_failure_gallery.md` |
| `P6-T4` / C-25 | Write the one-line interpretation for every ablation row, as each result lands | `C-25_ablations.md` |
| `P6` / C-27 | Choose the qualitative cases from the shortlist — the ones that show the mechanism, not just the biggest margin | `C-27_qualitative_wins.md` |
| `P7-T2` / C-29 | Write the methods draft and the one-page summary. Only you know why each choice was made | `C-29_handoff/` |
| Every capture | The two-or-three-sentence meaning in the `.md` note, written while you remember | every `C-nn_*.md` |

### What is never a screenshot

Plots, tables, terminal output, config files, and diffs are all saved directly by the thing that
produces them — `savefig`, `to_csv`, `tee`, `git diff >`. If you find yourself about to screenshot
one of those, the script is missing an output path. Add it instead: a screenshot of a figure cannot
be re-plotted for a column width, a screenshot of a table cannot be typeset, and a screenshot of a
terminal cannot be searched.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mamba-ssm` won't build on sm_120 | Medium-high | FiLM cross-attention fallback, thesis unchanged (Gate 0-A) |
| Aletheia's weighting overlaps the hinge | **Open** | Gate 0-B partially verified — IDs are real, but Aletheia has TWO mechanisms (one input-derived). Needs full PDF. Thesis likely distinct but differentiation table not finalized. |
| Reliability estimator won't correlate with CRF | Medium | Gate at `P5-T2`, before anything depends on it |
| Conditioning gives no gain over static gating | Medium | Gate at `P5-T3`; oracle-gate ablation quantifies available headroom |
| DFDC download/processing overruns | Medium | Part-loop validated on part 00; DFDCP unblocks development meanwhile |
| Celeb-DF EULA delayed | Low | FF++ suffices through Phase 5; CDF-v2 needed only for final eval |
| Improvement within seed variance | Medium | Three seeds + bootstrap CIs from the start, not retrofitted |

---

## Standing rules

1. **Never train on anything but FF++.** Every other dataset is evaluation-only, forever.
2. **Never tune on a test set** — not hyperparameters, not early stopping, not thresholds. Use the
   FF++ validation split.
3. **Every reported number gets a `results_log.csv` row** with its git commit hash.
4. **Every citation verified in `C-02`** before it is written down.
5. **Never delete a capture.** Superseded results demonstrate progress and defend the protocol.
6. **Three seeds minimum** for any claimed improvement.
7. **Gates are gates.** A failed gate means diagnose, not proceed-and-hope.
8. **Every figure ships as PNG + PDF + its `_data.csv`.** A figure your teammate cannot re-plot is a
   figure you will be asked to regenerate after the checkpoint is gone.
9. **Screenshots are only for other people's UIs** — email, licence portals, dataset pages. Anything
   your own code produces gets saved by your own code.
10. **Take the `[MANUAL]` captures the week they become available**, per the checklist above. They
    are the only artifacts that expire.
