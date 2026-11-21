# Research Analysis: From Capstone to Publication

**An independent technical audit of the existing pipeline and the four AI-generated research documents, with a recommended research pathway.**

Prepared: 2026-08-08
Target: top-tier journal (Elsevier-tier or above)
Hardware: Lenovo Legion Pro 7i Gen 10 — Core Ultra 9 275HX (24C), 64 GB DDR5-6400, RTX 5070 Ti Laptop (12 GB GDDR7), 2 TB PCIe Gen5 SSD

---

## Executive summary

Nine findings, ordered by how much they change what you should do next.

1. **The headline result in `README.md` is not a cross-dataset result.** 905 of 5000 training
   videos are Celeb-DF. The reported "External (Celeb-DF) AUC ≈ 0.65" was measured largely on data
   the model trained on. Your true generalization gap is **worse than 0.65**.
2. **This is leverage, not just a defect.** A larger honest gap is a stronger motivation section.
   The paper's opening result becomes "detectors fail under distribution shift far worse than the
   literature's optimistic protocols suggest" — which is both true and publishable.
3. **The tri-stream + SBI + calibration proposal is not publishable at your target tier.** Every
   pairwise interaction in it is already published (CAST owns spatial+temporal cross-attention;
   FSBI owns SBI+frequency). The critique file's assessment is correct, and better-supported than
   the two documents that dispute it.
4. **The four research documents cannot be cited from.** One cites a university course-assignment
   page and a pharmaceutical proceedings as sources; another cites an Instagram reel for hardware
   guidance; a third presents a results table whose bottom row is labelled *"Simulated/Target"* —
   invented numbers surrounded by prose that reads as measured. Details in §3.
5. **The heavy alternatives they push (rPPG, Hamiltonian dynamics, 3D transformers, "UPI-HT") are
   not recommended** — one is refuted by a paper in the documents' own reference list, one rests on
   unverified citations, and one does not fit in 12 GB. §3.4.
6. **There is a clean open gap:** no existing multi-stream detector conditions its fusion on
   *estimated per-input evidence reliability*. §6.
7. **The gap and the Mamba direction are the same idea.** A selective SSM's selection parameter Δ
   *is* input-dependent gating. This lets three separate "pathways" collapse into one mechanism with
   one hinge — a considerably stronger paper than three novelties bolted together. §6.2.
8. **One hardware risk needs a day-one decision:** your GPU is Blackwell (sm_120), and `mamba-ssm`
   ships CUDA extensions that must compile against it on Windows. A fallback that preserves the
   entire thesis is specified in §6.4, so this cannot kill the paper.
9. **Your dataset is third-party preprocessed**, which blocks both a methods section and any
   frequency-domain claim. Rebuilding from source is correct; the spec is §5.

**Recommended pathway:** reliability-conditioned selective fusion (§6).
**Recommended primary venue:** *Information Fusion* (Elsevier, top-tier) — the contribution is
literally a fusion mechanism. §6.7.

---

## 1. Audit of the existing pipeline

Read from source rather than from `README.md`, because the two disagree in load-bearing places.

### 1.1 What is actually built

| Stage | Implementation | Location |
|---|---|---|
| Spatial | EfficientNet-B3 via `timm`, `num_classes=0` → 1536-D. Input **224×224** | `src/ground_truth/models.py:31` |
| Temporal | 2-layer Bi-LSTM (512, bidirectional → 1024) + attention pooling | `src/ground_truth/models.py:95` |
| Attention pool | Single `Linear(H,1)` → masked softmax → weighted sum | `src/ground_truth/models.py:73` |
| Fusion | **Not a neural network.** 5-D vector → sklearn `LogisticRegression` | `src/ground_truth/build_features.py:178` |
| Calibration | Platt scaling, `CalibratedClassifierCV(method="sigmoid", cv="prefit")` | `src/ground_truth/train_ensemble_final.py` |
| Frames | **8 per video**, uniformly spaced | `notebooks/extract_frames.ipynb:34` |
| Face detection | **None in the pipeline.** Crops are pre-made; `face_crop.py` is dead code | `src/process_data/face_crop.py` |

The 5-D fusion vector is `[s_mean, s_max, s_std, s_top3, t_prob]` (`build_features.py:178`).

Three consequences worth naming plainly:

- **Input resolution is off-spec.** EfficientNet-B3's native training resolution is 300×300; you
  feed 224×224. You are running a B3 at roughly B0's input scale, paying B3's parameter cost
  without its resolution benefit.
- **"Fusion" is a 5-number summary.** The spatial stream is compressed to four scalars before the
  classifier sees it. Whatever spatial-temporal interaction exists is unavailable to the model —
  this is late fusion in the weakest sense, and it is the single biggest architectural limitation.
- **8 frames cannot support a temporal claim.** At 8 uniformly-spaced frames from a multi-second
  clip, consecutive samples are hundreds of milliseconds apart. Micro-expression dynamics, blink
  kinematics, and flicker are all aliased away. A reviewer will ask, and there is no good answer.

### 1.2 Verified defects

Each was confirmed against source; each is cheap to fix; two are load-bearing for the paper.

**(a) The Celeb-DF leak — the critical one.**
`notebooks/split_dataset.ipynb:180` carries the comment
`# Celeb-DF (optional) - we include in entries but you'll not use them for training`, then appends
Celeb-DF entries to `entries`, which is shuffled into all splits. Measured on disk:

| Split | Celeb-DF videos | Total |
|---|---|---|
| `data/train.txt` | **905** | 5000 |
| `data/val.txt` | 159 | 937 |
| `data/test_internal.txt` | 69 | 314 |

`notebooks/test_CelebDF.ipynb` then evaluates on all 1168 Celeb-DF videos — approximately 78 % of
which appear in a training or validation split. **The README's 0.65 is not a cross-dataset number.**

**(b) `results.txt` mislabels its test set.**
`build_features.py:218` iterates `["train","val","test","test_internal","reserved_200"]`, but
`ensemble_features_final/` contains only `train.npz`, `val.npz`, `test.npz`, `test_celebdf.npz` —
**no `test_internal.npz`**. So `train_ensemble_final.py:50-54` falls through:

```python
X_test, y_test = load_data("test_internal")
test_name = "Test (Internal)"
if X_test is None:
    X_test, y_test = load_data("test")
    test_name = "Test (External)"
```

`checkpoints/ensemble_final/results.txt` therefore reports **`Test (External) AUC: 0.9695`** for
what is the *internal* held-out test set (255 samples). Any paper draft quoting 0.9695 as external
would be quoting a bug.

**(c) The `-1e9` mask blocks mixed precision.**
`models.py:85` uses `scores.masked_fill(mask, float("-1e9"))`. In fp16 (max ≈ 65504) this overflows
to `-inf`, and `-inf` rows produce NaN after softmax — which is why `train_temporal.py` disables AMP
with the comment that FP32 is more stable, and why `notebooks/reextract_one_embedding.ipynb` exists
to repair the corrupt `train/837.npy`. The fix is `torch.finfo(scores.dtype).min`. This re-enables
AMP for the temporal model, roughly halving its training time.

**(d) Stale absolute paths.** `data/train.txt` contains
`c:\Users\lkmah\OneDrive\Desktop\Lokesh\VS Code\...`, unresolvable from `E:\`. Any re-extraction
fails until the splits are regenerated — which they will be anyway (§5).

**(e) Unused compression infrastructure.** `src/process_data/augmentations.py` implements working
ffmpeg CRF re-encode helpers (`reencode_video`, `reencode_h264_qf`) that **nothing calls**. This is
precisely the infrastructure the recommended pathway's central experiment needs — a free head start.

**(f) Unsupported UI claim.** `app.py:866` renders `acc_col.metric("Accuracy", "99.2%")`. No
measurement in the repository supports this. Harmless in a demo, but it must not migrate into the
manuscript.

### 1.3 What is genuinely good here

Worth stating, because it determines how much you keep: the OOF stacking in
`train_ensemble_final.py:70-82` is leak-free and correctly implemented; checkpointing/resume is
disciplined; `check_embeddings.ipynb` verifies split disjointness and NaN-checks embeddings; seeds
and `cudnn.deterministic` are set. **The engineering is sound. The research protocol is what
needs replacing** — which is a much better position to be in than the reverse.

---

## 2. What the four documents actually say

| Document | Position | Load-bearing claim |
|---|---|---|
| *Previously Decided Research Decision (Gemini + GPT)* | Build tri-stream + SBI + calibration | "adds real novelty through tri-stream fusion and SBI-driven generalization" |
| *Claude Critic and Analysis* | Component-level novelty is largely absent | "not novel at the component level… modestly novel at the system level, and only if you frame it correctly" |
| *Latest Architectural Novelty Research (Gemini)* | Agrees it is a "compositional trap"; proposes Mamba, degradation gating, tri-dimensional SBI, native calibration loss | "every fundamental interaction within this proposed system has already been robustly published" |
| *Latest Research… (NotebookLM)* | Proposes physics-informed direction (Hamiltonian, rPPG, frequency masking), unified as "UPI-HT" | Table claiming UPI-HT reaches 0.985 CDF-v2 / 0.971 DFDC |

**Independent verdict: the critique document is substantially correct, and the two documents that
dispute it are the weaker-sourced ones.** The original proposal's own justification contains no
citations, no datasets, no metrics, and no baselines — it is a seven-item list of intents. Its
strongest claim ("adds real novelty") is asserted, not argued.

One thing all four documents miss: **none of them read `split_dataset.ipynb`.** The most important
fact about this project — that its headline number is invalid — appears in none of them. They
critique the architecture while the protocol underneath it is broken. That is the difference between
reviewing a proposal and auditing a project.

---

## 3. Literature synthesis

### 3.1 Sourcing warning — read before citing anything

**The four documents are leads, not citations.** Verifiable without web access:

- The Gemini novelty document's reference list contains items that are not research sources:
  `scde.jntuh.ac.in — "Assignment 15"` (a university course-assignment page) and
  `bhumipublishing.com — "Next Generation Approaches in Pharmaceutical, Chemical, Biological and
  Biomedical Research"` (unrelated domain).
- *Chat with Google AI mode (Pro)* cites **an Instagram reel** as source [1] for its hardware
  claims, and a Medium post on 100-million-token context windows as source [2] for transformer
  sequence limits. Its central architectural recommendation is unsourced.
- Its specific technical advice is also wrong in a checkable way: it states a Bi-LSTM can ingest
  "~300 frames (10 seconds)" while transformers cap at "16 to 32 frames." Both figures are
  presented as hardware limits; neither is. They are design choices, and the LSTM's advantage here
  is illusory — a Bi-LSTM over 300 frames has a *longer* effective path than attention over 32.
- The Mamba recommendation's supporting citation is **IJFMR**, a predatory-tier journal.
- The NotebookLM comparison table labels its own bottom row **"Proposed UPI-HT
  (Simulated/Target)"** — the 0.995/0.985/0.971 figures are invented. The surrounding prose
  presents them alongside real measurements without distinction.

**PARTIALLY VERIFIED (September 2026) — Gate 0-B still open** (`ROADMAP.md` §Gate 0-B). The five
arXiv IDs resolve to real papers with matching titles, but **C-02 has not been produced**, so under
invariant 9 none of them is citable yet. Two author attributions in earlier drafts were wrong and
are corrected here:
- `2604.16486` — **Aletheia / PhyLAA-X** — Devendra Ghori, Apr 2026.
- `2605.04405` — **HAAD**, *Detecting Deepfakes via Hamiltonian Dynamics* — Harry Cheng et al.,
  May 2026. (**Not** Miyachi.)
- `2605.00630` — **CMTA** — Hang Wang et al., May 2026. (**Not** Hwang.)
- `2607.13234` — **BMF / BitMind Forensics** — Ken Jon Miyachi & Dylan Uys, Jul 2026.
- `2607.17441` — **MSCA-FFT** — Pamela Kirui et al., Jul 2026. Image-level, not video.

**On Aletheia's 0.966:** an earlier revision "corrected" this to 90.8% AUC. That correction was
backwards — **0.966 is the reported DFDC AUC; 90.8% is accuracy.** The original flag stands: a
0.966 cross-dataset DFDC AUC is far above the 0.70–0.85 range typical for FF++-trained detectors,
and until the training protocol is confirmed from the full paper (in particular, whether DFDC or
other eval data entered training), **do not adopt it as the bar to beat.** This project has
already produced one such number, and it was leakage. Re-read every Aletheia figure from the PDF
during C-02 rather than from secondary summaries.

**Assessed real and safe to build against:** SBI (Shiohara & Yamasaki, CVPR 2022); LAA-Net
(arXiv 2401.13856); CAST (arXiv 2506.21711 → *Knowledge-Based Systems*, 2026); FSBI; FreqBlender
(OpenReview); F3-Net; ISTVT (IEEE TIFS); DeepfakeBench (SCLBD).

> **Standing rule for the manuscript: every citation verified against a DOI or arXiv listing before
> it is written down.** A fabricated citation found by a reviewer is a desk-reject, and these
> documents demonstrably contain some.

### 3.2 Saturated vs. open territory

| Territory | Status | Owner |
|---|---|---|
| Spatial CNN artifact detection | Saturated | LAA-Net, and many |
| Bi-LSTM temporal modelling | Dated — will draw reviewer fire | superseded by Transformer/SSM |
| Spatial + temporal cross-attention | **Owned** | **CAST 2026** |
| Spatial + frequency cross-attention | Owned | Uddin 2025, CAE-Net |
| SBI as augmentation | Owned | Shiohara & Yamasaki 2022 |
| SBI + frequency (DWT) | **Owned** | **FSBI 2025** |
| Frequency-domain pseudo-fake synthesis | Owned | FreqBlender |
| Post-hoc / ensemble calibration | Owned | Jin 2025, CDC |
| **Reliability-conditioned fusion** | **Open** | — |
| **SSM as heterogeneous fusion operator** | **Largely open** | Mamba work is audio / audio-visual / spatial-only |

### 3.3 Why the tri-stream proposal fails at your target tier

The proposal's defenders correctly note that no single paper combines all five elements. That is
true and insufficient. The problem is that **every pairwise interaction is published**:

- spatial × temporal via cross-attention → **CAST**
- spatial × frequency via cross-attention → Uddin 2025 / CAE-Net
- SBI × frequency → **FSBI**
- calibration × deepfake detection → Jin 2025 / CDC

A tri-stream model is the union of those pairs. When every edge of the graph is known, adding the
third node is engineering, and the three unanswerable reviewer questions are:

1. Why does this need cross-attention rather than concatenation?
2. How is this not CAST with a frequency branch?
3. How is this not FSBI with a temporal branch?

The proposal has no mechanism-level answer to any of them. The recommended pathway answers all
three from a single premise (§6.3).

### 3.4 Verdict on the heavy alternatives

**rPPG — do not use.** Physiological signals degrade under exactly the compression this domain
lives with. Decisively, the NotebookLM document's *own reference list* includes *"On Using rPPG
Signals for DeepFake Detection: A Cautionary Note"* and *"Applicability of rPPG-Based Deepfake
Detection"* — it recommends rPPG while citing the papers that caution against it, and its own prose
concedes rPPG detectors "degrade under heavy lossy compression (such as the H.264 c40 profile)."
Since c40 robustness is a headline experiment for any generalization paper, this is
self-defeating.

**Hamiltonian / symplectic dynamics — do not use.** Rests on unverified citations (§3.1). Beyond
provenance: a learnable potential `V(q)` plus leapfrog integration means backpropagating through
K integration steps, each requiring `∇_q V(q)` — a double-backward per step. Training cost and
instability rise sharply for a benefit that no verified measurement supports. High mathematical
sophistication is not the same as high expected value.

**3D transformers / ViViT / full "UPI-HT" — infeasible here.** Quadratic attention over
`T×N` tokens exceeds 12 GB at any sequence length that would justify the architecture. UPI-HT
additionally stacks 3D-DWT, rPPG, cross-attention, and Hamiltonian dynamics into one model whose
only reported numbers are self-labelled "Simulated/Target." It also inherits precisely the
gradient-conflict problem the same document warns about (`L_total = λ₁L_cls + λ₂L_rec + λ₃L_align`).

**Tri-dimensional modality-specific SBI — partially adopt.** The idea that each stream should see
a *different* pseudo-fake is genuinely good and defeats the FSBI overlap. It is not a strong enough
contribution to headline a top-tier paper, but it makes an excellent supporting component. Adopted
in §6.2 in reduced form.

---

## 4. Hardware feasibility

### 4.1 The constraint that actually matters

The Google AI chat concludes "you are not hardware limited for any standard Deepfake research."
That is broadly true for the workloads discussed and misses the one thing that will actually bite:

> **The RTX 5070 Ti Laptop is Blackwell — compute capability sm_120.** It requires CUDA 12.8+ and a
> recent PyTorch build. `mamba-ssm` and `causal-conv1d` ship custom CUDA extensions that must
> compile against your architecture. **Windows + a brand-new architecture is the worst case** for
> those packages: prebuilt wheels typically target Linux and older arch lists.

Treat this as a **day-one go/no-go gate**, not a week-six discovery. Mitigations in order of
preference: (1) WSL2 with CUDA passthrough; (2) prebuilt Vision-Mamba wheels if any target sm_120;
(3) a pure-PyTorch associative-scan SSM — slower, no compilation, always works. And critically,
**the §6.4 fallback preserves the entire thesis** if none of these land.

### 4.2 VRAM budget at 12 GB

| Configuration | Fits? | Notes |
|---|---|---|
| Frozen CLIP ViT-B/16 + adapters, bs 32 @ 224px, AMP | Comfortable | frozen backbone ⇒ no optimizer states for 86 M params |
| EfficientNet-B3 @ 300px, bs 24–32, AMP | Comfortable | legacy baseline, now at native resolution |
| Bi-SSM over 32-frame interleaved multi-stream tokens | Comfortable **if it installs** | linear in sequence length |
| Reliability estimator (< 1 M params) | Negligible | ~4 conv layers on a downsampled input |
| Transformer fusion, 32 frames × 3 streams | Tight but feasible | needs grad checkpointing |
| Full 3D transformer / ViViT @ 32 frames | **No** | why §3.4 rules out the composite blueprints |

Your 64 GB RAM is a genuine asset for the dataloader: 24 cores at 8–12 workers will keep the GPU
saturated during PNG decode, which is the actual bottleneck at these batch sizes.

**Feature caching is the real accelerator.** With a frozen backbone, extract embeddings **once**,
then train the fusion mechanism on cached tensors — the same trick your current pipeline already
uses with `embeddings/*.npy`. This turns each fusion experiment from hours into minutes and is what
makes a large ablation grid affordable overnight. Budget one full extraction pass (~a few hours) per
backbone choice, then dozens of cheap fusion runs.

---

## 5. Dataset rebuild

Rebuilding from original sources is **mandatory**, for four reasons beyond provenance:

1. **Methods section.** You cannot describe preprocessing you did not perform. "We used
   face-cropped videos from a third-party repository" is not a reproducible protocol.
2. **Compression robustness.** You cannot faithfully synthesize c40 from someone else's already-
   compressed c23 crops. Degradation must be applied by you, from known-quality sources.
3. **Frequency claims.** Unknown re-encoding by the third party has already imposed unknown
   quantization on the spectrum. Any frequency-domain result on that data is uninterpretable.
4. **Identity-disjoint splits.** These require identity labels, which live in original filenames
   (`id0_id1_0001.mp4`) and directory structure.

### 5.1 Scope (your figures, confirmed against official pages)

| Dataset | Role | Size |
|---|---|---|
| FF++ base (1000 real + 4000 fake), c23 + c40 | **training set** | 12 GB |
| FF++ DFD extension (363 actor + 3068 fake), c23 + c40 | cross-dataset eval | 28.4 GB |
| Celeb-DF v2 | cross-dataset eval | 10 GB |
| DFDC full (50 parts) | cross-dataset eval | ~470 GB |

Full DFDC accepted per your decision — it does strengthen the results table for a top-tier
submission, and your download → process → delete → next-part loop resolves the disk ceiling
against 1.4 TB free. Two process notes: **validate the loop end-to-end on part 00** before
committing to all 50, and keep DFDCP (~5 GB) as an early sanity check so method development is
never blocked on the 470 GB pull.

### 5.2 Preprocessing spec — this settles the resolution question

"Resolution" turns out not to be the operative axis. Three distinct things matter:

**Source compression (matters most).** Train on **c23**, evaluate on **c40**. This is the community
standard and the axis reviewers actually check. Raw is ~2 TB and effectively unused. Your question
about "resolution counterparts" resolves here: FF++'s variants are *compression* levels, not
resolutions, and you want both.

**Stored crop size: 320×320.** Larger than any training resolution you might choose (224 for ViT,
300 for B3, 299 for Xception), so you never re-extract to change input size. Re-extraction is the
expensive step; storage is cheap.

**Crop margin: 1.3× the detected box — likely more important than pixel count.** Blending
boundaries live at the *edge* of the face. A tight crop removes the exact artifact being detected,
and SBI needs the landmark hull plus surrounding context to blend at all. I suspect your current
third-party crops are tight, which would partly explain the weak generalization — worth measuring
against your own crops as a diagnostic.

**Storage format: PNG, not JPEG.** JPEG applies its own 8×8 DCT quantization *on top of* the video's
H.264 quantization. For a model reading spectral statistics, that is a second, uncontrolled
artifact source layered over the one you are trying to measure. At ~200–250 KB/frame, budget
~100 GB of crops in total — affordable against 1.4 TB, and it removes an entire class of reviewer
objection.

**Other parameters:**
- **32 frames per video**, up from 8. Eight cannot support a temporal claim (§1.1).
- One documented detector across **all** datasets — mismatched detectors between train and test
  introduce a domain gap that looks like a generalization result.
- **Identity- and dataset-disjoint splits.** Train on FF++ only; CDF-v2 / DFDC / DFD never seen
  during training or model selection. Non-negotiable — this is the fix for §1.2(a).

---

## 6. Recommended pathway

### 6.1 The thesis

> **Forensic evidence has input-dependent reliability.** Compression destroys frequency evidence;
> low frame-rate and motion destroy temporal evidence; spatial evidence survives longest. Every
> existing multi-stream detector fuses streams with weights learned as an *average over the training
> distribution*, so none can express **"ignore this stream, for this input."** A detector that
> estimates evidence reliability per input and conditions fusion on it should be both **more robust
> under distribution shift** and **better calibrated** — where the second property follows from the
> first as a consequence of the mechanism, rather than being attached post-hoc.

Two properties make this a top-tier thesis rather than a good idea. It is **falsifiable** — if
learned reliability does not track measured degradation, the claim fails visibly (§6.5, experiment
4). And it predicts a *second, independent* result (calibration) from a single mechanism, which is
what separates a contribution from a trick.

### 6.2 Mechanism: reliability-conditioned selective fusion

**Backbone — a component, not a contribution.** Frozen CLIP ViT-B/16 (or DINOv2) + lightweight
trainable adapters. Cheap at 12 GB, strong published cross-dataset numbers, and enables the feature
caching that makes the ablation grid affordable. Framed honestly as a component, this is a strength;
framed as novelty, it would be the paper's weakest claim.

**Fusion and temporal modelling — one operator.** A bidirectional selective SSM over an
**interleaved multi-stream token sequence**, replacing both the Bi-LSTM and the cross-attention.

**The hinge — why this is one idea and not three.** Condition the SSM's selection parameter **Δ**
(and the B/C projections) on an estimated per-stream reliability descriptor. In a selective SSM, Δ
governs how strongly the current token is written into the hidden state. So Δ is not an arbitrary
place to inject reliability — it is *the* natural home for "trust this stream less, right now."
Selectivity already *is* input-dependent gating; conditioning it on estimated reliability makes
degradation-awareness and sequence modelling the same operation. This is the difference between
"we used Mamba and also added a gate" (two bolted novelties) and "reliability enters exactly where
the operator already decides what to remember" (one mechanism).

**Reliability estimator.** A small CNN (< 1 M params) predicting an interpretable degradation
descriptor: 8×8 blockiness energy, high-frequency spectral ratio, local entropy, blur. **Trained
self-supervised on your own CRF sweep** — you applied the degradation, so ground truth is free. No
annotation, no extra data.

**Stream-specific pseudo-fakes (adopted from the Gemini document, reduced).** Each stream sees a
different SBI variant — spatial blends texture boundaries, temporal perturbs inter-frame alignment,
frequency injects spectral artifacts — so the streams learn non-redundant evidence rather than three
views of one signal. Supporting component; also the direct answer to the FSBI overlap.

**Calibration.** A reliability-weighted confidence term, so calibration is a *predicted consequence*
of the mechanism. This converts the critique document's Option C from a bolt-on into a finding.

### 6.3 Why reviewers accept it

It answers the three questions that kill the tri-stream proposal, from one premise:

| Question | Answer |
|---|---|
| Why not concatenate? | Concatenation cannot represent input-conditional unreliability. The reliability signal has nowhere to enter. |
| How is this not CAST? | CAST's fusion is content-dependent but not *reliability*-dependent, and it has no frequency stream — so it never encounters the failure mode this addresses. |
| How is this not FSBI? | FSBI is an augmentation scheme. This is a fusion operator. They are orthogonal, and FSBI becomes a baseline row. |

And the contribution is **forced by a documented physical failure mode** rather than chosen for
convenience — which is the specific quality the critique document identified as missing.

### 6.4 Fallback that preserves the thesis

If `mamba-ssm` will not build on sm_120/Windows (§4.1), the identical mechanism expresses as
**reliability-conditioned cross-attention**: FiLM the K/V projections, or scale attention logits by
reliability. Same thesis, same experiments, same claims — different operator. Roughly one week of
work. **The tooling risk cannot kill the paper**, which is why the risky component is deliberately
not load-bearing.

### 6.5 Experiments

1. **Cross-dataset generalization.** Train FF++ c23 only → CDF-v2, DFDC, DFDCP, DFD.
2. **Compression sweep** — the money figure. c23 → c40 → simulated social-media re-encode, using
   the already-written `augmentations.py` helpers. Baselines should degrade faster than the
   reliability-conditioned model; that gap *is* the contribution.
3. **Calibration under shift.** ECE, MCE, Brier, NLL, with and without reliability conditioning.
   Report MCE and Brier alongside ECE — ECE alone is gameable by a non-discriminative model, a
   pathology the Gemini document correctly flags.
4. **Mechanism verification** — the figure that makes the paper credible. Plot learned
   frequency-stream reliability against measured CRF. If it closes monotonically as compression
   rises, the mechanism visibly does what the thesis claims. If it does not, the thesis is wrong
   and you will know early.
5. **Ablations.** No-gate / static-gate / learned-gate / **oracle-gate (upper bound)**; SSM vs
   Transformer at matched parameters; interleaved vs separate-then-fuse; frozen vs adapted backbone;
   shared vs stream-specific SBI.
6. **Baselines via DeepfakeBench** for protocol-comparable numbers, plus a reproduced CAST-style
   spatial-temporal baseline as the nearest competitor.

### 6.6 Paper framing

**Title:** *Know When Not To Look: Reliability-Conditioned Selective Fusion for Robust and
Calibrated Deepfake Detection*

**Contributions, as they will appear:**
1. Formalizing evidence reliability as an input-dependent, *estimable* quantity in multi-stream
   forensics.
2. A fusion operator conditioning selective state-space dynamics on estimated reliability.
3. Self-supervised reliability estimation from controlled degradation — no annotation.
4. Demonstrating that calibration under distribution shift follows from reliability-aware fusion.
5. An honest cross-dataset protocol, including a documented account of how optimistic protocols
   (like this project's own prior splits) inflate reported generalization.

Contribution 5 costs nothing, is unusually candid, and directly serves the field. It also turns
§1.2(a) from an embarrassment into a result.

### 6.7 Venue

| Venue | Fit |
|---|---|
| **Information Fusion** (Elsevier, top-tier) | **Primary target.** The contribution is literally a fusion mechanism — this is the natural home, and it clears your senior's Elsevier bar with margin. |
| IEEE TIFS | Strong fit — forensics + calibration + robustness. |
| Pattern Recognition (Elsevier) | Solid fit, slightly broader scope. |
| Knowledge-Based Systems | CAST's own venue; direct-comparison framing works well. |
| IEEE TPAMI | Reach. Realistic only if the numbers land well *and* the mechanism analysis is deep. |
| CVPR / ICCV | Not recommended as the primary target: high variance, and the journal route better rewards the thorough robustness and calibration studies that are this work's strength. |

### 6.8 Alternatives considered and rejected

Kept for completeness, since you asked for distinct pathways — but the unified mechanism dominates
both, so they are documented as *rejected*, not as live options.

**A — Frozen foundation backbone + adapters only.** Best raw numbers, cheapest to train. Rejected
as a headline: CLIP+LoRA for deepfake detection is crowded (IncoTune, facial-component-guided
adaptation). **Becomes the strongest baseline row** in the recommended pathway.

**B — Pure SSM swap (Bi-LSTM → Mamba).** Modernizes the architecture and is defensible via linear
complexity enabling longer sequences. Rejected as a headline: without the reliability hinge it reads
as a block swap, and the reviewer question "why is this not just a newer RNN?" has no strong answer.
Its good part — the SSM operator — is absorbed into the recommendation.

---

## 7. Roadmap

| Phase | Work | Gate |
|---|---|---|
| **0** | **Day one:** `mamba-ssm` + `causal-conv1d` build test on sm_120. Verify the four unresolved arXiv IDs. | Decides §6.2 vs §6.4. Both viable — this only picks which. |
| **1** | Download FF++ c23+c40 + DFD, CDF-v2. Start DFDC part-loop (**validate on part 00 first**). | Data on disk with a documented pipeline. |
| **2** | New preprocessing: 32 frames, 1.3× margin, 320px PNG, identity- and dataset-disjoint splits. | `grep -ci celeb data/train.txt` → **0** |
| **3** | Retrain baseline on **FF++ only**. Produce the honest cross-dataset number. | Expect worse than 0.65. **This number is the paper's motivation.** |
| **4** | Reliability estimator (self-supervised on CRF sweep) → conditioned fusion → calibration term. | Experiment 4 (§6.5) must show reliability tracking CRF. |
| **5** | Full eval matrix, compression sweep, calibration study, ablation grid. | — |
| **6** | Write. Verify every citation against DOI/arXiv. | Zero unverified citations. |

**Fix alongside phase 2** (each is minutes, two affect results): the `-1e9` → `torch.finfo().min`
mask fix, which re-enables AMP; the `test_internal` fallback that mislabels the test set; the
`app.py:866` accuracy claim; and regenerating splits with relative paths.

---

## 8. Open items

- **arXiv IDs — partially verified.** All five exist with matching titles; two author attributions
  corrected (§3.1). Not citable until C-02 is produced and the user confirms each verdict.
- **PhyLAA-X's 0.966 DFDC AUC — open.** 0.966 is the AUC (90.8% is accuracy). Do not use it as the
  bar to beat until the training protocol is confirmed from the full paper (§3.1).
- **Aletheia overlap — open, needs the full PDF.** The abstract describes two mechanisms:
  (1) physics feature volumes (optical-flow curl, specular skewness, rPPG spectra) fed into LAA-X
  attention via cross-attention gating — input-derived conditioning, closer to ours than first
  assessed; (2) uncertainty-aware ensemble weighting — prediction-confidence-based. The thesis is
  likely still distinct (SSM Δ-conditioning on self-supervised degradation descriptors is not
  cross-attention gating on physics volumes), but **do not claim "no conflict" anywhere** until the
  mechanism is read from the paper or code. Note: `github.com/devghori1264/Aletheia` describes an
  unrelated Django/NLP system — cite cautiously.
- **DFDC download logistics** — part-loop validated on part 00 before committing to all 50.
- **Detector choice** for preprocessing (RetinaFace vs MTCNN vs S3FD) — settle in `P3-T1` and hold
  it fixed across every dataset.
- **Crop-margin diagnostic** — measure your own 1.3× crops against the third-party crops on the
  same videos. If generalization improves from margin alone, that is worth a paragraph in the paper.

---

## Appendix: claim verification

Every code claim in this document was checked against source during preparation.

| Claim | Verified by |
|---|---|
| 905 Celeb-DF in `train.txt`; 159 val; 69 test_internal | `grep -ci celeb` on each split file |
| Celeb-DF appended to `entries` despite comment | `notebooks/split_dataset.ipynb:180-184` |
| 5-D fusion vector → LogisticRegression | `build_features.py:178`; `train_ensemble_final.py:61` |
| `test_internal.npz` absent → "Test (External)" mislabel | `ls ensemble_features_final/`; `build_features.py:218`; `train_ensemble_final.py:50-54` |
| `results.txt` reports `Test (External) AUC: 0.9695` | `cat checkpoints/ensemble_final/results.txt` |
| `-1e9` fp16 overflow blocks AMP | `models.py:85`; AMP-disabled comment in `train_temporal.py` |
| 8 frames per video | `notebooks/extract_frames.ipynb:34` (`FRAMES_PER_VIDEO = 8`) |
| EfficientNet-B3 at 224px | `models.py:31-43`; `IMG_SIZE = 224` |
| Unused ffmpeg CRF helpers | `src/process_data/augmentations.py`; no callers found |
| `app.py` unsupported 99.2% claim | `app.py:866` |
| Stale OneDrive paths in splits | `head -3 data/train.txt` |
| Non-source citations in Gemini doc | its own reference list |
| Instagram reel as hardware source | *Chat with Google AI mode (Pro)*, source [1] |
| UPI-HT numbers self-labelled "Simulated/Target" | NotebookLM doc, comparison table |
| rPPG cautionary papers in NotebookLM's own refs | its own reference list |

Dataset sizes in §5.1 are yours, checked against official pages. Sizes I could not verify this
session are marked as such in §8.
