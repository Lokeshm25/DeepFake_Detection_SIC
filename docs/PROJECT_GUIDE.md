# Project Guide — where this project came from, what changed, and where it stands

**Written for:** the owner (Lokesh), so you can take back control of your own project.
**Written by:** Claude, 2026-11-24. The facts here come from the repository, the git history and
the past-session prompts (`docs/past_session_prompts.md`).
**Plain-language rule:** every technical term is explained the first time it appears, and again in
the glossary (§13). If something here is unclear, that's a defect in this guide. Ask, and it gets
rewritten.

---

## 0. The 60-second version

- **February–March 2026:** you built a working deepfake detector as a capstone. It reported good
  numbers, including about 0.65 on a "different" dataset (Celeb-DF).
- **August 2026:** you decided to turn it into a journal paper, and asked for an independent
  audit. The audit found that **905 Celeb-DF videos had accidentally been put into the training
  data**. So the 0.65 wasn't a fair test, because the model had already seen most of those videos.
- That discovery changed everything. We kept your old system as the **"before" picture**, and
  planned a **new model with one clear research idea**: the model should judge, for each video,
  *how much each type of evidence can be trusted*, and lean on the trustworthy evidence.
- **September and November 2026:** a lot of planning documents, a lot of fixing of those documents,
  and then a working method for two AI assistants (Claude and Codex) to work on the project
  safely under your control.
- **Today:** planning is finished. The first piece of real code, the results logbook, is built,
  tested and committed. **No new model has been trained yet.** The next step is to archive the old
  system and measure exactly how much the leak inflated its result. That measurement becomes
  Figure 1 of the paper.

---

## 1. Where it started — your capstone system

This is what you built, in plain terms. It still exists, untouched, in `src/ground_truth/` and
`src/process_data/`.

1. **Input:** face-cropped videos, which came already cropped from a third party. We never ran
   our own face detector.
2. **8 frames per video** were taken, evenly spaced.
3. **Spatial model — EfficientNet-B3.** It looks at each frame on its own, like a photo, and turns
   it into a list of 1536 numbers that describe what it sees. ("Spatial" means within one image.)
4. **Temporal model — a Bi-LSTM with attention.** It reads those 8 frame descriptions *in order*
   and looks for something wrong in how the face changes over time. ("Temporal" means across time.)
5. **Fusion — logistic regression.** It combines both opinions into one final "fake" probability.
   It only saw 5 summary numbers: four from the spatial model and one from the temporal model.
6. **Calibration (Platt scaling).** This adjusts the probabilities so that "80% fake" really means
   about 80%.
7. **A Streamlit demo app** (`app.py`) to try it on a video.

**Reported numbers (README):** about 0.97 on its own test set, and about 0.65 on Celeb-DF.

---

## 2. What went wrong — the audit (August 2026)

You gave Claude four AI-generated research documents (from Gemini, ChatGPT, NotebookLM and Google
AI mode) and your code, and asked for an independent analysis. The full result is
`RESEARCH_ANALYSIS.md`. These are the findings that matter, in plain words:

| Problem | What it means | Status |
|---|---|---|
| **The Celeb-DF leak** | `split_dataset.ipynb:180` has a comment saying Celeb-DF is *not* used for training, but the code put it in the training list anyway. 905 of 5000 training videos were Celeb-DF. | Will be fixed when the data is split again (P3-T4). Before that, we **measure** it (P1-T2). |
| **"0.9695 External" is mislabelled** | `results.txt` says "Test (External) AUC: 0.9695". It's actually your *internal* test set. A missing file made the code fall back silently. | Never quote 0.9695 as external. Fix in P1-T3. |
| **"99.2%" in the app** | A hardcoded accuracy number that no measurement supports. | Remove in P1-T3. |
| **AMP was turned off** | A `-1e9` value overflows in half-precision maths and produced errors (NaN), so faster training had been disabled. | One-line fix in P1-T3, which makes training roughly 2× faster. |
| **8 frames is too few** | Frames are hundreds of milliseconds apart, so quick facial movements are lost. A reviewer would attack any "temporal" claim. | New pipeline uses 32 frames (P3-T3). |
| **B3 run at 224px** | EfficientNet-B3 is designed for 300px images. | Handled in P4-T2. |
| **The fusion is weak** | 5 summary numbers can't capture how the two streams relate. | Replaced by the new mechanism (Phase 5). |
| **The research documents can't be cited** | Several cite an Instagram reel, a course-assignment page and a pharmacy proceedings, and one shows *invented* "simulated/target" numbers next to real ones. | Every citation must be checked by hand (invariant 9, capture C-02). |

**Why this is good news, not just bad news:** the leak means the *honest* cross-dataset number will
be lower than 0.65. That makes the paper's motivation stronger: "detectors fail under real-world
shift much worse than people report." That's a true and publishable story.

---

## 3. What we're building now — the research idea in plain words

### 3.1 The idea

A deepfake leaves several kinds of evidence:

- **spatial** — what a single frame looks like (blending edges, texture);
- **frequency** — hidden patterns in the image's fine detail, which a computer sees after a
  mathematical transform;
- **temporal** — how the face moves from frame to frame.

These kinds of evidence are **not equally trustworthy for every video**. Heavy compression, such as
a WhatsApp re-upload, wipes out the frequency evidence. Blur and low frame rate damage the temporal
evidence. Spatial evidence survives the longest.

Existing detectors combine the streams with fixed habits learned during training, the same mix for
every video. They can't say *"for this particular video, don't trust the frequency stream."*

**Our model can.** It first measures how damaged each frame is, then uses that to decide how much
to listen to each stream. **Title idea:** *"Know When Not To Look."*

**Analogy:** three witnesses describe a crime. One was standing in fog. A good judge doesn't
average all three equally. They give less weight to the one in the fog. Our model is the judge, and
the "fog meter" is the reliability estimator.

### 3.2 The parts of the new model

| Part | What it does | Plain-language note |
|---|---|---|
| **Backbone: frozen CLIP ViT-B/16 + small adapters** | Turns each face into numbers. | A big, already-trained image model that we *don't* retrain (it's "frozen"). We only train small add-ons called adapters. It's cheap, and it's known to generalise well. Treated as a tool, not our contribution. |
| **Reliability estimator** | Gives every frame 4 scores: blockiness, high-frequency ratio, entropy (detail), blur. | Our "fog meter". It learns from videos we compress ourselves, so the right answers come for free. Under 1M parameters. |
| **Selective SSM (Mamba) fusion** | Reads the frames' spatial, frequency and temporal tokens in sequence and decides what to remember. | An SSM is a sequence model, a modern alternative to the LSTM. Mamba has a built-in knob called **Δ (delta)** that controls how strongly each new piece of input is written into memory. |
| **The key trick** | The fog-meter scores adjust Δ, separately for each stream. | "Trust this stream less right now" goes exactly where the model already decides what to remember. This is the paper's main contribution. |
| **Stream-specific fake examples (SBI)** | Each stream trains on its own kind of synthetic fake. | So the three streams learn different evidence instead of copies of the same thing. |
| **Calibration** | Confidence drops when the evidence is unreliable. | Better calibration becomes a *result* of the idea, not an add-on. |

**If Mamba won't install on your GPU:** the same idea works with "cross-attention", another way of
combining streams. It's called the FiLM fallback. Same thesis and same experiments, just a
different engine. Gate 0-A (task P0-T1) decides which engine we use.

### 3.3 Old vs new, side by side

| | Capstone (old) | New model |
|---|---|---|
| Face crops | Third-party, pre-made | Our own detector, chosen by a test (P3-T1), 320px, 1.3× margin |
| Frames per video | 8 | 32 |
| Image model | EfficientNet-B3, trained, at 224px | Frozen CLIP + small adapters |
| Evidence streams | Spatial + temporal | Spatial + frequency + temporal |
| Time model | Bi-LSTM | Selective SSM (Mamba), or the FiLM attention fallback |
| Fusion | Logistic regression on 5 numbers | Reliability-conditioned fusion inside the SSM |
| Training data | FF++ **+ leaked Celeb-DF** | **FF++ only**, always |
| Test datasets | Celeb-DF (contaminated) | Celeb-DF v2, DFDC, DFDCP, DFD, FF++ c40, never trained on |
| Seeds | 1 | 3 (42, 1337, 2024), reported as mean ± std |
| Target | Capstone grade | Journal: *Information Fusion* (Elsevier) |

The old system isn't thrown away. It's retrained honestly on FF++ only (Phase 4) and becomes the
**baseline**, the thing the new model has to beat.

---

## 4. Cross-dataset ("cross-domain") testing — why it's the heart of the paper

- **In-domain test:** train on FF++ and test on *other* FF++ videos. Almost every detector scores
  well here, so it proves little.
- **Cross-dataset test:** train on FF++ and test on a *completely different* dataset (Celeb-DF,
  DFDC…), made with different fakes, cameras and compression. This is what happens in the real
  world, and it's where detectors collapse.
- **The rule that protects it (invariant 1):** the test datasets must **never** be used in training,
  in validation, or for any choice we make. The capstone broke this rule by accident. That's the
  leak.
- **Rule of thumb:** a cross-dataset AUC above about 0.95 from an FF++-trained model is suspicious
  and stops work until explained, because it usually means leakage.

**Datasets in the plan:** FF++ c23 and c40 (the *only* training data); FF++ DeepFakeDetection
extension, Celeb-DF v2, DFDCP and DFDC (evaluation only). "c23" and "c40" are compression levels;
c40 is heavier. **On disk today** are only the old third-party face crops (`Dataset/`, about
1.6 GB) and the capstone's checkpoints and embeddings. **The original source datasets haven't been
downloaded yet** (Phase 2).

---

## 5. Timeline — what happened, in order

| When | What happened |
|---|---|
| **Feb 10 – Mar 6, 2026** | Capstone built: frame extraction, embeddings, spatial and temporal training, ensemble, Celeb-DF test, README, licence. |
| May 4 | README title changed. |
| **Aug 6–9** | Research phase. Independent audit, `RESEARCH_ANALYSIS.md`, `ROADMAP.md` (phases, gates, the 29 "captures"), screenshot rules for manual captures. |
| **Sep 19** | 12 planning/infrastructure files created (`IMPLEMENTATION.md`, `docs/ARCHITECTURE.md`, `docs/API_CONTRACTS.md`, configs, `CLAUDE.md`, `AGENTS.md`…). Environment set to Python 3.11. A September audit listed inconsistencies. |
| Nov 21 | Lost chat sessions recovered as far as possible (the old Claude replies were deleted by the 30-day cleanup; your prompts were exported). Chat retention set to 365 days. All September audit fixes done and verified. Two design decisions settled: (1) reliability is scored per frame, with its own response for each stream; (2) the ablation uses whatever engine Gate 0-A picks. |
| Nov 21–22 | Two-assistant workflow set up: `RESEARCH_PROTOCOL.md`, project skills, SPEC and handoff templates. Codex onboarded, reviewed twice by Claude, accepted by you. |
| Nov 22–24 | **First real code:** the results logbook (P0-T5) plus run records (SPEC-001). Built by Codex, reviewed twice by Claude, accepted by you, committed and pushed (`b444c5e`). |
| Nov 24 | Overnight batch issued to Codex: rest of Phase 0 + all of Phase 1 (`docs/handoffs/BATCH-01_phase0-1.md`). |

**Why the plan felt like it kept shifting:** almost every change since August has been *correcting
or tightening* the same plan, not replacing it. The research idea has stayed the same since
August 8. What changed was the detail: fixed author names, a backwards AUC/accuracy "correction",
made-up citations removed, documents brought into line with each other, and now rules for two
assistants. The direction is stable.

---

## 6. What we can rely on today (verified)

| Item | Evidence |
|---|---|
| The leak is real: 905 / 159 / 69 Celeb-DF videos in train / val / internal test | Counted from `data/*.txt` (now tracked in git as evidence). |
| 0.9695 is the internal test set, not external | Traced in code: `build_features.py:218` → `train_ensemble_final.py:50-54`. |
| The 5 arXiv papers exist with the right titles; two author names were corrected | Checked against arXiv in September. **Not yet citable:** you still have to confirm each one in C-02. |
| **Results logbook + run records (P0-T5 / SPEC-001)** | 28 automated tests pass, plus code-style, formatting and type checks. Run fresh by Claude on Nov 24, independently of Codex. Committed in `b444c5e`. |
| Documents are consistent with each other | Full verification pass on Nov 21. |
| Codex follows the rules | Onboarding reviewed twice, accepted by you. |

**What you must NOT rely on or quote:** the 0.65 Celeb-DF AUC (contaminated), 0.9695 (mislabelled),
99.2% (unsupported), and any number from the four AI research documents. Also don't use Aletheia's
0.966 DFDC AUC as "the bar to beat" until we've read the paper and know how it was trained.

---

## 7. How we work now — and what your approvals mean

### 7.1 Who does what

| Who | Job |
|---|---|
| **You (owner)** | Decide the research direction. Approve big changes. Settle disagreements. Accept finished work. Authorise commits. |
| **Claude** | Research, literature, method design, writing SPECs (task specifications), reviewing Codex's code. |
| **Codex** | Writes and runs the code: data pipelines, training, evaluation, tests. |
| Antigravity | Not used (inactive). |

Every rule both assistants follow is in `RESEARCH_PROTOCOL.md`. If a rule anywhere else disagrees
with it, the protocol wins.

### 7.2 Levels A, B and C — how big is a change?

Before changing anything, an assistant labels the change with a level. **The level tells you
whether you need to be asked.**

| Level | Plain meaning | Examples | Do you need to approve? |
|---|---|---|---|
| **A — Engineering** | Changes *how* the code works, never *what results it gives*. | Faster code, better error messages, tests, tidier files. | **No.** It's just reported. |
| **B — Could move the numbers** | Doesn't change the research question, but might change a result. | Batch size, learning rate, augmentation, image size, early stopping. | **Yes, before it runs.** It's tested fairly: old vs new, same seeds, one thing changed. **Then you decide** whether to adopt it. |
| **C — Changes the research** | Changes what we're claiming or how we measure it. | The model design, datasets, splits, how metrics are computed, gates, the captures, any fixed setting (32 frames, 320px, seeds…). | **Yes, in writing, before any work.** |

**Config values** (settings in the `configs/*.yaml` files) come in three kinds:

- **Fixed:** written in `IMPLEMENTATION.md`, or adopted by you. Changing one is Level C.
- **Provisional:** a placeholder guess in a file marked `STUB`, such as the learning rate. Setting
  it properly is Level B.
- **Spec-written:** a value a task is *told* to fill in, such as which engine (P0-T1) or which face
  detector (P3-T1). Filling it in is just doing the task.

### 7.3 SPECs and handoffs — the paperwork

- **SPEC** means *specification*: a written description of exactly what to build, how to test it,
  and what's allowed to change. For most tasks, the task's section in `IMPLEMENTATION.md` already
  *is* the SPEC. A separate SPEC file (`docs/specs/SPEC-nnn_*.md`) is only written for a new idea
  or a Level B/C change.
- **SPEC status:** DRAFT (proposed) → APPROVED (you said yes, Codex may build it) → IMPLEMENTED →
  REVIEWED (Claude checked it) → ACCEPTED (you signed off; it's folded into `IMPLEMENTATION.md`).
- **Handoff record** (`docs/handoffs/<task>.md`): one shared notebook page per task. Codex writes
  its plan, test results and any deviations. Claude writes review findings. Codex answers each
  finding. You write the final decision. The assistants can't see each other's chats, so
  **everything they need to share goes in this file**.
- **Review verdicts:** CONFORMANT (matches the SPEC), CONFORMANT WITH FINDINGS (mostly right, some
  fixes needed), NON-CONFORMANT (doesn't match, redo).

### 7.4 SPEC-001, explained simply

**Problem:** every result goes into one logbook (`paper_assets/results_log.csv`). Before SPEC-001,
a row said *which code version and settings* produced a number, but not *which data files*, *which
library versions*, or *which saved model file*. If the data were later re-split, or a model file
overwritten, an old number could no longer be traced or defended.

**What SPEC-001 added:**

1. Every run gets a **run ID**, a name like `20261201-213000_P4-T1_baseline_b3_seed42` (time, task,
   model, seed). Every logbook row carries it.
2. Every run writes a **run record** (`paper_assets/runs/<run ID>/manifest.yaml`). It holds the
   question the run answers; a *fingerprint* (sha256 hash) of every data-split file used; the
   Python/PyTorch/CUDA/GPU versions plus a full package list; and fingerprints of the model files
   the run **produced** and the ones it **loaded**.
3. **Evaluation runs get their own run ID** (your decision) and point back to the training run that
   made the model they tested.
4. **Amendment 1** (your approval): an evaluation can load *several* model files, and old capstone
   model files are allowed even though they have no run record. They're marked as "made before
   SPEC-001", and their fingerprint is still saved.

**Why it matters:** if a reviewer asks "are you sure that number didn't come from leaked data?", we
can show the exact data files and model file behind it. That's the question that sank the capstone
number.

A **fingerprint (hash)** is a short code computed from a file's contents. If even one byte changes,
the code changes completely. So it proves the file is exactly the same one.

### 7.5 Your decision cheat-sheet

When an assistant asks you to approve something, ask:

1. **What level is it?** A means just be informed. B means a fair test first, then you decide.
   C means a research change, so read the reason carefully.
2. **Does it touch the ten invariants (§9)?** If yes, the answer is almost always no.
3. **What happens if I say no?** An assistant should always tell you. If it didn't, ask.
4. **Can I say "explain in plain words first"?** Always. You're never expected to approve something
   you don't understand.

---

## 8. The paper's evidence plan — "captures"

A teammate who wasn't here will write the paper. So every important result is saved **at the moment
it's produced**, as a numbered *capture* (C-01 … C-29) in `paper_assets/`, each with a short note
saying what it means.

- **[AUTO]** (22 of them): a script makes everything.
- **[HUMAN-IN-LOOP]** (6): a script makes the files, then **you** add a judgement, such as checking
  some crops or writing an interpretation line.
- **[MANUAL]** (1, plus one folder): only you can make it. **C-09** is screenshots of dataset
  licence approvals, taken *when the email arrives*. Inside **C-02** there's a folder of screenshots
  of the arXiv pages.
- When an assistant reaches a human or manual capture it must **STOP and wait** for you. It never
  invents your judgement.

Figures are always saved as PNG + PDF + a CSV of the plotted numbers + a note, never as
screenshots of code output.

---

## 9. The ten rules that are never broken (invariants), in plain words

1. **Train only on FF++.** The other datasets are for testing only, forever.
2. **Never use a test set to make a choice.** All choices use FF++'s validation part.
3. **Check the data splits before every training run** (`tools/verify_splits`).
4. **Log every number** in the logbook, with the code version.
5. **Never delete anything in `paper_assets/`.**
   5b. **Every evaluation saves a score for every video,** so figures can be redrawn later without
   the model.
6. **Three seeds** (repeat with 3 different random starts) for any claimed improvement.
7. **New "trust" parts start switched off** (zero-initialised), so "with vs without" is a fair
   comparison.
8. **Confidence ranges are computed per video, not per frame** (frames of one video are too
   similar to count separately).
9. **Every citation checked by hand** before it's written anywhere.
10. **Gates are gates.** If a check fails, fix it. Don't push on and hope.

---

## 10. Current state — exactly (2026-11-24)

**Done:**
- All planning documents, consistent with each other (`RESEARCH_ANALYSIS.md`, `ROADMAP.md`,
  `IMPLEMENTATION.md`, `docs/ARCHITECTURE.md`, `docs/API_CONTRACTS.md`, configs).
- The working rules (`RESEARCH_PROTOCOL.md`, `CLAUDE.md`, `AGENTS.md`), skills and templates.
- **P0-T5 + SPEC-001:** the logbook and run records. Built, tested, accepted, pushed.

**In progress (Codex, overnight — BATCH-01):**
P0-T6 figure helpers → P0-T4 capture checker → P0-T3 environment freeze → P0-T1 Mamba install test
(decides the engine) → P1-T1 archive the old system → P1-T2 measure the leak (Figure 1) →
P1-T3 the four cheap fixes → Phase 1 exit check. Codex commits each task locally (no push) and
writes a summary at the bottom of the batch file. Claude reviews each task next.

**Waiting on you:**
- **P0-T2 / C-02, the citation audit:** open the 5 arXiv pages, confirm each one, take dated
  screenshots. Claude can prepare the table first.
- **Dataset access (Phase 2):** request FF++, Celeb-DF v2 (needs your institutional email) and
  DFDC access. **Screenshot each approval when it arrives (C-09).**
- A WSL2 Ubuntu install, only if Codex reports that native Windows Mamba failed and you want to try
  attempt 2 before falling back.

**Not started:** Phases 2–7.

---

## 11. What's left — the broad phases

| Phase | What | Roughly | Main output |
|---|---|---|---|
| **0 — Gates & environment** | Engine test (Mamba?), citation check, environment freeze, helpers | Nearly done (overnight) | Engine decision; C-01, C-02, C-03 |
| **1 — Freeze old project & measure the leak** | Archive the capstone, measure seen vs unseen Celeb-DF, 4 quick fixes | Overnight | **Figure 1** (C-05/C-06) — the honest starting number |
| **2 — Datasets** | Download FF++, Celeb-DF v2, DFDCP (first), DFD, DFDC (≈470 GB, in parts) | Days, mostly waiting on approvals | Data on disk; provenance table; licence screenshots |
| **3 — Preprocessing** | Pick a face detector by test, 32 frames, 320px crops, rebuild the splits correctly | ~1–2 weeks | Clean splits, with proof of no leak (C-13) |
| **4 — Honest baseline** | Retrain your old model on FF++ only; two published baselines; full test matrix | ~1–2 weeks | **"Honest vs inflated" figure** (C-15), Table 1 |
| **5 — The mechanism** | Backbone, reliability estimator, reliability-conditioned fusion, calibration | Main research work | **Mechanism verification figure** (C-20): does trust drop as compression rises? |
| **6 — Full evaluation** | Final matrix (3 seeds), compression sweep, calibration, ablations, efficiency, robustness | ~2–3 weeks GPU | Tables 2–4, the "money figure" (C-24) |
| **7 — Writing handoff** | Assemble everything for your teammate, methods draft, final citation pass, submit | — | Submission to *Information Fusion* |

**The decision points ahead that only you can make:** which engine (after P0-T1), which face
detector (P3-T1 includes a judgement from you), whether the mechanism actually works (Phase 5
gate: if reliability doesn't track compression, the idea is wrong and we'll know early), and the
final paper claims.

---

## 12. Where to find things

| You want… | Look at |
|---|---|
| The *why* — research argument, literature, audit | `RESEARCH_ANALYSIS.md` |
| The *what and when* — phases, tasks, gates, captures | `ROADMAP.md` |
| The *how* — exact code specs per task | `IMPLEMENTATION.md` |
| The *rules* | `RESEARCH_PROTOCOL.md` |
| Model diagrams / code interfaces | `docs/ARCHITECTURE.md`, `docs/API_CONTRACTS.md` |
| Settings | `configs/*.yaml` |
| Task status, reviews, decisions | `docs/handoffs/` |
| Proposals awaiting or past approval | `docs/specs/` |
| Results and paper evidence | `paper_assets/` (not in git: back it up yourself) |
| Your old prompts | `docs/past_session_prompts.md` |
| This guide | `docs/PROJECT_GUIDE.md`, updated at the end of each phase |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Deepfake** | A video where a face was swapped or manipulated by AI. |
| **AUC** | One number for how well a detector separates real from fake. 1.0 is perfect, 0.5 is coin-flipping. Doesn't depend on picking a threshold. |
| **AP, EER, accuracy** | Other scores. AP: precision across thresholds. EER: the error rate where false alarms equal misses. Accuracy: % correct at one chosen threshold. |
| **Calibration / ECE / MCE / Brier** | Whether "80% fake" really means 80%. ECE is the average gap, MCE the worst gap, Brier the overall probability error. Reported together, because ECE alone can be gamed. |
| **Train / validation / test split** | Train: the model learns from it. Validation: used to make choices. Test: used only at the end to measure. Mixing them is leakage. |
| **Leakage** | Test data sneaking into training, so scores look better than they are. |
| **Identity-disjoint split** | The same *person* never appears in both training and test, so the model can't just recognise faces. |
| **In-domain / cross-dataset (cross-domain)** | Test on the same dataset vs a different one (§4). |
| **FF++, Celeb-DF, DFDC, DFDCP, DFD** | Public deepfake datasets. FF++ is for training; the rest are for testing only. |
| **c23 / c40 / CRF** | Video compression levels. Higher number means more compressed and lower quality. CRF is the compression setting we use to make our own compressed copies. |
| **Seed** | The starting number for randomness. Same seed means same run. We use 42, 1337 and 2024. |
| **Checkpoint** | A saved trained model file. |
| **Embedding** | The list of numbers a model produces to describe an image. |
| **Backbone / frozen / adapter** | Backbone: the big image model. Frozen: not retrained. Adapter: a small trainable add-on. |
| **CLIP ViT-B/16** | A large, publicly available image model, used as our frozen backbone. |
| **EfficientNet-B3, Bi-LSTM** | The image model and the sequence model in your capstone. |
| **SSM / Mamba / Δ (delta)** | A sequence model; Mamba is a modern, fast one. Δ is its built-in "how much to remember this input" knob. Our idea feeds reliability into Δ. |
| **FiLM / cross-attention fallback** | Another way to combine streams, used if Mamba won't install. Same idea. |
| **sm_120 / Blackwell** | Your GPU's architecture (RTX 5070 Ti). It's new, so some libraries may not compile for it yet. That's Gate 0-A. |
| **AMP / fp16** | Faster training using lower-precision numbers. Blocked in the capstone by the `-1e9` bug. |
| **Reliability descriptor** | The 4 "fog meter" scores per frame: blockiness, high-frequency ratio, entropy, blur. |
| **SBI (self-blended images)** | Synthetic fakes made by blending a face with itself, used to teach detectors. |
| **Zero-init** | A new part starts at zero so it changes nothing until trained, which keeps comparisons fair. |
| **Ablation** | Removing or swapping one part to prove that part is what helps. |
| **Bootstrap / CI** | Re-sampling the test videos many times to get a confidence range (CI) around a score. |
| **Gate** | A pass/fail check that must pass before moving on. |
| **Capture (C-nn)** | A saved paper artifact with an explanation note (§8). |
| **Invariant** | One of the ten never-break rules (§9). |
| **Level A / B / C** | How big a change is, and whether you must approve it (§7.2). |
| **SPEC / handoff / verdict** | See §7.3. |
| **Run ID / manifest / hash** | See §7.4. |
| **Commit / push** | Commit saves a snapshot in your local git history. Push uploads it to GitHub. |
| **DeepfakeBench** | A public toolkit for reproducing published detectors fairly. Used for baselines. |
| **Information Fusion** | The target journal (Elsevier). |
