# SPEC-004 — Production-grade repository, `.venv` + WSL environment, and the notebook→script workflow

**Status:** DRAFT — awaiting owner approval
**Change level:** mostly A (structure, tooling). The environment switch is **B** (it could move a
number), so it is gated by the parity check in §9. The `operator` change is **C**, already authorised
by the owner on 2026-11-28 ("make it fully compatible").
**Related task IDs:** P0-T1 (operator), P0-T3 (environment), all later phases (workflow)
**Author:** Claude   **Date:** 2026-11-28
**Handoff record:** `docs/handoffs/SPEC-004.md`

**Owner direction (2026-11-28):** match the structure of the owner's production repos (Candidra,
Veyns); move day-to-day work to a `.venv`, keeping the conda `aiml` build as the untouched backup;
use notebooks for research exploration and scripts for the final run of each step; install Ubuntu
in WSL2 so the fast Mamba kernel can be used.

## 1. Objective

The research code must be as reproducible and reviewable as the owner's production work:

1. One dependency definition (`pyproject.toml` + `uv.lock`) that rebuilds the environment exactly.
2. GPU training in **WSL2 Ubuntu 22.04** with the native `mamba-ssm` CUDA kernel. Windows stays
   for editing, docs, tests and the demo app.
3. Automatic quality gates (pre-commit + CI), so a broken test or an unformatted file can't land.
4. A fixed rule for notebooks vs scripts, so no paper number ever comes from a notebook.

**Fact (repo survey, 2026-11-28):** Candidra and Veyns both use `uv` with `.venv` and a committed
`uv.lock`, pinned `ruff`, `.pre-commit-config.yaml` (ruff, ruff-format, whitespace/EOF, YAML/JSON
checks, large-file guard, private-key detection), GitHub Actions CI (`uv sync` → ruff check →
ruff format --check → pytest), `.editorconfig` (LF, UTF-8), and `CONTRIBUTING.md`,
`docs/SETUP.md`, `docs/DECISIONS.md`. Candidra adds a `Makefile`. This repo has none of these.
It has an unpinned `pyproject.toml`, a `requirements.txt` and conda freezes.

## 2. What stays exactly as it is

- `src/ground_truth/`, `src/process_data/` (legacy; protocol §6 forbids restructuring them).
- `paper_assets/` contents (invariant 5), `checkpoints/`, `embeddings/`, `data/*.txt`, `Dataset/`.
- The conda `aiml` environment and its freezes (`environment.yml`, `conda-win-64.lock.txt`,
  `environment-after-P0-T1.yml`). They are the reference environment for every Phase 0–1 result.
- `IMPLEMENTATION.md`'s **target repository layout**. It already defines the right package
  structure (`src/data`, `src/models`, `src/train`, `src/eval`, `tools/`, `configs/`). This SPEC
  adopts it rather than inventing a new one.

## 3. Environment design

```
Windows (E:\VS Code\DeepFake_Detection_SIC)          WSL2 Ubuntu-22.04
  .venv/            ← uv, Python 3.11, torch 2.8 cu128   ~/venvs/deepfake  ← uv, same lock + mamba-ssm
  edit, docs, tests, app.py, light notebooks            training, eval, heavy notebooks (Jupyter)
            └──────────── same git working copy (/mnt/e/VS Code/DeepFake_Detection_SIC) ────────────┘
  Raw datasets: D:\datasets\ (read once by preprocessing)
  Preprocessed crops: WSL ext4 disk (~/data/crops), moved to D: (see 3.3)
```

### 3.1 One lock, two platforms
- `pyproject.toml` pins exact versions; `uv.lock` is committed. `torch==2.8.0` and `torchvision`
  come from the cu128 index (`[tool.uv.sources]` + `[[tool.uv.index]]`).
- Linux-only extras: `mamba-ssm` and `causal-conv1d` with the marker `sys_platform == 'linux'`.
  They're built from source with `TORCH_CUDA_ARCH_LIST=12.0` (the recipe is recorded in C-01).
- Windows: `uv sync --extra dev` → `.venv\`. WSL: `UV_PROJECT_ENVIRONMENT=~/venvs/deepfake uv sync
  --extra dev --extra gpu`. The Linux environment lives outside the shared working copy, so the two
  `.venv`s never collide, and Python imports don't go through the slow `/mnt/e` bridge.
- `requirements.txt` is kept but marked legacy (the capstone's list). The conda files stay as the
  backup.

### 3.2 Engine
If the WSL build passes (`paper_assets/C-01_wsl_attempt2.md`), `configs/fusion_ssm.yaml:operator`
becomes `mamba_cuda`. `src/models/selective_scan.py` stays as the test oracle: a unit test checks
the CUDA kernel against it to 1e-3 (fp32) on random inputs. Native Windows can't import
`mamba-ssm`, so on Windows `operator: mamba_cuda` must fail loudly, never fall back silently.

### 3.3 Data placement
- **Fact:** the WSL disk is on C: (215 GB free). D: has 456 GB free and E: 364 GB.
- **Recommendation:** move the distro to D: once (`wsl --shutdown; wsl --manage Ubuntu-22.04 --move
  D:\WSL`). Keep raw downloads on D:/E: and write preprocessed PNG crops into WSL ext4, because PNG
  decoding through `/mnt` is several times slower and decoding is the training bottleneck
  (protocol §6). This is a **hypothesis**, to be measured in P3-T3 with a 1,000-image read
  benchmark.
- Paths come from a git-ignored `configs/paths.local.yaml` (already covered by
  `configs/*.local.yaml` in `.gitignore`), never hardcoded (protocol §6).
- DFDC (≈470 GB raw) won't fit on any single drive alongside everything else. It's downloaded and
  preprocessed in parts, deleting each raw part once its crops are verified. This needs the owner's
  OK when Phase 2 starts.

## 4. Repository structure changes

| Change | Why |
|---|---|
| `notebooks/legacy/` ← the 9 capstone notebooks + root `DeepfakevFinal.ipynb`, `Run_DeepFake.ipynb` (`git mv`, contents untouched) | Separates the frozen capstone evidence from new research notebooks. Their saved outputs are evidence (they proved the leak never reached training), so they are never re-run or stripped. |
| `notebooks/research/<TaskID>_<slug>.ipynb` | One exploration notebook per research task. |
| `src/data`, `src/train`, new modules in `src/models` and `src/eval` | As already specified in `IMPLEMENTATION.md`'s target layout. Created when their task starts, not empty now. |
| `.editorconfig`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | Same gates as Candidra/Veyns (§1). |
| `Makefile` (runs in WSL) | `make setup`, `make test`, `make lint`, `make train TASK=… CONFIG=…`. |
| `docs/SETUP.md` | Step-by-step: WSL, uv, both environments, data paths, the Mamba build. |
| `docs/DECISIONS.md` | A dated, one-line-per-decision index of owner decisions, each linking to its SPEC/handoff. Today they are scattered across chat and handoffs. |
| `CONTRIBUTING.md` | Branch/commit conventions, the handoff loop, and "how to add a task" for a human teammate. |
| Notebook path fix | Legacy notebooks use `ROOT = Path.cwd().parent`; after the move they need `parent.parent`. **They are not edited.** `notebooks/legacy/README.md` says to run them from `notebooks/` only if ever needed. |

**Not adopted, and why:** Docker (no deployment target; WSL already isolates the GPU stack);
`nbstripout` (notebook outputs are evidence here); a `.env` file (no secrets in this project).

## 5. The notebook → script rule

**Root cause of "notebooks only use one worker" (Fact):** on Windows, PyTorch's DataLoader starts
worker processes with *spawn*, and each worker must re-import the Dataset class. A class defined
*inside a notebook* can't be imported, so it crashes or hangs, and `num_workers=0` "fixes" it.
The capstone notebooks hit exactly this: `train_spatial.ipynb` has "num_workers=0, # IMPORTANT: set
to 0 to avoid multiprocessing issues", and `train_temporal_clean.ipynb` has "NUM_WORKERS = 0 # keep 0
in notebooks". The scripts worked with 4 workers because their classes
live in a `.py` file.

**The rule:**
1. **All reusable code lives in `src/`**: Dataset, model, loss and metric code. Notebooks
   `import` it and never define it. That alone makes 8–12 workers work inside notebooks too. On
   WSL/Linux, workers use *fork* anyway.
2. **Notebook = exploration.** Small subsets, sanity checks, plots for intuition, trying ideas.
   A notebook number never goes in the paper.
3. **Script = the final run.** `python -m src.train.<name> --config configs/<name>.yaml`. It's
   config-driven, calls `start_run`/`log_result`/`finish_run`, writes per-video scores, and saves
   figures through `save_figure`. Only script outputs feed `paper_assets/`.
4. **One code path.** The notebook and the script call the same `src/` functions, never
   copy-pasted versions. **Fact:** the capstone had two separate copies of `AttentionPool`, both
   with the `-1e9` bug (`src/ground_truth/models.py` and `src/process_data/train_temporal.py`).
   That is what copy-paste does.

## 6. Data and preprocessing

n/a. No data changes. Paths only (§3.3).

## 7. Training methodology

n/a. No hyperparameter changes. `num_workers` 8–12 is already the protocol §6 value.

## 8. Evaluation methodology

n/a, except the parity check in §9.

## 9. Acceptance tests

1. `uv lock` resolves. `uv sync` succeeds on Windows (`.venv`) and in WSL (`~/venvs/deepfake`).
2. The full test suite passes in **both** environments, and still passes in `aiml`.
3. **Environment parity (the Level B guard):** the frozen legacy temporal model, run on the same
   200 cached `embeddings/test_celebdf` sequences in `aiml` (Windows) and in the WSL venv, gives
   video probabilities equal within 1e-5, and identical AUC. A mismatch stops adoption.
4. `mamba_cuda` vs the `selective_scan` oracle: match within 1e-3 (fp32), forward and backward.
5. A DataLoader with `num_workers=8` runs from a notebook in both environments.
6. `pre-commit run --all-files` passes (or the one-time reformat is its own commit). CI is green on
   GitHub.
7. `git mv` only for the moved notebooks, so `git log --follow` keeps their history, and their
   bytes (including outputs) are unchanged.

## 10. Expected artifacts

`uv.lock`, the updated `pyproject.toml`, `.editorconfig`, `.pre-commit-config.yaml`,
`.github/workflows/ci.yml`, `Makefile`, `docs/SETUP.md`, `docs/DECISIONS.md`, `CONTRIBUTING.md`,
`notebooks/legacy/README.md`, C-01 and C-03 addenda (the WSL build, the venv freezes), and the parity
output saved with `save_console`.

## 11. Permitted engineering optimisations (Level A)

Formatting-only reformat commits, CI caching, and a CPU-only torch wheel in CI.

## 12. Restricted changes (need the owner)

Editing legacy code or notebooks, deleting the conda env or its freezes, changing any config value
other than `operator`, moving or deleting anything in `paper_assets/`, `checkpoints/`, `embeddings/`
or `data/`.

## 13. Unresolved decisions

1. **Move the WSL disk to D:?** Recommended (§3.3). A one-time command, about 2 minutes.
2. **CI on GitHub Actions.** Free for public repos. If the repo is private, the free tier's minutes
   are ample for lint + CPU tests. Recommended: yes.
3. **Where the Jupyter server runs for GPU notebooks.** Recommended: JupyterLab inside WSL, opened
   from the Windows browser, or VS Code's WSL extension, which runs notebooks in the WSL kernel
   directly.
