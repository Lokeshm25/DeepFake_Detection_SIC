# Handoff — CODEX-ONBOARDING

**SPEC:** Owner-supplied CODEX HANDOFF INSTRUCTIONS and `RESEARCH_PROTOCOL.md` §§1, 3, 5.
Level A work requires no new SPEC (§5).
**Status:** ACCEPTED
**Change level:** A — align project instructions with the already-authorized collaboration protocol.
**Date:** 2026-11-21

## 1. Implementation plan (Codex)

1. Read the binding protocol, project instructions, roadmap, implementation coverage and acceptance
   tests, SPEC and handoff templates, and SPEC-001's status. Check Git status before editing.
2. Modify only `AGENTS.md` and create this handoff. Preserve the pre-existing changes to `.gitignore`,
   `AGENTS.md`, `CLAUDE.md`, the protocol, Claude-only skills, and templates.
3. Align instruction precedence, Codex/Claude/owner roles, Level A/B/C change control, permitted
   specialist-agent use, scientific safeguards, protected paths, and the review/acceptance loop.
4. Run document conformance and local-reference checks, `git diff --check`, and SHA-256 preservation
   checks against the pre-edit snapshot. Review the final diff and record fresh output in §3.
5. Hand over for Claude's review; respond to each finding as Fixed, Disputed, or Deferred and
   re-verify after fixes. The owner decides acceptance. No commit or push is authorized for this task.

**Review-response plan — round 1 (Level A alignment):**

1. Read Claude's four findings and the recorded owner decisions on findings 2–3. Use the updated
   `RESEARCH_PROTOCOL.md` §3 as the source of truth; preserve its new wording and Claude's review.
2. Update `AGENTS.md`'s change-control table, config-value definitions, `configs/` ownership row,
   and operator/detector guidance to distinguish fixed, provisional, and spec-written values.
   Keep owner approval before Level B runs and adoption after comparison on FF++ validation.
3. Apply finding 4 by qualifying the handoff implementation-plan step for implementation tasks.
4. Refresh the document checks for the approved rules, run `git diff --check`, and compare the
   28 protected-file hashes and preserved review/owner sections against this round's pre-edit
   snapshot. Record actual output in §3 and a disposition for each finding in §5.

**Fact:** The initial checkout is `main` at `beb49990383f07c032e68af10ffea1a0cb43893d`, with
pre-existing uncommitted work. No standalone task tracker was found in the targeted file search;
`ROADMAP.md` supplies task IDs, and SPEC/handoff records supply their review state.
**Fact:** `docs/specs/SPEC-001_run_manifest.md` says DRAFT — awaiting owner approval. Its
implementation and acceptance tests are outside this task. No research phase or capture is executed.

## 2. Deviations (Codex)

| What differs from the SPEC | Why | Level (A/B/C) | Owner decision |
|---|---|---|---|

No implementation deviations. No Level B/C changes were made. Level B/C deviations stop work
until the owner decides.

**Fact — corrections within the authorized Level A scope:**

- Replaced Claude-centric invariant authority and generic Research/Engineering roles with the
  binding protocol and its owner/Claude/Codex collaboration model (§1).
- Removed unrestricted config-edit permission. After the owner-approved review clarification,
  fixed changes require Level C approval, provisional STUB-only settings are Level B, and
  spec-written operator/detector outputs execute their approved tasks (§3). Baselines are preserved.
- Replaced mandatory automatic commits with per-task owner authorization (§5).
- Made split verification mandatory before training, with task-specific evaluation checks (§2/§4).
- Added the omitted per-video scores, validation-only selection, zero-init conditioning, and
  suspicious-AUC stop requirements (§2/§4); distinguished the mislabelled internal AUC from the
  contaminated Celeb-DF evaluation (§9).
- Replaced the unspecified tracker/commit-message handoff with the shared record, fresh output,
  explicit finding dispositions, Claude review, and owner acceptance (§5).
- Pointed capture stops to the shared protocol, including C-02's manual folder (§7).

**Resolved handoff reference:** Section H was absent from the original handoff and repository.
The owner supplied Claude's six-item checklist in this conversation on 2026-11-21. Its requirements
and Codex's implementation responses are saved in §7 so future review can use repository records.

## 3. Verification (Codex)

**Review-response round 1 — 2026-11-21.** Checks cover the fixed / provisional / spec-written
rules and the implementation-only plan requirement. The preservation baseline was captured after
Claude's owner-approved protocol clarification and before this round's Codex edits. Diff reference:
working-tree `AGENTS.md` versus `beb49990383f07c032e68af10ffea1a0cb43893d`.

**`git diff --check` — exit 0.** Actual output (line-ending notices for pre-existing edits):

```text
warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'CLAUDE.md', LF will be replaced by CRLF the next time Git touches it
```

**Read-only PowerShell document checks — exit 0.** Exact command:

```powershell
$ErrorActionPreference = 'Stop'
$agents = Get-Content -LiteralPath 'AGENTS.md' -Raw -Encoding UTF8
$flat = [regex]::Replace($agents, '\s+', ' ')
$checks = [ordered]@{
    protocol_precedence = $flat.Contains('`RESEARCH_PROTOCOL.md` is binding and wins on conflict.')
    codex_approved_specs = $flat.Contains('| Codex | Implements approved specifications:')
    classify_before_changes = $flat.Contains('Classify every change as A, B, or C before changing anything')
    bc_stop_and_owner = $flat.Contains('A Level B/C change or deviation stops implementation and goes to the owner')
    fixed_config_approval = ($flat.Contains('**Fixed** — specified in `IMPLEMENTATION.md`, or adopted by the owner after a Level B comparison. Changing it is Level C.') -and -not $flat.Contains('or a committed config'))
    provisional_config_rule = $flat.Contains('**Provisional** — present only in a config marked `STUB` and specified nowhere in `IMPLEMENTATION.md`. Setting it is Level B; it becomes fixed once the owner adopts the result.')
    spec_written_config_rule = ($flat.Contains('**Spec-written** — a value an approved task is instructed to write') -and $flat.Contains('`fusion_ssm.yaml:operator`') -and $flat.Contains('`preprocess.yaml:detector`') -and $flat.Contains('Writing it executes that approved task; changing it afterwards is Level C.'))
    config_ownership_and_guidance = ($flat.Contains('Fixed changes: Level C; provisional STUB-only settings: Level B; spec-written values: approved task execution') -and $flat.Contains('provisional STUB-only settings are Level B, with owner OK before the comparison and adoption decided afterwards.'))
    level_b_run_and_adoption = ($flat.Contains('Run only after the owner OKs the run') -and $flat.Contains('selected on FF++ validation.') -and $flat.Contains('Owner decides adoption afterwards.'))
    handoff_plan = ($flat.Contains('**For implementation tasks, open `docs/handoffs/<ID>.md`**') -and $flat.Contains('implementation plan before changes'))
    fresh_verification = $flat.Contains('paste fresh output into the record')
    finding_dispositions = ($flat.Contains('**Fixed**') -and $flat.Contains('**Disputed**') -and $flat.Contains('**Deferred**'))
    owner_acceptance_and_commits = ($flat.Contains('Only the owner approves SPECs and accepts final work') -and $flat.Contains('Commit or push only when the owner authorizes it for that task'))
    verify_before_training = $flat.Contains('`python -m tools.verify_splits` to pass before any training')
    external_data_separation = $flat.Contains('External datasets never enter training, validation, or model selection.')
    per_video_scores = $flat.Contains('video_id,dataset,true_label,pred_prob')
    seeds_zero_init_video_bootstrap = ($flat.Contains('`42, 1337, 2024`') -and $flat.Contains('Zero-init all conditioning parameters') -and $flat.Contains('Bootstrap over videos'))
    gate_and_capture_stops = ($flat.Contains('cross-dataset AUC above 0.95') -and $flat.Contains('STOP execution of the current task') -and $flat.Contains('WAIT for explicit user confirmation'))
    protected_setup_and_draft = ($flat.Contains('Playwright MCP') -and $flat.Contains('SPEC-001_run_manifest.md` is DRAFT'))
}
$failed = @($checks.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object { $_.Key })
if ($failed.Count -gt 0) { throw ('Failed documentation checks: ' + ($failed -join ', ')) }
Write-Output ('PASS: ' + $checks.Count + ' collaboration-rule checks')
$references = @('RESEARCH_PROTOCOL.md', 'CLAUDE.md', 'ROADMAP.md', 'IMPLEMENTATION.md', 'RESEARCH_ANALYSIS.md', 'docs/handoffs/_TEMPLATE.md', 'docs/specs/_TEMPLATE.md', 'docs/specs/SPEC-001_run_manifest.md')
foreach ($path in $references) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw ('Missing reference: ' + $path) } }
Write-Output ('PASS: ' + $references.Count + ' existing local document references')
foreach ($path in @('AGENTS.md', 'docs/handoffs/CODEX-ONBOARDING.md')) {
    $content = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    if ($content -match '(?m)[ \t]+\r?$' -or -not $content.EndsWith("`n")) { throw ('Whitespace or final-newline error: ' + $path) }
}
Write-Output 'PASS: changed documents have no trailing whitespace and end with a newline'
```

Actual output:

```text
PASS: 19 collaboration-rule checks
PASS: 8 existing local document references
PASS: changed documents have no trailing whitespace and end with a newline
```

**Preservation check.** The command below inventories the same 28 protected files and captures
Claude's §4 review (including owner decisions and fix instructions) and §6 owner-decision text.
`functions.exec` compares before/after file paths and SHA-256 hashes, review/owner text (normalising
line endings), `paper_assets/` existence, HEAD, and Git status. Exact inventory command:

```powershell
$protectedPaths = @('.gitignore', 'CLAUDE.md', 'ROADMAP.md', 'IMPLEMENTATION.md', 'RESEARCH_PROTOCOL.md', 'RESEARCH_ANALYSIS.md', 'docs/handoffs/_TEMPLATE.md')
$protectedFiles = @($protectedPaths | ForEach-Object { Get-Item -LiteralPath $_ -ErrorAction Stop })
$protectedFiles += @(Get-ChildItem -LiteralPath 'configs', 'src/ground_truth', 'src/process_data', 'docs/specs', '.claude' -Recurse -File)
$records = @($protectedFiles | Sort-Object FullName -Unique | ForEach-Object { [pscustomobject]@{ path = $_.FullName; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
$handoff = Get-Content -LiteralPath 'docs/handoffs/CODEX-ONBOARDING.md' -Raw -Encoding UTF8
$reviewSection = [regex]::Match($handoff, '(?ms)^## 4\. Review.*?(?=^## 5\. Response)').Value
$ownerSection = [regex]::Match($handoff, '(?ms)^## 6\. Owner decision.*?(?=^## 7\.)').Value
if (-not $reviewSection -or -not $ownerSection) { throw 'Missing review or owner-decision section' }
[pscustomobject]@{ protected_files = $records; review_section = $reviewSection; owner_section = $ownerSection; paper_assets_exists = (Test-Path -LiteralPath 'paper_assets'); head = (git rev-parse HEAD); status = @(git status --porcelain=v1 --untracked-files=all) } | ConvertTo-Json -Depth 4
```

Actual comparison output:

```text
PASS: 28/28 protected file hashes unchanged
PASS: Claude review and owner-decision text preserved
PASS: paper_assets/ remains absent
PASS: HEAD unchanged (beb49990383f07c032e68af10ffea1a0cb43893d)
PASS: existing Git status entries preserved; no new paths
```

**Validation scope:** Documentation only; no Python or config values changed. Training, evaluation,
pytest, Ruff, and type checks are outside this response. No research phase acceptance or capture
is claimed. `tools/` and `tests/` are not present in the inspected file listing.

## 4. Review (Claude) — round 1

**Diff reviewed:** working-tree `AGENTS.md` vs `beb4999`, against `RESEARCH_PROTOCOL.md` and
Section H (§7 of this record). Date: 2026-11-21.
**Verdict:** CONFORMANT WITH FINDINGS. All six Section H items are addressed (Fact: checked each
against `AGENTS.md`). Claude's files were not modified (Fact: their mtimes predate this task).

| # | Severity | Category | Location | Finding (Fact/Hypothesis) | Evidence | Suggested fix |
|---|---|---|---|---|---|---|
| 1 | major | methodology / spec-drift | `AGENTS.md:184`, `:215-216`; root cause also in `RESEARCH_PROTOCOL.md` §3 (Claude's wording) | **Fact.** "Any `configs/*.yaml` value change is Level C" and "Changing [operator's] committed value requires Level C" would block config writes that approved specs *require*: P0-T1 must write `operator`, P3-T1 must set `detector`. | `IMPLEMENTATION.md:119` ("Write the result to `configs/fusion_ssm.yaml`"), `:440` (`detector: null # REQUIRED — set from the P3-T1 result`) | Writing a value that an approved task's spec says to produce is executing that spec (within the task's scope), not a change. Claude corrects the protocol wording; Codex mirrors it in `AGENTS.md`. |
| 2 | major | methodology — **needs owner decision** | `AGENTS.md:184`; `RESEARCH_PROTOCOL.md` §3; `configs/fusion_ssm.yaml` `training:` | **Fact.** Some committed STUB values are specified nowhere in `IMPLEMENTATION.md` or `ROADMAP.md` — e.g. `fusion_ssm.yaml` lr `5.0e-5`, batch 16, grad accumulation 2, warmup 3. Under the current rule they become frozen Level C values although nobody chose them on evidence. | `grep` for these values in `IMPLEMENTATION.md`/`ROADMAP.md` returns nothing; the four STUB configs say "values finalised during Phase N" | Recommended: a value that appears only in a STUB config and not in `IMPLEMENTATION.md` is **provisional**. Setting it is Level B (controlled comparison on FF++ val), and it becomes fixed once the owner adopts it. |
| 3 | minor | spec-drift | `AGENTS.md:82` vs `RESEARCH_PROTOCOL.md` §3 | **Fact.** `AGENTS.md` makes Codex stop for an owner decision *before running* a Level B comparison; the protocol only requires the owner for *adoption*. Stricter, not unsafe. | the two table rows | Recommended: keep the stricter `AGENTS.md` version (a comparison costs GPU-hours across 3 seeds) and align the protocol to it, so both files say the same thing. Owner's call. |
| 4 | nit | engineering | `AGENTS.md:54`, `:65-66` | **Fact.** "Mandatory First Steps for ANY Agent" step 5 (open a handoff, write an implementation plan) only fits implementation tasks, not e.g. a literature review. | heading vs step 5 | Reword to "for implementation tasks". |

**Owner decisions on findings (2026-11-21, given to Claude in conversation):**
- Finding 2 → stub values found only in a `STUB` config are **provisional**; setting them is Level B.
- Finding 3 → **owner OKs a Level B run before it starts**, and decides adoption afterwards.

**Claude's fix (done):** `RESEARCH_PROTOCOL.md` §3 now defines fixed / provisional / spec-written
config values and the Level B "OK before running" rule (findings 1–3). This is a pure alignment with
owner decisions, reported here per §3 "Documents".

**For Codex:** mirror §3 in `AGENTS.md`. Line 184 and lines 215-216 should distinguish
fixed / provisional / spec-written values, keeping P0-T1's `operator` write and P3-T1's `detector`
write inside their tasks. Line 82 already matches. Apply finding 4. Then re-verify and respond in §5.

## 5. Response (Codex) — round 1

| # | Fixed / Disputed / Deferred | Note or evidence |
|---|---|---|
| 1 | Fixed | **Fact:** `AGENTS.md:92`, `:195`, and `:227` distinguish spec-written values. P0-T1's operator write and P3-T1's detector write execute their approved tasks; changing them afterwards is Level C. This mirrors `RESEARCH_PROTOCOL.md` §3 and the required outputs in `IMPLEMENTATION.md:119` and `:440`. |
| 2 | Fixed | **Fact:** `AGENTS.md:88` and `:90` mirror the owner's fixed/provisional decision: a STUB-only value specified nowhere in `IMPLEMENTATION.md` is provisional (setting it is Level B); owner adoption makes it fixed, and fixed changes are Level C. No config value was edited. |
| 3 | Fixed | **Fact:** `AGENTS.md:83` retains and explicitly states owner OK before a Level B run, unchanged baseline, same seeds/splits, one variable, FF++ validation selection, and adoption decided afterwards. This matches the owner's decision and the updated protocol. No comparison was run. |
| 4 | Fixed | **Fact:** `AGENTS.md:65` now starts "For implementation tasks"; the handoff implementation-plan step is qualified by task type. |

Fresh verification output is recorded in §3: 19 collaboration-rule checks and eight existing
document references passed; all 28 protected-file hashes and Claude's review/owner-decision text
were preserved. No findings are disputed or deferred. The response is ready for Claude's re-review;
owner acceptance remains pending in §6.

## 4b. Review (Claude) — round 2 (2026-11-22)

| # | Verdict | Evidence |
|---|---|---|
| 1 | Resolved | **Fact:** `AGENTS.md:92-94`, `:195`, `:226-230` treat P0-T1 `operator` and P3-T1 `detector` as spec-written; matches `RESEARCH_PROTOCOL.md` §3. |
| 2 | Resolved | **Fact:** `AGENTS.md:86-91` reproduce the fixed/provisional definitions; `:83` puts provisional values under Level B. |
| 3 | Resolved | **Fact:** `AGENTS.md:83` requires owner OK before the run and leaves adoption to the owner. |
| 4 | Resolved | **Fact:** `AGENTS.md:65` begins "For implementation tasks". |

No config, legacy code or `paper_assets/` file is modified (`git status`). `git diff --check` is clean.
No new findings.

**Verdict: CONFORMANT.** Ready for owner acceptance (§6).

## 6. Owner decision

**Accepted** by the owner, 2026-11-22. Commit authorised together with the SPEC-001 approval.

## 7. Owner-supplied Section H — requirements and Codex responses

Source: Claude's collaboration-setup handoff, 2026-11-21, pasted by the owner after Codex requested
the missing section. These are implementation responses to the setup requirements, not a verdict
from Claude on the resulting diff.

| # | Requirement from Section H | Fixed / Disputed / Deferred | Implementation evidence in `AGENTS.md` |
|---|---|---|---|
| 1 | Replace generic Research/Engineering roles and Purpose with owner/Claude/Codex; Antigravity inactive; update delegation and ownership. | Fixed | Purpose, Agent Roles, Scientific Change Control, and File Ownership now follow protocol §§1/3. |
| 2 | Replace automatic commits and "never leave uncommitted results" with the shared handoff loop and per-task commit/push authorization. | Fixed | Handoff Protocol Between Agents uses the shared template, Claude review, owner acceptance, and explicit commit/push authorization. |
| 3 | No task tracker exists; use `docs/handoffs/` or remove the reference. | Fixed | Mandatory First Steps points to SPEC state and per-task handoff records; no generic tracker reference remains. |
| 4 | Replace stale Claude-specific invariant/capture pointers with the shared protocol, especially §7. | Fixed | Purpose and Mandatory First Steps name the binding protocol; Safety Rails and STOP-AND-WAIT point directly to §7. |
| 5 | Include `RESEARCH_PROTOCOL.md` and `RESEARCH_ANALYSIS.md` in document change control, alongside the roadmap/implementation; pure corrections must still be reported. | Fixed | Scientific Change Control lists all four shared research documents and the reportable pure-correction exception; Claude-specific edits require explicit authorization. |
| 6 | Qualify all "no approval" implementation/config permissions by A/B/C, especially values fixed in a spec or committed config. | Fixed | Scientific Change Control and File Ownership apply the owner-approved fixed / provisional / spec-written rule: fixed changes are Level C, provisional STUB-only settings Level B, and operator/detector outputs remain within approved task execution. |

The additional requested rules are also recorded: binding protocol precedence; classification
before changes and B/C escalation; Fact/Hypothesis/Assumption labels; and Claude-only skills
remaining unchanged, without loading or copying them into Codex. All future material meant for
another agent should be saved in its repository handoff record, rather than left only in chat.
