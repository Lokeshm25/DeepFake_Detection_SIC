---
name: deepfake-literature-review
description: Review, compare and verify deepfake-detection literature for this project — paper analysis, evidence matrices, related-work positioning, research gaps, and citation/claim verification against arXiv/DOI (invariant 9, capture C-02). Use whenever a paper, citation, reported benchmark number, or "is this novel / does X overlap our method" question comes up.
---

# Deepfake literature review

Binding rules: `RESEARCH_PROTOCOL.md` (invariant 9, §1 claim labels, §7 capture protocol).
Literature position of record: `RESEARCH_ANALYSIS.md` §3. Changing that position is Level C.

This project already has fabricated and misattributed citations in its source material, and one
"correction" that flipped an AUC and an accuracy. Treat every secondary summary — NotebookLM,
Gemini, ChatGPT, Google AI mode, blog posts, and earlier drafts of our own docs — as a **lead to
check, never as evidence**.

## 1. Verify before using

For each cited work, open the primary source yourself (WebSearch / web-fetch on the arXiv abs page,
DOI resolver, or publisher page) and record:

| Field | Note |
|---|---|
| ID | arXiv `YYMM.nnnnn` or DOI |
| Title, authors, date | exactly as on the page — authorship errors are common here |
| Venue | "preprint" unless a published version is found |
| Verdict | `verified` / `mismatch` (say what differs) / `not found` |
| Checked | date, URL |

- Do not trust a number until you have read it in the paper itself, in context (table + caption).
  Note whether it is AUC, accuracy, AP or EER, frame- or video-level.
- Reject non-research sources (course pages, social media, predatory venues) as evidence.
- Context7 is for library docs, not papers — don't use it here.

**C-02 is `[HUMAN-IN-LOOP]` with a `[MANUAL]` screenshot folder.** You may draft rows for
`paper_assets/C-02_citation_audit.md`, but only the owner confirms verdicts and takes the dated arXiv
screenshots in `paper_assets/C-02_citation_audit/`. When you reach that point, follow STOP-AND-WAIT
(`RESEARCH_PROTOCOL.md` §7): finish the draft, stop, and tell the owner exactly which IDs need a
screenshot and a verdict. Never mark a row verified on the owner's behalf.

## 2. Protocol-match before comparing numbers

A cross-dataset number is comparable to ours only if the protocol matches. For each reported result
record: training data (FF++ only? which compression? any eval data in training?), test set and
version, frame vs video level, metric, and whether code/weights are public. A number from a
mismatched protocol goes in the matrix flagged, never as "the bar to beat". Cross-dataset AUC above
~0.95 from an FF++-trained model is a red flag to explain, not a target.

## 3. Evidence matrix

Produce one row per paper:

`paper | ID | verified? | mechanism (1 line) | train data | test sets | metric/level | reported numbers | code? | overlap with our mechanism | gap it leaves`

**Overlap analysis** compares against our actual claim: the SSM selection parameter Δ (and B/C) is
conditioned on a self-supervised, frame-level degradation descriptor through a per-stream zero-init
response, over interleaved spatial/frequency/temporal tokens (`IMPLEMENTATION.md` §P5-T3). Read the
method section, not just the abstract — the Aletheia overlap was missed once by stopping at the
abstract.

## 4. Gaps and positioning

State each gap as **Fact** (no verified paper does X — list what was searched), **Hypothesis**
(X would help because...), or **Assumption**. Say how a reviewer would attack the claimed gap.

## 5. Output

- Working notes and matrices: `docs/research_context/<topic>_review.md` (create the file; mark it
  DRAFT and dated).
- Proposed changes to `RESEARCH_ANALYSIS.md` are a Level C SPEC or an explicit request to the owner,
  not a direct edit.
- End with: what was verified, what is still unverified, and what the owner must confirm.
