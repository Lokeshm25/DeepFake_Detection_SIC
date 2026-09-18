# Research Context Files — Index

## Purpose

This directory contains the original AI-generated research advice and analysis documents that
informed the project's research direction. These are **reference materials, not authoritative
sources** — see `RESEARCH_ANALYSIS.md` §3.1 for known sourcing issues.

---

## Files

| File | Source | Quality Notes |
|---|---|---|
| `Chat with Google AI mode (Pro).txt` | Google AI Chat | ⚠️ Cites an Instagram reel as hardware source, contains wrong LSTM vs Transformer claims |
| `Claude Critic and Analysis on Previous Research Decision.txt` | Claude | ✅ Rigorous novelty audit — identifies CAST as key competitor |
| `Previously Decided Research Decision (Gemini and GPT).txt` | Gemini + GPT | Original "Calibrated Tri-Stream" blueprint — superseded by SSM fusion approach |
| `Prompts I used for the different researches.txt` | User | Prompts used to generate the other documents |
| `Latest Deepfake Architectural Novelty Research (By Gemini).md` | Gemini | ⚠️ Reference list contains non-research sources (university assignment, pharmaceutical proceedings) |
| `Latest Research on Architectural and Theoretical Advancements... (BY NotebookLM).md` | NotebookLM | ⚠️ Contains self-labelled "Simulated/Target" numbers. arXiv IDs verified as real (Sept 2026). |

---

## How to Use These

1. **NEVER cite directly** from these documents — verify every claim against the primary source
2. Use as **leads** for literature search — the arXiv IDs are verified real
3. The Claude critique (`Claude Critic...`) is the most rigorous and informed the research pivot
4. The superseded "Calibrated Tri-Stream" direction is documented for provenance

---

## Key Decision Trail

1. **Original direction** (Gemini + GPT): Calibrated tri-stream with cross-attentive fusion
2. **Claude critique**: Showed this is compositional, not novel — CAST owns spatial-temporal cross-attention
3. **Research pivot**: Reliability-conditioned selective fusion via SSM Δ-conditioning
4. **Current direction**: Documented in `RESEARCH_ANALYSIS.md` §6 and `ROADMAP.md`
