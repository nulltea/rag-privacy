---
type: handoff
status: current
created: 2026-06-02
updated: 2026-06-02
tags: [sok, confidential-llm-rag, evaluation-framework, scope, corpus-evaluation]
companion: [2026-06-01-confidential-llm-rag-sok-corpus-finalized]
---

# Handoff — SoK: Confidential Transformer Inference & RAG

**This session** overhauled the paper's *scope thesis* and *framework structure*.
**Next session** applies that framework to the corpus: fill the evaluation tables with
real, normalized per-scheme numbers and codings.

Branch **`paper`** (local, not pushed) in `/Users/timofey/repos/rag-privacy`. Paper in
`paper/manuscript/`. Build: `cd paper/manuscript && make` (pdflatex+bibtex, NO biber).
Currently **clean: 19pp, 0 undefined refs/cites**. Do NOT use `/compile-latex` (that's Beamer).

## Don't re-derive — read these
- **Session state / all locked decisions:** `~/.claude/primer.md` (rewritten this session; the
  top blocks cover deployment-readiness thesis, hardware tiers, taxonomy refactor, unified §2).
- **Curation source of truth:** `paper/manuscript/survey-corpus.md` (~56 cited schemes; threat/perf
  columns; scope notes #1–#3; the [x] set is the cite list).
- **This session's 4 commits** (`git log --oneline -6`): `1657807` corpus+refs · `3e90512` table/scope
  alignment · `d3467cd` deployment-readiness + tiers · `fbd64fd` unified Evaluation Framework. Read the
  commit bodies for the rationale of each change.
- **Earlier plan:** `paper/quality_reports/plans/2026-06-02_manuscript-sok-alignment.md` (Steps 1–6, done; gitignored).
- **Prior handoff (corpus finalized):** `docs/handoffs/2026-06-01-confidential-llm-rag-sok-corpus-finalized.md`.

## What changed this session (scope + structure overhaul)
1. **Corpus re-audit** (the "PACMANN" thread): added ~12 in-scope schemes — graph-RAG cluster
   (PACMANN, ZKGraph, XorMM/FLASH, H₂O₂RAM, PeGraph), SHAFT, TransLinkGuard, Portcullis,
   DP-RAG/DP-KSA, "Depth Gives a False Sense of Privacy", ZKIFV. Graph-RAG re-balanced to LightRAG's
   real retrieval surface (GNN / graph-analytics demoted to mention-only; GORAM = 3PC reserve).
2. **Thesis reframe → DEPLOYMENT READINESS**: the SoK no longer champions a family. Organizing lens =
   likelihood of real deployment on 3 criteria: **performance · fidelity · threat-model fit** (fit =
   *appropriate*, not maximal). "protagonist/central focus" jargon removed everywhere.
3. **Hardware-trust tiers**: Tier 1 none · Tier 2 CPU-TEE (commodity GPU untrusted) · Tier 3 confidential
   accelerator. Focus Tiers 1–2; Tier 3 = baseline, deferred to the VERIFIED accelerator-TEE SoK
   (`accel_tee_sok`, NDSS 2026, Zhang et al.).
4. **Taxonomy refactor**: 6→5 categorical axes — Adversary behavior · Trust anchor · Attacker knowledge ·
   Confidentiality scope (merged observation+protection) · **Security basis** (crypto/IT/DP/hardware/
   heuristic — the NEW axis separating provable from heuristic schemes). Side-channel demoted axis→finding.
5. **Unified §2 "Evaluation Framework: Security and Cost"**: §2.1 "Taxonomy: Threat Model and Security"
   (5 axes + landscape table + capability budget) · §2.2 "Cost: Performance and Fidelity" (normalization +
   use-case grouping + `tab:yardstick`). Comparisons grouped **by use case** (generation/embedding/
   reranking/retrieval), not by family. Deleted old `02_threat_model.tex` + `02b_comparison_method.tex`.

## NEXT SESSION — Survey-Corpus Evaluation per the Evaluation Framework
The framework scaffolds are built; the **data cells are `\needswork`**. The task is to populate them
from the corpus, normalized per the framework. Concretely:

1. **Code every [x] scheme on the 5 security axes** (§2.1) → the per-scheme appendix table
   (`app:classification` in `99_appendix.tex`, currently a stub) + verify the representative-row
   `tab:threat-taxonomy` cells. Assign each a **Security basis** value.
2. **Fill the per-use-case performance table** (`tab:performance` in `08_synthesis.tex`) with normalized
   figures per `tab:yardstick`: ×-overhead-vs-plaintext where reported, else regime-bucketed; flag each
   head-to-head vs self-reported. Pull numbers from `survey-corpus.md` Perf column + source PDFs.
3. **Fill fidelity** as Δ-vs-plaintext (or "exact") per use case.
4. **Justify the deployment-readiness scores** (`tab:deployment` H/M/L cells) from the filled numbers.
5. Carry the **non-comparability caveat**; nothing cross-paper is head-to-head unless marked.

Keep the salami partition: the SoK claims only the AloePri-breaks *finding* (the vignette); attack
method + the hybrid fix stay in the deferred companion paper.

## Known open items (flag at draft time)
- `refs.bib`: all non-★ entries TODO-marked (years/venues/IDs UNVERIFIED). Run `/validate-bib` +
  `/verify-claims` before submission. The 24 ★ EdgeQuake-indexed entries have verified authors/titles.
- §2.1 side-channel finding needs cited evidence (Spectre/Foreshadow/PLATYPUS + TEE side-channel surveys).
- Vignette headline recovery number is a companion-paper dependency (still `\needswork`).
- Intro Motivation / Why-a-New-SoK / Roadmap subsections are still `\needswork` stubs.
- ~48+ `\needswork` markers remain BY DESIGN (deep per-family Approach/Performance/Limitations bodies).
- Subtitle still "…and the Performance Frontier" — open question whether to broaden to "Deployment Frontier".

## Working preferences (memory)
- grill-me sessions use AskUserQuestion selectors (`feedback-grill-me-use-selectors`).
- Section/subsection titles spell out "and", never "&" (`feedback-spell-out-and-in-titles`).
- Don't over-engineer simple cleanups (`feedback-no-overengineering-simple-cleanup`).
- Bash cwd resets to `paper/` between calls — `cd paper/manuscript` before `make`.
- Commits go to the local `paper` branch directly (matching prior commits); not pushed, no PR, unless asked.

## Suggested skills
- **`/audit-reproducibility`** or a structured data-fill pass — for populating the normalized
  performance/fidelity tables from the corpus + source PDFs.
- **`/validate-bib`** — once refs are being cited heavily; cross-check + flag TODO metadata.
- **`/verify-claims`** — CoVe-check the perf/fidelity numbers and citation metadata (many web-sourced).
- **EdgeQuake MCP** (`document_get_md`, `query`) — pull exact numbers from the 24 ★-indexed PDFs; +OpenAlex
  for metadata verification.
- **`grill-me`** — for any remaining framework/structure forks before drafting.
- **`/review-paper`** — later, once the evaluation tables are filled and prose drafted.
