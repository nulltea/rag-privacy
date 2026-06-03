---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-01
tags: [sok, confidential-llm-rag, survey-corpus, scope]
---

# Handoff — SoK: Confidential Transformer Inference & RAG

**Focus for next session:** survey corpus is finalized; scope/focus/families are set.
Move from *curation* into *drafting* (bib + synthesis tables + first mechanism section).

## Where things live
- **Repo:** `/Users/timofey/repos/rag-privacy` (parent), branch **`paper`**, remote `nulltea/rag-privacy`.
  Paper lives in `paper/manuscript/`. (Commits are **local — not pushed**; the old `paper/.git`
  pointed at an unpushable upstream and was removed, folding `paper/` into the parent repo.)
- **Manuscript:** `paper/manuscript/` — build with `cd paper/manuscript && make`
  (pdflatex + bibtex; **no biber/latexmk**; cleveref is guarded → `\autoref` fallback).
  Builds clean (10 pp). **Do NOT use `/compile-latex`** (that's for Beamer/`Slides/`).
- **Curation artifact:** `paper/manuscript/survey-corpus.md` — the ~45-scheme selected set,
  threat/perf columns, foundational/superseded section, scope block. **This is finalized.**
- **Decision record (gitignored, local):** `paper/quality_reports/specs/2026-05-29_confidential-llm-rag-stage1.md`
  — read the **PIVOT block** + **Scope corrections** section for all locked decisions.
- **Session state:** `~/.claude/primer.md`.
- Recent commits (read these instead of re-deriving): `git log --oneline -6` →
  `f425f8b` (final selections), `8c4cc65` (scope corrections + §7 reframe),
  `300b10e` (retitle + corpus), `007c3b9` (pivot to SoK), `c996840` (§2 taxonomy).

## What the paper IS (don't re-litigate — all decided)
- **Pure SoK**, genre = *mechanism × threat-model × performance*. Title: *"SoK: Confidential
  Transformer Inference and Retrieval-Augmented Generation — Mechanisms, Threat Models, and
  the Performance Frontier."*
- **Positioning:** complements **Bodea et al. SoK** (TUM 2025 — risk-centric, naive-RAG, no
  perf axis, excludes graph-RAG). We fill those 3 gaps.
- **Two scope corrections (latest):** (1) comparison tables hold only *practical/SOTA* schemes;
  foundational/superseded → lineage mention only. (2) *commodity hardware* — **no confidential
  GPU (H100/B200)**; pure-TEE is the **baseline**, **hybrid-split is the protagonist**.
  Framing line: *"confidential transformer inference on commodity GPUs."*
- **Deferred companion paper** owns: the unified Static(AloePri)+Dynamic(GELO/TwinShield)
  design, the ArrowMatch/Sequence-IMA attack *methodology*, GELO optimizations, TEE-split
  measurements. **Salami rule:** the SoK claims only the *finding* (the AloePri-breaks vignette);
  the companion claims the method+fix.

## ⚠️ Section-numbering gotcha (manuscript ≠ corpus)
The two files number sections differently:
- **`survey-corpus.md`:** §1 MPC/SS · §2 FHE/HE · §3 TEE · §4 Obfusc · §5 DP · §6 Hybrid ·
  §7 Targeted-Verification · §8 Attacks · §9 Surveys · Foundational · §10 Supporting · §11 Commercial.
- **Manuscript (`main.tex` \input order):** §1 Intro · §2 Threat-model taxonomy (**drafted**) ·
  §3 Cryptographic (MPC **+** FHE combined) · §4 TEE (Baseline) · §5 Obfuscation · §6 DP ·
  §7 Hybrid split · §8 Synthesis (matrix+gap-map+perf+graph-RAG) · §9 Related work · §10 Conclusion.
- Mapping: corpus §1+§2 → manuscript §3; corpus §3→§4; §4→§5; §5→§6; §6→§7; corpus §7
  (verification) is **cross-cutting** in the manuscript (threat-model integrity axis + §8), not its own family section.

## Manuscript status
- Only **§2 (threat-model taxonomy)** is written as real prose. All other sections are
  scaffolded stubs with `\needswork{}`/`\dnote{}` markers (grep them).
- `refs.bib`: ~14 seed entries, **TODO-marked unverified metadata** (years/venues/IDs). Many
  more needed from the ~45 checked schemes.

## Next steps (highest-leverage first)
1. **Generate `refs.bib`** from the `[x]` schemes in `survey-corpus.md` (+ attacks + the 2
   surveys) — verify TODO-marked metadata against EdgeQuake/OpenAlex.
2. **Build §8 synthesis**: the stage×mechanism matrix → gap-map + the **performance table**
   (normalize by regime — BERT-encoder / 7B-decoder / ANN@corpus — NOT one flat column;
   carry the explicit non-comparability caveat).
3. **Draft the protagonist section** (manuscript §7 hybrid split) — best-grounded family.

## Known open items (deferred by the user, flag at draft time)
- **§2 side-channel finding lacks cited evidence** — add TEE.Fail / WeSee (in corpus §8, unchecked).
- **Positioning surveys** uncited — Private-Transformer-Inference Survey (2412.08145),
  SoK:Accelerator-TEE (both in corpus §9, unchecked).
- **Reranking** is thin — report as a *gap-map finding* (matches Bodea's own gap); cite HR+QDA.
- **Verify `\tq` cells** in the §2 threat table (Petridish, Compass).
- Replace every `\needswork`/TODO before submission; verify all `refs.bib` TODO fields.
- Venue: arXiv first → trim to **PoPETs** (rolling CFP).

## Working preferences / tooling
- **grill-me sessions use AskUserQuestion selectors** (not open prose) — memory
  `feedback-grill-me-use-selectors`.
- EdgeQuake MCP: docs in workspace **`rag-privacy`**; the 24 indexed papers in the **default**
  workspace. OpenAlex + WebSearch/WebFetch for citation metadata verification.
- Parallel sub-agents (general-purpose) worked well for fan-out extraction across the
  `../docs/research/*.md` files.

## Suggested skills
- **`/validate-bib`** — once `refs.bib` is populated, cross-check against citations.
- **`/verify-claims`** — CoVe-check the perf numbers + citation metadata (lots are
  TODO-marked / web-sourced and unverified).
- **`grill-me`** — for any remaining planning forks (e.g., §8 table structure).
- **`/review-paper`** — later, once sections are drafted (mechanism-primary SoK).
- Reach for **`lit-review`** only if expanding a family beyond the curated set.
