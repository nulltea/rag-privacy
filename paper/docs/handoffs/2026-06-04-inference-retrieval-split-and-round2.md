---
type: handoff
status: current
created: 2026-06-04
updated: 2026-06-04
tags: [sok, refactor, retrieval, cryptographic-inference, round2-review, confidential-llm, rag]
companion: [scheme-categorization, survey-corpus]
---

# Handoff — Confidential LLM/RAG SoK: inference/retrieval split + round-2 review

**Repo:** `rag-privacy`, branch `paper` (pushed to origin/`paper`; NOT merged to `master`).
**Tree:** clean, all committed. **Build:** `cd paper/manuscript && make` → 27 pp, 0 undefined refs.

Reference-only; does not duplicate the plans/handoffs/commits below — read those.

## This session's arc (commits, newest first)
- `311933e` — **structural refactor**: split crypto inference (§3) vs retrieval (new §8). Read its message.
- `cbff63f` — hardened `address-comments` skill (Completeness Contract) + round-2 audit.
- `c820aae` — client-reliance axis (§2 trust-anchor + §9 finding) + 2 follow-up handoffs.
- `7741448` — §3 round-2 PDF-comment fixes + CryptoMoE/Yona routing attack + Li-et-al adoptions.
- `930c54a`, `f9c9380` — §3 crypto prose finalization + attack investigation + curation propagation.

## Current section spine (post-refactor — NOTE the renumber)
§1 intro · §2 framework (`sec:framework`) · **§3 Cryptographic INFERENCE** (`sec:crypto`) ·
§4 TEE (`sec:tee`) · §5 obfuscation · §6 DP · §7 hybrid-split · **§8 Confidential Retrieval &
Storage** (`sec:retrieval`; §8.2 dense/naive-RAG, §8.3 graph/LightRAG, §8.4 limits) ·
**§9 synthesis** (`sec:synthesis`, was §8) · §10 related (was §9) · §11 conclusion (was §10) · App.
Cross-refs are label-based and resolve; only *verbal* "§8 = synthesis" references are now stale.

## Decision records / artifacts — read before touching related cells
- Plans (gitignored, local): `paper/quality_reports/plans/2026-06-04_retrieval-section-split.md`,
  `..._client-reliance.md`, `..._section3-prose-and-attack-investigation.md`.
- Attack investigation: `docs/research/crypto-attack-surface.md` (property→attack→scheme + transitivity).
- PDF-comment audit (33 rows, all resolved + reconciliation note): `paper/quality_reports/pdf-review-round2.md`.
- Curation truth: `paper/manuscript/survey-corpus.md`; corrections-of-record =
  `docs/research/scheme-categorization.md` **Part E** (read before changing scheme cells).

## Open follow-ups (next-session candidates)
1. **Client-reliance sweep** — classify ~50 schemes thin/stateful/thick; fill `\tq` cells in §9
   `tab:deployment`. Brief: `docs/handoffs/2026-06-04-client-reliance-sweep.md`.
2. **Δacc sweep** — extend Li-et-al plain/enc/loss (now in §3 `tab:crypto-inf`) to non-crypto families.
   Brief: `docs/handoffs/2026-06-04-delta-accuracy-sweep.md`.
   *(NOTE: those two task-handoffs were committed under the REPO-ROOT `docs/handoffs/`, not here under
   `paper/docs/handoffs/` where session handoffs live. Consider relocating them for consistency.)*
3. Deferred deep prose / `\needswork` remain in several sections (`grep -rn needswork`).
4. Pre-submission: `/validate-bib` + `/verify-claims` on refs.bib TODO fields; confirm author lists
   flagged in `crypto-attack-surface.md` §6 (ArrowMatch, arXiv 2602.11088, CryptoMoE order).

## Gotchas (will bite you)
- **cleveref NOT installed** → `\autoref` fallback is ACTIVE; multi-arg `\cref{a,b}` → one undefined ref.
  Use `\cref{a} and~\cref{b}`.
- **`make` overwrites the annotated `main.pdf`** → with `/address-comments`, EXTRACT first, then build.
  Backups: `paper/quality_reports/annotated-pdfs/` (gitignored).
- **`quality_score.py` reports 0/100** for any section — false positive (hardcodes `Bibliography_base.bib`;
  manuscript uses `refs.bib`). Real gate = bibtex undefined-count in `main.log`.
- **EdgeQuake**: one aspect per `query`; many aspects of one doc → `document_get_md`. PDF→MD drops
  table numbers to placeholders (Part E #20).
- **Commit convention**: work lives on `paper`, never merged to `master`; `/commit`'s new-branch +
  merge-to-main steps do NOT apply — commit directly on `paper`.
- Pre-existing cosmetic ~43 pt overfull at `tab:refmetrics` (§2) — ignore.

## Suggested skills (next session)
- `/compile-latex` — after any edit (pdflatex+bibtex via manuscript Makefile; NOT xelatex/biber).
- `/address-comments` — if the PDF is re-annotated (now has the hardened reconciliation gate).
- `/verify-claims` — pre-submission, on new numeric/citation claims (esp. §8 retrieval, Li-et-al Δacc).
- `/commit` — to ship (on `paper`, no master merge).
- `/grill-me` — user prefers grilling via AskUserQuestion selectors before non-trivial changes.

## User working preferences (observed)
Grill via selectable options, not open prose. One clear next action per turn. Flag uncertainty.
Remind to commit at session end. Honest status (no "all done" without proof). Don't bundle EdgeQuake
queries. Never silently drop reviewer comments.
