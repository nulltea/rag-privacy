---
type: handoff
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [sok, scheme-categorization, cost-tables, edgequake, confidential-llm, rag]
companion: [scheme-categorization]
---

# Handoff — Confidential LLM/RAG SoK: scheme categorization + cost tables

**Repo:** this one (`rag-privacy`), branch `paper` (not pushed).
**Primary artifact:** `docs/research/scheme-categorization.md` (repo-root `docs/`, NOT the
gitignored `paper/docs/`).
**Companion repo:** `/Users/timofey/repos/edgequake` (branch `edgequake-main`) — bug reports only.

Read first: `~/.claude/primer.md`, the project memory index
`~/.claude/projects/-Users-timofey-repos-rag-privacy-paper/memory/MEMORY.md`, and the commits
below. Don't re-derive what's already in them.

## What this work is

Mapping every **approved (`[x]`) scheme** in `paper/manuscript/survey-corpus.md` onto the §2
evaluation framework (`paper/manuscript/sections/02_evaluation_framework.tex`). The doc has:
- **Part A** — security per mechanism family (7 §2.1 axes, one row/scheme).
- **Part B** — cost per use case, **8 columns**: Scheme · Target model(s) · Hardware ·
  ×-overhead vs plaintext · Comm (absolute) · Preprocessing/offline · Fidelity · Flag.
  Use cases: Generation · Embedding (encoder) · Vector retrieval · Graph retrieval.
  Reranking dropped (no scheme reports standalone private-rerank perf → open gap, §8).
- **Part C** — §7 composable integrity (separate table).
- **Part D** — upload tracker (`Uploaded [x]/[ ]` = in EdgeQuake) + PDF links.
- **Part E** — 20 numbered findings: source-vs-corpus corrections, scope drops, CoVe + hardware
  + table-dropping discoveries. **Read Part E before changing anything** — it records why cells
  are the way they are.

## State (all committed)

Commit chain on `paper`: `49d916f` (baseline) → `941a148` → `478e80b` → `4940c32` → `4f0c2fe` (HEAD).
EdgeQuake `edgequake-main`: `bd3b3b69`/`b9607a74` (query-recall bug), `cc4b2371` (table-dropping bug).
Manuscript builds clean (`cd paper/manuscript && make` → 20pp, 0 undefined refs/cites) as of `478e80b`.

Scope decisions locked (Part E): dropped **GraSS** (#1, 83 h@1M non-deployable), **Amulet** (#3)
and **SecureInfer** (#15) (on-device model-IP — inverted threat model, out of scope; §1 Scope
prose updated), **SIGMA** (#16, superseded by SHAFT), **Collaborative Obfuscation** (= AloePri dup).
~25 schemes grounded ★ from EdgeQuake primary sources; cost figures CoVe-verified (#19; SHAFT
corrected to NDSS camera-ready, H100 CC 4→8%).

## Two EdgeQuake bugs discovered (filed in edgequake repo, reproductions included)

1. `issues/2026-06-02-hybrid-query-misses-hardware-chunks.md` — hybrid `query` is recall-incomplete:
   drops experimental-setup/hardware prose chunks present in the doc. 6 confirmed false-negatives.
   **Lesson: use `document_get_md` (full MD), not `query`, for completeness-sensitive extraction.**
2. `issues/2026-06-02-md-conversion-drops-table-content.md` — PDF→MD conversion drops **table
   content** to `![tbl_…]` placeholders → communication cost / per-component latency (table-bound)
   are unrecoverable via query OR document_get_md. **Standing rule (memory
   `feedback-edgequake-issue-reporting.md`): if EdgeQuake tools misbehave, stop and file an issue.**

## Open work (priority order)

1. **Recover table-bound comm/latency from original PDFs** (EdgeQuake can't surface them):
   SecFormer, Fission, CryptoMoE (absolute GB), TwinShield, FLASH/PeGraph, CipherFormer Table IV.
   Path: WebFetch `https://ar5iv.labs.arxiv.org/abs/<arXiv-id>` (ar5iv renders tables as HTML; arXiv
   IDs in Part D / `paper/manuscript/refs.bib`). ePrint blocks bots (HTTP 403) — use ar5iv/arXiv.
2. **Ground GORAM + H₂O₂RAM** — not in EdgeQuake (Part D `[ ]`); hardware still `⚠`. Either user
   uploads to EdgeQuake or WebFetch (GORAM arXiv 2410.02234 / PVLDB vol18 p3601; H₂O₂RAM 2409.07167).
3. **`refs.bib` non-★ TODO fields are UNVERIFIED** — run `/validate-bib` + `/verify-claims` before
   submission (flagged in primer).
4. **Deferred deep prose** — ~48 `\needswork` markers in `paper/manuscript/sections/*.tex`
   (Approach/Performance/Limitations bodies, vignette headline number) — separate, larger task.

## Methodology notes for the next agent

- Extraction is **fetch-and-extract**, done **inline, not via subagents** (memory
  `feedback-ground-inline-not-agents.md`) — except `/verify-claims`, which the user explicitly invokes.
- `document_get_md` on large docs returns a **saved file path** (not inline); `grep` the file
  (this shell is zsh — no word-splitting of unquoted vars; `head`/`sort` sometimes absent, use
  `grep -m`). Many saved MD files already exist under the session `tool-results/` dir.
- `chunk_count:0` docs are not vector-indexed → `query` can't see them; use `document_get_md`.
- vs-plaintext overhead for crypto/MPC schemes is genuinely usually absent (they report vs prior
  scheme) — `— (only relative)` is correct, not a miss.
- Grilling: ask via AskUserQuestion selectors with a recommended option first (memory
  `feedback-grill-me-use-selectors.md`). Section titles spell out "and" not "&".

## Suggested skills

- `/verify-claims` — before trusting any newly-fetched figures and on `refs.bib` TODO fields.
- `/validate-bib` — cross-check refs.bib (structural; `--semantic` for drift/DOIs).
- `/compile-latex` (or `make` in `paper/manuscript`) — verify the manuscript still builds after edits.
- `/grill-me` — if the next task is another structural/normalization decision on the tables.
- `/commit` — the user commits explicitly and often; commit at logical units (both repos when
  EdgeQuake issues change).
