# PDF review round 3 — comment audit (2026-06-04)

Source: `pdfannots` extraction of the annotated `main.pdf` (annotated copy overwritten by
rebuilds; comments preserved here). Reconciliation: script markdown = 12 comments, JSON
authoritative count = 12 — match, no dropped annotations. **12 of 12 comments resolved** (12 ✅).
Build: `make` clean — exit 0, 0 undefined refs/cites, 29 pp (was 27); new `tab:retr-attacks`
resolved (Table 12 after a later mid-doc table insertion); orphaned `ssec:crypto-inf` label removed.
Reconciliation gate: M = 12 = audit rows = Σ terminal statuses; 0 rows `⬜`. PASS.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred (reason + where tracked) · ❌ won't-fix (reason) · ⬜ pending (transient only).

Decisions (grilled via selectors, 2026-06-04):
- C1 → split §3.1 into **two** paragraphs: MPC/SS + FHE.
- C5 → **drop** flat/IVF taxonomy, keep "dense ANN retrieval".
- C10 → PrivGemo numbers from EdgeQuake (`document_get_md`); GORAM via WebFetch.
- C12 → **one** banded attack table (Dense / Graph bands).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | §3.1 (p9) | Split "3.1 Approach" into 2–3 paragraphs (MPC, FHE) | ✅ | Approach now two `\emph{}`-led paragraphs: *Secret sharing and MPC* / *Fully-homomorphic encryption* (SS kept with MPC per grilled decision C1). |
| 2 | §3.2 (p9) | Remove "3.2 Cryptographic Inference" subsection; distribute content into §3.1 + section intro | ✅ | `\subsection{Cryptographic Inference}` heading + `ssec:crypto-inf` label removed; "we don't re-systematize / Li et al." framing moved to §3 intro; exemplar prose + `tab:crypto-inf` now flow under §3.1. |
| 3 | §8.1 (p15) | Remove "8.1 Approach" subsection (redundant; approaches live in 8.2/8.3) | ✅ | `\subsection{Approach}` + `ssec:retr-approach` removed; access-hiding primitive descriptions folded into §8 intro as a second paragraph. |
| 4 | §8.2 (p15) | Explain clearly how/what dense retrieval leaks and why it is a problem | ✅ | New opener spells out the three dense channels: query embedding (Vec2Text-invertible), access pattern (reconstruction), at-rest vectors; frames schemes by which channels they close. |
| 5 | §8.2 (p15) | flat/IVF taxonomy relevant? If not, drop it; keep dense retrieval via ANN | ✅ | Dropped per decision C5 — "flat or inverted-file (IVF) vector index" → "dense ANN retrieval" in §8 intro, §8.2 opener, and `tab:crypto-vec` caption. |
| 6 | §8.2 (p15) | PHE/CKKS/HE scoring should be its own paragraph (separate subclass) | ✅ | §8.2 split into three `\paragraph`s: *Oblivious fetch (PIR)* / *Homomorphic scoring and reranking* (TRSE, RemoteRAG, RAGtime-PIANO CKKS) / *DCPE*. |
| 7 | §8.3 (p16) | Explain clearly how/what graph search leaks (node access, adjacency volume) and why a problem | ✅ | Opener now itemizes three graph channels: node access pattern (subject identity), adjacency volume (degree fingerprint → re-identification), `source_id` fan-out (links query→source docs). |
| 8 | §8.3 (p16) | LightRAG needs a citation | ✅ | Added `\citep{guo_lightrag}` at the LightRAG-surface mention (ref already in refs.bib). |
| 9 | §8.3 (p16) | XorMM: we have an implementation in the parent dir — add as a to-do (measure it) | ✅ | Added `\needswork{…}` table note under `tab:crypto-graph`: co-measure XorMM query latency on the LightRAG adjacency workload (renders red `[TODO:…]`). |
| 10 | §8.3 (p16) | GORAM & PrivGemo: find latency and communication | ✅ | GORAM filled from arXiv 2410.02234: latency 58 ms–36 s @1.4B edges, init <3 min, comm n/r (3PC (2,3)-SS sqrt-ORAM). PrivGemo has **no crypto cost** by design (anonymization); recorded its operational figure ≈70% subgraph reduction from EdgeQuake `document_get_md`. |
| 11 | §8.3 (p17) | Anonymization/PrivGemo paragraph is vague — clarify how the approach secures graph-view retrieval | ✅ | Rewrote: local "Hand" explores, sends remote "Brain" only a sanitized subgraph; structure pruning/clustering removes identifying motifs (≈70% node reduction, 2.4M→≈732K on CWQ); HMAC pseudonymizes entities; flagged as heuristic obfuscation with unquantified residual leakage. |
| 12 | §8.4 (p17) | Limitations: concrete attack names+citations; do NOT mix dense/graph; per-scheme defenses; add a table | ✅ | Split §8.4 into separate *Dense* / *Graph* paragraphs with named+cited attacks and per-scheme defenses; added banded `tab:retr-attacks` (Attack(cited) \| Leakage \| Schemes at risk \| Defence, with Dense and Graph row-bands). |

## Post-review corrections (2026-06-04)

**Attack-table padding (user-flagged, twice).** Fixed-`p{}` widths underfilled the ~16.5 cm text
block, leaving a right gap (and justified stretching). Final fix: both attack tables
(`tab:retr-attacks`, `tab:crypto-attacks`) converted to `tabularx{\textwidth}` with a weighted
ragged-right `Y{w}` column type (`\newcolumntype{Y}[1]{>{\raggedright\arraybackslash\hsize=#1\hsize}X}`
added to `main.tex`; weights 1.05/1.0/0.85/1.1 sum to the 4-column count). Sizes to `\textwidth`
exactly — no width-guessing. (NB: `Y{w}` goes in the preamble unbraced — `{Y{w}}` triggers an
"Illegal pream-token" array error.)

**"Rerank" terminology (user-flagged) — correction of record.** "Reranking" in RAG is reserved
for a *cross-encoder* stage over ANN top-$k$. The RemoteRAG paper (EdgeQuake
`82e9a235-…`) never uses "rerank"/"cross-encoder" (0 hits); it is **two-stage retrieval** — a
DistanceDP-perturbed coarse retrieval to $k'$ candidates, then exact cosine top-$k$ over those
candidates under PHE (user decrypts + sorts), optional $k$-out-of-$k'$ OT for fetch. Prior draft
mislabeled this (and TRSE/p\textsuperscript{2}RAG/Opal scoring steps) as "rerank". Fixed:
`tab:crypto-vec` Stage `retrieve,rerank`→`retrieve,score` (p2RAG, TRSE, RemoteRAG); RemoteRAG
primitive `PHE rerank`→`PHE top-$k$`; §8.2 heading `…and reranking`→`Homomorphic scoring`; §8.2
prose, §6 (`06:40`), and §9 parentheticals (`RemoteRAG's HE rerank`/`Opal's in-enclave rerank`
→ `…scoring`) corrected. **Retained** (legitimate sense): the canonical reranking *stage* and
the §9 finding that a private cross-encoder **reranking stage is an open gap** — that distinction
is now internally consistent (schemes "score"; true reranking = open gap).

**PrivGemo fidelity cell (user-flagged).** "quality-only" → "lossy; QA-pres." with a caption
gloss (retrieval not exact — subgraph pruned by design — but downstream answer quality preserved).

**Gap-map float placement (user-flagged).** `tab:gapmap` was `table*[t]` and floated above §8's
Table; re-pinned `\begin{table}[H]` so it stays in §9 after the attack table.

**New §3 attack table (user-requested).** Added `tab:crypto-attacks` (now Table 5, p11) mirroring
`tab:retr-attacks`: banded MPC/secret-sharing vs FHE — Muse model-extraction, party collusion,
permutation inversion (Fission), Hidden-No-More prompt reconstruction, IND-CPA\textsuperscript{D}
key recovery — with the assumption/heuristic exploited, schemes at risk, and defence. Inserting it
mid-document renumbered later tables (`tab:retr-attacks` 11→12, `tab:gapmap` 12→13, etc.).

**BUILD GOTCHA (recorded).** `make` keys on file mtime; after an Edit the new `main.pdf` can end up
with an mtime ≥ the edited section, so `make` reports "Nothing to be done" and silently skips the
rebuild. Always confirm `make` actually ran (3 pdfTeX passes in the log, or check `main.aux` mtime);
if in doubt, `touch sections/*.tex main.tex && make` or `make clean && make`.
