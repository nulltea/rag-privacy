# Manuscript — SoK: Confidential LLM Inference and RAG

arXiv working draft. Genre: **pure SoK**. Contribution: a taxonomy
(threat-model + mechanism) → **gap-map + performance axis**, spanning LLM
inference *and* RAG including **graph-RAG**. Positioned as the
mechanism/threat-model/performance complement to Bodea et al.'s risk-centric
RAG-privacy SoK — filling its three self-named gaps (practical cost, advanced
architectures, crypto/TEE depth).

Decisions of record: `../quality_reports/specs/2026-05-29_confidential-llm-rag-stage1.md`
(see the **SCOPE PIVOT** block, grill #3).

## Build

```bash
make          # pdflatex + bibtex + pdflatex x2  -> main.pdf
make quick    # one fast pass
make clean
```

Toolchain: `pdflatex` + `bibtex` (no `biber`/`latexmk`). `natbib` for portability.

## Layout (mechanism-primary)

| File | Holds |
|---|---|
| `sections/00_abstract.tex` | abstract (write last) |
| `sections/01_introduction.tex` | motivation · why-a-new-SoK (Bodea) · scope · contributions |
| `sections/02_threat_model.tex` | **the 6-axis threat-model taxonomy** (cross-cutting; drafted) |
| `sections/03_cryptographic.tex` | MPC / FHE / secret-sharing (inference + private retrieval) |
| `sections/04_tee.tex` | trusted execution environments |
| `sections/05_obfuscation.tex` | static obfuscation **+ the breaks vignette** (finding only) |
| `sections/06_differential_privacy.tex` | differential privacy |
| `sections/07_hybrid_split.tex` | TEE + obfuscation hybrids (GELO/ObfuscaTune/TwinShield) |
| `sections/08_synthesis.tex` | stage×mechanism matrix · gap-map · performance · graph-RAG |
| `sections/09_related_work.tex` | Bodea positioning + prior surveys |
| `sections/10_conclusion.tex` | open problems (from the gap-map) |
| `sections/99_appendix.tex` | SLR methodology · per-scheme classification |
| `refs.bib` | seed citations — **TODO-marked fields are UNVERIFIED** |

## Deferred companion paper (stub — keep separate from this SoK)

A second paper owns all original empirics + the design:

- **GELO optimizations** + a **unified Static (AloePri) + Dynamic (GELO/TwinShield)**
  obfuscated-inference design.
- The full **ArrowMatch / Sequence-IMA** attack methodology + AloePri replication.
- The TEE-split measurement campaign.

**Partition rule (avoid salami-slicing):** this SoK claims only the *finding*
that static obfuscation under-protects (a gap-map data point, `ssec:vignette`);
the companion paper claims the *method* and the *fix*. The companion cites this
SoK for framing.

## Before submission

- Replace every `\needswork{…}` / `\dnote{…}` (grep: `grep -rn needswork sections`).
- Verify every `refs.bib` entry carrying a `TODO` note (incl. Bodea, Opal,
  ObfuscaTune, PermLLM, Petridish, STIP) against EdgeQuake / OpenAlex.
- Expand the surveyed corpus ~24 → ~40–50 papers (OpenAlex citation-graph).
- Verify the `\tq` cells in `tab:threat-taxonomy` and `tab:gapmap`.
- Add the performance-table figures + the explicit non-comparability caveat.
