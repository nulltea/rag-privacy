# Manuscript — Confidential LLM/RAG in Practice

arXiv v1 working draft. Genre: **SoK + measurement**. Headline: *what holds and
what breaks under a semi-honest cloud*. Decisions of record:
`../quality_reports/specs/2026-05-29_confidential-llm-rag-stage1.md`.

## Build

```bash
make          # pdflatex + bibtex + pdflatex x2  -> main.pdf
make quick    # one fast pass (refs may be stale)
make clean
```

Toolchain: `pdflatex` + `bibtex` (this machine has no `biber`/`latexmk`).
Bibliography uses `natbib` + `bibtex` for portability (arXiv + PoPETs both build it).

## Layout

| File | Holds |
|---|---|
| `main.tex` | preamble, title, `\input` order, drafting macros (`\needswork`, `\dnote`) |
| `sections/00_abstract.tex` | abstract (write last) |
| `sections/01_introduction.tex` | motivation, gap, contributions, roadmap |
| `sections/02_threat_model.tex` | **locked** semi-honest adversary, trust anchors, out-of-scope |
| `sections/03_background_survey.tex` | survey **by trust mechanism** + tradeoff table |
| `sections/04_tee_split_inference.tex` | *what holds* — TEE+GPU split (overhead + attack resistance) |
| `sections/05_static_obfuscation_aloepri.tex` | *what breaks* — AloePri repro + omitted attacks |
| `sections/06_evaluation.tex` | like-for-like holds-vs-breaks (the headline table) |
| `sections/07_at_rest_storage_caprise.tex` | RAG measurement (Caprise repro; may move to appendix) |
| `sections/08_unified_design.tex` | unified static+dynamic **proposal** (eval = v2 future work) |
| `sections/09_related_work.tex` | positioning; Compass deferred |
| `sections/10_conclusion.tex` | conclusion + future work |
| `sections/99_appendix.tex` | repro details, attack harness |
| `refs.bib` | seed citations — **TODO-marked fields are UNVERIFIED** |

## Before submission

- Replace every `\needswork{...}` and `\dnote{...}` (grep them: `grep -rn needswork sections`).
- Verify every `refs.bib` entry carrying a `TODO` note against EdgeQuake / OpenAlex.
- Resolve the three still-open decisions in the stage-1 record (survey coverage
  target, reproduction tolerances, model + TEE platform).
- The unified design (Sec. 8) stays a proposal for v1; it gates the PoPETs v2.
