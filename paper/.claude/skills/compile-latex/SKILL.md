---
name: compile-latex
description: Compile the arXiv manuscript with pdflatex + bibtex (via the manuscript Makefile, NOT XeLaTeX). Use when the user says "compile", "build the PDF", "rebuild the manuscript", "run latex", "render the tex", or asks why a `.tex` isn't producing a PDF. Operates on `manuscript/main.tex` → `main.pdf`.
argument-hint: "[make target: all (default) | quick | clean]"
allowed-tools: ["Read", "Bash", "Glob"]
---

# Compile the Manuscript

The manuscript builds with **pdflatex + bibtex** through `manuscript/Makefile` — **not** XeLaTeX,
no `Slides/`, no `Preambles/`/`TEXINPUTS`. `refs.bib` lives in `manuscript/` (local, not repo root).
Biber is **not** used.

## Quick start

```bash
cd manuscript && make            # full build: pdflatex, bibtex, pdflatex x2
```

Targets: `make quick` (one pass, refs may be stale), `make clean` (remove artifacts),
`make watch` (rebuild on change; needs `entr`). The harness resets cwd between commands, so
always `cd manuscript && …` in one command.

`make` is incremental: if no source changed it prints *"Nothing to be done for `all'"*. To force
a rebuild, `touch main.tex` first or run `make clean && make`.

## Workflow

1. **Build** — `cd manuscript && make` (capture output).
2. **Check the LaTeX log** (`manuscript/main.log`, not just make's stdout):
   - Undefined refs/cites: `grep -ciE "undefined" main.log` (expect 0); also grep `Citation .* undefined` / `Reference .* undefined` / `Label(s) may have changed` (the last means another pass is needed).
   - Overfull boxes: `grep -c "Overfull \\hbox" main.log`; list any `>20pt` (small ones are cosmetic).
   - Page count: `grep "Output written on main.pdf" main.log` → `(N pages, …)`.
3. **Report** — success/failure, undefined count, overfull count (+ any >20pt and which table/section), page count. Don't `open` the PDF yourself (can't view it); suggest `open manuscript/main.pdf` for the user.

## Why the 4-step cycle

pdflatex (writes `.aux` with cite keys) → bibtex (reads `.aux`, writes `.bbl`) → pdflatex
(pulls in bibliography) → pdflatex (resolves cross-references + final page numbers).

## Notes

- **Always pdflatex**, never XeLaTeX, never biber — `bibtex` only.
- A known cosmetic ~43pt overfull at `tab:refmetrics` is pre-existing; flag it, don't chase it.
- If the build hard-fails, `-halt-on-error` stops at the first error — read the last ~20 lines
  of make output for the offending file/line.
