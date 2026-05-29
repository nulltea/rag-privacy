# Build the manuscript with the available toolchain (pdflatex + bibtex; no latexmk).
# Usage:  make        -> full build (pdflatex, bibtex, pdflatex x2)
#         make quick   -> single pdflatex pass (fast, refs may be stale)
#         make clean   -> remove build artifacts
#         make watch   -> rebuild on change (requires entr; optional)

MAIN := main
TEX  := $(wildcard sections/*.tex) $(MAIN).tex refs.bib

.PHONY: all quick clean watch

all: $(MAIN).pdf

$(MAIN).pdf: $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	-bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

quick:
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).pdf sections/*.aux

watch:
	ls $(TEX) | entr -c make all
