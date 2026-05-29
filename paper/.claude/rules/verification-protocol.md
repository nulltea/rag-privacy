---
paths:
  - "manuscript/**/*.tex"
  - "Figures/**/*"
  - "scripts/**/*.R"
---

# Task Completion Verification Protocol

**At the end of EVERY task, Claude MUST verify the output works correctly.** This is non-negotiable.

## For the LaTeX Manuscript:
1. Build from the `manuscript/` directory (`make`, or the pdflatex/xelatex + bibtex + pdflatex×2 cycle) and check for errors
2. Confirm the PDF was produced with non-zero size and the expected page count
3. Open the PDF to verify figures and tables render (`open` on macOS, `xdg-open` on Linux)
4. Check the log for undefined `\ref`/`\cite` and overfull `\hbox` warnings

## For TikZ Figures:
1. Compile the diagram standalone and confirm it builds without errors
2. Run the prevention pre-check (`scripts/check-tikz-prevention.py`) and the `tikz-reviewer` measurement passes
3. Verify labels/arrows clear their neighbors (see `tikz-measurement.md`)

## For R Scripts:
1. Run `Rscript scripts/R/filename.R`
2. Verify output files (PDF, RDS) were created with non-zero size
3. Spot-check estimates for reasonable magnitude

## Common Pitfalls:
- **Assuming success**: always verify output files exist AND contain correct content
- **Undefined refs/cites**: a clean early pass can still hide unresolved `\ref`/`\cite` — check the final pass
- **Stale figures**: a regenerated figure must actually be re-included in the build

## Verification Checklist:
```
[ ] Output file created successfully
[ ] No compilation errors; no undefined refs/citations
[ ] Figures/tables render correctly
[ ] Opened in viewer to confirm visual appearance
[ ] Reported results to user
```
