---
name: address-comments
description: Extract reviewer comments and highlights from an annotated manuscript PDF (Adobe Acrobat / Preview / Skim) into Markdown, then apply or answer each one in the LaTeX source and rebuild. Use when the user says "address my comments", "read my PDF comments", "check my annotations", "I annotated the PDF", "apply my PDF feedback", or after they say they saved an annotated PDF.
---

# Read PDF Comments

Turn reviewer annotations on the compiled PDF into actionable edits in the `.tex` source.
The extraction is deterministic — never visually parse the rendered PDF for comment text
(note text lives in popups that don't render); always use the script.

## Quick start

From `paper/`:

```bash
bash .claude/skills/address-comments/scripts/extract-annots.sh manuscript/main.pdf
```

Prints Markdown — each highlight's page, section, quoted source text, and the reviewer's
note. Add `-o quality_reports/pdf-comments.md` to also save it. The script auto-installs
`pdfannots` via `pipx` on first run and stamps the PDF's save-time so you can confirm freshness.

## Workflow

1. **Extract** — run the script on the annotated PDF (default `manuscript/main.pdf`); read its output.
2. **Triage** — classify each comment: edit / question-to-answer / won't-fix. Group by section file (`sections/NN_*.tex`).
3. **Locate** — Grep the quoted text in the source; the quote is verbatim from the PDF, so it matches the `.tex` modulo LaTeX markup.
4. **Apply** — make the edit; for a question, answer it to the user (and edit if the answer implies a change). Skip nothing silently.
5. **Verify** — rebuild (`cd manuscript && make`); confirm 0 undefined refs/cites and no new overfull boxes.
6. **Report** — one line per comment: what it asked, what you did.

## Prerequisites

- The PDF must be a **local** file with annotations saved in. Acrobat: **File ▸ Save As → on this Mac**, not Adobe Cloud (cloud saves never touch local disk, so nothing to read).
- Highlight the specific text and attach the note to it — that pairs your words with the exact sentence + page. Free-floating sticky notes extract too, but without anchored context.

## Notes

- Re-run after every annotation round; if output says "no comments found," the save went to the cloud or wasn't saved.
- Works with Acrobat, Preview, and Skim (standard PDF annotations). Scanned PDFs won't yield quoted text — ours is real text, so quotes are exact.
