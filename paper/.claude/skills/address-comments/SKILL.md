---
name: address-comments
description: Extract reviewer comments and highlights from an annotated manuscript PDF (Adobe Acrobat / Preview / Skim) into Markdown, then apply or answer each one in the LaTeX source and rebuild. Use when the user says "address my comments", "read my PDF comments", "check my annotations", "I annotated the PDF", "apply my PDF feedback", or after they say they saved an annotated PDF.
---

# Read PDF Comments

Turn reviewer annotations on the compiled PDF into actionable edits in the `.tex` source.
The extraction is deterministic — never visually parse the rendered PDF for comment text
(note text lives in popups that don't render); always use the script.

## Completeness Contract (NON-NEGOTIABLE)

**Every extracted comment must reach exactly one explicit terminal status. Silently
dropping, skipping, merging-away, or forgetting a comment is a hard failure of this skill.**

- **No silent omission.** If you cannot address a comment, you do not skip it — you mark it
  `❌ won't-fix` (with a written reason) or `⏸ deferred` (with a reason + where it's tracked).
  "I didn't get to it" is `⏸ deferred`, never absence.
- **Conservation of comments.** `extracted count == audit-doc rows == task count == Σ terminal statuses`.
  These four numbers must be equal at the end. If they are not, you are not done — halt and reconcile.
- **No batching that loses identity.** You may *group* edits, but every individual comment keeps
  its own numbered audit row and its own terminal status. A section-file task may cover several
  comments only if each of those comments is still listed and individually statused in the audit doc.
- **Extraction completeness, not just processing completeness.** The script may bucket
  annotations into more than one section (notes, highlight-only, free-floating). Carry **all**
  buckets. Reconcile the script's comment count against the PDF's raw annotation total (Step 1)
  before trusting it — a comment the extractor drops is still a comment you ignored.
- **Report every comment, not a summary.** The final report enumerates all N with their status;
  never report "all resolved" from a tally alone — the enumeration is the proof.

Violation example to avoid: extracting 25 comments, processing 25, and reporting "all 25
resolved" while the PDF actually held 28 annotations (3 highlight-only notes the script's
first section omitted). That is silently ignoring 3 comments.

## Quick start

From `paper/`:

```bash
bash .claude/skills/address-comments/scripts/extract-annots.sh manuscript/main.pdf
```

Prints Markdown — each highlight's page, section, quoted source text, and the reviewer's
note. Add `-o quality_reports/pdf-comments.md` to also save it. The script auto-installs
`pdfannots` via `pipx` on first run and stamps the PDF's save-time so you can confirm freshness.

## Workflow

1. **Extract** — run the script on the annotated PDF (default `manuscript/main.pdf`). Read
   **all** of its output, every bucket (detailed notes, highlight-only, free-floating), not
   just the first section. Then **reconcile the count against the PDF's raw annotation
   total** so the extractor didn't silently drop any. The markdown the script prints can show
   fewer items than there are annotations (highlight-only marks, notes on figures/tables); the
   JSON format lists *every* annotation as a flat list:
   ```bash
   # authoritative annotation count — ALL annotations, not the markdown buckets
   pdfannots -f json manuscript/main.pdf | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"
   ```
   (Use the `pdfannots` CLI — it lives in a pipx venv, so system `python3` cannot
   `import pdfannots`; always go through `-f json`, never an `import`.) If this number exceeds
   the comments you extracted from the markdown, dump the JSON and recover the missing ones
   (`pdfannots -f json … | python3 -m json.tool`) before proceeding. Record the reconciled
   total `M` — the denominator everything else must match.
2. **Triage** — classify each comment: edit / question-to-answer / won't-fix. Group by section file (`sections/NN_*.tex`). Triage **all M**; classifying is not resolving — a "won't-fix" still needs a row and a reason.
3. **Track** — set up the two trackers *before* touching the source (see "Tracking" below). **Create one audit-doc row for every one of the M comments** (start each `⬜`), and a task per comment (or per section-file batch — but every comment stays individually listed). **Assert `audit rows == M` now**; if not, you've already dropped one.
   - **Audit doc** — `quality_reports/pdf-review-roundN.md`, modeled on `pdf-review-round1.md`.
   - **Task list** — `TaskCreate` (survives context loss; visible to the user).
4. **Locate** — Grep the quoted text in the source; the quote is verbatim from the PDF, so it matches the `.tex` modulo LaTeX markup.
5. **Apply** — work through **every** row. Make the edit; for a question, answer it (and edit if the answer implies a change). As each is handled, set its task `completed`/`deferred` and flip its audit row to a **terminal status** (`✅`/`ℹ️`/`❌`/`⏸`) with the what-changed cell filled. **No row may stay `⬜` at the end** — a comment you can't or won't do is `❌`/`⏸` *with a reason*, never left blank or removed.
6. **Verify** — rebuild (`cd manuscript && make`); confirm 0 undefined refs/cites and no new overfull boxes.
7. **Report** — run the **reconciliation gate** and finalize. Assert `M == audit rows == Σ terminal statuses` and that **zero rows are `⬜`**; if any check fails, HALT, say so, and fix before claiming completion. The header tally must read `X of M` with the breakdown (e.g. `20 ✅ · 3 ℹ️ · 2 ⏸`), and the report to the user must **enumerate every comment** with its status — never assert "all resolved" from the tally alone.

## Tracking

**Both trackers are mandatory once there is more than a single comment.** The audit doc is the
durable record (survives PDF rebuilds, which overwrite the annotated `main.pdf`); the task list
is the live progress view.

### Audit doc — format (mirror `quality_reports/pdf-review-round1.md`)

```markdown
# PDF review round N — comment audit (YYYY-MM-DD)

Source: `pdfannots` extraction of the annotated `main.pdf` (annotated copy overwritten by
rebuilds; comments preserved here). **X of M comments resolved** (k answered without a change).
Committed in `<hash>`.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred (reason + where tracked) · ❌ won't-fix (reason) · ⬜ pending (transient only — must not survive to the report).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | §1.2 | <verbatim comment> | ⬜ | |
```

Start every row `⬜` at step 3; flip each to a terminal status (`✅`/`ℹ️`/`⏸`/`❌`) during
step 5. `⬜` is a work-in-progress marker only — **no row may still be `⬜` when you report**.
The header tally is `X of M` with the status breakdown, and it must reconcile: rows = M,
and every row carries a terminal status with a non-empty reason for `⏸`/`❌`.

### Task list

- One task per comment (subject = `§loc: short gist`), or one per section-file batch if the
  comment count is large. Set `in_progress` when you start a comment, `completed` when its edit
  builds clean. Never mark a comment done before the rebuild in step 6 passes for that change.

## Prerequisites

- The PDF must be a **local** file with annotations saved in. Acrobat: **File ▸ Save As → on this Mac**, not Adobe Cloud (cloud saves never touch local disk, so nothing to read).
- Highlight the specific text and attach the note to it — that pairs your words with the exact sentence + page. Free-floating sticky notes extract too, but without anchored context.

## Notes

- Re-run after every annotation round; if output says "no comments found," the save went to the cloud or wasn't saved.
- Works with Acrobat, Preview, and Skim (standard PDF annotations). Scanned PDFs won't yield quoted text — ours is real text, so quotes are exact.
