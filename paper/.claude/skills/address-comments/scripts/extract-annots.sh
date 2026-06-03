#!/usr/bin/env bash
# extract-annots.sh — extract reviewer comments/highlights from an annotated PDF
# to Markdown (page · section · quoted source text · note). Fast + deterministic.
#
# Usage:  extract-annots.sh [pdf-path] [-o out.md]
#   pdf-path  defaults to manuscript/main.pdf (then main.pdf, ../manuscript/main.pdf)
#   -o out.md additionally writes the result to a file
set -euo pipefail

# 1. Resolve the PDF.
PDF="${1:-}"
if [[ -z "$PDF" || "$PDF" == -* ]]; then
  PDF=""
  for c in manuscript/main.pdf main.pdf ../manuscript/main.pdf; do
    [[ -f "$c" ]] && { PDF="$c"; break; }
  done
fi
[[ -n "$PDF" && -f "$PDF" ]] || { echo "error: no PDF found (pass a path; tried manuscript/main.pdf)" >&2; exit 1; }

# 2. Optional output file: "<pdf> -o out.md".
OUT=""
[[ "${2:-}" == "-o" && -n "${3:-}" ]] && OUT="$3"

# 3. Resolve pdfannots; install once via pipx (fallback pip3 --user) on first use.
PA=""
if command -v pdfannots >/dev/null 2>&1; then PA="pdfannots"
elif [[ -x "$HOME/.local/bin/pdfannots" ]]; then PA="$HOME/.local/bin/pdfannots"
elif python3 -c "import pdfannots" >/dev/null 2>&1; then PA="python3 -m pdfannots"
else
  echo "# installing pdfannots (one-time)…" >&2
  pipx install pdfannots >/dev/null 2>&1 || pip3 install --user pdfannots >/dev/null 2>&1 \
    || { echo "error: could not install pdfannots (need pipx or pip3)" >&2; exit 2; }
  if command -v pdfannots >/dev/null 2>&1; then PA="pdfannots"; else PA="$HOME/.local/bin/pdfannots"; fi
fi

# 4. Extract.
mtime="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$PDF" 2>/dev/null || date '+%Y-%m-%d %H:%M')"
hdr="# PDF comments — $(basename "$PDF") (saved $mtime)"
body="$($PA "$PDF" 2>/dev/null || true)"
[[ -n "${body// /}" ]] || body="_(no comments/highlights found — confirm the PDF was saved LOCALLY with annotations, not to Adobe Cloud)_"

if [[ -n "$OUT" ]]; then
  { printf '%s\n\n%s\n' "$hdr" "$body"; } > "$OUT"
  echo "wrote → $OUT" >&2
else
  printf '%s\n\n%s\n' "$hdr" "$body"
fi
