# Humanize Audit: 08_retrieval.tex

**Date:** 2026-06-09
**Word count:** ~1,490 (prose only; table environments excluded)
**Findings:** 23 (23 HIGH, 0 MED, 0 LOW) — all one category

The only problem is the hard no-em-dash rule: 23 prose em-dashes. Everything else
is clean. Cosmetic-but-pervasive fix, not a rewrite. §8 predates the no-em-dash
rule (consistent with the §3/§4/§6/§8 pending-sweep note).

## Per-category summary

| Category | HIGH | MED | LOW |
|---|---:|---:|---:|
| 1. Boilerplate transitions | 0 | 0 | 0 |
| 2. AI-cliché lexicon | 0 | 0 | 0 |
| 3. Em-dash / punctuation | 23 | 0 | 0 |
| 4. Symmetric paragraph shapes | 0 | 0 | 0 |
| 5. Tricolon abuse | 0 | 0 | 1* |
| 6. Hedging stacking | 0 | 0 | 0 |
| 7. "Not only X but also Y" | 0 | 0 | 0 |
| 8. Formulaic openers | 0 | 0 | 0 |
| 9. Hyphenation excess | 0 | 0 | 0 |
| 10. Sycophancy | 0 | 0 | 0 |

\* Tricolons present but discipline-legitimate (named mechanisms/channels); not flagged.

## Findings (Category 3 — prose em-dashes, all HIGH)

| Line | Current text | Suggested |
|---:|---|---|
| 11–12 | "predominantly cryptographic---PIR...encryption---but the layer also" | parenthesize the list |
| 20 | "practical deployment---one-to-three orders of magnitude cheaper" | sentence split |
| 30 | "comparisons survive---trading exactness" | sentence split |
| 39 | "(\cref{tab:crypto-vec})---each of which is a concrete leak" | sentence split |
| 42 | "touches or returns---the access pattern---reveals" | parenthesize |
| 46–47 | "three channels---query, access pattern, and at-rest index---it actually closes" | parenthesize |
| 93 | "(PHE)---the user decrypts" | sentence split |
| 94 | "locally---with optional...oblivious transfer" | sentence split |
| 102 | "plaintext embedding---so the ciphertext imposes" | comma |
| 109 | "is \emph{conditional}---by adding asymmetric noise" | colon |
| 120–121 | "node access pattern---which entity...---identifies" | parenthesize |
| 123 | "adjacency volume---how many neighbours...---leaks" | parenthesize |
| 125–126 | "fan-out---which chunks an entity maps to---links" | parenthesize |
| 169 | "load-bearing---Compass runs the ORAM controller" | colon |
| 170 | "traversal client-side---a deployment cost" | sentence split |
| 181 | "not the content---closed only by layering ORAM" | comma |
| 219 | "70\% node reduction---e.g....732K---explicitly" | parenthesize |
| 224 | "guarantee---a heuristic defence" | sentence split |
| 227 | "retrieval surface---entity/relation...---which is why" | parenthesize |
| 238 | "by construction---which is precisely the leak" | sentence split |
| 252 | "themselves encrypted---which is why the oblivious schemes" | sentence split |
| 257–258 | "leak \emph{volume}---postings-list...---so the same" | parenthesize |
| 267 | "structural fingerprint---a heuristic obfuscation" | sentence split |

## Em-dashes NOT flagged (structural — inside table environments)

Lines 50, 75, 153, 159 are inside `\begin{table*}...\end{table*}` (captions, footnotes,
cells). Structural; left as-is.

## Recommendation

~15 HIGH per 1000 words, but all one category. **Strip in place / cosmetic**, not a
rewrite. Prose is otherwise free of AI-voice tells — concrete vocabulary, substantive
openers, no hedging or sycophancy. After the sweep, rebuild (3 pdfTeX passes; watch for
new overfull hboxes where sentences were split).
