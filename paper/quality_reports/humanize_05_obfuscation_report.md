# Humanize Audit: 05_obfuscation.tex (2026-06-08)

**Word count:** ~1,580 (prose only). **Findings:** 9 (0 HIGH, 4 MED, 5 LOW).
**Recommendation:** 0 HIGH/1000w → **cosmetic cleanup only** (far below 5–8 strip / >8 rewrite thresholds).

## Per-category counts

| Category | HIGH | MED | LOW |
|---|---:|---:|---:|
| 1 Boilerplate transitions | 0 | 0 | 0 |
| 2 AI-cliché lexicon | 0 | 0 | 0 |
| 3 Em-dash / punctuation | 0 | 2 | 0 |
| 4 Symmetric paragraph shapes | 0 | 0 | 1 |
| 5 Tricolon abuse | 0 | 1 | 1 |
| 6 Hedging stacking | 0 | 0 | 0 |
| 7 "Not only X but also Y" | 0 | 0 | 0 |
| 8 Formulaic openers | 0 | 1 | 1 |
| 9 Hyphenation excess | 0 | 0 | 1 |
| 10 Sycophancy | 0 | 0 | 0 |

## MED findings (actionable)

- **L48-58 | punctuation (semicolon/run-on)** — KV-Cloak paragraph; the "With the secret matrices $S,M$..." sentence packs a cleft + nested apposition + trailing "though". Densest sentence in §5. Split.
- **L234-244 | punctuation (comma-splice run-on)** — "Accumulation" paragraph chains 4 long comma-spliced clauses. Split at "A trained inverter...".
- **L22-23 | formulaic opener** — "Every scheme in this family makes the same high-level choice:" reads as LLM scaffolding. State the choice directly.
- **L30-113 | tricolon cadence** — 3+ "X, Y, and Z" enumerations cluster (L30-32, L34, L113); each legitimate, but recast one for variety.

## LOW findings (author judgment, acceptable)

- L60 input-transform enumeration — legitimate.
- L152-160 single topic→two-classes→summary — isolated, below pattern threshold.
- L77/22/170 opener/title proximity — borderline, fine.
- L16-20 compound modifiers (input-transform, weight-transform = paper's own terms) — domain-standard.
- L205-212 "stronger transform, more noise, null-space, hardened training" — 4-element list, not a tricolon.

## Cleared

Em-dashes: ZERO prose `---` (math-mode operators only). Boilerplate: none. Cliché lexicon: none. Hedging: none. "Not only…but also": none. Sycophancy: none (keyinsight/openproblem boxes neutral).

## Top-3 concentrated paragraphs (refactor priority)

1. L48-58 KV-Cloak mechanism (densest sentence).
2. L234-244 Accumulation (comma-spliced run-on).
3. L22-32 Approaches opener (formulaic + tricolon).
