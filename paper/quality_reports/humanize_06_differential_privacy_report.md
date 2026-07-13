# Humanize Audit — 06_differential_privacy.tex (2026-06-09)

**Word count:** ~1,150 (prose; excludes table markup)
**Findings:** 7 (0 HIGH · 3 MED · 4 LOW). **0 em-dash-rule violations.**

## Per-category counts
| Category | HIGH | MED | LOW |
|---|---:|---:|---:|
| 1 Boilerplate transitions | 0 | 0 | 0 |
| 2 AI-cliché lexicon | 0 | 0 | 0 |
| 3 Em-dash / punctuation | 0 | 1 | 1 |
| 4 Symmetric paragraph shapes | 0 | 1 | 0 |
| 5 Tricolon abuse | 0 | 1 | 1 |
| 6 Hedging stacking | 0 | 0 | 0 |
| 7 "Not only X but also Y" | 0 | 0 | 0 |
| 8 Formulaic openers | 0 | 0 | 1 |
| 9 Hyphenation excess | 0 | 0 | 1 |
| 10 Sycophancy | 0 | 0 | 0 |

## Findings
| Line | Cat | Sev | Current | Suggested |
|---:|---|---|---|---|
| 27–33 | 3 | MED | 3-semicolon parallel enumeration (Gaussian/Laplace/exponential mechanism mapping) | optionally split into two sentences |
| 49–53 | 5 | MED | DP-Forward/SPARSE/NVDP triple: parallel "trains X" + closing parenthetical metric | vary sentence shape on one |
| 58–61, 65–86, 150–159 | 4 | MED | recurring topic→examples→summary closer ("The three are genuinely distinct…", "The two are complementary…", "fall into two classes…") | break the closing-summary tic in one paragraph |
| 31 | 3 | LOW | "Euclidean/Mahalanobis distance" slash | leave (domain shorthand) |
| 58–60 | 5 | LOW | paired tricolons (notions / what-is-trained) | leave (load-bearing) |
| 70 | 8 | LOW | "The output schemes invert the trust setup." opener | leave (single instance) |
| 91–95 | 9 | LOW | "near-plaintext"/"post-processing" compounds | leave (< 3/paragraph) |

## Concentration (top 3)
1. Local-DP ¶ (41–63) — the DP-Forward/SPARSE/NVDP parallel triple.
2. Approaches opener (27–34) — 3-semicolon enumeration.
3. Central-DP / Limitations openers — recurring topic→examples→closing-summary shape.

## Recommendation
**Cosmetic only (optional).** 0 HIGH/1000 words = clean by the calibration heuristic. No rewrite
warranted; the symmetry is content-justified exposition. If dampening desired, vary one closing
summary and/or one triple-cadence sentence so consecutive paragraphs don't echo.
