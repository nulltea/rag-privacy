# Humanize Audit: 07_hybrid_split.tex (2026-06-08)

**Word count:** ~960 (prose only). **Findings:** 9 (1 HIGH systemic, 4 MED, 4 LOW).
**Recommendation:** cosmetic on the AI-voice axes, BUT one mandatory item — **25 prose `---` em-dashes** violate the project hard rule. §7 predates the rule (committed `a2eed77`, round-4); never swept.

## Per-category counts

| Category | HIGH | MED | LOW |
|---|---:|---:|---:|
| 1 Boilerplate transitions | 0 | 0 | 1 |
| 2 AI-cliché lexicon | 0 | 0 | 0 |
| 3 Em-dash / punctuation | 1 | 3 | 0 |
| 4 Symmetric paragraph shapes | 0 | 0 | 0 |
| 5 Tricolon abuse | 0 | 1 | 1 |
| 6 Hedging stacking | 0 | 0 | 0 |
| 7 "Not only X but also Y" | 0 | 0 | 0 |
| 8 Formulaic openers | 0 | 0 | 1 |
| 9 Hyphenation excess | 0 | 0 | 0 |
| 10 Sycophancy | 0 | 0 | 0 |

## HIGH (mandatory) — prose em-dash sweep

25 literal `---` outside comments, verified by grep. Lines: 17, 19, 22, 23, 31, 32, 47, 48, 51, 53, 56, 58, 62, 63, 79, 132, 135, 136, 142, 143, 149, 159, 162, 165, 180, 181, 189, 191.

Grammatical roles:
- **Clause-joiners → sentence break:** 17, 19, 22/23, 79, 132, 142/143, 149, 159, 162, 165.
- **Parenthetical asides → parentheses/commas:** 31/32, 47/48, 51/53, 56/58, 62/63, 135/136, 180/181, 189, 191.

## MED

- **L17-24 opener** — 3 clause-joining em-dashes incl. a dash-bracketed aside nested in another dash clause. Densest. Rephrase to short sentences.
- **L42-70 Approaches ¶2** — GELO sentence ~6 clauses, TwinShield ~5; split the two longest.
- **L131-154 Performance ¶1** — TwinShield sentence (L142-150) runs ~9 lines (dash + per-head aside + participial tail). Split into 2-3.
- **L156-168 binding-constraint** — 4-scheme semicolon cadence (legitimate); only the embedded dashes (L159, L162) need touching.

## LOW (acceptable, author judgment)

- L33 tricolon (transform/inverse, non-linearities, keys) — technical, legitimate.
- L189 tricolon (inversion, collision, injection) — matches table terms.
- L30 opener "Every scheme … makes the same two-part design choice" — substantive, fine.
- L187 "Finally," — single sequential connector, fine.

## Cleared

No cliché lexicon (no leverage/delve/crucial role/underscore the importance), no sycophancy, no hedge-stacking, no "not only…but also," no symmetric paragraph shapes, no hyphenation pile-ups.

## Top-3 concentrated paragraphs

1. L13-24 scope-framing opener (3 clause-joining dashes, nested aside).
2. L142-154 Performance / TwinShield run-on (densest single sentence).
3. L156-168 binding-constraint synthesis (2 embedded dashes).
