# Attacks-tables audit synthesis — 2026-06-10

Scope: the six "Known Attacks" tables. Lenses L1 structure, L2 proofread,
L3 humanize, L4 term-audit, L5 redundancy-vs-prose, L6 staleness, L7 balance.

## Summary verdict per lens

- **L6 staleness: CLEAN.** Every number in the tables (§3 `1.3–2×`, §5 `>98%`,
  §6 `>75%`, `60→19%@ε=10`, §7 `>98%`, `~6 min`) matches current prose verbatim;
  that prose was primary-source-verified in the recent four-lens passes. No stale
  numbers. One framing note below (§7 ICA row self-duplication).
- **L3 humanize: CLEAN.** No em-dashes in any table or caption (they use
  `$\Rightarrow$`, `{=}`, semicolons). No AI-voice tells in cells.
- **L1 structure: the main problem.** Two style families; spacing violations.
- **L4 term-audit: one real issue** — `Defence` (§3, §8) vs `Defense` (§4/§5/§7).
- **L5 redundancy + L7 balance: several tightenings**, concentrated in the tall rows.

---

## A. Structural fixes (mechanical, high-confidence)

### A1. Row spacing — the headline inconsistency (MANDATED by the 3pt rule)
| Table | Current | Action |
|---|---|---|
| §3 crypto | **no `\addlinespace`** | add `\addlinespace[3pt]` between every body row |
| §8 retrieval | **no `\addlinespace`** | add `\addlinespace[3pt]` between every body row |
| §6 dp | mix `[3pt]`×2 / `[5pt]`×4 | normalize all to `[3pt]` |
| §7 hybrid | all `[5pt]`×5 | normalize all to `[3pt]` |
| §4 tee, §5 obfusc | `[3pt]` ✓ | reference standard, no change |

### A2. Cross-table style normalization
| Attribute | §3 / §8 (Family B) | §4–§7 (Family A) | Proposed |
|---|---|---|---|
| font | `\scriptsize` | `\footnotesize` | **grill** (see G1) |
| `\arraystretch` | none | `1.15` | add `1.15` to §3/§8 *iff* font unified |
| `\tabcolsep` | §8 `4pt`, §3 `5pt` | `5pt` | §8 → `5pt` |
| headers | plain | `\textbf` | bold all (grill G1) |
| cite cmd | §8 `\cite` | `\citep` | §8 → `\citep` |
| "Defence/Defense" | Defence | Defense | **Defense** everywhere (§3 header+caption, §8 header) |

### A3. Two column schemas (deepest structural divergence)
- Family B (§3, §8): `Attack (cited) | {Assumption|Leakage} exploited | Schemes at risk | Defence` — no "Effect" column.
- Family A (§4–§7): `Attack | Target | Effect | Defense (±status)`.

These are genuinely different content shapes (crypto/leakage tables vs.
attack→effect tables). **Grill G2:** unify to one 4-col schema, or keep two
intentional schemas and only unify the cosmetic style (A2)? Recommendation:
keep two schemas, unify cosmetics — forcing an "Effect" column onto §3/§8 would
manufacture filler.

---

## B. Redundancy-vs-prose trims (need judgment — each shown old → new)

The body paragraph after each table already narrates every row, so cells should
be the terse reference, not a second prose copy. Candidates, tallest rows first:

**§5 `tab:obfusc-attacks` — "Single-shot weight solving" (the tallest row).**
- Effect cell, old: *"with observable weights, direction similarity recovers
  ${>}98\%$ of weights and vocabulary-matching / precomputed-noise solve the
  static secret in one shot"*
  → new: *"direction similarity recovers ${>}98\%$ of weights; vocabulary-matching
  and precomputed-noise solve the static secret in one shot"* (drop "with
  observable weights" — already the band/Target premise).
- Defense cell, old: *"a direction-changing, noised transform closes it
  (AloePri's key matrices + noise; ObfusLM's anonymity), at an accuracy cost"*
  → new: *"a direction-changing, noised transform (AloePri keys+noise; ObfusLM
  anonymity), at an accuracy cost"*.

**§4 `tab:tee-attacks` — TEE.Fail row (tallest).**
- Effect cell, old: *"extracts keys and forges attestation from deterministic
  AES-XTS memory encryption"* → new: *"extracts keys and forges attestation"*
  (the AES-XTS mechanism is in the prose and is the determinism the Target
  implies). Shortens the tallest cell by a line.

**§6 `tab:dp-attacks` — "Nearest-neighbour under metric-LDP".**
- Effect cell, old: *"indistinguishability holds only in proportion to $\ell_2$
  distance in embedding space, so far-apart inputs stay separable"*
  → new: *"indistinguishability scales with $\ell_2$ distance, so far-apart
  inputs stay separable"*.
- "Embedding / inversion-model" Effect, old: *"reconstruct input or tokens from
  the noised embedding; IMA recovers ${>}75\%$ on DP-Forward at deployment
  $\varepsilon$"* → new: *"reconstruct tokens from the noised embedding; IMA
  recovers ${>}75\%$ on DP-Forward at deployment $\varepsilon$"* (drop "input or").

**§7 `tab:hybrid-attacks` — "ICA/BSS + anchor" (self-redundant row).**
- Effect cell, old: *"hidden-state reconstruction; fails under fresh $A$, partial
  only with many in-batch anchors"* — *"fails under fresh $A$"* duplicates the
  Defense cell ("fresh per-batch $A$; shields").
  → new Effect: *"hidden-state reconstruction; partial only with many in-batch
  anchors"* (let Defense carry "fresh $A$").
- "Gram-matrix leak" Effect, old: *"$U^{\top}U{=}H^{\top}H$ leaks the hidden-state
  covariance + token-similarity spectrum (passive; narrows the BSS search)"*
  → new: *"$U^{\top}U{=}H^{\top}H$ leaks hidden-state covariance (passive; narrows
  the BSS search)"* (drop "+ token-similarity spectrum"; covered in prose).

**§3 / §8:** already terse (scriptsize). No content trims proposed; structural only.

---

## C. Balance / compactness (L7 — apply LAST, after B)

- **§5**: the "Single-shot weight solving" row drives table height. After the B
  trims it shrinks; if still tall, widen the first col `p{2.9cm}`→`p{2.6cm}` is
  *not* needed — instead the two `X` cols carry it. Re-measure in PDF.
- **§4**: TEE.Fail Status col `p{2.7cm}` holds "none: AMD and Intel declare
  physical attacks out of scope" (~3 lines). After Effect trim, consider Status
  `p{2.7cm}`→`p{2.9cm}` to pull it to 2 lines; offset first col `p{2.9cm}`→`p{2.7cm}`.
- **§7**: `p{2.3cm}` (Attack) + X + X + `p{3.1cm}` is lopsided. Attack names are
  short ("Game of Arrows", "Gram-matrix leak") — `p{2.3cm}` is fine; Defense
  `p{3.1cm}` is the widest fixed col and justified (long defense phrases). Leave
  unless a row overflows after B.
- **§3 / §8**: `Y{1.05}Y{1.0}Y{0.85}Y{1.1}` weights — balanced already; adding
  `\addlinespace` is the only change. If font is unified to footnotesize (G1),
  re-measure for overflow.

All balance changes are **verify-in-PDF**, not blind edits.

---

## Decisions to grill (Phase 3)
- **G1** Style unification of §3/§8: full promotion to Family A look
  (footnotesize + arraystretch + bold headers) vs. minimal (only 3pt spacing +
  Defense spelling + §8 colsep/citep), keeping scriptsize.
- **G2** Column schema: keep two schemas (rec.) vs. force one 4-col schema.
- **G3** Which B-trims to accept (all / structural-only / none).

---

## RESOLUTION + EXECUTION OUTCOME (2026-06-10)

Grill answers: **G1 = Minimal (keep scriptsize)**, **G2 = Keep two schemas**,
**G3 = Accept all trims**.

Applied:
- A1 spacing: §3 + §8 gained `\addlinespace[3pt]` between all rows; §6 (4×) and
  §7 (5×) normalized `[5pt]`→`[3pt]`. §4/§5 unchanged (already 3pt). All six now
  uniform 3pt.
- A2 spelling: `Defence`→`Defense` in §3 (header + caption) and §8 (header).
- B trims (all 7): §5 single-shot-solve Effect+Defense; §4 TEE.Fail Effect
  (dropped "from … AES-XTS memory encryption"); §6 embedding Effect ("input or"
  dropped) + metric-LDP Effect ("scales with ℓ₂ distance"); §7 Gram Effect
  (dropped "+ token-similarity spectrum") + ICA Effect (dropped self-duplicating
  "fails under fresh A").

Two in-flight reversals of the plan (engineering judgment on seeing full files):
- **§8 `\cite` kept (NOT changed to `\citep`).** §8 uses `\cite` consistently
  across ALL its tables (perf tables included); switching only the attacks table
  would break §8 internal consistency to fix a near-invisible cross-section diff.
- **§8 colsep already `5pt`** — the `4pt` seen in the initial grep was §8's
  *perf* table, not the attacks table. No colsep edit needed.

Balance (L7) — FOLLOW-UP after user flagged §5 still blown up: the initial pass
was wrong to call balance "no change needed" (0-overfull ≠ balanced). §5's
`Single-shot weight solving` row wrapped its `p{2.7cm}` Defense cell to ~8 lines.
Fix: (a) further trimmed that Defense cell ("AloePri keys+noise; ObfusLM
anonymity" → "AloePri, ObfusLM"); (b) replaced the rigid `p{}`+`X`+`X`+`p{}`
layout in §4/§5/§6/§7 with proportional `Y{w}` ragged columns (weights sum to
4.0), the same mechanism §3/§8 already use. Now ALL SIX tables share one column
mechanism. Per-table weights:
- §4 `Y{0.85}Y{1.1}Y{1.0}Y{1.05}` · §5 `Y{0.8}Y{1.0}Y{1.25}Y{0.95}`
- §6 `Y{0.85}Y{0.7}Y{1.5}Y{0.95}` (Effect-heavy) · §7 `Y{0.62}Y{1.28}Y{1.3}Y{0.8}`
§5's tall row dropped from ~8 lines to ~4. Build is 0 overfull / 0 underfull /
0 table-width warning; all six verified in the PDF (pp. 7, 9, 12, 15, 16, 20).
Page count 29 (unchanged). 0 undefined.

**Out-of-scope defect spotted (NOT fixed):** §9.1 (p20) still carries a red
`[TODO: narrate the headline gaps …]` block in the Stage×Mechanism gap-map.
Pre-existing, unrelated to the attacks tables.
