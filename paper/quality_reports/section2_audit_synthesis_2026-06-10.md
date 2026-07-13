# §2 Evaluation Framework — Four-Lens Audit Synthesis (2026-06-10)

File: `manuscript/sections/02_evaluation_framework.tex` (~2,400 prose words + 2 tables).
Lenses: `/humanize` · `/term-audit` · `/proofread` · `/verify-claims` (forked claim-verifier, fresh context).

## Headline

Voice is clean (0 HIGH AI-tells: no boilerplate transitions, no sycophancy, no hedging stacks,
no symmetric paragraph shapes). The work is concentrated in four buckets:

1. **Em-dash sweep** — §2 was MISSED in the prior §3/§4/§6/§8 sweeps. ~25 prose `---` violations.
2. **THREE factual errors in the extrapolation footnote** (L237–240) caught by verify-claims.
3. **One table contradiction** (GELO CPU-TEE vs GPU-TEE) — primer-flagged, still unfixed.
4. **Light term/proofread cleanup** — uncited "Li et al.", `Mb` unit, residual "H100 CC" name.

---

## A. verify-claims — factual (RESOLVED, primary-source-checked)

The footnote-extrapolation investigation went two rounds. The first-round verifier produced
**two false positives** that deeper checks (correct eprint IDs, full tables) reversed. Final outcome:

### A.1 The crypto 1-hop extrapolations were both unsound → replaced with own-figures

| Scheme | Old (extrapolated) | Finding | Applied fix |
|---|---|---|---|
| SHAFT L198 | `≈1.4 s/inf†; ≈0.6 GB/inf†` | The 1-hop divided SIGMA's **pure-GPU-compute** 1.72 s by SHAFT's only reported SIGMA speedup ("1.8–2.4×", BERT 2.32×), which is a **LAN end-to-end running-time** ratio — a unit mismatch. The row numbers matched neither 2.32× nor the 70% comm reduction. SHAFT reports its own BERT-base directly (Table VII, eprint **2025/2324**): 41.25 s LAN / 281.36 s WAN / 10.46 GB. | → `41/281 s LAN/WAN; 10.5 GB/inf` |
| Euston L200 | `≈10.7 s/inf†; ≈37 MB/inf†` | `10.7 ≈ 37.3/3.5` = NEXUS **GPU** latency ÷ Euston **CPU** speedup — a cross-config fabrication; the two are also on different hardware (4×A100 vs 1×RTX 6000 Ada). Euston's own amortized BERT-base GPU latency = **46 s/inf** (Fig. 6; already per-input, do NOT ÷32). Euston reports **no absolute MB** comm — only relative ("4.4× less than NEXUS"); the 37 MB was never Euston's. | → `46 s/inf (GPU)`; comm dropped |

Consequence: no row carries `†`, so the `†` footnote (L237–240) and the now-inapplicable
"One-hop relative normalization" paragraph (L289–296) were **removed**. This also resolved the
uncited "per Li et al." (it lived in the deleted footnote).

### A.2 Two FALSE-POSITIVE flags — reversed, NO change made

| Flag | First-round claim | Reversal |
|---|---|---|
| SHAFT "41 s LAN" wrong | "should be 28.6 s" | **41 s is correct.** The 28.6 came from the **wrong paper** (eprint 2025/2287 = "MIOPE", not SHAFT). Correct SHAFT 2025/2324 Table VII = 41.25 s LAN. Footnote was right; kept. |
| PermLLM "20 Mb/tok" | "verify bits vs bytes; likely MB" | **20 Mb (megabits) is correct**, verbatim from PermLLM arXiv 2405.18744 §1. NO change. |

### A.3 Verified SUPPORTED (no change)
AloePri direct quote (verbatim, arXiv 2603.01499 §5.1) · GELO per-batch fresh mixing matrix +
single-batch BSS (arXiv 2603.05035v2) · Chen et al. WAN-miss finding (eprint 2026/491, verbatim) ·
NEXUS Δacc upper bound **0.40%** not 0.38% (moot — NEXUS line deleted with the footnote).

**Lesson reinforced:** verify eprint IDs and read the actual table before "fixing" a number; two
plausible-looking corrections were themselves wrong. Don't fill/replace a number from a single
verifier pass when the source is ambiguous.

---

## B. Table contradiction (primer-flagged, structural) — RESOLVED + APPLIED

GELO's architecture verified verbatim (arXiv 2603.05035v2): **two devices** — a *trusted
confidential GPU* (H100/H200, the TEE) applies the secret mixing matrix `A`; only the heavy
self-attention matmuls offload to a *separate untrusted commodity GPU* (L40S), which sees only
`U=AH`; the TEE un-mixes via `Q=A⁻¹Y`. No CPU-TEE anywhere. The confidential GPU is the trusted
**baseline**; the untrusted GPU is the throughput *add-on* (inverse of the original premise).
→ `tab:inf-perf` L228 fixed **CPU-TEE → GPU-TEE**; matches L78, §7, Table 2.

**Second-order flag (left for §7 per user):** this makes GELO a Tier-3 scheme, unlike the other
four §7 hybrids (TwinShield/ObfuscaTune/SCX/CMIF = CPU-TEE/Tier-2). §2 L117–120's generalization
"the hybrid split's defining move is to deliver Tier-3 perf at Tier 2" does not cover GELO. Left
as a §7 concern.

---

## C. Em-dash sweep (HARD RULE — §2 was missed)

~25 prose em-dashes. Three need sentence-level recast; the rest are mechanical (→ colon / period / comma / parens).

- **Recast (stacked, 3 dashes):** L10–11 (two-kinds-of-dimension), L109–110 (Tier-1 thin-client), L117 (Tier-2→3 jump).
- **Comma pairs:** L142 (residual leakage, not the stated goal), L166 (places: nothing/CPU-TEE/GPU-TEE), L171 (Fourth, and most consequential), L258 (threat-model fit, not maximality).
- **Colon/period swaps:** L14, L22, L94, L107, L113, L114, L133, L136–137, L139, L247, L281–282, L287.
- **NOT voice tells (separate decision):** 10 table band-label separators `\emph{Family} --- \cref{}` (L48/55/61/70/77, L197/204/211/220/227).

## D. Proofread

| Line | Sev | Issue | Fix |
|---|---|---|---|
| 239 | HIGH | "per Li et al." named, **uncited** | add `\citet{...}` (the deferred MPC/FHE survey, 2505.10315) |
| 201 | MED | PermLLM "20\,Mb/tok" (megabits) breaks the GB/MB byte convention | confirm bits-vs-bytes; if bytes → "MB" |
| 205 | MED | "GPU-TEE (H100 **CC**)" = residual of renamed "Nvidia-CC" | drop "CC" / use "H100/H200-class" |
| 217, 208 | MED | densest `tab:inf-perf` rows (ObfusLM, Opal), non-wrapping `lllll` | **verify in PDF** for overfull hbox |
| 23 | LOW | "defence" (UK) vs project "Defense" | → "defense" |
| 266 | LOW | "artefact" (UK) vs US spelling elsewhere | → "artifact" |
| 58 | LOW | `PCIe \texttt{act}.` trailing period | drop period |
| 319 | LOW | leftover relocation comment | remove (no-correction-commentary) |

## E. term-audit (word-choice / register)

| Line | Term | Issue | Suggested |
|---|---|---|---|
| 103 | "buys performance or simplicity" | colloquial metaphor (×appears twice as framing) | "trades a stronger trust assumption for performance or simplicity" |
| 93–94 | "mines every observable" | colloquial metaphor | "inspects/examines every observable" |
| 164 | "stake everything on an ephemeral secret" | colloquial | "depend entirely on an ephemeral secret" |
| 257–258 | "strawman schemes that never claimed that bar" | "strawman" (verb) + "that bar" colloquial | "misrepresent schemes that never claimed it" |
| 155 | "reads most heavily" | metaphor | "weights most heavily" |
| 171 | "cleaves the table" | metaphor | borderline; keep or "separates the table" |

Borderline-KEEP (domain-idiomatic, not flagged): "load-bearing" (client), "thin-client", "overpays in performance", "sharpest escalation/predictor".
