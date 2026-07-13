# §3 Cryptographic Inference — Four-Lens Audit Synthesis (2026-06-10)

Lenses: `/humanize` (humanize-auditor) · `/term-audit` (audit phase) · `/proofread` (proofreader) · `/verify-claims` (claim-verifier, fresh context). Target: `manuscript/sections/03_cryptographic.tex` (~720 prose words).

**Headline:** Voice is clean (0 HIGH AI-tells; no cliché lexicon, no boilerplate, no sycophancy, no hedging). The one systemic issue is **12 prose em-dashes** (§3 predates the no-em-dash rule). **One HIGH factual error**: CryptoMoE cost direction is reversed.

---

## P0 — HIGH factual error (must fix)

**F1. CryptoMoE direction reversed (line 42).** Draft says CryptoMoE recovers sparsity "at $2.8$--$3.5\times$ dense-MPC cost" — framed as a cost ABOVE dense MPC. Primary source (arXiv 2511.01197 abstract) reports **2.8–3.5× end-to-end latency *reduction* over a dense baseline** (plus 2.9–4.3× comm. reduction, 99.2% accuracy retained). CryptoMoE is ~2.8–3.5× **cheaper** than dense MPC, not more expensive.
→ Reword to e.g. "recovering MoE sparsity at $2.8$--$3.5\times$ lower cost than a dense-MPC baseline."

---

## P1 — Citation correctness + structural prose (MED)

**F2. `\citep` → `\citet` for textual citations (lines 14, 15, 32).** "Li et al.~\citep{...}", "Chen et al.~\citep{...}", "Li et al.~\citep{li_pti_survey} measure them" use the author name as the sentence subject, so the parenthetical form double-prints the author. Switch all three to `\citet`.

**F3. CryptoMoE sentence run-on + dangling participle (lines 39–42).** Mixed participle/finite chain ("hiding ... and has been used"); trailing "recovering sparsity..." dangles. Restructure into 2–3 sentences. **Folds together with F1** (same sentence).

**F4. Overfull-hbox verification (table rows, lines 65/67/71).** Densest `\scriptsize` rows — Muse row (4 long cells), permutation-inversion row ("non-information-theoretic"), FHE row ("Approx.-ciphertext"). House rule: verify in the actual PDF, not just a clean build.

---

## P2 — Register / terminology (term-audit + proofread overlap)

| ID | Line | Item | Proposed |
|---|---|---|---|
| F5 | 88 | "buy speed" (figurative; named in register rule) | "trade for speed" / "gain speed by" |
| F6 | 103 | "It bites only where" (colloquial) | "It applies only where" / "is exploitable only where" |
| F7 | 95 | "best read as a heuristic speed/leakage trade" | "is best characterized as a heuristic speed/leakage trade-off" |
| F8 | 13 | "at one-to-three orders of magnitude in latency or communication" (missing head noun) | "...order-of-magnitude **overhead** in latency or communication" |
| F9 | 49/55/62 vs 87/95 | "speed heuristic" coined + named 3 ways | standardize (e.g. "efficiency heuristic" / "soundness-relaxing relaxation") |
| F10 | 80 | "collapse additive sharing back to plaintext" (figurative; table L66 already says it canonically) | "recombine additive shares to recover the plaintext" |
| F11 | 82 | "quietly co-locate its 'non-colluding' parties" | "covertly co-locate the nominally non-colluding parties" |
| F12 | 95 | "drives leakage down as ${\sim}\log N$" (N undefined here) | "reduces leakage as ${\sim}\log N$ in the number of evaluators $N$" |

---

## P3 — Low / cosmetic (proofread)

- F13 (32): `$>$85\%` → `${>}85\%$` to match `${\approx}60\%$` (line 33) typography.
- F14 (42): "GBs" → "GB" (units not pluralized).
- F15 (44 vs 50/54/70): "vs.\ " (body) vs "vs" (table) — standardize.
- F16 (33–35): "per-pass overhead and interaction rounds multiply by the generation length" → "...are multiplied by..." (passive disambiguates).
- F17 (66 vs 80/83): "honest-majority" vs "non-colluding" — confirm intentional (honest-majority ≠ 2PC non-collusion). Technical check, not a typo.

---

## Em-dash sweep (12 prose occurrences — all MED, all must go)

Lines: **11, 12, 13–14, 16, 25, 28, 32–33, 33, 81–82, 83, 93–94, 95.** Several are appositive pairs better converted to parentheses/colons than split (11–12, 13–16, 25); the rest split into shorter sentences. Table en-dashes (`$1.3$--$2\times$`) and `$\Rightarrow$` are NOT prose em-dashes — leave them.

Most concentrated: intro (11–16, 4 dashes), Approaches (25–33, 4 dashes), Permutation-secrecy para (93–95, 2 dashes).

---

## Verify-claims: the other 7 (all SUPPORTED)

- Li et al. >85% MPC / ~60% HE non-linear share — **exact match** (caveat: two different cited sub-works, not one head-to-head; current "Li et al." survey attribution is fair).
- Muse malicious-client weight extraction from semi-honest 2PC — confirmed (22×–312× fewer queries).
- SIMC malicious-client "at semi-honest cost" — confirmed; exact 1.3–2× ratio plausible but not pinned to a table (LOW — confirm vs PDF if load-bearing).
- Hidden No More: fixed-permutation + decoder injectivity, names PermLLM/STIP/Centaur — confirmed verbatim.
- IND-CPA^D / Li–Micciancio CKKS key recovery + noise flooding — confirmed.
- yona_moe_steal = "Stealing User Prompts from MoE" (arXiv 2410.22884) tie-break co-batch full recovery — confirmed.
- RNS-CKKS leveled / bounded depth / expensive bootstrap — confirmed.
