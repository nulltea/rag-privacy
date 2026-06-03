# PDF review round 1 — comment audit (2026-06-03)

Source: `pdfannots` extraction of the annotated `main.pdf` (annotated copy since overwritten by
rebuilds; comments preserved here). **All 31 comments resolved** (1 answered without a change);
see the resolution note below. Committed in `80a3f42` (content/terminology) and `8b3a51b`
(structural items).

Legend: ✅ done · ℹ️ answered, no change.

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | §1.2 | "defer" — standard verb? | ✅ | §1.2 + §9 → "refer the reader to"; §3 → "focus on". (Deferring to *future*/companion work kept — standard.) |
| 2 | §1.3 | "read off the categorical taxonomy" unclear | ✅ | → "determined by". |
| 3 | §1.3 | "(section 8)" use shorthand | ✅ | Global ref style → `§8` (cleveref+autoref both). |
| 4 | §1.3 | don't reference "subsection 2.1" like that | ✅ | Now `§2.1` (compact); kept subsection refs in §X.Y form per your choice. |
| 5 | §1.4 | side-channel blind spot — why here / contradicts out-of-scope | ✅ | Reworded to "surfacing the field-wide blind spot we report as a finding in §8"; exclusion preserved. |
| 6 | §1.4 | "mechanism" — better term? | ℹ️ | Kept "mechanism family" (defense classes) vs "scheme" (individual systems) — distinct; rationale noted. |
| 7 | §1.5 | "Roadmap" → Organization? | ✅ | Renamed to "Organization" + real mapping paragraph. |
| 8 | §2 | "...other two criteria" tautology | ✅ | Cut. |
| 9 | §2.1 | "vignette" standard term? | ✅ | "vignette"→"case study" throughout; the case study itself moved to `app:validation`. |
| 10 | §2.1 | "ciphertexts plus distance" — only cosine distance | ✅ | → "cosine-distance structure (the distance, not the ciphertext, is the leak)". |
| 11 | §2.1 | "...nonetheless root trust..." hard to read | ✅ | Rewrote without "nonetheless". |
| 12 | §2.1 | caption `ct+dist` → cosine distance | ✅ | Caption updated. |
| 13 | §2.1 | are AloePri/STIP the only static-obf schemes? | ✅ | §2 landscape now states the table is representative; full rosters in §3–§7. |
| 14 | §2.3 | unbounded adversary — static or dynamic (GELO)? info-theoretic? | ✅ | Reworded precisely: static obf breaks; GELO per-batch concedes nothing across batches (single-batch BSS, *not* info-theoretic); SGT's info-theoretic claim flagged contested. |
| 15 | §2.3 | "of section 1" — too much retroactive referencing | ✅ | Dropped; trimmed the recurring "deployment-readiness criteria of §1" backward cite across §3/§4/§5/§6. |
| 16 | §2.3 | vignette constrained to AloePri — one-sided | ✅ | Built: appendix `app:validation` covers the heuristic-basis set (AloePri/SGT/OSNIP/GELO/ObfuscaTune); finding + reproduction protocol; recovery figures `\needswork` (companion). |
| 17 | §2.2 | "fraught" wording | ✅ | → "hard to compare fairly". |
| 18 | §2.2 | "yardstick" uncommon | ✅ | → "reference metrics" (visible word gone; internal label `tab:yardstick` unchanged, invisible). |
| 19 | §2.2 | head-to-head / self-reported / provenance unclear | ✅ | Restructured to "indicative by default / co-measured"; swept §2+§8 (the §5 residual is gone with the vignette block). |
| 20 | §3 | "place this section's weight on" → focus on | ✅ | §3. |
| 21 | §3 | Table mixes vector + graph retrieval — separate | ✅ | Split into `tab:crypto-vec` + `tab:crypto-graph`. |
| 22 | §3.1 | "absolute figures below" → cite a table | ✅ | → "in `\cref{tab:crypto-inf}`". |
| 23 | §3.1 | "(subsection 2.2)" retroactive ref | ✅ | Now `§2.2`; retroactive-ref pass done (see #15). |
| 24 | §4 | TEE table appears before its reference — anchor after | ✅ | Per-family scheme tables (§3–§7) pinned with `[H]`; `table*` summaries kept `[t]`. |
| 25 | §5 | "token substitution" — hallucinated? | ✅ | Confirmed real (SANTEXT/RANTEXT); clarified inline. |
| 26 | §5 | "unintelligible values" — vague | ✅ | → "values the server cannot interpret". |
| 27 | §5 | also runs on vLLM/llama.cpp (deployment factor) | ✅ | Added infra-compatibility sentence. |
| 28 | §5 | "TCB of none or light" — vague | ✅ | → "no trusted hardware (Tier 1)". |
| 29 | §5 | "study it offline" — arbitrary | ✅ | → "analyze it in advance". |
| 30 | §5 | vignette surface/companion paragraph — verbose/redundant | ✅ | Removed with the vignette restructure. |
| 31 | §6 | DP intro misses retrained-model factor | ✅ | Added the embedding-DP retraining divide. |
| + | §3 | **NEW:** Table 6 drifted off Table 5's page | ✅ | `[H]` (float package) pins both consecutively. |

## Resolution — all four structural items now done

- **Vignette restructure** (#9, #16, #19-residual, #30): moved §5 vignette → appendix
  `\section{Attack Validation of Deployment-Ready Schemes}` (`app:validation`), generalized to the
  heuristic-basis deployment-ready set (AloePri, SGT, OSNIP, GELO, ObfuscaTune) with a reproduction
  protocol + `tab:validation`; recovery figures `\needswork` (companion-owned). "vignette"→"case
  study" everywhere; verbose surface paragraph removed; all cross-refs → `app:validation`.
- **Table-placement** (#24): per-family scheme tables pinned with `[H]` (float pkg) so none float
  ahead of their section; full-width `table*` summaries left as `[t]`.
- **AloePri/STIP "only two?"** (#13): §2 landscape now notes the table is representative, full
  rosters in §3–§7.
- **Retroactive-ref reduction** (#15, #23): trimmed the recurring "deployment-readiness criteria of
  §1" backward cite in §3/§4/§5/§6 framings (kept the load-bearing §8 scoring-table ref).

Build after all items: 22 pp, 0 undefined refs/cites; only the pre-existing ~43 pt `tab:refmetrics`
overfull. Recovery figures in `tab:validation` and the companion-paper forward citation remain the
sole `\needswork` for this review round.
