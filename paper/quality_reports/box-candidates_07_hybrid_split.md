# §7 Insight / Open-Problem box candidates (2026-06-08)

Budget: doc-wide ≤2 Insight, ≤2 Open Problem. §5 already has **Insight 1** + **Open Problem 1**.
So §7 adds **Insight 2** + **Open Problem 2** (1 of each) to reach the cap. Counters are document-wide.

## INSIGHT candidates

- **I-1 (REC) — Round-trips, not FLOP%, set latency.** The §7 headline. VERIFIED + enriched:
  SCX's own latency breakdown shows TEE↔GPU **memory movement dominates 94% (prefill) / 86% (generation)**
  of latency even though it offloads ~100% of compute; GPU→CPU bandwidth (2.89 GB/s) is the asymmetric
  bottleneck. TwinShield offloads 87% FLOPs but its per-head OutAttnMult/OutSoftMax structure maximizes
  crossings → serialization-bound. Span: TwinShield O(HL)/token … CMIF single handoff. Strongest, best-anchored.
- **I-2 — Exact correctness, no fidelity tax.** VERIFIED: GELO float32 exact; SCX provably lossless (ΔPPL=0,
  100% acc); TwinShield exact; ObfuscaTune near-exact (condition-number error). Unlike DP, no accuracy–privacy
  tradeoff; the cost is entirely latency. Clean but less surprising.
- **I-3 — Security reduces to refresh.** VERIFIED (GELO): fresh per-batch mixing → single-batch BSS, no
  cross-batch gain (I(H_t;U_{1:T})=I(H_t;U_t)); static breaks under multi-run attacks. BUT overlaps §5's
  static-reuse thesis + §5 Open Problem 1 (re-obfuscation scheduling). Redundancy risk across sections.

## OPEN-PROBLEM candidates

- **OP-2 (REC) — Close second-order leakage under refresh, cheaply.** VERIFIED (GELO §3.2.1): even with
  fresh per-batch A, orthogonal mixing leaks the Gram matrix (U^TU=H^TH = feature covariance; UU^T = token-
  similarity spectrum). Fixes cost: non-orthogonal A = O(n³) inverse/batch + ill-conditioning (enforce κ<100);
  shield vectors (5%, 4–10× row norm) displace user tokens → throughput loss. GELO itself lists "stronger
  formalization" as future work and disclaims a cryptographic proof. Security-axis; complements I-1 (perf-axis).
- **OP-1 — Minimize round-trips / PCIe traffic without growing TCB.** VERIFIED: SCX reaches O(1) crossings/token
  via intermediate-layer key sharing, but relaxes per-step key freshness and is still memory-movement-bound.
  Pairs tightly with I-1 (states the problem the insight raises). Risk: same axis as I-1.
- **OP-4 — Malicious-accelerator generality.** VERIFIED: only TwinShield's U-Verify (first to verify nonlinear
  SoftMax, Freivalds generalization) handles a malicious accelerator; others are honest-but-curious. Narrower.
- **OP-3 — Common-footing benchmarking.** §7 prose-supported (GELO synthetic microbenchmark vs others not on a
  common footing). More methodological than research; weaker as a boxed Open Problem.

## ⚠ Verification caveat on an EXISTING §7 claim (not a box)

The §7 prose calls ObfuscaTune's static matrices "fatal" and the table attributes Game of Arrows / static-basis
breaks to it. Primary check: the **ObfuscaTune paper does NOT state its matrices are reused across inferences or
"refreshed periodically"** (it only says different matrices per block don't matter), and it argues inversion is
"not applicable by design" via TEE authentication. Game of Arrows' demonstrated scope is on-device TSLP
(TransLinkGuard/SOTER), not ObfuscaTune. The table already hedges ("ObfuscaTune-class", "shares it") = transitive.
Recommend: do NOT harden "ObfuscaTune is broken" in any new box; keep the refresh framing about the demonstrated
schemes. (Same over-attribution pattern as the §5 fix.)

## Recommendation

Insight 2 = **I-1** (round-trips/serialization, perf-axis). Open Problem 2 = **OP-2** (second-order leakage under
refresh, security-axis). Complementary; both strongly primary-source-verified; neither leans on the shaky
ObfuscaTune-broken claim.
