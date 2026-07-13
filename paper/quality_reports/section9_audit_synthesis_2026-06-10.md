# §9 Synthesis — four-lens audit synthesis (2026-06-10)

Target: `manuscript/sections/09_synthesis.tex` (just-rewritten prose, post round-9). Lenses:
/humanize (voice), /proofread (grammar), /verify-claims (facts), /term-audit (register, done inline).
Agent reports: humanize-auditor, proofreader, claim-verifier (all read-only).

## Verdict
- **verify-claims: 13/13 SUPPORTED, 0 warnings.** No factual fixes needed.
- **humanize: cosmetic (0.6 HIGH/1000w).** No boilerplate/cliché/hedging/em-dash tells.
- **proofread: ~6 MED grammar + several LOW.** Clean scheme-name usage otherwise.
- **term-audit: light** (a few register nits, below).

## verify-claims (claim-verifier) — all SUPPORTED
13 claims checked vs §3–§8 + web (SPDZ, Freivalds, Blackwell TEE-I/O). All internally consistent +
externally corroborated. **One nuance (not an error):** "per-token TEE↔GPU round-trips" is a
family-level generalization; CMIF (single handoff) and SCX (O(1)/token) sit at the low end. The
generation-weakness claim holds because every member multiplies per-pass cost by (1+T). → optional soften.

## humanize (voice) — cosmetic
| # | line | cat | sev | item | resolution |
|---|---|---|---|---|---|
| H1 | 15–28 (9.1 ¶) | symmetric shape | MED | five-fold "X attains … and pays …" parallel cadence | GRILL: break rhythm vs keep clean geometry |
| H2 | 29–32 | tricolon/4-list | MED | four-element "a real-time service … a regulated workload … a cost-sensitive … a quality-critical" list in same ¶ as H1 | with H1 |
| H3 | 190 | sycophancy | MED | "The gap is real and worth naming:" | APPLY: drop, state finding directly |
| H4 | 15–33 | hyphenation | HIGH(low-stakes) | ≥3 compound hyphenates in 9.1 ¶ (deployment-readiness, security-basis, near-plaintext ×2, cost-sensitive, quality-critical) | APPLY: de-hyphenate 1–2 |
| — | 9, 220 | formulaic/superlative | LOW | "This section draws…", "the clearest open problem" | leave (discipline-legitimate) |

## proofread (grammar) — apply batch
| # | line | sev | current | fix |
|---|---|---|---|---|
| P1 | 27–28 | MED | "pays in the TEE↔GPU serialization its offloading induces" | add "that": "serialization that its offloading induces" |
| P2 | 139,141 | MED | "One is structural, where …"; "The other is performance, where …" | "where" misused for reasons → colon: "One is structural: …"; "The other is performance: …" |
| P3 | 206–207 | MED | TEE.Fail "which … and which …" chained relatives | split into two sentences |
| P4 | 163–164 | MED | long refresh sentence, ambiguous "they", tricolon mid-clause | recast; parenthesize the (activations, attention, KV-cache) list; fix antecedent (em-dash-free) |
| P5 | 183 | MED | "Nvidia-CC" vs "NVIDIA" (L150/153) | KEEP: "Nvidia-CC" is the canonical scheme name in the tables; "NVIDIA" is the vendor. Different referents → no change (ℹ️) |
| P6 | 111–112 | MED | op:no-e2e box appositive run-on | split: "…uncharacterized (\cref{op:dp-compose}). The prior RAG-privacy SoK reaches the same gap~\citep{bodea_sok}." |
| P7 | 152 | LOW–MED | "the wider platform space is narrow" (self-contradiction) | "the platform space beyond these is narrow and still forming" |
| P8 | 28–29 | LOW | "forced by the basis it rests on rather than incidental" dangles | "…rests on, not incidental" |
| P9 | 199 | LOW | nested appositive "power analysis and the memory bus among them" | parenthesize: "(power analysis and the memory bus among them)" |
| P10 | 101 | LOW | "takes … at a storage overhead" | "at the cost of storage overhead" |
| P11 | 218 | LOW | "access hiding by Opal or ORAM" (category-mixed) | "access hiding by Opal (TEE) or a client-side ORAM index" |
| P12 | 124 | LOW | "(\cref{sec:crypto},~\cref{sec:retrieval})" odd tie | ", " : "(\cref{sec:crypto}, \cref{sec:retrieval})" |

## term-audit (register) — light
- "sore point" (L69, generation) — colloquial; → "weak point" or "bottleneck". APPLY.
- "buys" (L141, "buys lower server cost") — register; the paper standardized "buys"→"trades/offers" in §2/§6 audits. → "lowers server cost or saves rounds". APPLY.
- "sidesteps" (L153) — acceptable; leave.
- "throat-clearing" H3 already covered.

## Grill decisions — RESOLVED + APPLIED (2026-06-10)
- G1: **Break the 9.1 cadence** — varied 3 of the 5 family sentences (matches/but its basis is · is the cheapest/but rests on · keeps/adds/paying · recovers/pays instead), de-hyphenated "security-basis"→"security basis". APPLIED.
- G2: **Keep** the per-token-roundtrips family summary (verified fair; §7 carries the full range). No change.
- G3: **Applied** the full batch: P1 "that", P2 where→colon ×2, P3 TEE.Fail split, P4 refresh-sentence parenthetical, P6 op:no-e2e split, P7 "beyond these", P8 "not incidental", P9 power/memory-bus parenthetical, P10 "at the cost of", P11 "Opal (TEE) or a client-side ORAM index", P12 ,~ fixed; term sore-point→weak-point, buys→lowers/saves; H3 "The gap is real and worth naming"→"A gap remains". **KEPT Nvidia-CC** (canonical scheme name vs NVIDIA vendor — ℹ️ no change).
- Build after: 29pp, 0 undefined, 0 overfull, no prose em-dashes; sole `??` = deferred tab:deployment caption (out of scope).
- verify-claims: 13/13 SUPPORTED — no factual edits needed.
