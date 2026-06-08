# PDF review round 6 — comment audit (2026-06-06)

Source: `pdfannots` of annotated `main.pdf` (16 annotations; JSON==markdown==16, no dropped
buckets) **plus 2 section-wide meta-rules** from the command args (terminology audit, em-dash
sweep). Box (G3) maps to #12; attack-subtype (G4) maps to #15. All target §5 Static Obfuscation.
**M = 18.** **18 of 18 resolved** (all ✅). Build clean: 25pp, 0 undefined, no errors, no em-dashes, no overfull. Committed in `8b710f8`.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred (reason + where tracked) · ❌ won't-fix (reason) · ⬜ pending (transient).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | p8 intro | white-box adversary statement too thin (proactive gather → train deobfuscator; knows algorithm+params). | ✅ | Intro now: adversary "knows the obfuscation algorithm and its parameters, and can proactively collect the intermediary results of its own computation and use them to train a de-obfuscation model." |
| 2 | p8 intro | don't claim fidelity good across the board. | ✅ | Changed to "leads on performance and is often near-lossless. Fidelity is not uniform across the family … depends on how hard each scheme tunes its hiding." |
| 3 | p8 Approaches | observation surface = everything offloaded to untrusted accelerator. | ✅ | Rewrote: "everything the scheme offloads to the untrusted accelerator … offloaded attention, the externalized KV-cache, and the intermediate activations of the offloaded matrix multiplications." |
| 4 | p8 Approaches | "condition number" arbitrary → concrete. | ✅ | Now: larger $h$ "makes the obfuscation matrices more ill-conditioned, which amplifies floating-point error layer by layer and is what degrades accuracy." |
| 5 | p8 Approaches | KV-Cloak standalone vs composed; what does securing KV-cache protect? | ✅ | Clarified: KV-Cloak "is narrower … secures only the KV-cache … against an attacker who reads the externalized cache"; TEE holds only keys, attention runs on GPU; "blocks reconstruction of the prompt from a leaked cache"; "composes with an input/output obfuscation for end-to-end protection." |
| 6 | p8 Approaches | EE operators: linear or non-linear? | ✅ | "replaces the model's linear layers, together with a fixed set of non-linear activations (ReLU, GeLU, SiLU, RMSNorm, LayerNorm)." |
| 7 | p8 Approaches | EE covariant? grouping inconsistent. | ✅ | EE now stated as "a weight-transform scheme like AloePri"; in Perf&Accuracy grouped as "weight-transform schemes (AloePri, KV-Cloak, EE)" — consistent. |
| 8 | p9 Perf&Acc | Eguard 1.7× vs 1.6–3.4× discrepancy. | ✅ | Disambiguated: ${\approx}1.7\times$ is the inference overhead; $1.6$–$3.4\times$ is the per-training-step cost, "separate from the inference cost." |
| 9 | p9 Perf&Acc | "server-side hours" vague → exact cost. | ✅ | Replaced with the paper's figure: "over ten hours of server time" (ObfusLM Table 11). |
| 10 | p9 Perf&Acc | "algebraic identities" term check. | ✅ | Replaced: "their transforms are exactly undone: EE is equivariant by construction, and KV-Cloak's matrices cancel in the attention product." |
| 11 | p9 Perf&Acc | h not the only fidelity param. | ✅ | "its accuracy is not set by one parameter. The keymat expansion $h$, the additive Gaussian noise, and the per-head permutation window all trade accuracy for hiding." Trade-off table row updated to list all three. |
| 12 | p9 Perf&Acc | box the key insight. | ✅ | Added a `keyinsight` callout box (xcolor-only env in preamble; no tcolorbox dependency) with the deployment-difficulty insight. |
| 13 | p9 Limitations | "under-protects" too aggressive → deployment-difficulty reframe. | ✅ | Dropped "under-protects vs reported guarantees." New framing: sweet spot reachable but needs ablation; params don't transfer; tradeoff distinction (training-free-but-param-sensitive vs trained-network), stated in Perf&Accuracy + the box; Limitations opens "not as proof that obfuscation fails, but as a map of how hard each scheme has to work." |
| 14 | p9 Limitations | "glide-reflection-style" term source. | ✅ | Dropped the obscure qualifier; now "recovers input tokens directly" from obfuscated embedding matrices. |
| 15 | p9 Limitations | name ridge; add learned-inverter-on-offloaded-externalities class. | ✅ | Renamed to "closed-form (ridge) recovery"; added a "learning-based inversion" prose cluster + attack-table row (trained inverter on offloaded externalities, Kerckhoffs, no open weights), citing GEIA + Vec2Text + Depth-False-Privacy. New `geia` bib entry. |
| 16 | p10 Limitations | flawed dynamic-randomness throughline (static AloePri + hybrid GELO). | ✅ | Removed the throughline. Replaced with the deployment-difficulty lesson (Perf&Accuracy + box) and a correctly-scoped closing: only the broken schemes (STIP/Centaur/KV-Shield) "reuse a single static secret." No GELO/§7 example in the static section. |
| G1 | §5 all | terminology audit. | ✅ | Standardized: ridge / learning-based inversion / generative inversion (GEIA), equivariance, ill-conditioning. Removed invented terms (glide-reflection-style, "algebraic identities"). |
| G2 | §5 all | em-dash sweep. | ✅ | All `---` removed from §5 prose (verified 0 remain); sentences split. New standing rule saved to memory (`feedback-no-em-dashes`). |
