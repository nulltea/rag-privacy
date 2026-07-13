# PDF review round 9 — comment audit (2026-06-10)

Source: `pdfannots` extraction of annotated `main.pdf` (saved 21:21; overwritten by rebuilds, preserved here).
All 48 comments are on **§9 Synthesis** (the just-written prose, pp20–22). Reconciled: JSON=48 == markdown=48 == rows=48.
Resolution driven by `/grill-me` (5 selector decisions) + EdgeQuake/web verification. Build after: 29pp, 0 undefined, 0 overfull.
**48 of 48 resolved (all ✅).**

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred · ❌ won't-fix · ⬜ pending (none remain).

Grill decisions: (D1) 9.2 reframed application-constraint-driven, no "leaders", hybrid-split repositioned to embedding/reranking;
(D2) §1 softened to keep the not-way-stations contrast without "leaders"; (D3) MPC niche = collaborative inference (9.2 sentences);
(D4) finding #4 reframed to "integrity surcharge: cryptographic OR hardware" + MPC+MAC gap; (D5) vector-retrieval two-tiered, CAPRISE credited.

| # | Page / loc | Comment (gist) | Status | What changed |
|---|---|---|---|---|
| 1 | p20 intro | Headline contradicts "no scheme satisfies all three"; wrong to call families "leaders" | ✅ | Intro headline → "no family dominates; which scheme is deployable depends on the application"; dropped the "frontier held by hybrid+obfusc" claim |
| 2 | p20 9.1 | Duplication: 9.1 repeats intro "maxes ≤2" — remove from intro | ✅ | Removed the "each maximizes ≤2" mechanism from the intro; kept only in 9.1 |
| 3 | p20 9.1 | availability, not owning | ✅ | "owning the accelerator" → "availability of the accelerator" |
| 4 | p20 9.1 | DP = probabilistic formal | ✅ | "formal threat model" → "formal but probabilistic threat model" |
| 5 | p20 9.1 | Conclude: choice depends on application needs (latency, security budget…) | ✅ | 9.1 closing now names the deciding constraints: latency budget / provable basis / hardware on hand / fidelity to spare |
| 6 | p20 9.2 | "Turning the geometry" — arbitrary | ✅ | Opening → "The same trade reads differently in each use case" |
| 7 | p20 9.2 | NOT hybrid-split for generation; TEE+obfusc lead; hybrid = embedding/reranking | ✅ | Generation reframed: TEE (if owned) else static obfusc; hybrid noted as weak fit (per-token round-trips); hybrid moved to embedding/reranking bullet |
| 8 | p21 9.2 | Conf accelerator usually not owned | ✅ | "hardware is scarce and few deployments own it" |
| 9 | p21 9.2 | MPC justifiable only for collaborative inference; research | ✅ | Added: MPC/FHE natural setting = collaborative inference among mutually-distrusting data holders (e.g. disjoint institutional records) |
| 10 | p21 9.2 | "leaders/way-stations" arbitrary | ✅ | Removed entirely; replaced by constraint-driven recommendations |
| 11 | p21 9.2 | Hybrid relevant for embedding | ✅ | Embedding/reranking bullet: "this is the hybrid split's natural setting" (CPU-TEE + commodity GPU, short non-autoregressive workload) |
| 12 | p21 9.2 | "oblivious ones" — meaning | ✅ | Replaced with "access-hiding schemes"; spelled out PIR (returns record without revealing which) + HE scoring (values stay encrypted) |
| 13 | p21 9.2 | CAPRISE/DCPE promising; research | ✅ | Two-tier vector retrieval: CAPRISE credited (conditional DCPE + query-DP, ~1× cost, blocks inversion, hides inter-doc structure unlike SAP); deciding question = is the residual comparability leak tolerable |
| 14 | p21 9.2 | Forgetting Pacmann | ✅ | Pacmann added to vector ANN and to graph access-hiding (batched PIR, cryptographic) |
| 15 | p21 9.2 | Graph bullet has no trade-offs | ✅ | Rewrote graph bullet as cost-vs-channel trade-offs (node-access: client crypto vs TEE; volume: VH-EMM at storage cost; else anonymization); "no single scheme closes all three" |
| 16 | p21 9.3-1 | TCB does say where work runs; artifact we close; concise | ✅ | Opening: trust-anchor axis records where *trusted* compute runs and by construction brackets the client (a client is not its own trusted base); this finding surfaces the bracketed work |
| 17 | p21 9.3-1 | "reading down the families" confusing | ✅ | Removed |
| 18 | p21 9.3-1 | "covariant" static obfuscation | ✅ | "Static weight obfuscation" → "Covariant weight obfuscation" |
| 19 | p21 9.3-1 | How does DP-Forward show tension? Contradiction w/ operator fine-tune | ✅ | Separated the two: client runs the input-side layers online (more layers = more utility + more client compute); the offline fine-tune that calibrates noise runs on the operator — standing burden is the online pass, not setup |
| 20 | p21 9.3-1 | Don't reiterate SPARSE/NVDP offline perf | ✅ | Deleted the SPARSE-25min / NVDP-layer sentence |
| 21 | p21 9.3-1 | SCX every-token? replaces TEE? verify | ✅ | Verified: SCX = on-cloud enclave + client encodes each token & recovers first/last blocks (cloud handles the rest); reframed as "light but standing per-token role," does NOT replace the TEE |
| 22 | p21 9.3-1 | Name the oblivious graph schemes | ✅ | "Compass keeps an ORAM position map and runs a multi-round controller, and Pacmann preprocesses the database client-side" |
| 23 | p21 9.3-1 | TEE = workaround not counterpoint | ✅ | "the TEE is the workaround rather than a counterexample … trading the client's burden for trust in the hardware" |
| 24 | p21 9.3-2 | "coarse-grained" word | ✅ | Heading → "Confidential accelerators are scarce and enterprise-only" |
| 25 | p21 9.3-2 | specify confidential-accelerator/GPU-TEE | ✅ | "pure-TEE path" → "pure confidential-accelerator path, a GPU- or NPU-TEE trusted end to end" |
| 26 | p21 9.3-2 | "trades … for" not "converts" | ✅ | "trades the confidentiality problem for a hardware-procurement one" |
| 27 | p21 9.3-2 | verify only two platforms | ✅ | Reframed to "the confidential accelerators *our corpus draws on*" + noted the wider space is narrow/still-forming (recent, enterprise-only, inline TEE-I/O only with Blackwell) — verified via NVIDIA/AMD SEV-TIO sources |
| 28 | p21 9.3-2 | "part"? | ✅ | "confidential part" → "confidential accelerator" |
| 29 | p21 9.3-2 | "right-size" colloquial | ✅ | "right-size it" → "match the smaller workload" |
| 30 | p22 9.3-3 | "accumulate observations" → externalities/residual leakage | ✅ | "accumulate the externalities the scheme offloads, the obfuscated activations, attention, and externalized KV-cache" |
| 31 | p22 9.3-3 | "hidden quantity identifiable" — concrete + ":" | ✅ | "…until they pin down what the secret was meant to hide: the hidden states, and through them the input" |
| 32 | p22 9.3-3 | "fall to" wrong | ✅ | "fall to" → "are broken by" |
| 33 | p22 9.3-3 | "schemes that stand" colloquial | ✅ | Heading → "a refreshed secret is what survives accumulation"; body "schemes that stand" → "schemes that survive" |
| 34 | p22 9.3-3 | SCX one-time-key-per-token verify | ✅ | Verified against SCX paper (one-time keys per decoding step by default); kept |
| 35 | p22 9.3-3 | refresh doesn't hide Gram; name attack | ✅ | Made concrete: $U^{\top}U{=}H^{\top}H$ for any orthogonal mixing, so per-batch refresh still leaks pairwise inner products → feeds blind-source-separation + anchor attacks (§7) |
| 36 | p22 9.3-4 | "buys a hardware dependency" confusing | ✅ | Heading → "Malicious-operator security is an added integrity layer, not a free upgrade" |
| 37 | p22 9.3-4 | "costs more still" | ✅ | "costs more again" |
| 38 | p22 9.3-4 | why ref §3 | ✅ | §3 ref now naturally motivated ("a malicious client can already extract the model from a semi-honest protocol") |
| 39 | p22 9.3-4 | full malicious security vs malicious-operator defense | ✅ | Added: "defending a deviating operator is narrower than full malicious security, which also covers a cheating client and costs more again; the surveyed schemes provide neither" |
| 40 | p22 9.3-4 | not only hardware — Opal/TwinShield crypto | ✅ | Reframed to two routes: cryptographic (SPDZ MACs, TwinShield Freivalds/U-Verify, Opal Merkle freshness) OR hardware attestation; "not hardware-exclusive" |
| 41 | p22 9.3-4 | MPC+MAC malicious security; classification gap | ✅ | Verified SPDZ active security via IT-MACs; added the gap: surveyed crypto schemes are semi-honest, MACs could secure the linear core but permutation-reveal tricks sit outside MAC protection |
| 42 | p22 9.3-4 | not only trusted hardware | ✅ | Covered by the two-route reframe (crypto OR hardware) |
| 43 | p22 9.3-5 | "Reading down" AI tell | ✅ | "Reading down the residual-leakage axis…" → "The residual-leakage axis…" |
| 44 | p22 9.3-5 | power is a physical channel | ✅ | "microarchitectural, cache, timing, and physical side channels, power analysis and the memory bus among them" |
| 45 | p22 9.3-6 | (??) don't reference table | ✅ | Removed `\cref{tab:gapmap}` from the graph-RAG finding (and from the op:no-e2e box); the prose `??` is gone |
| 46 | p22 9.3-6 | ARoG covers construction? don't say "no treatment at all" | ✅ | Reframed: "the setting the field, ours included, covers least"; ARoG kept credited for anonymization; "one stage stays uncovered across the corpus: building the graph index under confidentiality" (ARoG = query-time KGQA anon, not index construction) |
| 47 | p22 9.3-6 | drop "generating over subgraph" / GNN | ✅ | Removed the "generating over the retrieved private subgraph" component entirely |
| 48 | p22 9.3-6 | last sentence arbitrary/duplicating | ✅ | Cut; closes on "Confidential graph construction is the clearest open problem in this setting" |

## Reconciliation gate
- Extracted M = 48 (JSON) == 48 (markdown) == 48 audit rows == 48 terminal statuses. Zero ⬜.
- Breakdown: **48 ✅** · 0 ℹ️ · 0 ⏸ · 0 ❌.
- Build: 29pp, 0 undefined, 0 overfull, no em-dashes in §9 prose, no `??` in §9 prose.

## Flagged (NOT one of the 48; deferred-table scope)
- `tab:deployment` caption still says "see \cref{tab:performance}", but `\label{tab:performance}` and `\label{tab:gapmap}`
  are **commented out** in the source (user's table-deferral WIP), so the caption renders `(??)`. This is inside the
  deferred-tables work the user set aside; left for the table revisit. Fix when un-deferring the tables (re-enable the
  labels or drop the cross-ref). §1 edit verified clean (no `??` on pp1–2).
