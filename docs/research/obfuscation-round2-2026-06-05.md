---
type: research
status: current
created: 2026-06-05
updated: 2026-06-05
tags: [static-obfuscation, hybrid-split, kv-cache, permutation-attacks, sok-corpus]
companion: [scheme-categorization, private-llm-inference, aloepri-vs-gelo]
---

# Static-Obfuscation / Hybrid-Split research round 2 (2026-06-05)

Second sweep to surface static-obfuscation private-transformer-inference schemes the
corpus (§4/§5 obfuscation, §6/§7 hybrid split) missed. Sources: OpenAlex, WebSearch,
EdgeQuake full-text (KV-Cloak), arXiv. Trigger: §7 Hybrid Split write-up + the
"Shadow in the Cache" KV-cache paper (already in EdgeQuake).

## Correction of record: KV-Cloak

**KV-Cloak is NOT a separate paper and is NOT a compute-offload TEE-hybrid.** It is the
defense proposed *inside* "Shadow in the Cache: Unveiling and Mitigating Privacy Risks of
KV-cache in LLM Inference" (Zhifan Luo et al., **NDSS 2026**, arXiv **2508.09442**;
ZJU + Huawei) — already in EdgeQuake (`9ed851a6...`, completed, 38 chunks).

- **Mechanism = pure-software obfuscation:** `K' = S·P̂·(K+A)·M` — persistent secret
  invertible matrices `S` (b×b), `M` (d×d) fused offline into attention weights via
  **operator fusion**; a **one-time random permutation `P̂` per PagedAttention block**
  (the dynamic part that severs cross-query algebraic relations); an additive **beacon
  mask `A`** giving zero-storage stateless permutation recovery (high-magnitude outliers
  encode the permutation, so `P̂` need not be stored).
- **TEE role = key custody only.** Paper *assumes* the MB-scale secret matrices live in a
  TEE ("we assume these secrets are protected via TEEs"; 898 KB for LLaMA-3.1-8B). It does
  **not** offload compute to/through a TEE the way GELO/TwinShield do. So it belongs in
  **§5 (obfuscation)** with a "TEE key-storage" note, **not §7 hybrid split.**
- **Perf:** 15.41 ms/GB ≈ **0.45% of prefill latency** (vs AES 3020.9 ms/GB); accuracy
  overhead "mostly <1%"; lossless (exact attention equivalence). vLLM-integrable.
- **Security:** CPA-style; key-recovery complexity `O((b²d+bd²)·b!)`; the b! from per-block
  permutation blocks brute force.
- ⇒ primer's "KV-Cloak (NDSS'26, **TEE-anchored**) → §7" is **wrong on both counts** (it's
  §5 obfusc, TEE only stores keys). Patch the primer's "§5 Static Obfuscation — DEFERRED"
  block.

### Its 3 KV-cache attacks (→ §8 attacks / §5 motivation)
- **Inversion Attack** — direct reconstruction of user input from cached K/V.
- **Collision Attack** — the potent, broadly-applicable one; exploits algebraic weight
  correspondence. Survives fine-tune mismatch (gray-box) but fails cross-architecture.
  **Breaks KV-Shield** (fixed shuffle preserves the statistical distribution).
- **Injection Attack** — semantic, leverages a pristine model to parse the cache.

## New schemes — static obfuscation (§5 candidates)

| Scheme | Ref | Mechanism | Threat / perf | Status |
|---|---|---|---|---|
| **ObfusLM** | Lin et al., **ACL 2025** (2025.acl-long.58) | model-obfuscation module w/ **(k,ε)-anonymity**; classification + generation | HbC/none; **+10% utility vs prior, ~80% EIA-resistance**; provable vs embedding-inversion | known target (primer); confirm static vs per-query in full text. **Not yet in EdgeQuake** |
| **KV-Cloak** | Luo et al., NDSS 2026 (2508.09442) | reversible matrix + per-block one-time perm + beacon mask (see above) | HbC/none + TEE key-custody; <1% overhead; lossless | **in EdgeQuake**; add §5 row + correct primer |
| **Equivariant Encryption (EE)** | Buban et al. 2025 (2502.01013); Nillion | selectively obfuscates internal reps, **preserves linear AND a prescribed set of non-linear ops**, near-zero overhead, "blind inference" | HbC/none; near-plaintext throughput; CNN→LLM | commercial-adjacent (Nillion); claims weak on formal security — grill. **Not in corpus** |
| **CodeCipher** | Lin et al. 2024 (2410.05797) | token-to-token confusion mapping on embedding matrix, discrete-optimized | HbC/none; preserves LLM output on code tasks | domain-specific (code LLMs); lineage/mention. **Not in corpus** |
| **Instance Obfuscation Inference (IOI)** | Yao et al. 2024, NAACL | instance-level obfuscation for classification | HbC/none | **already uploaded to EdgeQuake** (`57105106...`) but absent from corpus §4 list |
| **Centaur** | (2412.10652) 2024/25 | random-permutation replaces SMPC non-linears; plaintext-speed | HbC; **broken by Hidden-No-More** (already noted §8) | cited-as-broken lineage (like STIP); not a comparison row |

## New schemes — hybrid split / federated obfuscation (§7 candidates)

| Scheme | Ref | Mechanism | Threat / perf | Status |
|---|---|---|---|---|
| **FedRAG** | Mao et al. 2026 (2605.25716) | **Scrambled Distributed Attention**: feature scrambling + token permutation; decouples attention from data localization across institutions | HbC (semi-honest); **no TEE, no retraining**; **<0.1% util loss; 62× vs secure baselines**; defends intermediate-state inversion | strong §7/§8 (federated RAG) candidate. ⚠ permutation-reveal → may be exposed to the shuffling attack below. **Not in corpus** |
| **KV-Shield** | cited [41] in GELO | hybrid TEE-GPU: secret perm `R` in TEE, permuted weights `W_P=WR` on GPU | **BROKEN**: open-weights solve `W_P=WR`; Collision Attack; RoPE-incompatible | cited-as-broken §7 lineage (the STIP-analogue for KV-cache). **Not in corpus** |

## New attacks (§8)

| Attack | Ref | Target | Effect |
|---|---|---|---|
| **Shuffling-Defense break** (activation alignment) | 2605.04901 (2026) | permutation/shuffle-**reveal** schemes (STIP, Centaur, KV-Shield, FedRAG-class) | aligns differently-shuffled activations across queries → recovers permutation → **extracts weights** (L1 1e-4..1e-2); **~$1/query**. **Family-wide multi-run limitation** for §5/§7 permutation schemes |
| **LeftoverLocals** | cited [2] in GELO | GPU local memory | practical interception of KV-cache from GPU local mem → reconstruct responses; motivates KV-cache defenses |

## Borderline / likely out-of-scope (text-level sanitization or niche)
- **EmojiPrompt** (NAACL 2025) — generative prompt obfuscation, text-level → sanitization, not representation obfuscation.
- **PCAE / Covert Prompt Transmission** (2504.21311) — permutation enc + compression for *wireless edge*; niche.
- **Protecting Privacy in Classifiers by Token Manipulation** (Harel et al. 2024, 2407.01334) — token-mapping, minor/lineage.
- **SecMoE** — OT-based private MoE (crypto, §3 lineage alongside CryptoMoE, not obfuscation).

## Recommended actions
1. Patch primer "§5 Static Obfuscation — DEFERRED": KV-Cloak = §5 obfusc (TEE key-custody),
   not §7 TEE-anchored; unblock ObfusLM only on its own EdgeQuake upload.
2. Add to corpus §4 (static obfusc): **KV-Cloak**, **ObfusLM**, **IOI**; **EE** as a row or
   commercial mention; **CodeCipher** + **Centaur** as cited-as-broken/lineage prose.
3. Add to corpus §6/§7 (hybrid split): **FedRAG** (deployable, permutation-RAG); **KV-Shield**
   as broken lineage.
4. Add to §8 attacks: **Shuffling-Defense break (2605.04901)** — the load-bearing limitation
   for the whole permutation family; **LeftoverLocals**; Shadow-in-the-Cache's 3 attacks.
5. SoK narrative tension to exploit: KV-Cloak's *per-block one-time permutation* and GELO's
   *per-batch fresh mixing* are the design answer to the Shuffling-Defense break — "dynamic,
   never-reused randomness" is what separates surviving obfuscation from broken static
   permutation (STIP/KV-Shield/Centaur). This is the §5→§7 throughline.

## Correction (2026-06-05): "Shuffling-Defense break / 2605.04901" is UNVERIFIED
The "Shuffling-Defense break (activation alignment), arXiv 2605.04901" listed above could
**not be confirmed** by web search — no such arXiv id surfaced, and its description
(align differently-shuffled activations across queries → recover permutation → reconstruct)
matches **Hidden No More** (Thomas et al., ICML 2025, arXiv **2505.18332**), which is the
real, verified prompt-reconstruction attack on permutation+noise private inference. **Do not
cite 2605.04901.** Use `hnm` (2505.18332) for the multi-run reconstruction attack. The §7
attack table (`tab:hybrid-attacks`) cites HNM + LeftoverLocals (2401.16603) accordingly.
