---
type: research
status: current
created: 2026-06-02
updated: 2026-06-02
tags: [sok, taxonomy, evaluation-framework, confidential-llm, rag, scheme-categorization]
companion: [survey-corpus]
supersedes: []
---

# Scheme Categorization — Confidential LLM/RAG SoK

Maps every **approved (`[x]`) scheme** in `../../manuscript/survey-corpus.md` onto the
two-part evaluation framework of `../../manuscript/sections/02_evaluation_framework.tex`:

- **Security → per mechanism family** (§2.1's five categorical axes, applied to every
  scheme rather than one representative). One row per scheme. → **Part A**.
- **Cost → per use case** (§2.2's "compare by use case, not by family"). A multi-stage
  scheme appears once per use case it serves. → **Part B**.
- **§7 composable integrity** is *not* a confidentiality family — it gets its own table
  keyed by what it composes with. → **Part C**.

This document is the structural bridge between §2.1 and §2.2; it invents nothing, it
instantiates them. It feeds the §3–§7 family sections (security) and the §8 synthesis (cost).

## Grounding tiers and provenance

- **★ grounded** — extracted from the EdgeQuake-indexed primary source this session
  (21 scheme PDFs + 5 restored docs). Cells are from the paper.
- **† carried** — no full-text grounding yet; cells from the corpus row, to be confirmed
  by `/verify-claims` before submission. PDF links + paywall status in **Part D**.
- A handful of **source-vs-corpus contradictions** surfaced during grounding; they are
  consolidated in **Part E** and are the most submission-relevant output of this exercise.

## Legend

**Security axes** (§2.1): **Behavior** = HbC / Malicious / HbC+verif. · **Trust anchor
(Tier)** = none / CPU-TEE / GPU-TEE / non-coll. *k*PC / 1-server-PIR / SSE, with hardware
tier (T1 = no trusted HW; T2 = CPU-TEE, GPU untrusted; T3 = confidential accelerator) ·
**Knows** = black-box / wb (white-box weights) / wb+s (+ obfuscation scheme, secret only) ·
**Protects** = asset hidden · **Residual leak** = what the server still sees · **Basis** =
crypto / info-theoretic / DP / HW / heur. (combos allowed; `crypto*` = cryptographic but
distance-leaking).

**Cost** (§2.2): **Perf** self-labels its regime — ×-overhead *or* absolute, with
(model; hardware; key hyperparams). **Prov.** = head-to-head (same-paper baseline) /
self-reported (cross-paper, indicative). **Threat-fit** = which adversary it right-sizes for
(3rd deployment-readiness criterion). **Fidelity** = exact (computes true function) or a
Δ vs plaintext in the use-case metric; `Δ n/r` = no quality delta reported.

---

# Part A — Security categorization (per family)

## A.1 Cryptographic — MPC / Secret Sharing (§1)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| PermLLM | HbC | non-coll. 3PC (T1) | wb | input+model | shares; permuted plaintext (to user) | crypto+heur. | ★ |
| SecFormer | HbC | non-coll. 2PC (T1) | wb | input+model | shares | crypto | ★ |
| CryptoMoE | HbC | non-coll. 2PC (T1) | wb | input+model+routing | shares | crypto | ★ |
| p²RAG | HbC (+malicious-user defense) | non-coll. 2-server (T1) | — | query+database | shares; bounded DB leakage | crypto | ★ |
| Panther | HbC | none / 1-server (T1) | — | query+access | access pat. | crypto | ★ |
| Pacmann | HbC | none / 1-server PIR (T1) | — | query+access | none* (DB public) | crypto | ★ |
| SHAFT | HbC | non-coll. 2PC (T1) | wb | input+model | shares | crypto | ★ |
| BumbleBee | HbC | non-coll. 2PC (T1) | wb | input+model | shares+ct | crypto | † |
| Fission | HbC (extends to malicious) | non-coll. MPC + evaluators (T1) | wb | input+model | shares; shuffled+split act. to evaluators (leak ↓ w/ #evaluators) | crypto+heur. | ★ |
| PIR-RAG | HbC | none / 1-server PIR (T1) | — | query (target cluster) | access pat. (cluster-level only) | crypto (LWE-PIR) | ★ |
| XorMM | HbC (VXorMM: malicious) | none / SSE (T1) | — | corpus+volume | access/search pat. | crypto | ★ |
| FLASH | HbC | none / SSE (T1) | — | corpus+volume (conjunctive) | access/search pat. | crypto (+DP variant) | ★ |
| PeGraph | HbC | non-coll. 2-server SSE+SS (T1) | — | graph+queries+result ranks | access pat. (L-leakage) | crypto | ★ |
| GORAM | HbC | non-coll. 3PC (T1) | — | graph+access | shares; access pat. | crypto | † |

## A.2 Cryptographic — FHE / HE (§2)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| Euston | HbC | none (T1) | wb | input+model | ct (RNS-CKKS) | crypto | ★ |
| CipherFormer | HbC | none (T1) | wb | input+model | ct (HE)+GC | crypto | ★ |
| Compass | **Malicious** | none (T1; client-side ORAM) | — | query+corpus+results+access | op-type only (search vs ins/del) | crypto | ★ |
| SAP / ADCPE (DCPE) | HbC | none (T1) | — | doc content (vectors) | ct+dist | crypto* | ★ |
| CAPRISE | HbC | none (T1) | — | doc content+query | ct+dist (query)+ap | crypto*+DP | ★ |
| RAGtime-PIANO | HbC | none / 1-server PIR (T1) | — | query+access+distances | none* (ct only) | crypto | ★ |
| NEXUS | HbC | none (T1) | wb | input+model | ct (RNS-CKKS) | crypto | ★ |
| TRSE | HbC | none (T1) | — | doc content+query | ct+access pat. (user-side ranking) | crypto | ★ |

## A.3 Trusted Execution Environments (§3)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| Opal | **Malicious** | TDX CPU-TEE + B200 GPU-TEE (T3) | — (black-box host) | data+query+access pattern | op-type (Query/Ingest), fixed trace shape | crypto+HW | ★ |
| PipeLLM | HbC | CVM (TDX/SEV) + GPU-TEE (T3) | — | input+model | PCIe-transit ct (AES-GCM, IV) | HW | ★ |
| H₂O₂RAM | HbC | CPU-TEE (T2, doubly-oblivious) | — | query+access pattern | access counts/op-type | crypto+HW | † |
| H100 CC baseline | HbC | GPU-TEE (T3) / CPU-TEE (T2) | — | input+model | PCIe-transit ct (AES-GCM) | HW | ★ |

## A.4 Static Obfuscation (§4)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| AloePri | HbC | none (T1) | wb+s | input+output | obf. I/O + obf. weights | heur. (Rényi-mDP-analyzed) | ★ |
| SGT (Stained Glass) | HbC | none (T1) | wb+s | input (embeddings) | obf. embeddings | info-theoretic (MI) — *contested* | ★ |
| OSNIP | HbC | none (T1) | wb+s | input (embeddings) | obf. embeddings (≈⊥-orthogonal) | heur. (d_X-privacy) | ★ |
| ARoG / ARROWCLOAK | HbC | none (T1) | black-box | KG entity semantics | abstracted concepts + relational structure | heur. (anonymization) | ★ |
| Eguard | HbC (DB-breach + adaptive) | none (T1) | wb+s | input (embeddings) | projected embeddings | heur. (dual-level MI optimization) | ★ |

## A.5 Differential Privacy (§5)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| DP-Forward | HbC | none (T1) | wb+s | input (embeddings) | noisy embeddings | DP ((ε,δ)-SeqLDP, matrix-Gaussian) | ★ |
| RemoteRAG | HbC | none (T1) | — | query (+embedding+top-k indices) | perturbed embedding (within (n,ε)); bounded ap | DP+crypto ((n,ε)-DistanceDP + PHE + OT) | ★ |
| SPARSE | HbC | none (T1) | wb+s | input (embeddings) | elliptical-noised embeddings | DP (Mahalanobis mechanism, concept-aware metric-DP) | ★ |
| DP-RAG / DP-KSA | HbC (query-only adversary) | none (T1) | — | generated output (vs corpus extraction) | output within (ε,δ)-DP budget | DP (output-DP; PTR + sample-and-aggregate keyword selection) | ★ |

## A.6 Hybrid Split — TEE + Obfuscation (§6)

| Scheme | Behavior | Trust anchor (Tier) | Knows | Protects | Residual leak | Basis | Flag |
|---|---|---|---|---|---|---|---|
| GELO | HbC | GPU-TEE (T3 anchor; offloads to untrusted commodity GPU) | wb | input (hidden states) | mixed activations U=AH (VRAM) | heur.+HW (per-batch non-identifiability / single-batch BSS) | ★ |
| TwinShield | HbC+verif. | CPU-TEE + SS (T2) | wb | data+model | masked activations (VRAM) | crypto+HW | ★ |
| ObfuscaTune | HbC | CPU/GPU-TEE (T2; ~5% params in TEE) | wb | data+model | obf. activations+weights (VRAM) | heur.+HW | ★ |
| SCX | HbC | GPU-TEE (T2/3) | — | input+output (KV-cache) | stateless encoded KV-cache | DP+HW ((ε,0)-DP per-session OTK) | ★ |
| Portcullis | HbC | CPU-TEE (Intel TDX, T2) | — | input PII | anonymized prompt to 3rd-party LLM | HW+heur. (TDX-attested NER anonymization) | ★ |
| PrivGemo | HbC (remote LLM API) | none + anon (T1) | — | KG entity semantics **+ structure** | anonymized view (HMAC φ_Q + structural de-uniqueness); raw KG stays local | heur. (anonymization) | ★ |

> **§7 composable-verification schemes** (U-Verify, V3DB, ANNProof, ZKGraph, ZKIFV, RATLS,
> corpus commitment, ZKPROV, model commitment) are **Malicious/integrity** by construction
> and live in **Part C**, not here.

---

# Part B — Cost categorization (per use case)

Multi-stage schemes recur across tables; every perf cell self-labels its regime, and
**numbers are not comparable across rows** unless both are head-to-head in the same regime.
Each scheme is assigned **by the model it actually evaluates** — encoder (BERT/RoBERTa/ViT)
→ Embedding; decoder (Llama/GPT/Qwen) → Generation. Two changes from the first draft
(2026-06-02, after the B.2/B.3 audit, see Part E #14): **Reranking is not a cost bucket** —
no surveyed scheme reports a standalone private cross-encoder reranking number (the figures
were all retrieval-pipeline numbers reused), so it is an **open gap** (note after B.4, fed to
§8); and **Retrieval is split by index structure** — vector (flat/IVF) vs graph-indexed
(graph-based ANN + graph-semantic) search/storage.

## B.1 Generation (decoder inference)

| Scheme | Perf | Prov. | Threat-fit | Fidelity | Flag |
|---|---|---|---|---|---|
| PermLLM | 3 s/token (ChatGLM-6B; 3×L20-class GPU; 2PC+dealer, WAN); ~20 Mb/token | self-reported | HbC cloud, 2 non-colluding + dealer; constrained (permutation not info-theoretic) | exact | ★ |
| CryptoMoE | 2.8–3.5× latency / 2.9–4.3× comm reduction vs dense MoE (DeepSeek/OLMoE/QwenMoE 6.9–16.4B; Xeon 8468; 2PC LAN/WAN) | head-to-head | HbC 2PC, non-colluding; routing-pattern adversary | task-acc. 99.2% retained (≈−0.8%) | ★ |
| GELO | 20–30% overhead (76% offload; Llama-2-7B; confidential GPU + untrusted GPU) | head-to-head | HbC shared-GPU, single-batch BSS adversary | exact (float32 recovered) | ★ |
| TwinShield | 4.0–6.1× vs prior verifiable (87% offload; vision/lang Transformers; SGX + GPU); 5.4× vs TEE-only | head-to-head | malicious cloud (data+model+integrity) | exact | ★ |
| ObfuscaTune | 1.5–4.3× (GPT2-small→XL; ~5% params in TEE) | self-reported | HbC cloud (dual model+data conf.) | exact-ish (invertible obfusc; numerical error grows w/ matrix condition number) | ★ |
| SCX | near-zero online; target <50 ms (7B; GPU-TEE; per-session OTK) | self-reported | HbC cloud; KV-cache reconstruction adversary | exact | ★ |
| Opal | KG-filter + synthesis = ~1.15 s of the 2.32 s/query path (gpt-oss-20b; B200 CC) | head-to-head | malicious host; access-pattern adversary | exact (model in enclave) | ★ |
| AloePri | ~0% (near-plaintext; vLLM/SGLang drop-in; Qwen/Llama/DeepSeek) | head-to-head | HbC cloud, **constrained** attacker (RDP budget) | 0–3.5% acc. loss; <5% tokens recovered by VMA/IMA/IA/ISA | ★ |
| SGT (Stained Glass) | ~0 ms client-side transform (Llama/Qwen3 decoders; token-embedding obfusc) | self-reported | HbC cloud, constrained attacker | downstream LLM util −0.29 to −0.5 pp (but AloePri reports SGT broken by IMA — see E #9) | ★ |
| OSNIP | 0.96 ms (Llama-3.2-1B/3B, Qwen3-14B/32B; null-space projection) | self-reported | HbC cloud, KNN/vocab-match adversary | near-lossless util; KNN-attack ASR ≈ 0 | ★ |
| BumbleBee | ~8 min/token (LLaMA-7B; CPU; 2PC); >10× vs Iron | self-reported | HbC 2PC, no dealer | exact | † |
| SHAFT | 2.6–3.7× faster than BumbleBee, 1.8–2.4× than SIGMA, 62–70% less comm (BERT/GPT/ViT; 2PC LAN; constant-round softmax + Fourier GELU) | head-to-head (vs SIGMA/BumbleBee) | HbC 2PC, non-colluding | accuracy ≈ plaintext (GELU max err 4.6e-3) | ★ |
| Fission | >8× vs CrypTen (BERT); 5× (ModernBERT); >3× (Llama-3-1B); ~seconds @1B (80 vCPU + 2×H100; MPC nodes + evaluator nodes) | head-to-head (vs CrypTen) | HbC distributed MPC, non-colluding incl. evaluators | exact-ish (linear=MPC, nonlinear in clear on shuffled shares; acc ≈ PyTorch) | ★ |
| Euston | 5.5× (GPT-2-1.5B) / 8.8× (LLaMA-3-8B) vs NEXUS-CPU (batch 32×128 tok; LAN) | head-to-head (vs NEXUS) | HbC cloud, non-interactive | approx. (poly GELU/Softmax/LN; Δ n/r) | ★ |
| NEXUS | 37.3 s/inference, 164 MB BW (BERT-base; non-interactive) | self-reported | HbC cloud, non-interactive | approx. (Δ n/r) | † |
| PipeLLM | cuts CC overhead from 52.8%/88.2% (OPT-30B/66B) to <19.6% throughput (OPT 13B–175B; H100-SXM; speculative pipelined PCIe-AES) | head-to-head (vs vanilla CC) | HbC cloud, GPU-TEE operator | exact | ★ |
| H100 CC baseline | GPU 4–7% throughput penalty (Llama2 7/13/70B; H100 CC); CPU-TEE <10% thr / <20% lat; RAG-in-TEE 7% | head-to-head | HbC cloud, Tier-3 (GPU-TEE) / Tier-2 (CPU-TEE) baseline | exact | ★ |
| Portcullis | 96× vs Hide-and-Seek (mask/unmask); 1.33% overhead vs raw LLM inference; ~3 s latency (TDX gateway, LLaMA-2-7B/vLLM) | head-to-head (vs Hide-and-Seek/InferDPT) | HbC; protects PII before 3rd-party LLM | response cosine-sim >0.7 (GPT-4o-mini); beats Hide-and-Seek by 0.1 on Enron | ★ |
| DP-RAG / DP-KSA | output-(ε,δ)-DP, backend-agnostic (PTR keyword extraction; δ=1e-4); F1 21.7→25.2 @ε=0→8 (Llama-3.2-3B, NQ; 80 ensembles); beats non-RAG @ε≥2 | head-to-head (vs non-RAG / non-private KSA) | HbC; query-only output-extraction adversary | utility ↑ with ε; > non-RAG at moderate ε | ★ |

## B.2 Embedding (encoder inference — BERT/RoBERTa/ViT)

| Scheme | Perf | Prov. | Threat-fit | Fidelity | Flag |
|---|---|---|---|---|---|
| SecFormer | 71 s/sample (BERT-base, 512 tok; 3×V100, LAN); 3.57× faster than PUMA | head-to-head | HbC 2PC, non-colluding | task-acc. −0.9–1.3% vs PUMA (2Quad softmax) | ★ |
| CipherFormer | 7.7–11.9× vs HErBERT (encoder, N=1–2, L≤128; 2-thread VM, HE+GC) | self-reported | HbC cloud, low-round | approx. (ReLU-Softmax; +3–11% acc vs HErBERT, Δ vs plaintext n/r) | ★ |
| Euston | 3.5× faster than NEXUS (BERT-Base; EPYC + RTX 6000 Ada; LAN, batch 32×128) | head-to-head | HbC cloud, non-interactive | approx. (poly approx.; Δ n/r) | ★ |
| NEXUS | 37.3 s/inference, 164 MB BW (BERT-base, 128 tok; WAN 100Mbps/80ms); non-interactive 1-round; 53.6× less BW vs BumbleBee, 372.5× vs BOLT; GPU 42.3× faster ($0.05/token) | head-to-head (vs BOLT/BumbleBee) | HbC 2-party, non-interactive | approx. (poly GELU/Softmax/LN; acc.–latency tradeoff) | ★ |
| SHAFT | 62–70% less comm + 1.8–2.4× faster than SIGMA (BERT-base; 2PC LAN) | head-to-head (vs SIGMA/BumbleBee) | HbC 2PC, non-colluding | accuracy ≈ plaintext (constant-round softmax + Fourier GELU) | ★ |
| TwinShield | 4.0–6.1× vs prior (BERT/ViT; SGX+GPU) | head-to-head | malicious cloud (dual) | exact | ★ |
| DP-Forward | ~94% SST-2 acc @ moderate ε (BERT encoders; matrix-Gaussian forward) | head-to-head (own non-private + DP-SGD) | HbC cloud; embedding-inversion adversary | task-acc. Δ vs plaintext per ε (≈−1.7pp @ε≈3 w/ label privacy) | ★ |
| SPARSE | ~0 overhead; leakage 60→19% & util 65% @ε=10 (STS12; GTR/T5/SBERT); Vec2Text leakage −92% @ε=5 | head-to-head (vs LapMech/PurMech) | HbC cloud; concept-specific metric-DP budget | util preserved on non-sensitive dims (−few pp) | ★ |
| Eguard | inversion F1 → ~4% (>95% tokens protected); 1.6–3.4× train, 16.3 vs 9.6 ms/batch inference (T5/MPNet/RoBERTa, 2×A6000) | self-reported | HbC cloud, DB-breach/Vec2Text adversary | >98% downstream task acc retained | ★ |

## B.3 Vector retrieval & storage (flat/IVF ANN; no graph index)

| Scheme | Perf | Prov. | Threat-fit | Fidelity | Flag |
|---|---|---|---|---|---|
| p²RAG | 3–300× vs PRAG (k=16–1024; 2-server SS) | head-to-head | HbC 2-server, non-colluding | exact | ★ |
| RAGtime-PIANO | 40× lower latency, 323× lower comm vs PIR-RAG/GraphRAG (CKKS + PIANO PIR; IVF-Flat) | head-to-head | HbC 1-server; *fully secure* (no doc/distance/access leak) | better accuracy (two-stage FHE+PIR) | ★ |
| RemoteRAG | 0.67 s + 46.66 KB @10⁶ docs (DistanceDP + PHE; flat) | head-to-head | HbC cloud; query-inversion adversary | no retrieval loss; Vec2Text BLEU 50→10 | ★ |
| Panther | 18 s/query @10M points (single-server; cluster/IVF ANN; 284 MB; SIFT/Deep1B) | self-reported | HbC single-server, no non-collusion | exact (same accuracy as plaintext ANN) | ★ |
| PIR-RAG | 16.84 s/query @5K docs (MS MARCO; LWE-PIR cluster-and-fetch; downlink up to 474 MB fetching full cluster, uplink 2.4–24 KB) | head-to-head (vs Graph-PIR/Tiptoe) | HbC 1-server PIR; hides target cluster | NDCG@10 0.799, P@10 0.710 (vs Graph-PIR 0.901; coarse clustering) | ★ |
| SAP / ADCPE | ~0 ms (on-the-fly symmetric enc.; ANN-preserving storage) | self-reported | HbC snapshot adversary | approx. ANN (α-factor preserved; recall Δ n/r) | ★ |
| CAPRISE | 2,339 vec/s encrypt (9× vs RemoteRAG; +15 ms over 79.5 ms embed; gtr-t5-base, A100, m=100K) | head-to-head | HbC cloud; repeated-query adversary | top-k expanded to k′ (recall preserved, Δ n/r); Vec2Text BLEU 83→12 | ★ |
| TRSE | ~100s ms (k′=100; 2-round SE + FHE relevance scoring) — *keyword SSE, not embedding-ANN* | self-reported | HbC cloud; multi-keyword top-k, user-side ranking | exact scoring (Δ n/r) | ★ |
| Tiptoe-class / SANNS | baselines (see corpus; mostly † or [ ]) | — | — | — | — |

## B.4 Graph retrieval & storage (graph-indexed ANN + graph-semantic search)

| Scheme | Perf | Prov. | Threat-fit | Fidelity | Flag |
|---|---|---|---|---|---|
| Compass | 0.57–1.28 s/query (SIFT1M / MS MARCO; cross-region WAN; **HNSW**-on-Ring-ORAM); 920× vs HNSW-on-ORAM | head-to-head | **malicious server**, no HW trust | exact (Recall@10 ≥ 0.9, MRR@10 on par) | ★ |
| Pacmann | 1.6 s LAN / 3.1 s WAN @100M SIFT (single-thread Xeon; **graph-ANN** via 1-server PIR; recall@10 0.90) | head-to-head (vs Tiptoe/NGT) | HbC 1-server, **public DB**; query-only privacy | recall ≈ 90% of NGT (−10pp); 2.5× recall@10 vs Tiptoe | ★ |
| Opal | 1.57× vs plaintext Opal (2.32 s/query; B200 CC; KG + ANN over ORAM, 524K entries, WAN); 29.5× throughput vs in-memory secure | head-to-head | malicious host; access-pattern adversary | exact retrieval; KG-filter +13 pp judged-acc, matches plaintext Graphiti ceiling | ★ |
| ARoG | no crypto cost (KG anonymization; LLM-side reasoning) | self-reported | HbC third-party LLM API; entity-semantics adversary | SoTA on WebQSP/CWQ/GrailQA (privacy-preserving scenario; Δ vs non-private n/r) | ★ |
| PrivGemo | no crypto cost (dual-tower: local Hand LLM + remote Brain on anonymized view); SOTA on 6 KGQA (+17.1% vs best); 3.5 cloud calls vs 21.7 (ToG) | head-to-head (vs ToG/ARoG) | HbC; semantic+structural KG-exposure adversary | CWQ 67→59% (plaintext→full-anon); WebQSP 75%; lets Qwen3-4B ≈ GPT-4-Turbo | ★ |
| XorMM | ~1.23n storage; optimal query comm (ℓ); 1.8× faster search + 76% less storage vs dprfMM (Patel CCS'19); VXorMM verifiable variant ~2.46n | head-to-head | HbC SSE, untrusted store; volume-hiding | exact (non-lossy EMM) | ★ |
| FLASH | 2× storage saving, 90× faster setup, ~180× faster search vs OXTMM (conjunctive volume-hiding, symmetric-key; DP variants optional) | head-to-head (vs OXTMM) | HbC SSE, untrusted store; conjunctive volume-hiding | exact (non-lossy conjunctive EMM) | ★ |
| PeGraph | <1 s/query over millions of entities (real social graph; 2-server SSE OXT + additive SS); 5× comp / 200× comm vs GraphSE² | head-to-head (vs GraphSE²) | HbC non-colluding 2-server; encrypted social-graph search | exact (exact/fuzzy/ranked queries) | ★ |
| GORAM | 58.1 ms–35.7 s/query @41.6M vertices / 1.4B edges (3PC; 5 ego-query types) | self-reported | HbC 3PC, non-colluding; federated multi-silo | exact (ORAM ego-graph) | † |
| H₂O₂RAM | ~10³× faster than prior O₂RAM; 5–44× less memory (TEE doubly-oblivious substrate; graph-RAG store role) | self-reported | HbC host; access-pattern observation | exact (oblivious storage) | † |

> **Reranking — open gap (no cost bucket).** No surveyed scheme reports a standalone private
> cross-encoder reranking benchmark; every figure is a retrieval-pipeline number reused:
> RemoteRAG (PHE rerank folded into 0.67 s retrieval), Opal (in-enclave rerank folded into
> 2.32 s/query), p²RAG (bisection top-k = its retrieval number), Panther (ANN top-k), TRSE
> (FHE relevance scoring, no wall-clock). Private reranking is therefore an **open gap**
> (consistent with Mu et al.); it remains a row in the §8 pipeline-stage gap-map but is
> excluded from the cost yardstick.

---

# Part C — Composable integrity (§7)

Not a confidentiality family. Each entry adds **one** integrity property at near-zero cost,
letting a §3–§6 scheme tolerate a server that is malicious *for that one property*.

| Scheme | Composes with | Integrity property proven | Adversary | Cost-to-add | Flag |
|---|---|---|---|---|---|
| U-Verify (TwinShield) | §6 hybrid split (TEE↔accelerator) | hash-check of outsourced compute, incl. **non-linear SoftMax** (Freivalds generalization, check-product protocol) | malicious accelerator | verifiable inf. 3.9–6.1× vs 4.9–7.7× privacy-only (~33% less than Freivalds on linear) | ★ |
| V3DB | §1–§2 vector retrieval | succinct ZK that top-k = committed-snapshot IVF-PQ semantics (Plonky2 + Merkle/Poseidon + multiset checks); hides embeddings/index | malicious retrieval operator (workflow deviation, version equivocation) | ≤22× faster proving than circuit-only, ms-verify, 40% less mem; **integrity only, not access-pattern privacy** | ★ |
| Lightweight proofs-of-inference | §6 outsourced inference | opens a few random output→input paths (rational-prover sampling), not full proof | malicious (rational) prover | proving min→ms (ResNet-18, Llama-2-7B); trades soundness for speed | † |
| ANNProof | §1–§2 vector retrieval | verifiable K-ANNS: soundness/completeness/freshness vs blockchain-committed ADS (Merkle HNSW node tree + Merkle vector-id tree); **integrity-only, public datasets, no confidentiality** | malicious SP (wrong results) + malicious DP (tampered dataset) | ms-level VO; 160×/120× gen/verify, VO −28× vs SOTA; ADS ≤2% of index; sharding 53× | ★ |
| ZKGraph | §1/§7 graph retrieval | PLONKish ZK that graph traversal/neighborhood is correct vs committed snapshot | malicious operator (GRAGPOISON/LogicPoison) | ZK (no perf nums; research-stage) | † |
| ZKIFV | §1 keyword/inverted-index retrieval | ZK of inverted-index search correctness + completeness | malicious operator | ZK (no perf nums) | † |
| RATLS / remote attestation | every §3/§6 TEE scheme | verifies the right enclave code/version runs (identity, not computation) | malicious host (code-substitution) | ~0 (one-time) | † (primitive) |
| Corpus commitment + freshness | §2 storage + RAG | answer grounded in authorized, untampered, fresh corpus (Merkle/version) | malicious operator (poisoning, rollback) | ~0 (hash) | † (primitive) |
| ZKPROV | §2 retrieval / e2e | ZK dataset-provenance (heavy-crypto variant of corpus commitment) | malicious operator | ZK provenance (heavier) | † |
| Model commitment | any inference scheme | served model = committed model (hash/Merkle + attested) | malicious operator (model swap/backdoor) | ~0 (hash) | † (primitive) |

---

# Part D — Non-★ PDF links (carried schemes)

For schemes carried from the corpus (cells unverified, queued for `/verify-claims`).
Most have open arXiv/ePrint PDFs; publisher-only entries are flagged **paywalled**.
**Uploaded** = already ingested in EdgeQuake (as of 2026-06-02; `[x]` = present, incl.
still-processing). Check the box after uploading the rest.

| Scheme | Uploaded | Link | Access |
|---|:---:|---|---|
| SHAFT | [x] | eprint.iacr.org/2025/2324 | open |
| BumbleBee | [ ] | ndss-symposium.org (2025) / eprint.iacr.org/2023/1678 | open |
| Fission | [x] | eprint.iacr.org/2025/653 | open |
| PIR-RAG | [x] | arXiv 2509.21325 | open |
| XorMM | [x] | eprint/CCS 2022 (author copy); code github.com/CDSecLab/XorMM | open (author) |
| FLASH | [x] | ieeexplore.ieee.org/document/11129933 (TDSC) | **paywalled** |
| PeGraph | [x] | ieeexplore IEEE TIFS 2022 (DOI 10.1109/TIFS.2022.3201392) | **paywalled** |
| GORAM | [ ] | vldb.org/pvldb/vol18/p3601-fan.pdf / arXiv 2410.02234 | open |
| NEXUS | [x] | eprint.iacr.org/2024/136 / NDSS 2025 | open |
| TRSE | [x] | IEEE TDSC 2013 | **paywalled** |
| SPARSE | [x] | arXiv 2602.07090 | open |
| DP-RAG / DP-KSA | [x] | arXiv 2602.14374 | open |
| Eguard | [x] | arXiv 2411.05034 | open |
| Portcullis | [x] | AAAI 2025 | **paywalled** (check arXiv) |
| PrivGemo | [x] | arXiv 2601.08739 | open |
| PipeLLM | [x] | arXiv 2411.03357 (ASPLOS '25) | open |
| H₂O₂RAM | [ ] | arXiv 2409.07167 (USENIX Sec '25) | open |
| H100 CC baseline | [x] | arXiv 2509.18886 / 2409.03992 | open |
| Lightweight proofs-of-inference | [ ] | arXiv 2603.19025 | open |
| ANNProof | [x] | FGCS 2024 (BIT) | **paywalled** (check) |
| ZKGraph | [ ] | arXiv 2507.00427 | open |
| ZKPROV | [ ] | arXiv 2506.20915 | open |

---

# Part E — Source-vs-corpus contradictions (act on before submission)

Surfaced during grounding; **source preferred** in the tables above. These are the
highest-value corrections for the manuscript and `refs.bib`.

1. **GraSS — DROPPED (resolved).** Corpus claimed "13 s @5K"; the paper (ePrint 2024/2012)
   reports **≈83 hours** for top-16 at **1M scale** (acc 0.918, 28.7× faster than prior FHE) —
   *hours*, not seconds, worst-on-performance by orders of magnitude. **Demoted to
   Foundational/Superseded (FHE-retrieval lineage) in `survey-corpus.md` 2026-06-02** per scope
   rule #1; removed from the security/cost tables above. Venue "EuroS&P" was also unverified.

2. **Compass threat model — FIXED (resolved).** §2 `tab:threat-taxonomy` had `HbC | TEE/client
   ORAM | crypto+HW`. The paper defends a **malicious / fully-compromised server**, relies on
   **no trusted hardware** (Tier 1; client-side Ring-ORAM), basis **crypto**, residual leak =
   operation *type* only. **Row corrected and the "Opal is alone in defending a malicious host"
   landscape prose updated to "only Opal and Compass" in `02_evaluation_framework.tex` 2026-06-02.**

3. **Amulet — DROPPED (resolved).** Corpus said `HbC/TEE`. The paper defends a **malicious
   on-device user** and protects the **model weights** (not user data — "we do not address
   the privacy of the input data"), with a **proven information-theoretic** guarantee (MI=0).
   This is an **inverted threat model** (model-from-user, not data-from-cloud) — out of scope
   for a confidential-inference/RAG SoK that protects user/corpus data from the cloud.
   **Demoted to Foundational/Superseded (on-device model-IP lineage, with ShadowNet/SOTER/
   TEESlice) in `survey-corpus.md` 2026-06-02**; removed from the tables above.

4. **PermLLM basis.** §2 says `crypto`. Authors state the nonlinear-core permutation
   "cannot achieve information-theoretic security" — it is heuristic; linear layers (SS) and
   argmax (BFV cPIR) are crypto. Tag **crypto+heur.** (PermLLM is closer to a hybrid than pure crypto).

5. **Opal basis + tier.** §2 lists basis `HW` and "TDX." Access-pattern hiding rests on the
   Ring-ORAM **cryptographic** primitive + AEAD → **crypto+HW**; hardware is **TDX CPU-TEE +
   B200 GPU-TEE = Tier 3** (combined, not TDX-only). Also the "29×" headline = 29.5×
   *throughput* / 15× *cost* vs an in-memory secure baseline, not "29× infra."

6. **CAPRISE basis.** Corpus `crypto*`. It is conditional ADCPE (`crypto*`) **plus
   DistanceDP** on the query → **crypto*+DP**. The Vec2Text "BLEU 83→12" is a *privacy* metric,
   not a retrieval-fidelity delta.

7. **Pacmann scope.** Protects **query + access pattern only over a public DB**; it does
   **not** hide document content. Residual leak `none*` reflects PIR over a public corpus.

8. **GELO tier — categorization nuance.** GELO's trusted device is a **confidential GPU
   (H200, Tier 3)**, offloading to an untrusted commodity GPU. Unlike TwinShield/ObfuscaTune
   (CPU-TEE, Tier 2), GELO is the **GPU-TEE-anchored** hybrid. The "hybrid split stays
   at Tier 2" framing applies to the CPU-TEE members, not GELO.

9. **SGT basis is contested.** SGT claims **information-theoretic** (MI/GMM) protection, but
   AloePri's evaluation reports SGT broken by IMA (>90% TTRSR, >50% accuracy loss). Present SGT's
   info-theoretic claim with the AloePri counter-result attached (it is a live security dispute).

10. **NEXUS metadata.** Venue is **NDSS 2025**, not "CCS 2024." Headline is **37.3 s/inference,
    164 MB** (BERT-base); the corpus "1.31 s amortized, 256-batch" is **unverified** against the
    paper's headline — keep flagged until confirmed.

11. **XorMM authorship.** Corpus credits "Patel et al. 2022." The XorMM paper is **Wang et al.,
    CCS 2022** (DOI 10.1145/3548606.3559345); Patel et al. (CCS 2019, dprfMM) is the *baseline*.
    Fix `refs.bib`.

12. **Fission basis.** Corpus `HbC/MPC` (implying pure crypto). Reveals *shuffled* activations to
    evaluator nodes for in-clear nonlinear eval (permutation-secrecy, like PermLLM) → **crypto+heur.**

13. **SecureInfer basis.** XOR-OTP = one-time-pad → **info-theoretic+HW**, not generic crypto.

14. **Cost-grouping audit — RESOLVED (2026-06-02).** B.2/B.3 audit found the use-case buckets
    were mis-cut. (a) **Reranking** had no native numbers — all 5 rows reused retrieval/pipeline
    figures (Opal even had "no separate baseline") → dropped as a cost bucket, recorded as an
    open gap (Mu et al.), kept only as a §8 pipeline-stage row. (b) **Embedding** was real (a
    genuine encoder line: SecFormer/NEXUS/CipherFormer/DP-Forward/Eguard/SPARSE/SIGMA/TwinShield
    on BERT/RoBERTa/ViT) but contaminated by decoder schemes reusing generation numbers
    (GELO/ObfuscaTune/H100 CC) and decoder-input obfuscation (SGT/OSNIP) → schemes now assigned
    by the model they evaluate (encoder→Embedding, decoder→Generation). (c) **Retrieval** split
    by index structure into vector (flat/IVF) and graph-indexed (graph-based ANN + graph-semantic).
    **Propagated to §2.2 `tab:yardstick` + prose and §8 `tab:performance` + gap note 2026-06-02.**

15. **SecureInfer — DROPPED (resolved).** Grounding (Nayan et al. 2025) showed SecureInfer is
    an **on-device model-extraction defense**: it protects the **model weights** from a
    malicious device user (SGX holds non-linear/attention-projection/FFN/LoRA; GPU does
    XOR-encrypted matmul), not user data from the cloud — the *same inverted threat model* for
    which Amulet was dropped (#3). **Dropped from scope 2026-06-02** (with Amulet, on-device
    model-IP lineage); removed from the tables. The manuscript Scope paragraph now states
    on-device private-model-inference schemes (SecureInfer, Amulet) are out of scope.

16. **SIGMA — dropped from scope (2026-06-02).** Removed from the comparison set per scope
    refinement; SHAFT (which it is benchmarked against, and which beats it by 1.8–2.4×) covers
    the 2PC-secret-sharing Transformer-inference point. SIGMA demoted to MPC lineage in the corpus.

> **Doc-ID note:** during grounding the EdgeQuake doc IDs for Compass (`7b372edb`) and
> Fuchsbauer SAP/ADCPE (`887283e3`) were each grounded by their actual content, not by label.

---

## Coverage check

- **Security rows (Part A):** 14 MPC/SS · 8 FHE/HE · 4 TEE · 5 obfuscation · 4 DP · 6 hybrid = **41**
  (dropped: GraSS #1, Amulet #3, SecureInfer #15, SIGMA #16; Collaborative Obfuscation = AloePri dup).
- **§7 composable integrity (Part C):** 10.
- **Grounded ★ (~40):** Round 1 (orig. EdgeQuake corpus) — PermLLM, SecFormer, CryptoMoE,
  Pacmann (MPC); Euston, CipherFormer, Compass, SAP/DCPE, CAPRISE (FHE); Opal (TEE); AloePri,
  SGT, OSNIP, ARoG (obfusc); DP-Forward, RemoteRAG (DP); GELO, TwinShield, ObfuscaTune, SCX
  (hybrid); U-Verify, V3DB (§7). Round 2 (2026-06-02, after user uploaded Part D PDFs; grounded
  via `document_get_md`/hybrid query) — XorMM, FLASH, PeGraph, TRSE, NEXUS, Fission, p²RAG,
  Panther, RAGtime-PIANO, PIR-RAG, SHAFT (crypto); PipeLLM, H100 CC (TEE); SPARSE, Eguard,
  DP-KSA (DP/obfusc); Portcullis, PrivGemo (hybrid/graph); ANNProof (§7).
- **† carried (6, not yet grounded):** BumbleBee, GORAM, H₂O₂RAM, Lightweight-proofs-of-inference,
  ZKGraph, ZKPROV (not yet uploaded — see Part D). Cells from corpus, queued for grounding + `/verify-claims`.
- Every `[x]` confidentiality scheme appears once in Part A and ≥once in Part B; every `[x]`
  §7 scheme appears in Part C.
