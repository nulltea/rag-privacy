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

**Cost-table columns (8) and conventions (rebuilt 2026-06-02, grill #N):**
`Scheme · Target model(s) · Hardware · ×-overhead vs plaintext · Comm · Preprocessing/offline · Fidelity · Flag`.
- **×-overhead vs plaintext** is the headline, comparable metric (cost relative to running the
  *unsecured* model on the same hardware), with the absolute figure in parentheses where given.
  Head-to-head "X× vs scheme B" numbers are **not** comparable and are not used as the headline.
  Where a scheme reports only relative speedups, the overhead is **extrapolated 1-hop within the
  same model+hardware regime** and flagged `≈… (via B)`; if no same-regime vs-plaintext anchor
  exists, the cell reads `— (only relative: …)` or gives the absolute figure only. Crypto (MPC/FHE)
  schemes mostly land here — they report vs prior schemes, rarely vs plaintext — whereas
  TEE/obfuscation/hybrid/DP schemes report clean vs-plaintext overhead.
- **Comm** = absolute communication (e.g. 164 MB/inference, 20 Mb/token); `—` where negligible
  (TEE/obfuscation), relative-only where a paper gives only "X× less than B".
- **Preprocessing/offline** = type + magnitude of one-time setup: MPC offline triples / FSS keys;
  FHE keygen; obfuscation *matrix setup* (cheap, e.g. AloePri/GELO) vs **offline training of a
  transform/projection/encryptor network** (SGT, OSNIP, Eguard, SPARSE) — a major hidden cost for
  "≈0 inference overhead" obfuscation/DP schemes.
- **⚠ = not reported / not yet extracted** (especially hardware on some crypto/RAG schemes —
  flagged for a follow-up fetch + `/verify-claims`).

## B.1 Generation (decoder inference)

| Scheme | Target model(s) | Hardware | ×-overhead vs plaintext | Comm | Preprocessing/offline | Fidelity | Flag |
|---|---|---|---|---|---|---|---|
| PermLLM | ChatGLM-6B | 3×L20-class GPU; WAN 10 ms/1 Gbps | — (only relative: "orders faster than MPC"; abs 3 s/tok) | ~20 Mb/tok | MPC offline: Beaver + permutation triples (dealer) | exact | ★ |
| CryptoMoE | DeepSeek/OLMoE/QwenMoE 6.9–16.4B | Xeon 8468 (48-core, 2.1 GHz); 2PC LAN 3 Gbps/0.2 ms, WAN 400 Mbps/40 ms | — (only relative: 1.7–5.9× vs dense-MPC; matches insecure baseline in some cases via batched MatMul; +18% for secure dispatch/combine) | 3.1–6.1× < dense-MPC (rel.; Table 2 per-token) | MPC offline (HE+SS) | task-acc 99.2% (≈−0.8%) | ★ |
| GELO | Llama-2-7B | confidential GPU (H200) + untrusted GPU (L40S) | **1.2–1.3×** (20–30% vs insecure offload) | — (TEE-local mixing) | per-batch fresh invertible matrix (cheap, online) | exact (float32) | ★ |
| TwinShield | vision/language Transformers (BERT/ViT/CLIP/LLaMA-8bit) | Xeon Gold 6342 (2.8 GHz, 512 GB) + A40 48 GB; SGX (FPGA/TPU also tested) | — (only relative: 5.4× vs TEE-only, 4.0–6.1× vs prior verifiable; 87% offload) | masked act. (⚠ abs n/r) | offline precompute of mask products (RW) | exact | ★ |
| ObfuscaTune | GPT-2 small→XL | 2 GPUs (1 simulates TEE); "middle-range" (model n/r); 1–8 GPU-h/exp | **1.5–4.3×** (vs unprotected) | — | one-time obfusc-matrix setup (<10 s/GPT2-XL on mid GPU; low-cond.-number) | exact-ish (numerical err ↑ w/ cond.) | ★ |
| SCX | 7B decoder | GPU-TEE + cloud GPU (⚠ testbed machine n/r — confirmed via full MD) | **~1×** (near-zero online; target <50 ms) | — | per-session one-time-key setup | exact | ★ |
| Opal | gpt-oss-20b + nomic-embed | TDX CPU-TEE + B200 GPU-TEE; WAN | **1.57×** (vs plaintext Opal; abs 2.32 s/query) | ORAM batches (⚠ abs n/r) | ORAM index build + KG construction (offline) | exact | ★ |
| AloePri | Qwen/Llama/DeepSeek (≤671B) | client 2×Xeon 8457C; server GPU cluster (vLLM) | **~1×** (near-plaintext) | — | one-time covariant obfusc of weights (offline matrix transforms) | 0–3.5% acc loss; <5% token recovery | ★ |
| SGT (Stained Glass) | Llama-1B, Qwen3 | transform-training: 1×A100 80GB (small) → up to 64×8 A100 80GB (large); inference client-side | **~1×** (≈0 ms client transform) | obf. embeddings (> token IDs) | **offline training of SGT transformer** (MI-loss; ~6 GPU-h small → ~2 days large, FSDP2+TP) | −0.29–0.5 pp util (⚠ AloePri reports broken by IMA, E #9) | ★ |
| OSNIP | Llama-3.2-1B/3B, Qwen3-14B/32B | ⚠ not reported (confirmed via full MD; client-side encryptor) | **~1×** (0.96 ms) | obf. embeddings | **offline training of encryption network** (gradient access to server LLM) | near-lossless; KNN-ASR ≈ 0 | ★ |
| BumbleBee | LLaMA-7B | CPU (⚠ machine n/r); 2PC | — (only relative; abs ~8 min/tok) | 0.1× < BOLT (rel.) | HE (RLWE) + OT offline | exact | † |
| SHAFT | BERT / GPT / ViT | 2×NVIDIA A40, 256 GB, Xeon Gold 5318Y; 2PC LAN | — (only relative: 1.3× vs SIGMA on BERT; 4.6–5.3× LAN / 2.9–4.4× WAN vs BumbleBee — *verified, E #17*) | 25–41% < SIGMA (rel.) | 2PC offline (SS triples) | acc ≈ plaintext (QNLI 90.4 vs 90.8) | ★ |
| Fission | BERT/ModernBERT/Llama-3-1B | 80 vCPU + 2×H100 | — (only relative: >8× vs CrypTen; abs ~s/inf @1B) | 8× < CrypTen (rel.) | MPC offline (triples) | exact-ish (acc ≈ PyTorch) | ★ |
| Euston | GPT-2-1.5B, LLaMA-3-8B | AMD EPYC 7542 (32-thread) + RTX 6000 Ada; LAN | — (only relative: 5.5–8.8× vs NEXUS) | 2.8–4.4× < NEXUS (rel.) | FHE (RNS-CKKS) keygen + offline SVD mask | approx. (poly GELU/Softmax/LN; Δ n/r) | ★ |
| NEXUS | BERT-base (128 tok) | CPU (GPU variant 42.3×); WAN 100 Mbps/80 ms | — (abs 37.3 s/inf; vs-plaintext × n/r) | 164 MB/inf | FHE (RNS-CKKS) keygen | approx. (Δ n/r) | ★ |
| PipeLLM | OPT 13B–175B | H100-SXM (CVM + GPU-TEE) | **~1.2×** (<19.6% throughput vs w/o-CC) | PCIe (internal) | none (runtime pipelining) | exact | ★ |
| H100 CC baseline | Llama2 7/13/70B | H100 CC; Intel TDX·SGX | **1.04–1.08×** (4–8% GPU); CPU-TEE <10% thr / <20% lat | PCIe AES-GCM | none | exact | ★ |
| Portcullis | gateway LLaMA-2-7B; LLM = GPT-4o-mini/Mistral | Intel TDX (Sapphire Rapids, 16 vCPU) | **~1.01×** (1.33% vs raw LLM) | — | none (NER masks) | response cosine-sim >0.7 (GPT-4o-mini) | ★ |
| DP-RAG / DP-KSA | Qwen2.5-3B, Llama-3.2-3B/3.1-8B (+DPR) | ⚠ not reported (confirmed via full MD) | **≈N×** generator calls (N=80 ensemble) | — | none (output-DP, PTR; δ=1e-4) | F1 21.7→25.2 @ε=0→8; >non-RAG @ε≥2 | ★ |

## B.2 Embedding (encoder inference — BERT/RoBERTa/ViT)

| Scheme | Target model(s) | Hardware | ×-overhead vs plaintext | Comm | Preprocessing/offline | Fidelity | Flag |
|---|---|---|---|---|---|---|---|
| SecFormer | BERT-base (512 tok) | 3×V100; LAN 10 GB/s | — (only relative: 3.57× vs PUMA; abs 71 s/sample) | ⚠ abs n/r (SS) | MPC offline (2Quad approx.) | task-acc −0.9–1.3% vs PUMA | ★ |
| CipherFormer | text-class. encoder (L≤128) | 2-thread VM; 128-bit | — (only relative: 7.7–11.9× vs HErBERT) | ⚠ abs n/r (HE+GC) | HE keygen | approx. (+3–11% acc vs HErBERT; Δ vs plaintext n/r) | ★ |
| Euston | BERT-Base | AMD EPYC 7542 + RTX 6000 Ada; LAN | — (only relative: 3.5× vs NEXUS) | 4.4× < NEXUS (rel.) | FHE keygen + offline SVD mask | approx. (Δ n/r) | ★ |
| NEXUS | BERT-base (128 tok) | CPU (GPU variant 42.3×); WAN 100 Mbps/80 ms | — (abs 37.3 s/inf) | 164 MB/inf | FHE (RNS-CKKS) keygen | approx. (Δ n/r) | ★ |
| SHAFT | BERT-base | 2×NVIDIA A40, 256 GB, Xeon Gold 5318Y; 2PC LAN | — (only relative: 1.3× vs SIGMA on BERT — *verified, E #17*) | 25–41% < SIGMA (rel.) | 2PC offline (SS triples) | acc ≈ plaintext (QNLI 90.4 vs 90.8) | ★ |
| TwinShield | BERT / ViT | Xeon Gold 6342 (2.8 GHz, 512 GB) + A40 48 GB; SGX | — (only relative: 4.0–6.1× vs prior verifiable) | masked act. (⚠ abs n/r) | offline mask-product precompute | exact | ★ |
| DP-Forward | BERT encoders (SST-2/QQP) | Tesla P100 GPU cluster | **~1×** (≈ non-private; ~3× less than DP-SGD) | — | none (forward-pass matrix-Gaussian noise; optional noisy pretrain) | task-acc Δ per ε (−1.7 pp @ε≈3 w/ label privacy) | ★ |
| SPARSE | GTR-base/Sentence-T5/SBERT | ⚠ not reported (confirmed via full MD) | **~1×** (≈0 inference overhead) | — | **offline differentiable mask-learning per privacy concept** | util 65% @ε=10 (STS12); leakage 60→19%; Vec2Text −92% @ε=5 | ★ |
| Eguard | T5/MPNet/RoBERTa | 2×NVIDIA A6000 | **~1.7×** inference (16.3 vs 9.6 ms/batch) | — | **offline training of projection network** (1.6–3.4× train) | >98% task acc; inversion F1 → ~4% | ★ |

## B.3 Vector retrieval & storage (flat/IVF ANN; no graph index)

| Scheme | Target model(s) | Hardware | ×-overhead vs plaintext | Comm | Preprocessing/offline | Fidelity | Flag |
|---|---|---|---|---|---|---|---|
| p²RAG | RAG corpus (k=16–1024) | ⚠ not reported (2-server) | — (only relative: 3–300× vs PRAG) | ⚠ abs n/r | 2-server SS offline | exact (bisection = true top-k) | ★ |
| RAGtime-PIANO | RAG corpus (IVF-Flat) | ⚠ not reported (1-server) | — (only relative: 40× vs PIR-RAG) | 323× < PIR-RAG (rel.); ⚠ abs n/r | FHE (CKKS) + PIANO-PIR client preproc | better acc than PIR-RAG | ★ |
| RemoteRAG | 10⁶ docs (MiniLM/MPNet/T5/OpenAI emb) | 2×Xeon Gold 5420+ (28-core) + 2×A40 48 GB; Ubuntu 22.04 | **~1×** retrieval (abs 0.67 s @10⁶) | 46.66 KB @10⁶ | PHE keygen; embed + AES DB | no retrieval loss; Vec2Text BLEU 50→10 | ★ |
| Panther | SIFT/Deep1B (10M pts) | ⚠ not reported (single-server) | — (abs 18 s/query @10M) | 284 MB/query | offline batch-PIR hint + SS/GC setup | exact (same acc as plaintext ANN) | ★ |
| PIR-RAG | MS MARCO 5K (bge-base emb) | ⚠ not reported (1-server) | — (abs 16.84 s/query @5K) | downlink ≤474 MB; uplink 2.4–24 KB | LWE-PIR client preproc + cluster build | NDCG@10 0.799, P@10 0.710 (vs 0.901 Graph-PIR) | ★ |
| SAP / ADCPE | vectors (storage layer) | n/a (symmetric enc., on-the-fly) | **~1×** (≈0 ms enc.) | — | none (deterministic DCPE) | approx. ANN (α-factor preserved; recall Δ n/r) | ★ |
| CAPRISE | gtr-t5-base (100K vec) | NVIDIA A100 | **~1×** (+15 ms over 79.5 ms embed; 2,339 vec/s enc.) | — | DCPE+DP enc. of DB (offline/on-the-fly) | recall preserved (k′); Vec2Text BLEU 83→12 | ★ |
| TRSE | encrypted docs (keyword) | server Xeon E5620; Linux | — (abs ~100s ms, k′=100) | ⚠ abs n/r | SE index build + HE keygen | exact scoring (Δ n/r) — *keyword SSE, not embedding-ANN* | ★ |
| Tiptoe-class / SANNS | baselines (see corpus) | — | — | — | — | — | — |

## B.4 Graph retrieval & storage (graph-indexed ANN + graph-semantic search)

| Scheme | Target model(s) | Hardware | ×-overhead vs plaintext | Comm | Preprocessing/offline | Fidelity | Flag |
|---|---|---|---|---|---|---|---|
| Compass | SIFT1M / MS MARCO | GCP n2-standard-8 client (8 vCPU/32 GB) + n2-highmem-64 server (64 vCPU/512 GB); 3 Gbps/1 ms ↔ 400 Mbps/80 ms | — (only relative: 920× vs HNSW-on-ORAM; abs 0.57–1.28 s/query) | client 5.5 MB–0.5 GB index cache (⚠ per-query comm n/r) | ORAM build + Faiss HNSW/PQ; AES-256 | exact (Recall@10 ≥0.9, MRR@10 on par) | ★ |
| Pacmann | 100M SIFT | single-thread Xeon E5-2680; LAN/WAN | — (only relative: −22% lat vs Tiptoe; abs 1.6 s LAN / 3.1 s WAN @100M) | ~few KB/query (sublinear √n, Piano PIR) | client-preproc PIANO PIR (amortized, linear one-time) + client graph hints | recall ≈90% of NGT (−10pp) | ★ |
| Opal | gpt-oss-20b + nomic-embed (524K) | TDX + B200 CC; WAN | **1.57×** (vs plaintext Opal; abs 2.32 s/query) | ORAM batches (⚠ abs n/r) | ORAM index + KG build | exact; KG-filter +13 pp judged-acc | ★ |
| ARoG | KG (WebQSP/CWQ/GrailQA); LLM-side | n/a (3rd-party LLM API) | **~1×** (no crypto; LLM reasoning) | — | none (entity→machine-ID anonymization) | SoTA on 3 KGQA (Δ vs non-private n/r) | ★ |
| PrivGemo | KG (6 KGQA); Hand=Qwen3, Brain=GPT-4o-mini | n/a (local LLM + cloud LLM) | **~1×** (no crypto); 3.5 cloud calls vs 21.7 (ToG) | — | none (HMAC anon + structural de-uniqueness) | CWQ 67→59% (plaintext→full-anon); WebQSP 75% | ★ |
| XorMM | encrypted multimap (adjacency) | Intel i5-9500 @3 GHz, 8 GB RAM (CPU) | — (storage ~1.23n; only relative: 1.8× faster search, 76% less storage vs dprfMM) | optimal query comm (ℓ results) | SSE index build (XOR filter) | exact (non-lossy EMM) | ★ |
| FLASH | encrypted conjunctive multimap | Intel i5-10500 @3 GHz, 8 GB RAM (CPU) | — (only relative: 180× faster search, 2× less storage vs OXTMM) | near-optimal (⚠ abs n/r) | SSE index build (binary-fuse filter) | exact (non-lossy conjunctive EMM) | ★ |
| PeGraph | real social graph (millions) | Intel i7-10700K, 64 GB RAM; 2-server | — (only relative: 5× comp / 200× comm vs GraphSE²; abs <1 s/query) | ⚠ abs n/r | SSE (OXT) index + additive-SS setup | exact (exact/fuzzy/ranked) | ★ |
| GORAM | 41.6M vertices / 1.4B edges | ⚠ not reported (3PC) | — (abs 58.1 ms–35.7 s/query) | ⚠ abs n/r (3PC) | sqrt-ORAM ego-graph build | exact | † |
| H₂O₂RAM | oblivious store (graph-RAG substrate) | ⚠ not reported (TEE/SGX) | — (only relative: ~10³× vs prior O₂RAM) | — (TEE-local) | doubly-oblivious RAM build | exact | † |

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
    refinement; SHAFT (which it is benchmarked against) covers the 2PC-secret-sharing
    Transformer-inference point. SIGMA demoted to MPC lineage in the corpus.

17. **SHAFT numbers — version discrepancy (CoVe HIGH-WARN, resolved 2026-06-02).** The
    EdgeQuake-ingested SHAFT PDF *abstract* states "62–70% less comm, 1.8–2.4× vs SIGMA,
    2.6–3.7× vs BumbleBee (LAN)". An independent CoVe check against the **NDSS camera-ready**
    (Table VII / §C) found **1.3× vs SIGMA (BERT), 25–41% less comm, 4.6–5.3× LAN / 2.9–4.4×
    WAN vs BumbleBee** — the higher abstract figures appear to conflate a softmax-component
    micro-benchmark (Table V: 2.2–2.5×, 61–67% vs Zheng et al.) with the end-to-end SIGMA
    comparison. Cost cells corrected to the table-cited camera-ready values; the ingested
    preprint version differs and should be reconciled at citation time. Hardware (2×A40,
    256 GB, Xeon Gold 5318Y) and accuracy ≈ plaintext were confirmed.

18. **Hardware re-check + EdgeQuake query bug (2026-06-02).** A PDF cross-check (ar5iv) confirmed
    EdgeQuake's PDF→MD conversion is *faithful*, but hybrid `query` is **recall-incomplete**: it
    omitted experimental-setup/hardware chunks that exist in the docs. Re-pulling full MD via
    `document_get_md` fixed 5 false-negative `⚠ not reported` hardware cells — ObfuscaTune
    (2 GPUs, middle-range), DP-Forward (Tesla P100 cluster), RemoteRAG (2×Xeon Gold 5420+ /
    2×A40 48 GB), Compass (GCP n2-standard-8 / n2-highmem-64), SGT (training 1→64×8 A100 80GB).
    Confirmed genuinely-unreported: OSNIP, SPARSE, DP-KSA, SCX (+ PIR-RAG, Panther, p²RAG,
    RAGtime-PIANO from earlier full-MD grep).
    Bug filed: `~/repos/edgequake/issues/2026-06-02-hybrid-query-misses-hardware-chunks.md`.
    Remaining `⚠`: GORAM, H₂O₂RAM (not in EdgeQuake — web-fetch only).

19. **Cost-figure CoVe pass (2026-06-02).** Forked claim-verifier checked 17 headline cost
    claims vs primary sources (web; no EdgeQuake). **15 SUPPORTED**, 1 CONTRADICTED (SHAFT, #17),
    1 minor numeric fix (H100 CC 4–7%→**4–8%**, Chrapek et al.). Euston/Fission specifics
    CANNOT-VERIFY (ePrint 403 bot-block — not contradictions). NEXUS, PipeLLM, Pacmann, PIR-RAG,
    Opal, RemoteRAG, SecFormer, XorMM (Wang et al. confirmed), GraSS (83 h@1M), ObfuscaTune,
    DP-KSA, Portcullis all confirmed against source.

20. **Comm/overhead full-MD re-check + a second EdgeQuake bug (2026-06-02).** Re-checking the
    "only relative" overhead and `⚠ abs n/r` comm cells via full `document_get_md` (per the
    hardware-bug lesson) fixed: **TwinShield hardware** (Xeon Gold 6342 + A40 48 GB; query
    false-negative), **Pacmann comm** (~few KB/query, sublinear √n Piano PIR), and **CryptoMoE**
    figures (query gave 2.8–3.5×/2.9–4.3×; full MD = 1.7–5.9× latency / 3.1–6.1× comm vs dense-MPC,
    "matches insecure baseline in some cases"). **Key caveat:** most remaining `⚠ abs n/r` comm
    cells are *not* "paper omits it" — the absolute communication/latency lives in **paper tables
    that EdgeQuake's PDF→MD conversion drops to `![tbl_…]` placeholders** (CipherFormer Table IV,
    SecFormer Table 1, Fission Table 2, CryptoMoE Table 2, …). So those numbers are unrecoverable
    from EdgeQuake (query *or* document_get_md) and need the original PDF. Filed as a separate
    high-severity bug: `~/repos/edgequake/issues/2026-06-02-md-conversion-drops-table-content.md`.
    By contrast, vs-plaintext *overhead* for crypto schemes is usually a genuine absence (they
    report vs prior scheme, not vs plaintext) — those `— (only relative)` cells are correct.

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
