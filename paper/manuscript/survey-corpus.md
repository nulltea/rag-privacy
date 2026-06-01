# Surveyed Corpus — Confidential Transformer Inference & RAG

Working inventory of every paper/scheme surfaced across `../docs/research/*`
(8 docs, extracted in parallel) plus the EdgeQuake-indexed set. Organized by the
**mechanism families** that form the SoK spine (§3–§7), with cross-cutting
**Attacks**, **Verification/ZK**, **Surveys**, **Supporting**, and **Commercial**
buckets. Pipeline-stage tags: embed · store · retrieve · rerank · generate ·
graph-construct (gc) · graph-traverse (gt) · end-to-end (e2e).

- ★ = full-text indexed in the EdgeQuake corpus (24 papers).
- **Metadata is as recorded in the research docs — verify before citing**
  (years/venues/IDs unconfirmed; mirrors the `refs.bib` TODO discipline).
- Breadth target for the SoK is ~40–50 *cited* schemes; this inventory is larger
  (the full pool to select from). Counts at the end.

---

## 1. Cryptographic — MPC / Secret Sharing

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| PermLLM ★ | Zheng et al. 2024, NeurIPS | generate | A-SS + permutation triples; 3s/tok 6B WAN; broken by Hidden-No-More |
| SecFormer ★ | 2024, ACL | embed,generate | 2Quad + Goldschmidt + Fourier-GELU MPC |
| CryptoMoE ★ | Zhou et al. 2026 | generate | balanced-expert-routing private MoE inference |
| PUMA | 2023 (2307.12533) | embed,generate | 3PC, polynomial softmax/GELU |
| BOLT | 2024, S&P | embed,generate | 2PC HE+ASS, Remez softmax |
| BumbleBee | 2025, NDSS | embed,generate | 2PC HE+ASS; ~8 min/tok LLaMA-7B (cited) |
| MPCFormer | 2023, ICLR | embed,generate | 2Quad replacements + distillation |
| Iron | 2022, NeurIPS | embed,generate | 2PC HE+OT, SIRNN LUTs |
| SIGMA | 2024, PETS | embed,generate | GPU FSS LUTs for max/exp/recip |
| Nimbus | 2024, NeurIPS | embed,generate | distribution-aware poly nonlinearities |
| Ditto | 2024, ICML | embed,generate | quantization-aware 3PC |
| FABLE | 2025, USENIX | embed | MPC secret-table embedding lookup |
| Fission | 2025, ePrint 2025/653 | generate | A-SS + permutation triples |
| Cascade | 2025 (2507.05228) | generate | token-sharding, HNM-resistant, non-colluding |
| PRAG / Distributed Private Similarity Search | Zyskind et al. 2024, ACL PrivateNLP | retrieve,rerank | first MPC RAG retrieval; MPC-IVF |
| p²RAG ★(?) | Ming et al. 2026 (2603.14778) | retrieve,rerank | 2-server SS, arbitrary top-k; 3–300× over PRAG |
| SANNS | Chen et al. 2019/20, USENIX Sec | retrieve,rerank | MPC k-NN, LHE+ORAM+GC |
| Panther ★? | Li et al. 2025, CCS | retrieve,rerank | single-server ANN: PIR+SS+GC+HE (hybrid) |
| PIR-RAG | Wang et al. 2025 | retrieve | classical PIR into dense RAG |
| Multiple Millionaires' Problem | Tassa & Yanai 2024, PoPETs | rerank | secure max/argmax, top-k building block |
| Hiding Your Awful Online Choices | Mukherjee et al. 2024 | retrieve | HE+MPC secure k-NN, 100M scale |
| GORAM | Fan et al. 2025, PVLDB (2410.02234) | retrieve,gt | 3PC sqrt-ORAM ego-graph queries |
| Graphiti | Koti et al. 2024, CCS | gc,gt | size-independent-round MPC graph (SGA) |
| GraphSC | Nayak et al. 2015, S&P | gc,gt | foundational oblivious SGA MPC graph |
| OblivGNN | Xu et al. 2024, USENIX Sec | gt,generate | oblivious GNN inference via FSS |
| Influential Spreaders | Kukkala & Iyengar 2020, PoPETs | gt | MPC PageRank/k-shell/VoteRank |
| Local Clustering (heat-kernel PR) | Chakkaravarthy et al. 2023, PoPETs | gt | honest-majority 3PC PageRank |
| swanky | Galois Inc | — | Rust OT/GC/ZK/VOLE/PSI toolkit (tooling) |

## 2. Cryptographic — FHE / HE

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| Euston ★ | 2025 | embed,generate | NEXUS-successor RNS-CKKS FHE inference |
| NEXUS | 2024, CCS | embed,generate | non-interactive RNS-CKKS FHE transformer |
| CipherFormer ★ | Wang et al. | embed | FHE encoder inference, small classifiers |
| THE-X | 2022, ACL | embed | FHE BERT inference |
| HE-LRM | (2506.18150) | embed | pure-FHE large-vocab lookup, 56× |
| Panda | 2022, ePrint 2022/423 | embed | poly inverse-sqrt for CKKS (LayerNorm) |
| EncFormer | 2026 (2604.09975) | rerank,generate | 2-party FHE+MPC CKKS on A100 |
| IPFE SoK | Abdalla et al. 2022 (2204.05136) | generate | inner-product functional encryption survey |
| Tiptoe | Henzinger et al. 2023, SOSP | retrieve | LHE private NN search, 360M pages |
| RAGtime-PIANO | Notre Dame 2026, ePrint 2026/231 | retrieve,e2e | CKKS cluster + lattice PIR; "first fully secure RAG" |
| Compass ★ | 2025, OSDI (ePrint 2024/1255) | retrieve,gt | FHE similarity + ORAM access-hiding ANN |
| FRAG | Zhao 2024 (2410.13272) | retrieve | federated single-key HE vector DB |
| GraSS | Kim et al., EuroS&P | retrieve,gt | FHE graph-based ANN over encrypted queries |
| Revisiting Oblivious Top-k | Cong et al. 2025, SAC | rerank | oblivious top-k over BGV/BFV/TFHE |
| TRSE | Yu et al. 2013, TDSC | retrieve,rerank | 2-round searchable enc + HE scoring |
| Strong Simulation Queries | Lyu et al. 2021, ICDE | retrieve,gt | CPA-secure encrypted-graph pattern match |
| SimplePIR / PIANO | — | retrieve | O(√n) PIR primitives |
| CKKS / BFV | Cheon 2017 / Fan-Vercauteren 2012 | — | FHE scheme primitives |

## 3. Trusted Execution Environments (TEE)

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| Opal ★ | Kaviani et al. 2026 (2604.02522) | retrieve,rerank,generate,gt,e2e | TDX+B200, ORAM disk, in-enclave KG+ANN; malicious host |
| PrivANN | Chen et al. 2025, TrustCom | retrieve | fully oblivious ANN, TEE + read-opt ORAM |
| PipeLLM | 2025, ASPLOS (2411.03357) | generate | pipelined PCIe-AES, CC overhead <19.6% |
| TEESlice | (2411.09945) | generate | formal TEE/accelerator model-slicing |
| Goten | 2021, AAAI | generate | needs 2–3 non-colluding TEEs |
| DarkneTZ | (2004.05703) | embed | TrustZone shields last CNN layers |
| Oblix | Mishra et al. 2018, S&P | retrieve,rerank | doubly-oblivious SGX search index |
| Snoopy | Dauterman et al. 2021, SOSP | store,retrieve | scalable oblivious object store |
| Obladi / ZeroTrace / Metal | — | store,retrieve | oblivious-store / ORAM baselines |
| H100 CC baseline | (2509.18886 / 2409.03992) | embed,generate | whole-GPU enclave, PCIe AES-GCM, 4–8% |
| CC-GPU perf studies | (2505.16501; ACM Queue 2024; 2507.02770) | e2e | confidential-GPU measurement studies |

## 4. Static Obfuscation

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| AloePri ★ | Lin et al. (ByteDance) | embed,generate | covariant (data+weight) obfuscation; **the vignette target** |
| Collaborative Obfuscation | Lin et al. 2026 (2603.01499) | generate | covariant w/ dynamic covariance-structured masks |
| STIP | — | generate | 3-party static permutation; breaks for open weights |
| SGT / Stained Glass ★ | Protopia 2025 | embed | input-conditioned learned affine/Gaussian obfusc |
| OSNIP ★ | Cao et al. 2026 | embed | null-space projection; 0.96ms, KNN-attack 0.000 |
| TextObfuscator | Zhou et al. 2023, ACL Findings | embed | cluster-prototype token substitution |
| Eguard | (2411.05034) | embed | RoBERTa projector, MI-loss vs Vec2Text |
| CLUB | (2006.12013) | embed | MI lower-bound loss for private encoders |
| ARoG / ARROWCLOAK / Machine-IDs | Ning et al. 2025 ★ (2508.08785) | retrieve,generate,gt | KGQA anonymization (entity→machine IDs) |
| ADCPE / DCPE ★ | Fuchsbauer et al. 2022, SCN | store,retrieve | (approx) distance-comparison-preserving enc |
| CAPRISE / prRAG ★ | Ye et al. 2026 | embed,store,retrieve | conditional distance-preserving enc for RAG |
| INT8 quantization (as obfusc) | (2507.07700) | embed | absmax/zeropoint INT8 drops Vec2Text BLEU ~60% |
| SanText / CusText / CAPE | — | embed | metric-LDP token-replacement (also DP) |

## 5. Differential Privacy

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| DP-Forward ★ | Du et al. 2023, CCS | embed | matrix-Gaussian forward-pass (ε,δ)-SeqLDP |
| RemoteRAG ★ | Cheng et al. 2025, ACL Findings | retrieve,rerank | (n,ε)-DistanceDP query noise + PHE rerank |
| SPARSE | 2026, ICLR (2602.07090) | embed | concept mask + Mahalanobis-Laplace metric-DP |
| NVDP / NVIB | (2601.02307) | embed | variational IB layer w/ Rényi-DP |
| JL + Gaussian | Blocki et al. (1204.2606) | embed | random proj + noise, (ε,δ)-DP + dist preserve |
| ReuseKNN | Müllner et al. 2023, ACM TOIST | retrieve | DP KNN recommender, fixed-neighbour reuse |
| P-NGDB | Hu et al. 2024, KDD | retrieve,gt | adversarial-loss obfusc for neural graph DBs |
| DP-RAG / DP-SGD | — | retrieve | ε-DP retrieval / training baselines |

## 6. Hybrid Split (TEE + Obfuscation)

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| GELO ★ | Belikov & Fedotov 2026 | embed,generate | per-batch fresh invertible mask, TEE non-linears; **companion-paper substrate** |
| TwinShield ★ | 2025 (2507.03278) | embed,generate | info-theoretic masking + SS + U-Verify integrity |
| ObfuscaTune ★ | Frikha et al. 2024 | embed,generate | ~5% params in TEE, GPU runs obfusc activations |
| Slalom | Tramèr & Boneh 2019, ICLR (1806.03287) | generate | OTP linear-op offload + Freivalds (CNN) |
| DarKnight | 2021, MICRO (2207.00083) | generate | K-input coding-matrix generalization of Slalom |
| AsymML / 3LegRace | 2022, PoPETs | rerank,generate | TEE low-rank + GPU residual + DP noise |
| Delta (All Rivers…) | Niu et al. 2023 (2312.05264) | rerank,generate,e2e | asymmetric-flow split w/ formal DP |
| Shredder | Mireshghallah et al. | rerank,generate | learned additive noise on activations |
| SecureInfer | 2025 (2510.19979) | generate,e2e | XOR-OTP heterogeneous TEE-GPU split |
| SCX ★ | 2025, SIGCOMM | generate | per-session one-time-key; (ε,0)-DP not OTP |
| Amulet | 2025 (2512.07495) | embed,generate | per-round fresh invertible masks all layers |
| TOGES | Kane & Bkakria 2024, LNCS (2405.19259) | retrieve,gt | graph enc, Path-ORAM position-map in SGX |
| PrivGemo | Tan et al. 2026 (2601.08739) | retrieve,generate,gt,e2e | dual-tower KG-RAG, HMAC session anonymization |

## 7. Verification / ZK

| Scheme | Ref | Stage | Note |
|---|---|---|---|
| V3DB ★ | Qiu et al. 2026 | retrieve | audit-on-demand ZK for vector-search correctness |
| ZKIFV | — | retrieve | ZK over inverted-file index correctness |
| ANNProof | — | retrieve | verifiable-ANN proof scheme |
| Anchuri | 2026 | retrieve | statistical-ZK attested retrieval |

## 8. Attacks (the threat side — substantiates "what breaks")

| Attack | Ref | Target | Note |
|---|---|---|---|
| Vec2Text | Morris et al. 2023, EMNLP | embeddings | ~92% token recovery; the canonical inversion bar |
| EDNN | Lin et al. 2024 | embeddings | ~100% model-agnostic NN inversion |
| Lin Inversion-of-Obfusc-Embedding ★ | Lin et al. 2024 (2411.05034) | obfusc embeddings | ~100% recovery from glide-reflection obfusc |
| Kellaris et al. | 2016, CCS | access patterns | generic access/volume leakage proof |
| Kornaropoulos et al. | 2019, S&P | searchable enc | SE leakage impossibility style |
| ISA (Input Stealing) | AloePri §D.1 | hidden states | gradient-opt match to public model |
| IMA (Inversion Model) | AloePri §F.1 | obfusc embeddings | trained τ-invariant inverse |
| VMA (Vocab Matching) | AloePri §F.1 | token permutation | sorted-quantile row features + voting |
| TFMA / SDA | AloePri §7.6 | token freq / sequence | frequency/order leakage of τ |
| ArrowMatch / Game of Arrows | Wang Pengli et al. 2025, USENIX Sec (2602.11088) | obfusc weights | >98% weight recovery via direction-similarity |
| Sequence-IMA | (our companion paper) | obfusc sequence | **deferred — not claimed in the SoK** |
| Hidden No More (HNM) | Thomas et al. 2025, ICML (2505.18332) | permutation obfusc | 99% recovery; broke STIP/PermLLM/Centaur |
| TSQP | 2026 (2602.11088) | static-key obfusc | recovers Llama-3-8B layer in ~6 min |
| Mohaisen & Hong ICA | 2008 (0906.0202) | rotation/Hadamard mask | ICA attack on single-mask obfusc |
| Precomputed-Noise break | Saini, Jiang, Liu 2026 (2602.11088) | precomputed-basis TEE | breaks precomputed-noise schemes |
| Speculative-Decoding split leak | Cunningham 2026 (2602.16760) | fp16 activations | MLP inversion 59% top-1 (negative result) |
| TEE.Fail / WeSee | 2025 / (2404.03526) | TEE/SEV-SNP | CC-eroding side-channel/VMM attacks |
| Exposing Privacy Risks in Graph RAG | Liu et al. 2025 (2508.17222) | graph-RAG | black-box entity-listing extraction |
| AGEA | Yang et al. 2026 (2601.14662) | graph-RAG | agentic graph reconstruction under budget |
| GRAGPOISON | Liang et al. 2025 (2501.14050) | graph-RAG | targeted-relation corpus poisoning |
| LogicPoison | Xiao et al. 2026, ACL (2604.02954) | graph-RAG | zero-token cyclic entity-swap poisoning |
| POISONEDRAG | — | RAG | chunk-level poisoning baseline |
| LinkTeller | Wu et al. 2022, S&P | GNN edges | edge inference, defeats DP-GCN at ε>5 |
| MIA on KGs / GraphMI / Graph-Embedding-Leakage / PDP-Flames | Wang/Zhang/Duddu/Hu et al. | KG embeddings | membership-inference + model-inversion on graphs |

## 9. Surveys / SoKs (positioning)

| Work | Ref | Note |
|---|---|---|
| **Bodea et al. SoK** ★ | TUM 2025 | risk-centric RAG-privacy SoK, 72 papers — **the incumbent we complement** |
| Towards Secure RAG | Mu et al. 2026 | flags reranking privacy underdeveloped |
| Private Transformer Inference Survey | 2024 (2412.08145) | FHE/MPC/TEE survey |
| PPLLM-in-Practice comparative survey | 2026, ePrint 2026/105 | practical private LLM inference |
| SoK: Accelerator TEE Designs | 2026, NDSS | accelerator-TEE systematization |
| ETH Confidential-Inference benchmark | 2025 (2509.18886) | CPU/GPU TEE benchmark |
| Confidential GPU guides | Spheron 2026 / NVIDIA WP-11459 | practitioner/vendor refs |

## 10. Supporting / Foundational (not privacy schemes; cited as machinery)

Quantization/rotation: QuIP# · QuaRot · SpinQuant · ButterflyQuant · INT8.
Transforms/math: SRHT/FJLT (Ailon-Chazelle) · Tropp-SRHT · Mezzadri Haar-QR ·
Monarch · Pixelated Butterfly · fast-hadamard-transform · Kac-walk mixing ·
Householder-RNN · Maurer indistinguishability.
Retrieval/encoders: ColBERT/ColBERTv2 · token-pooling · Matryoshka (MRL) ·
RankGPT · BGE-reranker-v2-m3 · ms-marco-MiniLM · Qwen3-Reranker · Jina-Reranker-v3 ·
HR+QDA · LightRAG ★ · GraphRAG (MS) · SPIRAL/ConcurrentQA.
Attention kernels: FlashAttention-2/-3 · FLASH-D.
Representation geometry: Dimensional-Collapse (2508.16929) · Shape-of-Learning ·
Fisher-Approx-Shannon (2504.10016).

## 11. Commercial / Products

IronCore Cloaked AI (DCPE) · Privatemode (Edgeless) · Opaque · Fortanix ·
CyborgDB/Cyborg+cuVS · ConfidentialMind · NVIDIA Nemotron rerankers · DataKrypto ·
Lattica · Javelin AI · Corvex (B200).

---

## Counts

- Cryptographic (MPC/SS): ~28 · (FHE/HE): ~18
- TEE: ~14 · Static obfuscation: ~13 · DP: ~8 · Hybrid split: ~13
- Verification/ZK: 4 · Attacks: ~25 · Surveys/SoK: 7
- Supporting/foundational: ~30 · Commercial: ~11
- **EdgeQuake-indexed (★): 24** | **Core privacy schemes (families 1–7): ~95** |
  selection target for the SoK: **~40–50 cited**.

> Sources: `../docs/research/{private-llm-inference, private-embedding-research,
> private-reranking-research, private-information-retrieval, fhe-encrypted-vector-db,
> private-graph-search, privacy-rag-research, aloepri-attacks}.md` +
> EdgeQuake corpus. Family/stage tags are first-pass; reconcile during §8 matrix build.
