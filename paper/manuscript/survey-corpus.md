# Surveyed Corpus — Confidential Transformer Inference & RAG

Working inventory of every paper/scheme surfaced across `../docs/research/*`
(8 docs, extracted in parallel) plus the EdgeQuake-indexed set. Organized by the
**mechanism families** that form the SoK spine (§3–§7), with cross-cutting
**Attacks**, **Targeted Verification**, **Surveys**, **Supporting**, and **Commercial**
buckets. Pipeline-stage tags: embed · store · retrieve · rerank · generate ·
graph-construct (gc) · graph-traverse (gt) · end-to-end (e2e).

**Scope & target (paper corrections, 2026-06-01):**
1. **State-of-the-art / practical schemes only** in the comparison tables (§1–§7) —
   schemes with good security *and* low-enough overhead to be deployable.
   Foundational/superseded works are cited for lineage only (see the
   *Foundational / Superseded* section), never placed in comparison tables.
2. **Widely deployable; no specialized confidential-compute hardware.** Prioritize
   schemes that run on a CPU-TEE + **commodity GPUs**; do **not** assume expensive
   confidential GPUs (NVIDIA H100/B200 CC). Pure-TEE-on-CC-GPU is kept only as the
   performance/trust **baseline** (§3); the protagonist family is **hybrid split**
   (CPU-TEE + any GPU, §6).

- ★ = full-text indexed in the EdgeQuake corpus (24 papers).
- **Cover** column: `[ ]` = undecided → mark `[x]` to include in the ~40–50
  cited/systematized set. (Renders as plain text inside tables; that's fine for tracking.)
- **Threat** / **Perf** columns (scheme tables §1–§7 only): a compact adversary
  code (`intent/anchor` — HbC=honest-but-curious, Mal=malicious; anchor = none /
  CPU-TEE / GPU-TEE / TEE / 2-3PC / non-coll / 1-server / integrity) and a rough
  overhead/latency figure. **Perf numbers are NOT comparable across rows**
  (different hardware/models/datasets) — indicative only; this is the comparability
  caveat the §8 performance table must carry.
- **Metadata is as recorded in the research docs — verify before citing**
  (years/venues/IDs/numbers unconfirmed; mirrors the `refs.bib` TODO discipline).
- Breadth target for the SoK is ~40–50 *cited* schemes; this inventory is larger
  (the full pool to select from). Counts at the end.

---

## 1. Cryptographic — MPC / Secret Sharing

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | PermLLM ★ | Zheng et al. 2024, NeurIPS | generate | HbC/3-party | 3s/tok (6B WAN) | A-SS + permutation triples; broken by Hidden-No-More |
| [x] | SecFormer ★ | 2024, ACL | embed,generate | HbC/MPC | 71s (BERT-base); 3.6× vs PUMA | 2Quad + Goldschmidt + Fourier-GELU |
| [x] | CryptoMoE ★ | Zhou et al. 2026 | generate | HbC/MPC | 2.8–3.5× vs dense MoE | balanced-expert-routing private MoE |
| [ ] | PUMA | 2023 (2307.12533) | embed,generate | HbC/3PC | ~200s/tok (7B); proto-design | polynomial softmax/GELU |
| [ ] | BOLT | 2024, S&P | embed,generate | HbC/2PC | min-scale (BERT); token-prune | HE+ASS, Remez softmax |
| [x] | BumbleBee | 2025, NDSS | embed,generate | HbC/2PC | ~8min/tok (7B) | HE+ASS, segmented-poly exp; **2PC SOTA — most robust trust (no dealer/3-party)** |
| [ ] | MPCFormer | 2023, ICLR | embed,generate | HbC/MPC | ~68s (BERT-base); quad approx | 2Quad + distillation |
| [x] | SIGMA | 2024, PETS | embed,generate | HbC/2PC (FSS setup) | 45GB FSS key (BERT-large) | GPU FSS LUTs; **best online perf / billion-param scale**; weaker trust (setup) |
| [ ] | Nimbus | 2024, NeurIPS | embed,generate | HbC/2PC | ? | distribution-aware poly nonlinears |
| [ ] | Ditto | 2024, ICML | embed,generate | HbC/3PC | ? | quantization-aware |
| [ ] | FABLE | 2025, USENIX | embed | HbC/MPC | ? | secret-table embedding lookup |
| [x] | Fission | 2025, ePrint 2025/653 | generate | HbC/MPC | <20s (1B) | A-SS + permutation triples |
| [ ] | Cascade | 2025 (2507.05228) | generate | HbC/non-coll | ? | token-sharding, HNM-resistant |
| [ ] | PRAG / Distributed Private Similarity Search | Zyskind et al. 2024, ACL PrivateNLP | retrieve,rerank | HbC/MPC | sublinear comm | first MPC RAG retrieval; MPC-IVF |
| [x] | p²RAG ★(?) | Ming et al. 2026 (2603.14778) | retrieve,rerank | HbC/2-server | 3–300× vs PRAG | SS, arbitrary top-k |
| [ ] | SANNS | Chen et al. 2019/20, USENIX Sec | retrieve,rerank | HbC/2-server | 4.2s LAN @10M (clustering) | k-NN, LHE+ORAM+GC |
| [x] | Panther ★? | Li et al. 2025, CCS | retrieve,rerank | HbC/1-server | 18s @10M | ANN: PIR+SS+GC+HE (hybrid) |
| [x] | PIR-RAG | Wang et al. 2025 | retrieve | HbC/PIR | 16.8s @5K | classical PIR into dense RAG |
| [ ] | Hiding Your Awful Online Choices | Mukherjee et al. 2024 | retrieve | HbC/MPC | 100M entries | HE+MPC secure k-NN |
| [ ] | GORAM | Fan et al. 2025, PVLDB (2410.02234) | retrieve,gt | HbC/3PC | billion-edge | sqrt-ORAM ego-graph queries |
| [ ] | Graphiti | Koti et al. 2024, CCS | gc,gt | HbC/MPC | size-indep rounds | MPC graph SGA |
| [ ] | swanky | Galois Inc | — | — | — | Rust OT/GC/ZK/VOLE MPC toolkit (tooling) |

## 2. Cryptographic — FHE / HE

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | Euston ★ | 2025 | embed,generate | HbC/none | ~NEXUS-class (non-interactive) | NEXUS-successor RNS-CKKS FHE |
| [x] | NEXUS | 2024, CCS | embed,generate | HbC/none | 1.31s amortized (BERT, 256-batch) | non-interactive RNS-CKKS |
| [x] | CipherFormer ★ | Wang et al. | embed | HbC/none | 7.7–11.9× vs HErBERT | FHE encoder, small classifiers |
| [ ] | THE-X | 2022, ACL | embed | HbC/none | min-scale (BERT-tiny) | FHE BERT |
| [ ] | HE-LRM | (2506.18150) | embed | HbC/none | 56× lookup | pure-FHE large-vocab lookup |
| [ ] | Panda | 2022, ePrint 2022/423 | embed | HbC/none | ? | poly inverse-sqrt CKKS (LayerNorm) |
| [ ] | EncFormer | 2026 (2604.09975) | rerank,generate | HbC/2PC | 1.3–9.8× vs prior | FHE+MPC CKKS on A100 |
| [ ] | IPFE SoK | Abdalla et al. 2022 (2204.05136) | generate | HbC/none | ms/inner-prod | functional-encryption survey |
| [ ] | Tiptoe | Henzinger et al. 2023, SOSP | retrieve | HbC/1-server | ~2.7s (360M) | LHE private NN search |
| [x] | RAGtime-PIANO | Notre Dame 2026, ePrint 2026/231 | retrieve,e2e | HbC/none | 40× vs PIR-RAG | CKKS cluster + lattice PIR; "first fully secure RAG" |
| [x] | Compass ★ | 2025, OSDI (ePrint 2024/1255) | retrieve,gt | HbC/none+ORAM | not benchmarked (slowest) | FHE similarity + ORAM access-hiding ANN |
| [ ] | FRAG | Zhao 2024 (2410.13272) | retrieve | HbC/non-coll | ~1× (claimed, unverified) | federated single-key HE vector DB |
| [x] | GraSS | Kim et al., EuroS&P | retrieve,gt | HbC/none | 13s @5K | FHE graph-based ANN |
| [ ] | Revisiting Oblivious Top-k | Cong et al. 2025, SAC | rerank | HbC/none | low mult-depth (no wall-clock) | oblivious top-k over BGV/BFV/TFHE |
| [x] | TRSE | Yu et al. 2013, TDSC | retrieve,rerank | HbC/none | ~100s of ms (k'=100) | 2-round SE + HE scoring |
| [ ] | Strong Simulation Queries | Lyu et al. 2021, ICDE | retrieve,gt | HbC/none | ? | CPA-secure encrypted-graph match |
| [ ] | SimplePIR / PIANO | — | retrieve | HbC/PIR | O(√n) comm | PIR primitives |
| [x] | SAP / ADCPE / DCPE ★ | Fuchsbauer et al. 2022, SCN | store,retrieve | HbC/none | ~0ms (deployed) | Scale-and-Perturb (approx) distance-comparison-preserving enc; **parent of CAPRISE**; IronCore Cloaked AI |
| [x] | CAPRISE ★ | Ye et al. 2026 | store,retrieve | HbC/none | 2339 vec/s | conditional distance-preserving enc for RAG (builds on SAP) |
| [ ] | CKKS / BFV | Cheon 2017 / Fan-Vercauteren 2012 | store,retrieve | — | — | FHE scheme primitives |

## 3. Trusted Execution Environments (TEE)

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | Opal ★ | Kaviani et al. 2026 (2604.02522) | retrieve,rerank,generate,gt,e2e | Mal/TEE | 29× lower infra vs secure baseline | TDX+B200, ORAM disk, in-enclave KG+ANN; malicious host |
| [ ] | PrivANN | Chen et al. 2025, TrustCom | retrieve | HbC/TEE+ORAM | 2.4× vs FHE | fully oblivious ANN, read-opt ORAM |
| [x] | PipeLLM | 2025, ASPLOS (2411.03357) | generate | HbC/GPU-TEE | <19.6% | pipelined PCIe-AES |
| [ ] | Oblix | Mishra et al. 2018, S&P | retrieve,rerank | HbC/SGX+ORAM | 4.5–6.5× vs ZeroTrace | doubly-oblivious search index |
| [ ] | Snoopy | Dauterman et al. 2021, SOSP | store,retrieve | HbC/TEE-obliv | 13.7× vs Obladi | scalable oblivious object store |
| [x] | H100 CC baseline | (2509.18886 / 2409.03992) | embed,generate | HbC/GPU-TEE | 4–8% | whole-GPU enclave, PCIe AES-GCM |
| [ ] | CC-GPU perf studies | (2505.16501; ACM Queue 2024; 2507.02770) | e2e | — | varies | confidential-GPU measurement studies |

## 4. Static Obfuscation

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | AloePri ★ | Lin et al. (ByteDance) | embed,generate | HbC/none (wb+scheme) | ~0% (near-plaintext) | covariant (data+weight) obfusc; **the vignette target** |
| [x] | Collaborative Obfuscation | Lin et al. 2026 (2603.01499) | generate | HbC/none | near-plaintext | dynamic covariance-structured masks |
| [ ] | STIP | — | generate | HbC/3-party | near-plaintext | static permutation; breaks for open weights |
| [x] | SGT / Stained Glass ★ | Protopia 2025 | embed | HbC/none | ~0ms; −0.4% util | input-conditioned learned affine/Gaussian obfusc |
| [x] | OSNIP ★ | Cao et al. 2026 | embed | HbC/none | 0.96ms | null-space projection; KNN-attack 0.000 |
| [ ] | TextObfuscator | Zhou et al. 2023, ACL Findings | embed | HbC/none | ~0; −1pp util | cluster-prototype token substitution |
| [x] | Eguard | (2411.05034) | embed | HbC/none | ~0; F1 94→5 (Vec2Text) | RoBERTa projector, MI-loss vs Vec2Text |
| [ ] | CLUB | (2006.12013) | embed | HbC/none | ? | MI lower-bound loss for private encoders |
| [x] | ARoG / ARROWCLOAK / Machine-IDs | Ning et al. 2025 ★ (2508.08785) | retrieve,generate,gt | HbC/none | no crypto cost (anonymization) | KGQA anonymization (entity→machine IDs) |
| [ ] | INT8 quantization (as obfusc) | (2507.07700) | embed | HbC/none | ~0 (BLEU −60%) | absmax/zeropoint INT8 |
| [ ] | SanText / CusText / CAPE | — | embed | HbC/none (LDP) | near-plaintext | metric-LDP token-replacement (also DP) |

## 5. Differential Privacy

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | DP-Forward ★ | Du et al. 2023, CCS | embed | HbC/none | ~94% SST2 | matrix-Gaussian forward-pass (ε,δ)-SeqLDP |
| [x] | RemoteRAG ★ | Cheng et al. 2025, ACL Findings | retrieve,rerank | HbC/none | 0.67s | (n,ε)-DistanceDP + PHE rerank |
| [x] | SPARSE | 2026, ICLR (2602.07090) | embed | HbC/none | ~0; 65% util @ε=5 | concept mask + Mahalanobis-Laplace metric-DP |
| [ ] | NVDP / NVIB | (2601.02307) | embed | HbC/none | ? | variational IB layer w/ Rényi-DP |
| [ ] | JL + Gaussian | Blocki et al. (1204.2606) | embed | HbC/none | ~0; (1±ε) dist | random proj + noise, (ε,δ)-DP |
| [ ] | ReuseKNN | Müllner et al. 2023, ACM TOIST | retrieve | HbC/none | 17s @100M (Netflix) | DP KNN recommender, fixed-neighbour reuse |
| [ ] | P-NGDB | Hu et al. 2024, KDD | retrieve,gt | HbC/none | no runtime; −92% private MRR | adversarial-loss obfusc for neural graph DBs |
| [ ] | DP-RAG / DP-SGD | — | retrieve | HbC/none | ? | ε-DP retrieval / training baselines |

## 6. Hybrid Split (TEE + Obfuscation)

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
| [x] | GELO ★ | Belikov & Fedotov 2026 | embed,generate | HbC/GPU-TEE | 20–30% (76% offload) | per-batch mask, TEE non-linears; **companion-paper substrate** |
| [x] | TwinShield ★ | 2025 (2507.03278) | embed,generate | HbC+verify/CPU-TEE | 4–6× vs TEE-only (87% offload) | masking + SS + U-Verify integrity |
| [x] | ObfuscaTune ★ | Frikha et al. 2024 | embed,generate | HbC/TEE | 1.5–4.3×; 5% in TEE | GPU runs obfusc activations |
| [ ] | Delta (All Rivers…) | Niu et al. 2023 (2312.05264) | rerank,generate,e2e | HbC/TEE | 7.6–20× vs TEE-only | asymmetric-flow split w/ formal DP |
| [x] | SecureInfer | 2025 (2510.19979) | generate,e2e | HbC/TEE | 4.7× vs TEE-only; 2.06× vs GPU | XOR-OTP heterogeneous split |
| [x] | SCX ★ | 2025, SIGCOMM | generate | HbC/TEE | near-zero online | per-session one-time-key; (ε,0)-DP not OTP |
| [x] | Amulet | 2025 (2512.07495) | embed,generate | HbC/TEE | 2.8–4.8× vs GPU; 8–9× vs TEE | per-round fresh invertible masks all layers |
| [ ] | TOGES | Kane & Bkakria 2024, LNCS (2405.19259) | retrieve,gt | HbC/SGX+ORAM | wall-clock n/a (paywalled) | graph enc, Path-ORAM position-map in SGX |
| [x] | PrivGemo | Tan et al. 2026 (2601.08739) | retrieve,generate,gt,e2e | HbC/none+anon | no crypto cost (quality only) | dual-tower KG-RAG, HMAC session anonymization; **graph-RAG defense** |

## 7. Targeted / Composable Verification (lightweight integrity)

**Not a standalone family — a threat-model *upgrade* that composes with §3–§6.** Each
entry adds *one* cheap integrity property (hash / MAC / commitment / sampling), letting a
confidentiality scheme tolerate a server that is **malicious for that one property** at
near-zero cost — vs. proving the whole computation. **Out of scope (separate SoK):** full
verifiable inference — zkLLM, NANOZK, ZEN, zkCNN, Mystique, ezDPS, EZKL, zkLoRA — whose
proving overhead is 100×–1000s×.

| Cover | Scheme | Ref | Stage | Threat | Perf | Note |
|:---:|---|---|---|---|---|---|
|  | **A. Outsourced-compute integrity** — composes w/ §6 hybrid split |  |  |  |  |  |
| [x] | U-Verify (TwinShield) ★ | Xue et al. 2025 (2507.03278) | generate | Mal/integrity | 3.9–6.1× (verifiable inf.) | hash-check of TEE→accelerator compute; **first to verify non-linear SoftMax** (Freivalds-generalization) |
| [x] | Lightweight proofs-of-inference | Anchuri et al. 2026 (2603.19025) | generate | Mal/integrity (sampling; rational prover) | proving **min→ms**; ResNet-18, Llama-2-7B; trades soundness for speed | opens a few random output→input paths instead of proving all compute; **practical, not full-ZK**; composable |
|  | **B. Retrieval correctness & completeness** — composes w/ §1–§2 retrieval |  |  |  |  |  |
| [x] | V3DB ★ | Qiu et al. 2026 | retrieve | Mal/integrity | 22× vs circuit; ms-verify | audit-on-demand ZK that top-k matches the committed (encrypted) index |
| [x] | ANNProof | BIT 2024, FGCS | retrieve | Mal/integrity | ms-level VO; 160×/120× vs SOTA (gen/verify); VO −28×; +≤2% index-build | verifiable outsourced ANN via authenticated data structure (blockchain) |
|  | **C. Code / enclave attestation** — composes w/ every §3/§6 TEE scheme |  |  |  |  |  |
| [x] | Remote attestation / RATLS | TDX DCAP · SEV-SNP · H100 CC | setup | Mal/code-identity | ~0 (one-time) | verifies the *right enclave code/version* runs before trust — identity, not computation |
|  | **D. Corpus provenance / freshness / anti-rollback** — composes w/ §2 storage+RAG; defends §8 poisoning |  |  |  |  |  |
| [x] | Corpus commitment + freshness | Merkle/version (Opal-style) | store,retrieve | Mal/integrity | ~0 (hash) | answer grounded in an authorized, untampered, fresh corpus; **the cheap defense vs §8 poisoning** |
| [x] | ZKPROV\* | 2025 (2506.20915) | retrieve,e2e | Mal/integrity | ZK provenance (\*heavier) | ZK dataset-provenance — the heavy-crypto variant of the row above |
|  | **E. Model commitment / anti-substitution** — composes w/ any inference scheme |  |  |  |  |  |
| [x] | Model commitment | hash/Merkle + attested hash | setup | Mal/integrity | ~0 (hash) | served model = the committed one; blocks silent model swap/backdoor |

## 8. Attacks (the threat side — substantiates "what breaks")

| Cover | Attack | Ref | Target | Note (incl. recovery) |
|:---:|---|---|---|---|
| [x] | Vec2Text | Morris et al. 2023, EMNLP | embeddings | ~92% token recovery; the canonical inversion bar |
| [x] | EDNN | Lin et al. 2024 | embeddings | ~100% model-agnostic NN inversion |
| [x] | Lin Inversion-of-Obfusc-Embedding ★ | Lin et al. 2024 (2411.05034) | obfusc embeddings | ~100% recovery from glide-reflection obfusc |
| [x] | Kellaris et al. | 2016, CCS | access patterns | generic access/volume leakage proof |
| [x] | Kornaropoulos et al. | 2019, S&P | searchable enc | SE leakage impossibility style |
| [x] | ISA (Input Stealing) | AloePri §D.1 | hidden states | gradient-opt match to public model |
| [x] | IMA (Inversion Model) | AloePri §F.1 | obfusc embeddings | trained τ-invariant inverse |
| [x] | VMA (Vocab Matching) | AloePri §F.1 | token permutation | sorted-quantile row features + voting |
| [x] | TFMA / SDA | AloePri §7.6 | token freq / sequence | frequency/order leakage of τ |
| [x] | ArrowMatch / Game of Arrows | Wang Pengli et al. 2025, USENIX Sec (2602.11088) | obfusc weights | >98% weight recovery via direction-similarity |
| [x] | Sequence-IMA | (our companion paper) | obfusc sequence | **deferred — not claimed in the SoK** |
| [x] | Hidden No More (HNM) | Thomas et al. 2025, ICML (2505.18332) | permutation obfusc | 99% recovery; broke STIP/PermLLM/Centaur |
| [x] | TSQP | 2026 (2602.11088) | static-key obfusc | recovers Llama-3-8B layer in ~6 min |
| [x] | Mohaisen & Hong ICA | 2008 (0906.0202) | rotation/Hadamard mask | ICA attack on single-mask obfusc |
| [ ] | Precomputed-Noise break | Saini, Jiang, Liu 2026 (2602.11088) | precomputed-basis TEE | breaks precomputed-noise schemes |
| [ ] | Speculative-Decoding split leak | Cunningham 2026 (2602.16760) | fp16 activations | MLP inversion 59% top-1 (negative result) |
| [ ] | TEE.Fail / WeSee | 2025 / (2404.03526) | TEE/SEV-SNP | CC-eroding side-channel/VMM attacks |
| [x] | Exposing Privacy Risks in Graph RAG | Liu et al. 2025 (2508.17222) | graph-RAG | black-box entity-listing extraction |
| [x] | AGEA | Yang et al. 2026 (2601.14662) | graph-RAG | agentic graph reconstruction under budget |
| [x] | GRAGPOISON | Liang et al. 2025 (2501.14050) | graph-RAG | targeted-relation corpus poisoning |
| [x] | LogicPoison | Xiao et al. 2026, ACL (2604.02954) | graph-RAG | zero-token cyclic entity-swap poisoning |
| [ ] | POISONEDRAG | — | RAG | chunk-level poisoning baseline |
| [x] | LinkTeller | Wu et al. 2022, S&P | GNN edges | edge inference, defeats DP-GCN at ε>5 |
| [x] | MIA on KGs / GraphMI / Graph-Embedding-Leakage / PDP-Flames | Wang/Zhang/Duddu/Hu et al. | KG embeddings | membership-inference + model-inversion on graphs |

## 9. Surveys / SoKs (positioning)

| Cover | Work | Ref | Note |
|:---:|---|---|---|
| [x] | **Bodea et al. SoK** ★ | TUM 2025 | risk-centric RAG-privacy SoK, 72 papers — **the incumbent we complement** |
| [x] | Towards Secure RAG | Mu et al. 2026 | flags reranking privacy underdeveloped |
| [ ] | Private Transformer Inference Survey | 2024 (2412.08145) | FHE/MPC/TEE survey |
| [ ] | PPLLM-in-Practice comparative survey | 2026, ePrint 2026/105 | practical private LLM inference |
| [ ] | SoK: Accelerator TEE Designs | 2026, NDSS | accelerator-TEE systematization |
| [ ] | ETH Confidential-Inference benchmark | 2025 (2509.18886) | CPU/GPU TEE benchmark |
| [ ] | Confidential GPU guides | Spheron 2026 / NVIDIA WP-11459 | practitioner/vendor refs |

## Foundational / Superseded — privacy-scheme lineage (mention-only, excluded from comparison)

Cited briefly for lineage; **not** in the §1–§7 comparison tables (worst-on-performance
or obsoleted by descendants — per scope correction #1).

- **MPC/SS:** Iron (2022 — foundational 2PC HE+OT; superseded on comm by BumbleBee/BOLT) ·
  GraphSC (2015 — foundational oblivious SGA; 13h@1M) · Multiple-Millionaires (building
  block; circuit-depth only) · OblivGNN / Influential-Spreaders / Local-Clustering
  (graph-MPC niche; no LLM-scale perf).
- **TEE:** Obladi / ZeroTrace / Metal (oblivious-store baselines; superseded by Snoopy) ·
  DarkneTZ (TrustZone CNN-layer shielding; CNN-era) · Goten (needs 2–3 non-colluding TEEs;
  impractical) · **TEESlice** (security *analysis*, not a perf scheme — shows naive
  TEE↔GPU split breaks under public weights; cite as the **motivation** for obfuscated split).
- **Hybrid split:** Slalom (2019 — seminal additive-blinding offload; CNN/linear-only) ·
  DarKnight (2021 — coding-matrix generalization; CNN) · ShadowNet (2023 — weight-mask
  offload) · SOTER (2022 — fingerprint-integrity masking) · AsymML / 3LegRace
  (low-rank/residual split) · Shredder (learned-noise activations). All superseded on
  performance by the modern obfuscated-split schemes (GELO / TwinShield / ObfuscaTune /
  Amulet / SecureInfer).

## 10. Supporting / Foundational machinery (not privacy schemes; cited as building blocks)

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

Comparison-table sizes (after the practical-frontier reclassification):

- Cryptographic (MPC/SS): 22 · (FHE/HE): 20
- TEE (baseline): 7 · Static obfuscation: 11 · DP: 8 · Hybrid split: 9
- Targeted verification: 8 (full-ZK inference excluded — separate SoK) · Attacks: ~25 · Surveys/SoK: 7
- **Foundational/superseded (mention-only): ~16** · Supporting machinery: ~30 · Commercial: ~11
- **EdgeQuake-indexed (★): 24** | **In-comparison privacy schemes (§1–§7): ~91** |
  selection target for the SoK: **~40–50 cited**.

> Sources: `../docs/research/{private-llm-inference, private-embedding-research,
> private-reranking-research, private-information-retrieval, fhe-encrypted-vector-db,
> private-graph-search, privacy-rag-research, aloepri-attacks}.md` +
> EdgeQuake corpus. Family/stage/threat/perf tags are first-pass; reconcile during
> the §8 matrix + performance-table build. Perf numbers are non-comparable
> (different hardware/models/datasets).
