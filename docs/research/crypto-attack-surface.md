---
type: research
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [sok, cryptographic, attacks, leakage-abuse, transitivity, section-3]
companion: [scheme-categorization]
---

# Cryptographic Attack Surface — per-scheme attack investigation (§3)

Comprehensive attack/leakage investigation for the §3 cryptographic schemes of the
Confidential LLM/RAG SoK, organized by the **leakage property** each scheme exposes.
Built from 6 property-clustered research agents (EdgeQuake → OpenAlex → web,
2026-06-03). Primary axis: **property → attack → schemes → defense**, with a
**transitivity** layer (attack proven on scheme A via property X; scheme B shares X
but no attack is reported for B → FLAG).

**Scope:** §3 crypto schemes in full; §4–§7 schemes only where they share a flagged
property. **Blast radius this pass = annotate, not restructure** — findings feed §3.5
prose, table footnotes, and this doc; no demotions are enacted (candidates logged in
§"Demotion candidates").

**Provenance discipline:** every attack carries bib-ready metadata (authors, year,
venue, DOI/arXiv). `[UNVERIFIED]` marks anything an agent could not confirm against a
primary source. Cost numbers reused from `scheme-categorization.md` Part E #19/#21
(already CoVe-verified) are not re-fetched.

---

## 0. Corrections of record (act before submission)

High-value contradictions surfaced during grounding. **Source preferred.**

1. **DCPE attack mis-citation (the headline fix).** The §3 draft cites `lin_inversion`
   (arXiv 2411.05034) as the attack that breaks distance-comparison-preserving
   encryption (SAP/CAPRISE). **Wrong on two counts:** (a) arXiv 2411.05034 is the
   **Eguard defense** paper, not Lin's attack; (b) the real Lin et al. attack ("An
   Inversion Attack Against Obfuscated Embedding Matrix in Language Model Inference,"
   **EMNLP 2024**, DOI 10.18653/v1/2024.emnlp-main.126) targets **glide-reflection
   obfuscation** (Mishra et al. 2024), *not* DCPE. → Re-scope `lin_inversion` to §5
   obfuscation, fix its metadata (remove arXiv 2411.05034, add the EMNLP DOI), and cite
   the correct sources for DCPE (item below).
2. **XorMM DOI/authorship.** Corpus DOI `10.1145/3548606.3560593` is **Leakage
   Inversion** (Kornaropoulos et al. CCS 2022), not XorMM. Correct: **Wang, Sun, Li, Qi,
   Chen, "Practical Volume-Hiding Encrypted Multi-Maps...," CCS 2022, DOI
   10.1145/3548606.3559345.** Confirms Part E #11 (corpus mis-credited "Patel et al.";
   Patel CCS 2019 dprfMM is the *baseline*).
3. **FLASH metadata.** Verified **Wu, Wang, Sun, Chen, IEEE TDSC 2025**, DOI
   10.1109/tdsc.2025.3600572 (corpus said "2024"). Its abstract reports **>2× storage
   *saving* vs OXTMM**, not "2–3× storage overhead" — reconcile the table cell wording.
4. **CryptoMoE venue/year.** **NeurIPS 2025, arXiv 2511.01197** (corpus/draft "Zhou et
   al. 2026"). Reconcile year.
5. **CAPRISE venue/year.** Published **IEEE FLLM 2025**, DOI
   10.1109/fllm67465.2025.11391120, authors Ye, Guo, Liu, Lam (corpus/primer "2026").
   Formal name is *Conditional **Approximate** DCPE*. Reconcile year.
6. **HNM "99%" is a paraphrase.** Thomas et al. abstract says "**nearly perfect
   accuracy**"; the ">99%" figure is a local-note paraphrase. Quote as "near-perfect" or
   pull the exact PMLR-table figure before stating a percent.
7. **PermLLM reveal-to-user nuance (threat-model).** PermLLM §6.1: the permuted
   nonlinear inputs are revealed to **P1 = the user** (their own data), *not* to the
   model provider P0 — and PermLLM explicitly "cannot achieve provable security in an
   information-theoretic sense." So "HNM breaks PermLLM" holds for a **third-party /
   adversarial-compute-node** deployment, not for the reveal-to-user design as written.
   §3.5 must carry this caveat, not a blanket "broken."

---

## 1. Attack catalog (bib-ready)

### Permutation / shuffle-secrecy
- **HNM** — Thomas, Zahran, Choi, Potti, Goldblum, Pal. *Hidden No More: Attacking and
  Defending Private Third-Party LLM Inference.* ICML 2025 (PMLR v267, pp. 59434–59469).
  arXiv 2505.18332. Vocab-matching exploits decoder injectivity; near-perfect prompt
  recovery; broke STIP/PermLLM/Centaur (fixed permutations). HbC, white-box open-weights.
  EdgeQuake: N (referenced only).
- **Cascade** — same authors. *Cascade: Token-Sharded Private LLM Inference.* arXiv
  2507.05228. **The defense co-designed with HNM** (NOT an independent scheme). Security
  is **statistical, not cryptographic**, and only "for certain sharding choices."
- **ArrowMatch / Game of Arrows** — Wang Pengli, Zhang Ziqi, et al. *Game of Arrows...*
  USENIX Security 2025, DOI 10.5555/3766078.3766093. >98% weight recovery vs static-key
  weight obfuscation (TransLinkGuard); ArrowCloak defense = 6.5× attacker cost. Full
  author list `[UNVERIFIED]`.
- **Precomputed-noise / TSQP-class** — Saini et al.(?). *Vulnerabilities in Partial
  TEE-Shielded LLM Inference with Precomputed Noise.* arXiv 2602.11088 (Feb 2026).
  Recovers a LLaMA-3-8B layer in ~6 min; breaks SOTER/TSQP/TransLinkGuard. Author list
  `[UNVERIFIED]` (corpus disagrees: Saini/Jiang/Liu vs Wang).
- **Fission's own distribution attack** — Ugurbil et al. (Fission §3, ePrint 2025/653).
  Sorted-value (permutation-invariant) binary classifier: **94.1%** on un-permuted split
  inference, **68.7%** on PermLLM-style permuted reveal (vs 50.4% random). The
  scheme-internal evidence that permuted-reveal is heuristic.

### Distance / order-preserving encryption
- **Naveed, Kamara, Wright.** *Inference Attacks on Property-Preserving Encrypted
  Databases.* CCS 2015, DOI 10.1145/2810103.2813651. Plaintext recovery of the PPE family
  (DTE/OPE — DCPE's class) from ciphertext + auxiliary distribution. **Primary
  replacement for the DCPE mis-cite.**
- **Kellaris, Kollios, Nissim, O'Neill.** *Generic Attacks on Secure Outsourced
  Databases.* CCS 2016, DOI 10.1145/2976749.2978386. Reconstruction from access +
  volume; "applies directly" to DCPE-on-vector-DB deployments.
- **Fuchsbauer, Ghosal, Hauke, O'Neill.** *Approximate Distance-Comparison-Preserving
  Symmetric Encryption (SAP/ADCPE).* SCN 2022, DOI 10.1007/978-3-031-14791-3_6. The
  **property source** — distance-ordering preservation is the leak, by definition.
- **Vec2Text** — Morris, Kuleshov, Shmatikov, Rush. EMNLP 2023. Embedding→text inversion;
  the metric CAPRISE tunes β against (BLEU 83→12.4). EdgeQuake: Y.
- (background, not DCPE-breaking) Durak–DuBuisson–Cash (ORE, CCS 2016);
  Lacharité–Minaud–Paterson (S&P 2018, improved reconstruction).

### Access-pattern leakage
- **Kornaropoulos, Papamanthou, Tamassia.** *Data Recovery on Encrypted Databases with
  k-NN Query Leakage.* S&P 2019, DOI 10.1109/sp.2019.00015. **Most directly relevant** —
  k-NN/ANN reconstruction; rel. error 2.9%→0.003%. (S&P 2020, DOI
  10.1109/sp40000.2020.00029 — distribution-agnostic extension.)
- **Islam, Kuzu, Kantarcıoğlu (IKK).** *Access Pattern Disclosure on Searchable
  Encryption.* NDSS 2012. Origin of access-pattern query recovery.
- **Kellaris et al. CCS 2016** (above) — the "access pattern + volume are irreducible"
  result.
- Lacharité–Minaud–Paterson S&P 2018; Grubbs et al. S&P 2019 (DOI 10.1109/sp.2019.00030).

### Volume leakage (encrypted multimaps / SSE / adjacency)
- **Blackstone, Kamara, Moataz.** *Revisiting Leakage Abuse Attacks.* NDSS 2020, DOI
  10.14722/ndss.2020.23103. VolAn/SelVolAn volume-only query recovery at low known-data
  rate.
- **Grubbs, Lacharité, Minaud, Paterson.** *Pump up the Volume.* CCS 2018, DOI
  10.1145/3319535.3363216 (ePrint 2019/011). Volume-only reconstruction — works even
  against access-pattern-hiding (ORAM) schemes that leak volume.
- **Ando, George.** *The Cost of Suppressing Volume...* PoPETs 2022(4), DOI
  10.56553/popets-2022-0098. **Lower bound:** many VH-EMMs cannot beat naive padding —
  framing device for "perfect volume hiding is expensive."
- **Kornaropoulos et al.** *Leakage Inversion.* CCS 2022, DOI 10.1145/3548606.3560593
  (the DOI corpus wrongly gave XorMM). (baseline: Patel et al. dprfMM, CCS 2019, DOI
  10.1145/3319535.3354213.)

### FHE ciphertext-only / approximate-FHE / depth
- **Li, Micciancio.** *On the Security of Homomorphic Encryption on Approximate Numbers.*
  EUROCRYPT 2021, DOI 10.1007/978-3-030-77870-5_23 (ePrint 2020/1533). **IND-CPA^D**:
  full CKKS secret-key recovery **when the adversary observes decryptions**. Plain
  IND-CPA (server-curious, no decryption oracle) is unaffected — the scoping caveat.
- **Cheon, Hong, Kim.** *Remark on the Security of CKKS in Practice.* ePrint 2020/1581.
  Noise-flooding countermeasure (costs precision/depth).
- **Guo et al.** *Key Recovery Attacks on Approximate HE with Non-Worst-Case Noise.*
  USENIX Security 2024. Naive worst-case flooding still leaks.
- **Bossuat et al.** *Security Guidelines for Implementing HE.* CiC 2025, DOI
  10.62056/anxra69p1. Standards-track "do this in practice" reference.
- (depth/fidelity, not an attack) RNS-CKKS is *leveled*; nonlinears need low-degree
  polynomial approximation within depth L → NEXUS ~0.25% loss but "breaks on large input
  range."

### Non-collusion / semi-honest (MPC/SS)
- **Lehmkuhl, Mishra, Srinivasan, Popa.** *Muse: Secure Inference Resilient to Malicious
  Clients.* USENIX Security 2021 (ePrint 2021/1040). **The proof that semi-honest ≠
  secure:** a malicious client breaks server-input privacy → model extraction. Load-bearing.
- **Chandran et al.** *SIMC.* USENIX Security 2022 (+ SIMC 2.0, arXiv 2207.04637). Closes
  the malicious-client gap at ~1.3–2× cost — none of the §3 schemes adopt it.
- **Mosformer** — Shen et al. *Maliciously-secure 3PC transformer inference.* CCS 2025,
  ePrint 2025/1510. The malicious-3PC bar exists but no RAG-targeted §3 scheme meets it.
- (threat-class, not a named attack on these schemes) selective-failure /
  input-dependent abort.

---

## 2. Property → attack → scheme → defense

| Property | §3 schemes | Attack(s) | Mitigated by | Open? |
|---|---|---|---|---|
| **Permutation / shuffle-secrecy** | PermLLM, Fission | HNM (near-perfect, 3rd-party deployment); Fission's own 68.7% sorted-value classifier | Fission multi-evaluator chunking (log-decay) + query batching; reveal-to-user (PermLLM) limits exposure | Heuristic, not info-theoretic — residual persists; fails under evaluator–MPC collusion |
| **Distance/order-preserving enc** | SAP/DCPE, CAPRISE | Naveed CCS'15 (PPE inversion); Kellaris CCS'16 (access/volume); Vec2Text (embedding inversion) | β tuning; CAPRISE obfuscates inter-doc distances + query DistanceDP | Ordering leak is **by design**; CAPRISE closes the inter-doc channel, query access-pattern remains |
| **Access-pattern leakage** | Panther, PIR-RAG, Pacmann, Compass, TRSE | Kornaropoulos S&P'19 k-NN recovery; Kellaris CCS'16; IKK | PIR (Panther/Pacmann) / ORAM (Compass) **close** the channel | **Closed** for Panther/Compass/Pacmann; **open** for TRSE (no ORAM) and PIR-RAG (cluster-probe) |
| **Volume leakage** | XorMM, FLASH, PeGraph, GORAM | Blackstone NDSS'20; Grubbs CCS'18; Ando lower bound | VH-EMM (XorMM/FLASH) hides volume; GORAM ORAM-class hides both | XorMM/FLASH leak **query-equality/search-pattern**; PeGraph leaks access-pattern + ranking; GORAM **closed** (3PC cost) |
| **FHE ct-only / depth** | Euston, RAGtime-PIANO, Compass(FHE) | Li–Micciancio IND-CPA^D (CKKS key recovery) | Noise-flooding (Cheon-Hong-Kim); IND-CPA suffices when server sees no decryptions | **Weak flag** for non-interactive STFI (Euston/NEXUS, server-curious = IND-CPA); **correct model** where client decrypts (RAGtime-PIANO — but that lab is IND-CPA^D-aware) |
| **Non-collusion / semi-honest** | SHAFT, CryptoMoE, p²RAG, PeGraph, GORAM | Muse (malicious-client model extraction); selective-failure class | SIMC / Mosformer (not adopted); TwinShield U-Verify (cross-family) | **Open** — all §3 schemes semi-honest-only; guarantee degrades to "honest operator" under deviation/collusion |

---

## 3. Transitivity flags

1. **Fission shares PermLLM's permuted-reveal property — RESOLVED (MED, contingent).**
   Fission evaluators receive cleartext permuted **chunks** (not shares); a single
   evaluator ≡ PermLLM. HNM's exact vocab-matching is *weakened* by chunking (no node
   holds the full hidden state), but Fission's own paper concedes residual distribution
   leakage (68.7% → `0.68−0.02·log N`, eliminated only by batching) and that
   evaluator–MPC collusion reopens it. **Verdict:** same heuristic-permutation class;
   Fission improves on, does not eliminate. Cite Fission §3/§4.4 self-evidence.
2. **HNM/TSQP weight-side → TransLinkGuard, ObfuscaTune (§5/§6).** Static-key weight
   obfuscation shares the ArrowMatch-broken property. HIGH for TransLinkGuard (named);
   ObfuscaTune flagged (shares static-key) — MED.
3. **PPE-class inversion → any geometry-preserving obfuscation (OSNIP, SGT, §5).**
   "Geometry-preservation ⇒ ordering leak ⇒ potential inversion" is proven for DCPE/ADCPE;
   for OSNIP/SGT it is a FLAG (LOW–MED; OSNIP identity unconfirmed, SGT info-theoretic
   claim already contested — Part E #9).
4. **Access-pattern transitivity STOPS at oblivious schemes.** Kornaropoulos k-NN needs
   the server to see matched-record IDs — true for TRSE and any non-oblivious DCPE/CAPRISE
   storage layer (FLAG, HIGH for TRSE), **false** for Panther/Compass/Pacmann (PIR/ORAM
   genuinely closes it — *not* merely unattacked). Do not flag the oblivious ones.
5. **XorMM volume guarantee ↛ FLASH conjunctive / PeGraph ranked.** Single-key volume
   hiding does not cover FLASH's conjunctive KPRP surface (MED-HIGH) or PeGraph's
   access-pattern + ranking (HIGH). No published per-scheme attack — would be novel.
6. **Semi-honest-only → every §3 MPC/SS scheme.** Muse-class malicious-client / collusion
   gap is unclosed by all of SHAFT/CryptoMoE/p²RAG/PeGraph/GORAM; only TwinShield (§7)
   adds U-Verify integrity. HIGH. **The strongest cross-cutting §3.5 finding.**
7. **IND-CPA^D scoping (anti-overclaim).** Flag bites only where decryptions cross a
   boundary the adversary probes. Non-interactive STFI server-adversary = plain IND-CPA →
   weak flag for Euston/NEXUS. State the precondition; do not assert CKKS-RAG is "broken."

---

## 4. Cite-worthy shortlist for §3.5 (→ refs.bib if used)

Permutation: `hnm_thomas_2025` (2505.18332) · `cascade_thomas_2025` (2507.05228, defense).
Non-collusion: `muse_2021` (ePrint 2021/1040) · optionally `mosformer_2025` (ePrint 2025/1510).
FHE: `li_micciancio_2021` (EUROCRYPT, DOI 10.1007/978-3-030-77870-5_23).
DCPE: `naveed_ppe_2015` (CCS, DOI 10.1145/2810103.2813651) · `kellaris_generic_2016`
(CCS, DOI 10.1145/2976749.2978386) — `kellaris_generic_2016` also serves access-pattern + volume.
Access-pattern: `kornaropoulos_knn_2019` (S&P, DOI 10.1109/sp.2019.00015).
Volume: `blackstone_revisiting_2020` (NDSS, DOI 10.14722/ndss.2020.23103) ·
`ando_cost_2022` (PoPETs, DOI 10.56553/popets-2022-0098).
Fix: `lin_inversion` → retitle to EMNLP 2024 (DOI 10.18653/v1/2024.emnlp-main.126),
remove arXiv 2411.05034, re-scope to §5.

---

## 5. Demotion candidates (logged, NOT enacted — annotate-only this pass)

- **None warrant demotion.** No §3 exemplar is comprehensively/exactly broken in its
  stated threat model: PermLLM's permutation is reveal-to-user (heuristic but not a
  server-side breach as deployed); Fission quantifies and mitigates its residual; the
  oblivious retrieval schemes genuinely close access-pattern leakage. The honest
  framing is "heuristic-core caveat + semi-honest-only," handled in §3.5 prose, not
  table removal. Revisit if a third-party-node PermLLM deployment is the surveyed claim.

---

## 6. Open gaps / EdgeQuake ingestion

- **Fission HNM-transfer:** resolved from primary text (this pass). No further action.
- **Author lists to confirm before refs.bib:** ArrowMatch (Game of Arrows) full list;
  arXiv 2602.11088 (Saini vs Wang); CryptoMoE (NeurIPS 2025).
- **Whether Euston (ePrint 2026/046) / RAGtime-PIANO (2026/231) deploy noise-flooding** —
  not extracted from their security sections; verify if §3.5 leans on IND-CPA^D for them.
- **Ingest into EdgeQuake** (referenced but not held): Li–Micciancio EUROCRYPT'21,
  Kornaropoulos S&P'19/'20, Kellaris CCS'16, Muse USENIX'21 — the load-bearing §3.5 cites.
