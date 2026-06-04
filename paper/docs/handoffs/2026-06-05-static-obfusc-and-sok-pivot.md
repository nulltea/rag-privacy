---
type: handoff
status: current
created: 2026-06-05
updated: 2026-06-05
tags: [sok, static-obfuscation, aloepri, obfuslm, kv-cloak, competing-soks, refactor]
companion: [survey-corpus, scheme-categorization]
---

# Handoff — §5 Static Obfuscation research + deferred writing; pivot to SoK-refactoring

**Repo:** `/Users/timofey/repos/rag-privacy/paper` · branch `paper` (pushed, NOT merged to master).
**Build:** `cd manuscript && make` (pdflatex+bibtex; NO biber/xelatex). Watch the `make` mtime no-op trap
(touch `sections/*.tex main.tex` if it skips). cleveref NOT installed → `\autoref` is the active ref path.

This session did a research round for §5 Static Obfuscation, then the user **pivoted**: the new focus is
**refactoring the SoK to reference + justify against two newly-surfaced competing SoKs**, adopting targeted
features from them. The §5 writing is **deferred**. This doc captures both so nothing is lost.

---

## PART 1 — Static-obfuscation §5 research findings (this session)

### 1a. AloePri [29] ≡ Collab. Obfusc [30] — SAME PAPER (confirmed; act on this)
- Both `refs.bib` keys resolve to **one** work: **arXiv 2603.01499**, Yu Lin, Qizhi Zhang, Wenqiang Ruan,
  Daode Zhang, Jue Hong, Ye Wu, Hanning Xia, Yunlong Mao, Sheng Zhong (ByteDance + Nanjing Univ.).
- **Revision diff:** v1 (2026-03-02, 944 KB) titled "...via **Collaborative** Obfuscation (Technical Report)";
  v2 (2026-03-30, 786 KB) retitled "...via **Covariant** Obfuscation". Body calls the mechanism
  "covariant obfuscation" in BOTH. **Only substantive v2 change:** added **Rényi-metric DP (RmDP)** formal
  definition + composition theorems. **All empirical numbers identical** (acc loss 0.0–3.5% on
  DeepSeek-V3.1-Terminus 671B; TTRSR VMA 13.51 / IA 5.95 / ISA 0.0 / IMA 0.0; <5% recovery; attacks
  VMA/IA/ISA/IMA/NN/TFMA/SDA). EdgeQuake holds **v2** (doc id `812146cf-5c60-449b-855b-60e10ea8e003`,
  label "AloePri"). GitHub `sheng1feng/Aloepri` still carries the old "Collaborative" title (source of the split).
- **TODO when §5 resumes:** in `tab:obfusc` (05_obfuscation.tex L38–39) the two are TWO rows = a double-count;
  merge to one. Fix `lin_aloepri` year `2025`→`2026`, add arXiv 2603.01499; drop `lin_collaborative_obfusc`.
  Matches corpus note "−Collaborative Obfusc. = dup of AloePri" (survey-corpus.md L287). No manuscript
  footnote on the rename recommended (trivia); a one-liner in scheme-categorization Part E is enough.

### 1b. Newly-surfaced schemes (verified ABSENT from corpus/refs/research docs)
User decisions recorded:
- **ObfusLM** (Yu Lin et al., **ACL 2025**, long.58) — **ADD to §5**, but **DEFERRED**: user is uploading it
  to EdgeQuake and will signal. Same author group as AloePri (its **predecessor** → lineage point
  ObfusLM'25 → AloePri'26). Tier-1 model-obfuscation LMaaS vs embedding-inversion attacks (EIA);
  **(k,ε)-anonymity** ("provable security against EIAs"); classification **+ generation**; +10% utility,
  ~80% EIA resistance. Basis = anonymity-heuristic (NOT standard DP, NOT info-theoretic). No arXiv HTML
  (ACL PDF only; ACL PDF not text-extractable via WebFetch — read from EdgeQuake once uploaded).
  **Unresolved grill branch:** exact Basis cell + Protects cell (input embeddings vs model) — confirm from PDF.
- **KV-Cloak** (Zhifan Luo et al., **NDSS 2026**, arXiv 2508.09442; EdgeQuake id
  `9ed851a6-655b-462a-87ee-8e3a915dbf2d`) — **RECLASSIFIED to §7 Hybrid Split, next to SCX** (user-approved).
  NOT static obfuscation: security **fundamentally relies on a TEE** (OTP/permutation key matrices fresh
  per-block, held INSIDE the TEE; only obfuscated KV-cache externalized). Gray-box semi-honest server (sees
  plaintext weights + externalized cache, not TEE registers). ~0.45% prefill overhead, lossless (MMLU/SQuAD),
  1B–8B tested. Also contributes **3 attacks on naive KV-cache offload** (Inversion / Collision / Injection)
  → feed §8 attack tables. Nearest sibling = SCX (KV-cache, TEE-anchored, OTP-style, (ε,0)-DP).
  **TODO:** add as Part A.6/B.1 row + 07_hybrid_split.tex; weave its 3 attacks into §8.
- **NOT added:** AlienLM (prompt-level "language alienization", OpenReview 2025) — borderline, deferred.

### 1c. Other scheme leads surfaced (triage later, not yet acted on)
- Yixiang Yao et al. 2024 "Instance Obfuscation (IoI)" (arXiv 2402.08227) — a static-obfuscation SCHEME
  (concatenate obfuscator text; irreversible+unique embeddings). User flagged it as scheme-like (defer with ObfusLM).
  EdgeQuake id `57105106-529c-4ab9-ac8c-e7804de9ae54`.
- Mentioned-in-passing (not chased): KV-Cloak's GeoCache neighbor; PrivSplit (WebConf'26, hybrid-split);
  Selective Token-Level Cryptographic Redaction (arXiv 2606.03399, crypto).

---

## PART 2 — Deferred §5 writing work (full plan, paused mid-grill)

Target: `manuscript/sections/05_obfuscation.tex` (currently 4 `\needswork` bodies: Approach, Representative
Schemes table, Performance, Limitations/attacks — see file). User picked **"Full §5 prose + dedup + new rows"**.

Resolved grill branches:
1. **KV-Cloak → §7** (done above), so §5 new content = ObfusLM + AloePri dedup only.
2. KV-Cloak categorization = TEE-anchored hybrid (resolved).

Open grill branches (were mid-interview when user pivoted):
- ObfusLM Basis cell ((k,ε)-anonymity → heuristic-anonymity recommended) + Protects cell.
- **Narrative:** does adding ObfusLM dilute or sharpen the "static obfuscation under-protects" finding?
  Recommendation: ObfusLM (~80% EIA resistance = 20% leak, same group as the AloePri case-study target)
  **folds into the same caution** (sharpens, not dilutes); KV-Cloak in §7 reinforces "hybrid-split is the
  deployment-ready sweet spot."
- §5 prose bodies: Approach (token-subst / learned-embedding / covariant data+weight / null-space; HbC +
  white-box, Tier-1, privacy-vs-accuracy crux); Performance (cheapest family → why scrutiny); Limitations
  (weave Lin-inversion glide-reflection [lin_inversion], HNM static-permutation [hnm], ArrowMatch/TSQP
  weight-recovery [arrowmatch,tsqp], deeper-layer [depth_false_privacy], GELO Gram-matrix [belikov_gelo]).
  Note: KV-cache Inversion/Collision/Injection attacks belong in §7/§8, NOT §5 (since KV-Cloak moved).
- Existing §5 cautionary lineage prose (STIP, TransLinkGuard, TextObfuscator) stays prose-only, not in the table.

Current `tab:obfusc` rows: AloePri, Collab.Obfusc (DUP→merge), SGT, OSNIP, Eguard, ARoG. After edits:
AloePri (merged), SGT, OSNIP, Eguard, ARoG, +ObfusLM (when uploaded). ARoG placement is debatable
(KGQA anon — also lives in scheme-categorization A.4; leave as-is unless reconsidered).

---

## PART 3 — THE PIVOT (next session's primary focus): refactor wrt 2 competing SoKs

Full spike report is in the conversation; the substance:

### SoK A — Andreoletti et al., "Privacy-Preserving LLM Inference in Practice: Techniques, Trade-Offs, and
Deployability" (ePrint **2026/105**; SUPSI + Prem AI). EdgeQuake id `2d0b940f-17ca-469b-a923-1f97b0c7d689`.
- **Our CLOSEST competitor** — shares the *deployability* lens. 5 PET families (MPC/FHE/Hybrid/TEE/
  info-reduction), qualitative trade-off tables, thesis = "trust-minimising roadmap TEE→crypto-augmented→FHE."
- **Our deltas to assert:** (1) they have ~NO retrieval/PIR/ORAM/vector/graph-RAG; (2) we are quantitative
  (×-overhead, fidelity Δ, threat-fit, 1-hop extrapolation) vs their High/Med/Low; (3) **opposite thesis** —
  we elevate hybrid-split+obfuscation as readiness leaders; their unnamed "crypto-augmented mid-term" IS
  covariant-obfuscation/GELO, under-surveyed; (4) finer per-scheme 5-axis taxonomy; (5) systematic attacks +
  case study; (6) composable-integrity §7 + side-channel finding.

### SoK B — Chen et al., "SoK: Private Transformer-Based Model Inference" (ePrint **2026/491**; Northwest U.
+ Zhejiang U.). EdgeQuake id `28f72eca-5864-4b37-9d91-0008b3907247`.
- Crypto-ONLY (HE/MPC/hybrid), 61 frameworks, explicitly excludes TEE/obfuscation/DP/RAG. Distinctive:
  **reproducibility audit + re-benchmark** (P=paper vs R=reproduced, code-availability, crash flags;
  unified A10/Xeon, tc bandwidth), **network-aware re-eval**, **extensibility** (6/54 PPML extend to Transformers).
- **Our relation:** their scope = our §3, which we DEFER to Li et al. → Chen is a 2nd, more-current deferral
  target (depth + reproducibility). We span all 5 families + retrieval + graph.

### Adopt (targeted extensions — NOT duplication)
- **P/R + code-availability provenance marks** (from Chen) → cost-table legend. Cheap credibility; aligns with
  verify-claims. NOT a re-benchmark.
- **Network-sensitivity + fair-comparison checklist** (parties/quantization/network/hardware, from Chen) →
  §2.2 comparability conditions + §8 (corroborates our "numbers not comparable across regimes" rule).
- **Bottleneck framing** ("where cost comes from": linear / non-linear / autoregressive-multiplied /
  KV-cache+streaming, from Andreoletti) → §3 Approach + §4 leakage prose.
- **TEE deployment-workflow list** (attest→key-provision→secure-channel→output-handling; no plaintext logging,
  from Andreoletti) → §4 (we lack this operational detail).
- **"Deployability (no-infra-change)" axis** (Andreoletti) → make explicit in §8 (= our "obfuscation drops into
  vLLM/llama.cpp unmodified" point).
- **decision-framework/roadmap device** (both) for §8–§9 — but invert conclusion to OUR finding.

### Concrete edits the pivot implies
1. `tab:prior-soks` (01_introduction.tex): add 2 rows — Andreoletti (families all-but-retrieval; Threat
   operational; Cost ○ qualitative; Inference ●; RAG retr —; Graph —) and Chen (MPC/FHE/hybrid; Threat
   semi-honest; Cost ● incl. reproduction; Inference ●; RAG —; Graph —). Both reinforce
   "only-we = all-5-families + retrieval + graph + quantitative."
2. §1 + §10 (related_work) prose: Andreoletti = closest deployability competitor (distinguish hard); Chen =
   crypto-inference deferral cluster (defer depth+reproduction to "Li et al. and Chen et al.").
3. `refs.bib`: add `andreoletti_ppi_practice` (2026/105) + `chen_sok_pti` (2026/491). Current cluster: Bodea,
   Li (li_pti_survey), pti_survey_2024, Luo (luo_cplm_survey), Xue, Ma, Albadawi, accel_tee_sok.
4. Adopt items above (separate, smaller edits across §2.2/§3/§4/§8/§9 + cost-table legend).

---

## Where things live (read before editing — corrections-of-record)
- Curation truth: `manuscript/survey-corpus.md` (§4 = Static Obfuscation, L118–132).
- Scheme→framework map: `docs/research/scheme-categorization.md` — **Part E = corrections-of-record**;
  Part A.4 = obfuscation security rows; Part D = PDF links / upload status.
- Attack investigation: `docs/research/crypto-attack-surface.md`.
- Session primer (durable state): `~/.claude/primer.md`.
- Lessons: `paper/tasks/lessons.md`.

## Gotchas
- `make` mtime no-op trap (touch sources). cleveref absent (`\autoref` active). pdflatex overwrites annotated
  main.pdf. Per memory: grill via AskUserQuestion selectors; address-comments must reconcile every comment to a
  terminal status; EdgeQuake one-aspect-per-query (multi-aspect → document_get_md). EdgeQuake `document_get_md`
  on large docs returns a file path (read in chunks); hybrid `query` drops table/hardware chunks.

## Suggested skills for the next session
- **`grill-me`** — resolve the SoK-referencing/justification refactor branches (already requested by user).
- **`compile-latex`** — rebuild after edits (3 pdfTeX passes; verify 0 undefined).
- **`validate-bib`** then **`verify-claims`** — on the 2 new refs + any numeric claims before submission.
- **`review-paper`** — once §1/§9 positioning prose is refactored.
- (Deferred §5) re-enter the §5 grill once ObfusLM is uploaded to EdgeQuake.
