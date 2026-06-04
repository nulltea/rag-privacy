# PDF review round 2 — comment audit (2026-06-04)

Source: `pdfannots` extraction of the annotated `main.pdf` (saved 2026-06-04 13:33; raw dump
preserved in `quality_reports/pdf-comments-round2.md`, since rebuilds overwrite the annotated
PDF). Two annotation batches: **#1–26** on the 13:33 PDF, **#27–33** a continuation on the
15:21 PDF (`pdfannots -f json` M=7: 6 Highlights + 1 free-floating Text; preserved at
`annotated-pdfs/main.annotated-2026-06-04-1521.pdf`). **All 33 resolved** (0 undefined
refs/cites; only the pre-existing 43pt overfull; build 27pp). Structural calls #15/#21/#25
and the #27–33 batch taken with user sign-off via grill. Not yet committed.

**Reconciliation correction (2026-06-04, hardened skill).** The authoritative annotation count
is **M = 26** (`pdfannots -f json` → 25 Highlights + 1 Text), but the markdown extraction
rendered **25 bullets**: a free-floating Text note on p10 ("Fidelity and comms missing, separate
columns") was merged into the "Reported cost" bullet and originally folded into row #21, losing
its identity. It is now split out as **row #26**. Both halves were already addressed (the
Comm./storage + Fidelity columns and the latency-clarifying caption), so this is a bookkeeping
fix, not missed work — but per the Completeness Contract every annotation now has its own row.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred (reason) · ❌ won't-fix (reason) · ⬜ pending (transient).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | §3 intro (p8) | "we treat that sub-family briefly" → "we cover" | ✅ | "treat"→"cover". |
| 2 | §3.1 Approach (p9) | "Transformer-FHE" reads like a scheme name — disambiguate | ✅ | → "the RNS-CKKS scheme used in most FHE-based Transformer inference". |
| 3 | §3.1 Approach (p9) | TFHE w/ programmable bootstrapping — do we cover any? | ℹ️ | No — all covered FHE exemplars are leveled RNS-CKKS. Added a parenthetical in §3.1 noting TFHE/PBS LUT evaluation as an alternative we don't exemplify (deferred to the inference surveys). |
| 4 | §3.2 (p9) | Missing CryptoMoE description: why private MoE routing matters vs naive, how done | ✅ | Added a §3.2 sentence: naive private MoE must evaluate every expert or leak routing (experts specialize → routing reveals input type); CryptoMoE secret-shares routing + balances expert load, recovering sparsity at 2.8–3.5× dense-MPC. **Routing-leak threat grounded in a concrete attack** — Yona et al. 2024 "Stealing User Prompts from MoE" (arXiv 2410.22884), via `\citep{yona_moe_steal}` (+refs.bib). NB: CryptoMoE itself cites no attack (motivates via expert-specialization, ref [18] Yang et al. 2025 + its Fig 1b); Yona's cross-batch threat model differs from CryptoMoE's MPC-party leakage — noted in the bib entry. |
| 5 | §3.2 title (p9) | "(Exemplars; Depth Deferred)" — remove | ✅ | Title → "Cryptographic Inference (Exemplars)". |
| 6 | §3.2 (p9) | Long "X—Y—Z—" dash list is hard to read — reword | ✅ | Rewrote as a per-scheme parenthetical naming each exemplar (also clarifies which scheme is which setting). |
| 7 | §3.2 (p9) | "one-to-three orders of magnitude" repeats §3 intro — pick one | ✅ | Dropped the OoM restatement from §3.2 (kept canonical in §3 intro); §3.2 now keeps only the comm/fidelity detail. |
| 8 | §3.2 Table 4 (p10) | How does Li et al. report perf/fidelity; how normalize? | ✅ | §3.2 prose now states Li et al. don't normalize to plaintext — they tabulate self-reported latency/comm/task-accuracy, anchored to BERT-base where possible. |
| 9 | §3.2 Table 4 (p10) | SHAFT BERT-base — num params? | ✅ | Setting cell → "BERT-base (110M)". |
| 10 | §3.2 Table 4 (p10) | "MoE 6.9–16.4B" which model exactly? active params? | ✅ | Cell → "MoE (6.9–16.4B; 1.3–2.8B act.)"; models (OLMoE/QWenMoE/DeepSeekMoE) named in §3.2 prose. "B" = total clarified in caption. |
| 11 | §3.2 Table 4 (p10) | "MPC+evaluators, ModernBERT" — num params? | ✅ | Cell → "ModernBERT (149M)". |
| 12 | §3.2 Table 4 (p10) | "≈10.7 s†" non-informative — num tokens reported? | ✅ | Caption: encoder rows = one forward pass over 128-token sequence; "≈10.7 s/inf"; †footnote gives NEXUS anchor (128 tok). |
| 13 | §3.2 Table 4 (p10) | "≈1.4 s†" — num tokens reported? | ✅ | "≈1.4 s/inf"; same caption + † note (SHAFT via SIGMA, 128 tok); also noted SHAFT's own 41 s LAN. |
| 14 | §3.2 Table 4 (p10) | "<5 s/inf" — num tokens reported? | ✅ | Caption clarifies "/inf" = one forward pass over a 128-token sequence (encoder). |
| 15 | §3.3 (p10) | Retrieval must dive deeper per scheme/subfamily: privacy mechanism, threat model, what makes fast/slow; for graph schemes explain GraphRAG/LightRAG relevance | ✅ | [decision: by-subfamily] Added 4 §3.3 paragraphs — access-hiding (PIR/ORAM), volume-bounding (SSE/EMM), distance-preserving (DCPE), + "why retrieval is the deployable frontier" — each with mechanism, threat model, cost driver, and explicit LightRAG retrieval-surface tie-in. |
| 16 | §3.3 (p10) | "IVF" — define | ✅ | Glossed "flat or inverted-file (IVF) index" in prose; ANN spelled out in caption. |
| 17 | §3.3 (p10) | "Reported cost" — what does it measure (query)? | ✅ | Both tables restructured; "Latency" column + caption "latency = end-to-end query time unless noted (gt = graph-traversal query)". |
| 18 | §3.3 (p10) | "2-round SE + HE scoring" — what's SE? footnote | ✅ | tab:crypto-vec caption now defines SE = searchable encryption (+ PIR, HE). |
| 19 | §3.3 (p10) | "3–300× vs PRAG" — extrapolate | ✅ | Cannot extrapolate: p²RAG reports relative-only (Figure-only, no absolute latency); PRAG absolute not pinned. Marked "rel. only*" with a footnote stating the relative figures + that no absolute is reported. |
| 20 | §3.3 (p10) | "∼0 ms (deployed)" — do they report vec/s? | ✅ | SAP is a theory paper — no vec/s (kept "∼0 ms", symmetric transform); CAPRISE 2339 vec/s clarified as DB-encryption throughput ("enc."). |
| 21 | §3.3 (p10) | "Reported cost" highlight — "reported time: what this measures, query, gt, or?" | ✅ | Caption now states latency = end-to-end query time unless noted (gt = graph-traversal query); "Reported cost" → "Latency" column. |
| 22 | §3.3 (p10) | "SSE" — define, footnote | ✅ | tab:crypto-graph caption now defines SSE, (VH-)EMM, ORAM (also defined in §3.1). |
| 23 | §3.3 (p10) | "billion-edge" — clarify (GORAM scale) | ✅ | GORAM latency cell → "ego-query, billion-edge"; caption: "billion-edge denotes the largest graph tested, not a latency." |
| 24 | §3.3 (p11) | "we treat the attack" — arbitrary word, reword | ✅ | "we treat the attack"→"we examine the attack". |
| 25 | §3.4 Performance (p11) | Repeats §3.2/§3.3 — reconsider what we report here, or drop | ✅ | [decision: drop] §3.4 deleted; its one new line (inference-vs-retrieval regime contrast + comparability caveat) migrated into the §3.3 "why retrieval is the deployable frontier" paragraph. §3.5 Limitations is now §3.4. |
| 26 | §3.3 (p10) | **Free-floating note** (the one merged into #21 in round 2): "Fidelity and comms missing, separate columns" | ✅ | Both retrieval tables rebuilt to Scheme/Stage/Primitive/Latency/**Comm./storage**/**Fidelity** (scriptsize, tabcolsep 4pt; n/r where unreported). Now its own row per the Completeness Contract. |
| | | **— continuation batch (15:21 PDF, M=7) —** | | |
| 27 | §3.2 title (p9) | Remove "(Exemplars)" from the section heading | ✅ | Heading → "Cryptographic Inference". |
| 28 | §3.3 tab (p11) | Do SAP/DCPE/CAPRISE papers report storage overhead for encrypted vs plaintext vectors? | ✅ | Answered: no blow-up to report — DCPE ciphertext is same-dimension ($c=se+\lambda$), ${\approx}1\times$ storage. tab:crypto-vec cells "—"→"≈1×"; stated in the §3.3 DCPE paragraph. |
| 29 | §3.3 DCPE ¶ (p12) | (a) "orderings" imprecise — scheme preserves distance comparisons; (b) missing how CAPRISE differs from SAP + its contribution | ✅ | (a) "distance orderings"→"distance comparisons" (the scheme's own term, DCPE); (b) added: SAP preserves all pairwise comparisons (leaks corpus structure → vector-analysis attack); CAPRISE is conditional — asymmetric noise preserves only query↔doc comparisons, obfuscates inter-doc, + query DP. |
| 30 | §3.4 Limits (p12) | "SIMC)" — what is SIMC? | ✅ | Expanded inline → "SIMC, secure inference resilient to malicious clients [simc_2022]"; +refs.bib entry. |
| 31 | §3.4 Limits (p13) | "preserves distance orderings" — again "orderings"; is this the papers' term? | ✅ | → "preserves distance comparisons" (matches DCPE); also fixed the §3.1 one-liner and the paragraph title ("ordering"→"distance comparison"). |
| 32 | §3.4 Limits (p13) | "ANN/SSE schemes leak instead" — which schemes exactly? ANN is not a cryptographic scheme | ✅ | Named the schemes (Panther, Pacmann, Compass; XorMM, FLASH, PeGraph) + added "''ANN'' is a search method, not a scheme — the leakage is the index's, not the search's". |
| 33 | §3.4 Limits (p13) | **Free-floating note** (markdown-merged; recovered from JSON): "Which schemes are you referring to here? ANN is not a privacy scheme" | ✅ | Same fix as #32 (the two annotations are the same concern); now its own row per the Completeness Contract. |

**Li et al. methodology adoptions (this session, from the grill — not PDF comments).** (1) §3.1
now cites Li et al.'s non-linear-bottleneck quantification (>85% MPC / ~60% HE runtime). (2)
tab:crypto-inf Fidelity → **Δacc** (cryptographic schemes only; PermLLM 0, SHAFT/Fission ≈0,
CryptoMoE −0.8%, Euston approx†), crediting their plain/enc/loss convention; cross-family Δacc
deferred to `docs/handoffs/2026-06-04-delta-accuracy-sweep.md`. (3) hardware-basis caveat added
to the tab:crypto-inf caption (latencies on differing hardware, not co-measured).
