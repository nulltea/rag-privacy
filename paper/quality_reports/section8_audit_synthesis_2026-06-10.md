# §8 Confidential Retrieval — Four-Lens Audit Synthesis (2026-06-10)

Lenses: `/humanize` (humanize-auditor) · `/term-audit` (audit phase) · `/proofread` (proofreader) · `/verify-claims` (claim-verifier, fresh context, 13 claims). Target: `manuscript/sections/08_retrieval.tex` (~1,750 prose words).

**Headline:** Voice clean (no cliché lexicon, no hedging, no boilerplate; em-dash debt only). But this is a number-dense section and verify-claims surfaced **5 factual issues** — none fabricated, but several wrong labels/figures plus 2 unverifiable numbers. Storage cost-vs-saving polarity flagged independently by proofread (HIGH) and verify (C12).

---

## P0 — Factual corrections (verify-claims)

**V1. PIR-RAG primitive wrong (table L61).** "classical PIR" → PIR-RAG uses **lattice/LWE-based PIR** (homomorphic matrix-vector product). Source: arXiv 2509.21325 §4.3. Fix cell to "lattice PIR". (16.8s @5K and cluster-based: confirmed.)

**V2. Opal mislabeled "exact" (table L149).** Fidelity "exact ($+13$\,pp)" overstates — Opal retrieval is **approximate (IVF-PQ ANN + KG filter + rerank)**, not exact. Source: arXiv 2604.02522. (2.32s, 1.57×, +13pp: all confirmed.) → relabel fidelity. **[GRILL Q2]**

**V3. XorMM storage factor off (table L143).** "$1.5$--$2\times$ storage" matches neither variant: base XorMM ≈**1.23n**, verifiable VXorMM ≈**2.46n**. Source: CCS 2022. → correct. **[GRILL Q1]**

**V4. FLASH storage polarity (table L144 + prose L185).** Cell says ">2× storage **saving**"; prose L185 bundles FLASH into "$1.5$--$3\times$ storage **cost**". Contradictory polarity in/across the storage column. FLASH (TDSC 2025) frames it as **near-optimal storage overhead** (binary-fuse filter, within 13% of the lower bound), a saving *relative to prior VH-EMMs*; the specific ">2×" magnitude is **unconfirmed**. **[GRILL Q1]**

**V5 (LOW). Panther 284MB/query unconfirmed.** 18s@10M and the PIR+SS+GC+HE stack confirmed; the 284MB comm figure could not be independently verified (paywalled body). Low risk — keep unless Q3 pass says otherwise.

---

## P0 — Proofread HIGH

**F1. TRSE "100s ms" renders nonsensically (table L63).** `${\sim}100$s\,ms ($k'$=100)` reads as "100s ms" — ambiguous between "~100 ms" and "hundreds of ms". Must disambiguate. **[GRILL Q4]**

**F2. \needswork visible in PDF (table footnote L156).** XorMM co-measure TODO renders as a visible note. Tracked in primer (deliberate placeholder, not this session's scope) — leaving unless you say otherwise; flagged for submission-readiness.

---

## P1 — Unverifiable numbers (verify-claims MED)

**V6. PrivGemo node-reduction numbers (prose L223-224).** "${\approx}70\%$ node reduction" and ">2.4M-node CWQ → ~732K" could **not be located** in accessible text (likely appendix/figure). Design (Hand/Brain dual-tower, HMAC session IDs, crypto-free, QA preserved) fully confirmed; CWQ QA 67%→59% confirmed. Memory rule: no unverified numbers in the manuscript. **[GRILL Q3]**

---

## P1 — Structural prose / table semantics (proofread MED)

| ID | Line | Item | Fix |
|---|---|---|---|
| F3 | 92 | "reaching 0.67s" dangles off "The user…sorts" | split: "…final fetch. The scheme reaches $0.67$\,s at $100\%$ recall…" |
| F4 | 104-109 | 6-line SAP/CAPRISE sentence (colon+semicolon+colon) | split after "vector-analysis attack." |
| F5 | 141, 146 | Pacmann "$-$22\% lat", H2O2RAM "$\sim$1000× vs O2RAM" = relative-only in an absolute-latency column, no `$^*$` flag (tab:crypto-vec uses one) | add `$^*$`/"rel." marker for consistency |
| F6 | 148 | PrivGemo "${\approx}70\%$ subgraph" sits in Comm./storage column but is a fidelity/reduction figure | move to Fidelity or footnote |
| F7 | 205, 211 | `\citep` in tab:retr-attacks cells; other two tables use `\cite`; L211 nests parens "(bound (..))" | switch cells to `\cite`; L211 `\cite{ando_cost_2022}` |

---

## P2 — Register / terminology (term-audit + humanize sycophancy)

| ID | Line | Item | Proposed |
|---|---|---|---|
| F8 | 166, 178 | "load-bearing" (client) ×2 | paper's own "client-reliance"/"cost on the client" framing |
| F9 | 182 | adjacency multi-maps "graph retrieval **walks**" | "traverses" |
| F10 | 213 | "de-uniqueness sanitization" (coined) | "de-identification" / "structural anonymization" |
| F11 | 256 | oblivious schemes "**anchor the deployable end**" | "are the deployment-ready schemes" |
| F12 | 230 | "frontier **sits at** the retrieval layer" (sycophancy-adjacent + figurative) | "lies at" / ground the claim |
| F13 | 29 | "**push** the heavy…steps **onto** the client" | "shift the…steps to the client" |
| F14 | 47 | "index-storage **blow-up**" (caption) | "index-storage overhead/expansion" |
| F15 | 36/39 vs 40/43 | "leak" (count noun) vs canonical "leakage channel" | standardize on "leakage channel"/"channel" |

---

## P2 — Em-dash sweep (4 prose/caption `---`, all HIGH per rule)

- L47 caption: "storage---approximate" → ":" 
- L72 footnote: "vs PIR-RAG --- neither" → ";"
- L128 caption: "storage---graph-indexed" → ":"
- L156 \needswork body: "workload---fill" → ";"
- L150 table cell `---` (ARoG Fidelity placeholder): not prose; align to "n/r"/"lossy" convention (overlaps F-low below).

(En-dashes `--` in ranges like `58\,ms--36\,s`, `3--300$\times$`, `27--75\times` are correct, not flagged.)

---

## P3 — Low / cosmetic

- L28 "no-hardware designs" → "no-trusted-hardware" (house term).
- L150 ARoG Fidelity `---` → "n/r" (caption-defined convention; also an em-dash).
- L224 "CWQ" unexpanded → "ComplexWebQuestions (CWQ)".
- vs → vs.\ across L71/72/146/194 (I standardized §3 to "vs.\"; match here).
- L230 empty `\paragraph{}` — renders (build clean); optional `\medskip\noindent`.
- British "-our" (neighbour) + "-ize" (anonymize): Oxford convention, intentional — leave.

---

## Verify-claims: the SUPPORTED set (8/13 clean)

RemoteRAG (0.67s, 100% recall, DistanceDP+PHE two-stage) · CAPRISE (2339 vec/s, conditional ADCPE+DP, query-doc only) · SAP/DCPE (c=s·e+λ, all pairwise comparisons = the leak) · Pacmann (−22% latency @100M, 90% ANN, batched PIR) · GORAM (58ms–36s @1.4B, init <3min, 3PC sqrt-ORAM) · H2O2RAM (~1000× speedup — direction correct) · p2RAG (3–300× vs PRAG) + RAGtime-PIANO (40× vs PIR-RAG) · Ando bound. All direction-sensitive claims point the right way.
