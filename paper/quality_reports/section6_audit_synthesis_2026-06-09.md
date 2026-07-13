# §6 Differential Privacy — Four-Lens Audit Synthesis (2026-06-09)

File: `manuscript/sections/06_differential_privacy.tex`
Lenses: `/humanize` (AI-voice) · `/term-audit` (word-choice/register) · `/proofread` (grammar/typo/overflow) · `/verify-claims` (numeric facts, via EdgeQuake primary sources).
**No edits applied — report only.** Resolution is decided in the grill that follows.

---

## Headline

The section is **clean on every high-risk AI-voice category** (0 em-dashes, 0 boilerplate transitions, 0 cliché lexicon, 0 hedge-stacking, 0 sycophancy → cosmetic-only on humanize). The material findings are **one factual error**, a handful of **terminology/grammar fixes**, and **\emph overuse**.

Priority order: **P0 factual → P1 terminology/consistency → P2 grammar/clarity → P3 style/cosmetic.**

---

## P0 — Factual (must fix)

| # | Line | Issue | Primary-source resolution |
|---|---|---|---|
| F1 | 54 | SPARSE: "keeping ${\approx}65\%$ utility **at $\varepsilon{=}5$** while cutting leakage from $60\%$ to $19\%$" — but line 169 attributes the same $60\%\to19\%$ to **$\varepsilon{=}10$**. | **Source (Tsai et al., §4.2): "On the STS12 dataset at $\varepsilon{=}10$, SPARSE reduces privacy leakage from 60% to 19%... maintains 65% downstream utility."** Both numbers are at **$\varepsilon{=}10$, STS12.** → **Line 54 is wrong; change $\varepsilon{=}5 \to \varepsilon{=}10$.** Line 169 is correct. |
| F2 | 51 | DP-Forward "${\approx}94\%$ on SST-2" — **overstated; exceeds the non-private baseline (impossible).** | **Source (Du et al., DP-Forward).** Table 3 (SeqLDP): best SST-2 = **0.9266** (d=16, ε=16), **0.9186** at the deployment d=768 (ε=16). Table 5 (CDP): best = **0.9055** (ε≈8). **Non-private baseline = 0.9178.** No config reaches 94%. → **Change to "${\approx}92\%$ on SST-2 at $\varepsilon{\approx}16$"** (parity with SPARSE/NVDP, which state ε). |

## Verified-correct (no change)

- **DP-KSA N≈80** ✓ (paper: "set the number of ensembles to 80... the optimal number").
- **DP-KSA ε≈1 / ε≈2 behavior** ✓ (NQ: "at ε=1, performance tends to drop below or remain comparable to the non-RAG baseline"; TQA: "beginning at ε=2, DP-KSA consistently outperforms the non-RAG baseline"). "No-private-context baseline" correctly = non-RAG (ε=0) baseline.
- **SPARSE training cost** ✓ ("25.3 minutes" for 10k, "under 45 minutes" for 20k, RTX 3090 — exact).
- **SPARSE "frozen encoder" (L52) vs NVDP "pretrained" (L57)** ✓ — a *real* distinction, not an inconsistency: SPARSE trains only a mask+classifier over a fixed embedding model (Φ never updated); NVDP fine-tunes BERT. Keep both.
- **DP-Fusion m+1 passes/token** ✓ (consistent with bib + reframe).

## Unverified (confirm before final)

| Line | Claim | Status |
|---|---|---|
| 51 | DP-Forward "${\approx}94\%$ on SST-2" | Exact 94% **not in retrieved excerpts.** Plausible: DP-Forward matches non-private baseline (~91.8%) and beats DP-SGD variants' 92.5%. **But ε is unstated** — figures in the paper use ε=16 (moderate regime per the paper's own ε<10 strong / 10≤ε<20 moderate split). Recommend: confirm 94% against DP-Forward Table 3/5 and **state the ε** for parity with SPARSE/NVDP, which both cite ε. |
| 59 | NVDP "$83.0\%$ on MRPC at $\varepsilon_\mu{\approx}11$" | Carried from prior session (primer-verified), **not re-verified this pass.** Low risk. |

---

## P1 — Terminology / consistency

| # | Line | Issue | Fix |
|---|---|---|---|
| T1 | 119 | "as $\varepsilon$ **tightens**" — directionally ambiguous (tighter privacy = smaller ε) **and** inconsistent with the de-jargon pass (rest of §6 uses "smaller/lower"; caption L136 "smaller budget", L167 "Lowering ε"). | "as $\varepsilon$ **decreases**" (or "is lowered"). |
| T2 | 172 | prose "a higher-recall **named-entity recognizer**" conflicts with table (L149) and L171 "tagger". Primer records NER→tagger was a deliberate change. | "a higher-recall **tagger**". |
| T3 | 71 | "invert the **trust setup**" — "setup" is informal. | "trust assumptions" / "trust model". |
| T4 | 109 | "only when compute and memory are **spare**" — colloquial adjective. | "available" / "idle". |
| T5 | 44–45 | "model retraining is the **coarse mirror** of this" — metaphor. | Borderline; consider "tracks this inversely" or literal phrasing. (LOW) |
| T6 | 51 / 59 | precision style: "${\approx}94\%$" vs "$83.0\%$" (one approximate, one exact-looking decimal). | Pick uniform precision. |

---

## P2 — Grammar / clarity

| # | Line | Issue | Fix |
|---|---|---|---|
| G1 | 39 | "In **the** local DP the client perturbs" — wrong article; no comma; breaks parallel with L70 "central DP" (no article). | "In local DP, the client perturbs". |
| G2 | 177 | "**So** far-apart inputs can be separated..." — sentence-initial "So" is informal; reads as fragment. | Merge into the prior sentence with ", so". |
| G3 | 74 | "their confidentiality scope... **differs from the local schemes**" — compares scope to schemes. | "differs from **that of** the local schemes". |
| G4 | 84 | comma splice after colon: "DP-KSA protects corpus membership, DP-Fusion protects context spans." | semicolon, or "DP-KSA protects corpus membership **while** DP-Fusion...". |
| G5 | 108 | "for $m$ sensitive groups **and they parallelize**, so" — unclear antecedent / comma. | "for $m$ sensitive groups; **because these passes parallelize**, wall-clock...". |
| G6 | 117–119 | double-aside: "...as $\varepsilon$ tightens, the answer the model would give without the corpus or context at all:" — appositive dangling before a colon. | Parenthesize the appositive: "...toward its \emph{no-private-context} baseline (the answer the model would give without the corpus or context at all):". |
| G7 | 124–129 | single ~60-word sentence, many nested clauses. | Split at "whereas" into two sentences. |
| G8 | 64 | list "(nothing, which dimensions, the whole distribution)" mixes "nothing" with noun phrases. | "(none, which dimensions, the whole distribution)". |

---

## P3 — Style / humanize / cosmetic

| # | Line | Issue | Fix |
|---|---|---|---|
| S1 | 4–67 | **\emph overuse (~17 spans).** Contrastive italics that carry the analytical axes earn it (\emph{whose}/\emph{whom}, \emph{which}/\emph{where}/\emph{the noise distribution itself}). Vocal-stress italics tip into the AI pattern. | Thin: drop italics on \emph{statically to the model's weights} (L11), \emph{fresh per request} (L17), \emph{the input} (L67), \emph{with}/\emph{without} (L80–81). Keep the axis contrasts. |
| S2 | 62–64 | double parenthetical tricolon **re-lists** the DP-Forward/SPARSE/NVDP trio already walked through at L46–60. | State the trio once with force; trim the recap. |
| S3 | 99 / 106 | parallel paragraph openers "The local schemes pay \emph{offline}, once:" / "The central schemes pay \emph{online}:" back-to-back. | Intentional symmetry; optionally vary one opener's syntax. (LOW) |
| S4 | 162 | "by whether DP can answer them" duplicates the table caption (L134). | Vary the body phrasing. (LOW) |
| S5 | 60, 85 | trailing whitespace; L129→130 missing blank line before `\subsection`. | Cosmetic cleanup. |

---

## Cross-cutting verify item (project convention)

- **`\citep` (table, L147/149/153/155/157) vs `\cite` (prose).** Build is clean (0 undefined), so `\citep` resolves — but if the project standard is uniform `\cite` (autoref path, no natbib parenthetical distinction), the table's `\citep` is inconsistent. **Confirm intended command** before harmonizing.
- **Out-of-§6 bib note (low priority):** `tang_dprag` author order — paper is Tingting **Tang**, Yongqin **Wang**, James **Flemings**, Murali **Annavaram** (USC; arXiv 2602.14374). Bib currently "Tang and Flemings and Wang and Annavaram" (Wang/Flemings swapped).

---

## Recommendation

1 factual fix (F1) is **mandatory**. T1–T2 and G1–G4 are clean, low-risk corrections. S1–S2 are the only judgment calls worth a deliberate pass. Everything else is cosmetic. **No rewrite needed** — this is a strip-and-tighten pass.
