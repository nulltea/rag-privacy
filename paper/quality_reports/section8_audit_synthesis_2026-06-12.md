# §8 four-lens audit synthesis (2026-06-12)

Target: `manuscript/sections/08_retrieval.tex` (after the by-function refactor + round-10 comment fixes).
Lenses: /humanize (forked), /proofread (forked), /term-audit (forked general), /verify-claims (foreground EdgeQuake).

**STATUS: RESOLVED + APPLIED (grilled via selectors), build clean 28pp/0undef/0overfull/0??.**
Decisions: R1=parallel/drop-ordinals; R7=fix "one machinery"→"a single mechanism" (keep "encrypted-graph-database machinery");
R16=keep "non-colluding"; R12=apply "relaxation"→"option"; R14="re-ranking"→"reranking" (paper convention).
All HIGH+MED+LOW below applied (R1–R15). UNCOMMITTED.

## Headline
- **/humanize: 0 HIGH, 4 MED, 5 LOW per ~1750 words → cosmetic.** No boilerplate, no cliché lexicon,
  no hedge-stacking, no sycophancy, no prose em-dashes (all `---` are table band-labels).
- **/proofread: one HIGH** (intro "First,/Second," fragments), several MED/LOW. All 25 citations resolve;
  the four attack cites (kornaropoulos/blackstone/naveed/ando) correctly attributed.
- **/term-audit: 0 HIGH, 6 MED, 4 LOW.** Confidential/utility/oblivious terminology consistent.
- **/verify-claims: clean** except an Opal-precision wording fix.

## Resolution list (deduped across lenses)

### HIGH
- **R1** [proofread] §8 intro ¶2: "First, \emph{scoring…}, made confidential…  Second, \emph{hiding…}, secured…"
  are fragments (no finite verb); ordinals then drop to bare gerunds + "And" for the 3 graph functions.
  → restructure with finite verbs + parallel form.

### MED
- **R2** [proofread+term] L269 "preserve answer quality but are quality-preserving \emph{obfuscation}" tautology → drop redundant "quality-preserving".
- **R3** [proofread+term] British/American: L145 "neighbours", L146 "neighbourhood", L232 "neighbours" → American (paper uses defense/artifact/minimizing).
- **R4** [humanize] L110–126 scoring ¶: 3-semicolon chain + repeated "the shared cost is…/the shared benefit is…" closer → split sentence, vary one closer.
- **R5** [humanize] L238–251 adjacency ¶: ~75-word, 3-semicolon sentence (PeGraph/GORAM/H₂O₂RAM) → split.
- **R6** [term] L107 "Either way the price is exactness" → "the cost is".
- **R7** [term] "machinery": L122 "one machinery hides both", also L238 "encrypted-graph-database machinery" → "mechanism"/"primitives". JUDGMENT CALL.
- **R8** [term] L158 keyinsight "whether obliviousness costs" (intransitive) → "has a cost".
- **R9** [term] L289 "CAPRISE's query perturbation raises the bar" (idiom) → "increases the attacker's cost".
- **R10** [proofread] L195 stray space "ms/q ;" → "ms/q;"; verify XorMM/FLASH latency cells don't overfull in PDF.
- **R11** [verify-claims] L259 Opal "a 13-point gain in ANN accuracy" → "a 13-point accuracy gain over plain ANN"
  (verified: KG-filtering improves judged accuracy by 13pp over ANN-only; 1.57× is over Plaintext/non-oblivious Opal — wording holds).

### LOW / optional
- **R12** [term] L124 "the cheaper relaxation" → "cheaper option" (low confidence).
- **R13** [proofread+humanize] L62–66 split the ~55-word opening sentence after "compute-hiding".
- **R14** [proofread] L85 "re-ranking" → align to paper convention ("reranking"?).
- **R15** [proofread] L249–250 H₂O₂RAM "runs on top of, inside a TEE" stranded prep → reword.
- **R16** [verify-claims] p²RAG "two non-colluding servers": grounded in the verified threat table's "2-srv SS";
  corpus has no standalone p²RAG doc to re-confirm, but non-collusion is definitional for 2-server secret sharing.
  Keep or soften to "two-server secret sharing".

## verify-claims detail
- Opal +13pp = improvement over ANN-only (judged accuracy); 1.57× over Plaintext Opal. ✓ (R11 refines wording)
- FLASH conjunctive leaks search/query-equality over s-term/x-terms → "which keywords are queried together". ✓
- p²RAG 2-server SS: consistent with verified table; not independently re-surfaced this round (R16).
- All cited attacks correctly attributed (proofreader cross-checked refs.bib).
