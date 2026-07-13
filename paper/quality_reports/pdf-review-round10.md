# PDF review round 10 — comment audit (2026-06-12)

Source: `pdfannots` extraction of annotated `main.pdf` (saved 2026-06-11 23:45), all on the
refactored §8 (pp 18–21). Reconciled: **37 markdown comments == 37 JSON annotations**. M = 37.
**37 of 37 resolved (36 ✅ · 1 ℹ️).** Build clean: 28pp, 0 undefined, 0 overfull, 0 `??`.
Two clusters grilled (selectors): §8.3 trim depth, §9/§7 cross-ref purge scope.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred · ❌ won't-fix.

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | §8.x | Use bold scheme names when introducing schemes | ✅ | `\textbf{}` on first mention of every scheme in §8.1/8.2/8.4/8.5 (§5/§7 style) |
| 2 | §8.1 | "two-server secret sharing" — two-party? client a party? | ✅ | → "secret sharing split across two non-colluding servers" (verified: 2 servers, client not a party) |
| 3 | §8.1 | "coarse-retrieves" — meaning? | ✅ | → "first retrieves a coarse set of $k'$ candidates" |
| 4 | §8.2 | "dense function" — don't use "dense" as adjective | ✅ | → "The second function closes that channel" |
| 5 | §8.2 | "most accurate index" — meaning? | ✅ | → "the structure that gives the best accuracy at a given search cost for high-dimensional vectors" |
| 6 | §8.2 | Compass/Pacmann confusing — chunks not entities? | ✅ | Rewritten: proximity graph over the corpus's document-chunk embeddings, not a KG; each vertex = one stored chunk vector, not an entity; they hide the search over chunk vectors (answers: yes, embedded chunks) |
| 7 | §8.2 | "ANN quality" — accuracy or utility? | ✅ | → "90% of a non-private ANN run's accuracy" |
| 8 | §8.2 | "−22% latency" — more or less? | ✅ | → "cutting latency by 22%" |
| 9 | §8.2/8.5 | "TEE baseline" — drop "baseline" | ✅ | §8.2 → "Opal (a TEE design)"; §8.5 → "$1.57\times$ a non-oblivious run" |
| 10 | §8.3/8.5 | "RAG-native systems" — meaning? | ✅ | Term dropped in §8.3 (trimmed); §8.5 reworded to "purpose-built for RAG" |
| 11 | §8.3 | Remove PrivGemo-Hand clause | ✅ | Removed (§8.3 trimmed to gap statement) |
| 12 | §8.3 | Remove ARoG pre-built clause | ✅ | Removed |
| 13 | §8.3 | Remove "remains open (§9)" | ✅ | Removed |
| 14 | §8.4 | "fans each" — proper verb? | ✅ | → "maps each entity to its source chunks" |
| 15 | §8.4 | Remove "not by RAG-designed schemes" | ✅ | Removed |
| 16 | §8.4 | XorMM/FLASH order (less-performant first) | ✅ | Reordered: "from ≈1.2–2.5× (XorMM) down to near-optimal (FLASH)" |
| 17 | §8.4 | FLASH conjunctive keyword-pair — meaning? | ✅ | → "answers multi-keyword (AND) queries, additionally reveals which keyword pairs are queried together" |
| 18 | §8.4 | PeGraph secret sharing — with whom? | ✅ | → "across two non-colluding servers" (verified) |
| 19 | §8.4 | "initialization under 3 minutes" — meaning? | ✅ | → "a one-time index setup under 3 minutes" |
| 20 | §8.4 | H₂O₂RAM "supplies" — for what? | ✅ | → "provides the doubly-oblivious RAM layer that such a graph store runs on top of, inside a TEE" |
| 21 | §8.5 | "ANN top-n" — invented? | ℹ️ | No change: verified Opal's own notation ("Securely retrieve top-$n$ ANN results") |
| 22 | §8.5 | "Where even a TEE is unavailable" — wrong premise | ✅ | → "When no trusted hardware is used" |
| 23 | §8.5 | "+13 pp ANN quality" — better? | ✅ | → "a 13-point gain in ANN accuracy" |
| 24 | §8.5 | ARoG — no citation | ✅ | Added `\citep{ning_arog}` (renders [52]) |
| 25 | §8.5 | "KGQA" — what is this? | ✅ | Expanded "knowledge-graph question answering (KGQA)" at first prose use |
| 26 | §8.5 | "Hand" — double quotes | ✅ | → ``Hand'' |
| 27 | §8.5 | "Brain" — double quotes | ✅ | → ``Brain'' |
| 28 | §8.5 | "(anonymization lineage of §7)" — why? | ✅ | Removed the §7 parenthetical |
| 29 | §8.5 | §9-frontier sentence confusing | ✅ | Reworded standalone (kept insight, dropped §9 ref): "the deployable options remain the per-function schemes above" |
| 30 | §8.6 | "consume that leakage" → exploit | ✅ | → "exploit" (both occurrences) |
| 31 | §8.6 | "trade exactness" → utility | ✅ | → "trade utility" |
| 32 | §8.6 | "on ciphertext" → ciphertexts | ✅ | → "comparisons between ciphertexts" |
| 33 | §8.6 | PPE-invertible confusing | ✅ | → "Property-preserving ciphertexts can be inverted by an adversary that knows the underlying data distribution" |
| 34 | §8.6 | kNN geometry — surface CAPRISE doesn't defend? | ✅ | Answered in text: "CAPRISE's query perturbation raises the bar here, but only oblivious fetch with query DP (RemoteRAG) closes it" |
| 35 | §8.6 | HE/SS "avoid this class" — vec2text or geometry? | ✅ | → "neither the comparison inversion nor the result geometry is exposed" (both) |
| 36 | §8.6 | PIR/Compass — defend volume attacks? | ✅ | Answered: "Volume is a separate channel: a PIR fetch returns a fixed-size record and leaks none, whereas access-pattern hiding alone does not bound result-set size, which a graph store must pad for" |
| 37 | §8.6 | "deployable dense option (§9)" — stop ref §9 | ✅ | Dropped §9 ref → "the deployable dense option" |

**Reconciliation gate:** M=37 == 37 rows == Σ terminal (36 ✅ + 1 ℹ️) == 37. Zero `⬜`. PASS.

**§9/§7 refs in §8 after purge:** only the one framing-paragraph §9 signpost (deployment-readiness)
+ two legitimate "§3–§7 inference families apply" range refs. The editorial §9-frontier,
§9-deployable, §9-open, and §7-anonymization-lineage refs are removed.
