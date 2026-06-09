# PDF review round 7 — comment audit (2026-06-09)

Source: `pdfannots` extraction of the annotated `main.pdf` (saved 2026-06-09 14:51). All 28
comments are on §6 Differential Privacy (pp. 11–12). **28 of 28 resolved (28 ✅).** Applied via the
EdgeQuake-grounded research + grill workflow; build clean (27pp, 0 undefined, no new overfull).
Committed in `<pending>`.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred · ❌ won't-fix · ⬜ pending (none remain).

Reconciliation: JSON annotation total **M = 28** == markdown rows 28 == audit rows 28 == Σ terminal (28 ✅).

Key research findings (EdgeQuake + web, primary-source verified):
- Obfusc/hybrid DO borrow DP noise (AloePri Gaussian+RmDP; ObfusLM Laplace/(k,ε)-anon; SCX Laplace/(ε,0)-DP; CMIF RNM-DP), but as an adjunct to a **secret key**, applied **statically to weights**. DP-only family's distinction: guarantee from **randomness alone** (no secret), noise **fresh per request on data/outputs**.
- DP family splits **local** (input/embedding: client-side noise, untrusted operator, protects user's own input — DP-Forward/SPARSE/NVDP) vs **central** (output: trusted operator, server-side noise, protects operator-held corpus/context from the *consumer* — DP-KSA/DP-Fusion). Maps to Axis 4 confidentiality scope.
- DP-KSA retrieval = plain dense top-k (output-DP, stays §6, not §8). DP-Fusion: operator sees context in cleartext, no trusted HW, protects output recipient.
- Training: SPARSE mask 25.3min/10k (linear, RTX3090); NVDP = 1 IB layer over frozen BERT; DP-Forward ≈ ordinary FT (~3× < DP-SGD).
- Attacks: Vec2Text/GEIA inversion, IMA >75% on DP-Forward, GAN recon (NVDP/Bayesian-DP), MIA-against-RAG (corpus, DP-KSA), tagger-FN (DP-Fusion). Metric-DP distance = ℓ2 embedding metric (not ranking).

| # | Page / loc | Comment (gist) | Status | What changed |
|---|---|---|---|---|
| 1 | p11 intro "family rather than single mechanism" | Establish DP unique vs obfusc/hybrid that use DP internally | ✅ | Intro rewritten: names AloePri/ObfusLM/SCX as DP-borrowers, distinguishes by guarantee-from-randomness + per-request-data-vs-static-weight noise |
| 2 | p11 "Embedding-DP perturbs" | Embedding-DP needs client-side embedding = client TCB cost | ✅ | Local-DP paragraph: "client must run the encoder, moving compute and trust base onto the device" |
| 3 | p11 "provable bound" | Make provable concrete; (non-write) why crypto/TEE formal, obfusc not; info-theoretic | ✅ | "provable" made concrete (bound holds vs adversary knowing weights/mechanism/budget); Axis-5/info-theoretic answered in chat (no §6 expansion) |
| 4 | p11 "output-side composing orthogonally" | Clarify output-side meaning + dependence direction | ✅ | Central-DP paragraph: output = generated text; defends the consumer; composes orthogonally with any backend |
| 5 | p11 "trading fidelity as ε tightens" | "tightens" arbitrary | ✅ | Changed to "as the budget ε shrinks" (explicit direction) throughout |
| 6 | p11 "calibrated noise under budget ε" | Gaussian/Laplace/exponential → different DP notions + trade-offs | ✅ | Mechanism-mapping clause in Approaches: Gaussian→(ε,δ); Laplace/metric→ε/metric-LDP; exponential→discrete selection |
| 7 | p11 "task-agnostic mechanism calibrated" | Define task-agnostic; (non-write) is it why DP-Forward fine-tunes | ✅ | Defined inline; "the model is fine-tuned to perform on the noisy embeddings" answers the fine-tune link |
| 8 | p11 "NVDP replaces the fixed mechanism" | SPARSE also learned → "fixed" confusing | ✅ | Reframed as progression of what's learned (nothing / which dimensions / the distribution); "fixed mechanism" removed |
| 9 | p11 "stochastic posterior" | What is a posterior, standard term? | ✅ | Glossed: "a posterior (a learned distribution) over the multi-vector embedding" |
| 10 | p11 "shared representation" | Unclear + relation to DP noise | ✅ | Replaced with "the shared embedding" / "noised embedding it shares" |
| 11 | p11 "sensitivity-selective vs learned task-aware" | Both learned → difference? | ✅ | Distinguished as learns-where (SPARSE dimensions) vs learns-the-distribution (NVDP) |
| 12 | p11 "Output perturbation. Two schemes…" | (non-write) does embedding noise carry to output? | ✅ | Post-processing sentence added (local schemes protect input through to output); full answer in chat |
| 13 | p11 "treats the retrieved corpus as private dataset" | cosine vs inference? belongs in §8? | ✅ | Stated retrieval is ordinary dense top-k; privacy on the answer → stays §6, not §8 |
| 14 | p11 "runs model once per sensitive group" | Trusted hardware? operator sees plaintext? | ✅ | "operator runs the model and sees the context in the clear, with no trusted hardware"; protects the output recipient |
| 15 | p11 "applied per retrieved chunk" | What is a chunk? decoder or reranker? | ✅ | "per retrieved chunk"; both output schemes act at the generation stage |
| 16 | p11 "are not interchangeable" | Say what they ARE | ✅ | "complementary rather than interchangeable: DP-KSA protects corpus membership, DP-Fusion protects context spans" |
| 17 | p11 "Query perturbation. RemoteRAG" | Drop from §6 | ✅ | RemoteRAG paragraph removed entirely; query-DP lives in §8 |
| 18 | p12 "6.2 Representative Schemes" | Drop this subsection | ✅ | Dropped; representative grounding (models, headline numbers) folded into Approaches |
| 19 | p12 "6.3 Performance" | Report accuracy + perf×acc×security trade-off | ✅ | Renamed "Performance and Accuracy"; trade-off prose driving the three-way space |
| 20 | p12 "At inference … near-plaintext" | Does stronger noise/metric cost performance? | ✅ | Finding: ε = pure accuracy knob for local-DP; stronger privacy costs online compute (N, m) for central-DP; contrast with static obfusc |
| 21 | p12 "DP-Forward fine-tunes … SPARSE mask … NVDP NVIB" | Concrete training-overhead numbers | ✅ | SPARSE 25min/10k (linear, RTX3090); NVDP 1 IB layer over frozen encoder; DP-Forward ≈ ordinary FT (~3× < DP-SGD); inf-perf rows tagged offline-cost |
| 22 | p12 "DP-KSA N× … DP-Fusion m+1 passes" | Cover in detail + add to perf table | ✅ | Prose detail; tab:inf-perf cells updated (N≈80; m+1 pass/tok) |
| 23 | p12 "parallelize … near single pass" | Strong assumption; orchestration cost | ✅ | Softened: latency ≈1× only when compute/memory spare; m+1× real; needs orchestration (wrapper, not bespoke engine) |
| 24 | p12 "the shared currency" | Colloquial | ✅ | Replaced with "the cost the whole family ultimately pays" |
| 25 | p12 "degrading sharply as ε→0" | Slope shape + dependence | ✅ | Stated rate is mechanism/content-dependent (DP-Fusion log vs power-law); no universal sub/super-linear law |
| 26 | p12 "6.4 Limitations and Known Attacks" | Concrete attacks + citations; mitigable vs structural | ✅ | New tab:dp-attacks (mitigable/structural bands) + concise attack-focused prose, all cited |
| 27 | p12 "standard (ε,δ)-DP" | Not introduced earlier | ✅ | Standard (ε,δ)-DP introduced at DP-Forward in Approaches |
| 28 | p12 "in proportion to their distance" | What distance? which attack? | ✅ | Clarified: ℓ2 metric on the embedding vector (not a ranking score); exploited by nearest-neighbour search |
