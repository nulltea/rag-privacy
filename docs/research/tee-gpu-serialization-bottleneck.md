---
type: research
status: current
created: 2026-06-05
updated: 2026-06-10
tags: [hybrid-split, tee-gpu, serialization, gpu-utilization, pipelining, gelo, round-trips, kv-cache, threat-model, twinshield, scx, kv-cloak, pipellm]
companion: [scheme-categorization, obfuscation-round2-2026-06-05, gpu-offloaded-attention-with-value-cover]
---

# The TEE↔GPU serialization bottleneck — survey and GELO design spike

Two parts in one document. **Part I (literature survey)** asks *which papers,
schemes, and repositories specifically target the per-layer serial chain between
the trusted enclave and the untrusted accelerator, and with what techniques?*
**Part II (GELO design spike)** asks the same question from our deployment side:
*given GELO's threat model (SEV-SNP enclave ↔ VFIO-passed untrusted GPU,
`WEIGHTS-PUB` adversary), which of the surveyed mechanisms actually buy us
fewer or hidden round-trips, and which look promising but violate the cover
invariant?* Part II identifies the design directions worth a measurement spike
— not commitments.

Sources (Part I): WebSearch + WebFetch on primary arXiv/USENIX/ASPLOS/IEEE
sources, GitHub. Method note: synthesized from a fan-out search+fetch+verify
pass — 24 sources had their claims extracted and adversarially verified
(3-vote, "default to refuted"). Per-claim caveats and a 2026-06-08
self-correction are in §(e). This document does **not** modify the manuscript.

Companion read for GELO-specific cover details:
[`gpu-offloaded-attention-with-value-cover.md`](../dev/prototype/gpu-offloaded-attention-with-value-cover.md)
— how attention currently rides the GPU under a per-session non-orthogonal
`C_v` cover, which threats it defeats, and which it does not.

---

# Part I — Literature survey

## (a) The bottleneck, precisely

In the hybrid-split family (\S7 of the SoK: GELO, TwinShield, ObfuscaTune, SCX, and the
Slalom/ShadowNet/SOTER lineage), the model is partitioned by *operator type*:

- **Linear ops** — QKV projections, FFN/MLP GEMMs, output projection — run on the
  **untrusted GPU** over blinded/obfuscated activations (cheap on the GPU, and the bulk
  of FLOPs).
- **Non-linear ops** — Softmax, GELU/activation, LayerNorm/RMSNorm — run **inside the
  TEE** (typically a CPU enclave: SGX/TDX), because they are hard to evaluate securely
  under linear obfuscation.

This creates a **per-layer serial data-dependency chain**:

```
GPU matmul_i ─► encrypt+PCIe ─► TEE non-linear_i ─► encrypt+PCIe ─► GPU matmul_{i+1} ─► …
```

The GPU cannot begin `matmul_{i+1}` until the TEE returns non-linear output `i`. During
the TEE compute + the two PCIe crossings (each with AES-GCM encrypt/decrypt), the GPU is
**idle**. Because the CPU enclave is slow (encrypted memory, no SIMD throughput of a GPU)
and PCIe round-trips are frequent (one per layer, ×depth, ×decode-step), GPU utilization
collapses. This is the core tension: the partition exists for security, but the
data-dependency it induces serializes the two devices that should run in parallel.

**Key structural sub-costs** (from primary sources): on a confidential H100, the CC-mode
penalty is *dominated by the CPU↔GPU PCIe transfer*, not GPU compute
(arXiv:2409.03992; arXiv:2509.18886). In a CPU-enclave hybrid, per-tensor GPU→enclave
transfer costs **0.2–0.3 ms**, enclave→GPU **<0.01 ms** (SecureInfer, arXiv:2510.19979) —
so the round-trip is asymmetric and transfer-bound, not compute-bound. After offload,
in-TEE non-linear work can still be **>60 % of total execution time** (TwinShield,
arXiv:2507.03278).

---

## (b) Findings table — transformer-era hybrids + GPU/NPU-TEE hardware

*(CNN-era schemes are decoupled into their own section — see "CNN-era TEE↔GPU schemes"
below.)* Technique classes: **(1)** cross-layer/cross-request pipelining · **(2)** async /
speculative / prefetch · **(3)** batching to hide TEE latency · **(4)** reduce/eliminate
round-trips (non-linear on GPU under obfuscation, operator restructuring, fusion) ·
**(5)** overlap encryption with compute · **(6)** dependency-aware partition / scheduling
· **(7)** GPU-/NPU-TEE hardware that removes the round-trip · **(8)** decouple security
ops (async OTP/mask gen) from the critical path.

| Scheme | Venue / id | Class | What it overlaps / does about serialization | Reported number | Code | Relevance |
|---|---|---|---|---|---|---|
| **TwinShield** | 2025 · arXiv:2507.03278 | **(4)+(6)** | **Reduces round-trips by operator restructuring**: OutSoftMax (n exp→2n mult+n add in TEE), OutAttnMult (shifts O(mnp) mult to GPU; SoftMax on GPU under additive secret-sharing); permutations combine ops into single GPU calls. **Still sequential — no pipelining.** In-TEE residual >60%. | ~87% offload; 4.0–6.1× vs prior; 4.9–7.7× private inf | — | **Transformer-era; the round-trip-reduction frontier** |
| **SCX** | SIGCOMM'25 · DOI 10.1145/3718958.3750509 | (4) data-movement | Stateless KV-cache encoding (user-key permutation + optional additive noise), lossless; optimizes KV-cache *communication*, not GPU-compute overlap. | 36 ms LLaMA-7B; +85% KV-cache comm efficiency | yuanmu97/scx | Transformer; KV-cache layer |
| **KV-Cloak** (Shadow-in-the-Cache) | 2025 · arXiv:2508.09442 | (8) **aspirational** | Defense `K' = S·P̂·(K+A)·M` with **orthogonal `S, M` + magnitude-dominant additive `A`**; threat model gray-box weights-known (matches `WEIGHTS-PUB`). Async OTP-matrix generation to decouple masking from critical path is **named as future work, not implemented**. Software obfuscation, **not** TEE+GPU hybrid. | 15.41 ms/GB (0.45% prefill) vs AES 3020.9 | — | Parallel design to GELO's `C_v`; names the async-OTP lever; doesn't realize it |
| **SecureInfer** | 2025 · arXiv:2510.19979 | (4); names overlap as future work | SGX + GPU; privacy-critical tensors in enclave, linear matmul to GPU under OTP/XOR. Measures **asymmetric** transfer cost (GPU→CPU 0.2–0.3 ms/tensor, enclave→GPU <0.01 ms — the enclave-bound direction is ~20–30× more expensive) and **explicitly recommends** "caching or overlapping secure decryption with computation can further improve overall responsiveness." Does not implement; batching/continuous-batching also listed as future work in §VI. | 0.2–0.3 ms/tensor GPU→CPU; 4.7× vs TEE-only; 2.06× vs GPU-only; 8.44 tok/s | — | **Second paper (after KV-Cloak) naming async overlap as the right unexplored direction** |
| **PipeLLM** | ASPLOS'25 · arXiv:2411.03357 | **(2)+(5)** | **Speculative pipelined encryption**: overlaps *encryption* with GPU compute; predicts to-be-encrypted data from LLM serving patterns + relinquish-on-mispredict. Targets the **confidential-GPU PCIe path**, not the CPU-enclave non-linear round-trip. | CC drop 52.8%/88.2% (OPT-30B/66B) → **<19.6%** (13B–175B) | SJTU-IPADS/PipeLLM | **The single most on-target overlap system** (but different round-trip) |
| **TensorTEE** | ASPLOS'24 · arXiv:2407.08903 | **(5)+(7)** | Unifies CPU↔NPU TEE granularity at **tensor level** → direct transfer **without re-encryption**; tensor-granularity MAC + predictive execution avoids stalls. | **4.0×** LLM training vs prior het-TEE; 2.1% vs non-secure | — | Removes the per-transfer crypto that serializes |
| **Ascend-CC** | 2025 · arXiv:2407.11888 | **(7)** | On-NPU (Huawei Ascend) confidential computing, no host trust; encrypts data+params+operator binaries; **keeps full graph on accelerator → no host round-trip**. | "minimal overhead"; Llama2/3 | — | Discrete-NPU analogue to H100 CC |

---

## (c) Per-technique synthesis

**(1) Cross-layer / cross-request pipelining — essentially absent.** No surveyed system
actually overlaps GPU matmul for one request/layer with TEE non-linear work for another.
Slalom, ShadowNet, SOTER, SecureInfer, and TwinShield are all *architecturally
sequential*: TEE prepares masked data → GPU computes → TEE recovers/applies non-linear →
repeat. ShadowNet even names the CPU↔GPU interleaving (repeated GPU job setup) as an
overhead source but does not hide it.

**(2) Async / speculative — only PipeLLM ships it; SecureInfer measures the motivating
asymmetry and names it as future work.** PipeLLM's speculative pipelined encryption is
the only mature speculative overlap, but it targets the *encryption* step on a
confidential GPU, not the non-linear-in-CPU-enclave dependency. SecureInfer profiles the
enclave↔GPU transfer cost and finds it **asymmetric**: GPU→CPU 0.2–0.3 ms/tensor (the
secure-unwrap direction) vs enclave→GPU <0.01 ms — a ~20–30× gap. The paper concludes
"these findings suggest that caching or overlapping secure decryption with computation
can further improve overall responsiveness" and lists it as future work alongside
continuous batching. So the field has two papers (PipeLLM-shipped, SecureInfer-future-work,
plus KV-Cloak-future-work) all pointing at the same unexploited lever, with PipeLLM the
only working-system proof.

**(3) Batching to hide TEE latency — uniformly deferred.** SecureInfer lists
continuous/dynamic batching as *future work*; no hybrid system reports request- or
token-level batching that interleaves TEE and GPU work across a batch. This is the most
obvious unexploited lever for decode.

**(4) Reduce/eliminate round-trips — the active transformer-era frontier.** TwinShield is
the strongest example: OutAttnMult converts multiplicative attention to additive form
with precomputed offline values so attention offloads GPU-side, and OutSoftMax runs
Softmax on the GPU under additive secret-sharing — pushing non-linears *out* of the TEE
rather than overlapping the round-trip. AsymML shrinks the TEE side via low-rank
decomposition. These reduce how often/much you cross the boundary, but the residual chain
stays serial (TwinShield still spends >60% in-TEE).

**(5) Overlap encryption with compute — PipeLLM, TensorTEE.** PipeLLM overlaps AES with
GPU compute; TensorTEE removes per-transfer re-encryption entirely by unifying TEE
granularity at the tensor level (4.0× on LLM training). Both attack the *crypto* portion
of the round-trip, which on confidential GPUs is the dominant cost.

**(6) Dependency-aware partition / scheduling — Goten, SOTER.** Goten's round-optimal
protocol minimizes the *number* of TEE↔GPU communication rounds; SOTER's partition ratio
trades security for fewer enclave-resident ops. These are scheduling/partition choices,
not runtime overlap.

**(7) GPU-/NPU-TEE hardware — makes the bottleneck moot when available.** The cleanest
"fix" is to not split at all: run the whole graph (linear *and* non-linear) inside a
confidential GPU (H100 CC) or NPU (Ascend-CC). Then the CPU-enclave round-trip vanishes
and the only residual is PCIe DMA encryption (<7%, → ~0 for large models). TensorTEE
generalizes this to heterogeneous CPU↔NPU TEEs.

**(8) Decouple security ops from the critical path — named, not realized.** KV-Cloak
proposes async OTP-matrix generation to hide masking latency, but explicitly as *future
work*; Slalom's offline precomputed blinding is the only shipped instance, and it only
decouples *mask generation*, not the non-linear dependency.

---

## (d) Gap analysis — is this under-addressed for transformers?

**Yes, sharply.** Three findings:

1. **The overlap/pipelining lever is unexploited across the board, and entirely so for
   transformers.** Even the CNN/training era (Slalom, Goten, AsymML) attacked the
   bottleneck by *reducing communication* or *shrinking the TEE side*, never by
   overlapping GPU compute with enclave non-linear work across requests/layers. No
   transformer hybrid (TwinShield, SecureInfer, GELO, SCX) pipelines softmax/GELU-in-TEE
   against matmul-on-GPU. The decode phase — where a per-token non-linear round-trip
   recurs thousands of times and the batch dimension offers natural overlap candidates —
   is exactly where pipelining should pay off and exactly where nobody has tried it.

2. **The field is bifurcating into "reduce round-trips" (software) vs "remove them"
   (hardware), skipping "hide them" (scheduling).** TwinShield/AsymML reduce; H100 CC /
   Ascend-CC / TensorTEE remove. The middle path — keep the CPU-enclave split (for
   schemes that *need* it, e.g. when a confidential GPU is unavailable or the obfuscation,
   not the hardware, is the security root) but *hide* the latency via cross-request
   pipelining + transfer coalescing — is the open opportunity.

3. **SecureInfer's measurement motivates async overlap.** The 0.2–0.3 ms/tensor
   GPU→CPU vs <0.01 ms reverse asymmetry quantifies the cost of the secure-unwrap step
   that gates every TEE non-linear; the paper's own recommendation is to overlap this
   with computation. Combined with KV-Cloak's parallel "async OTP generation as future
   work" framing, three independent papers now point at the same lever from three angles
   (PipeLLM ships AES overlap on confidential GPU; SecureInfer recommends decryption
   overlap on CPU-enclave hybrid; KV-Cloak recommends OTP-gen overlap on commodity-GPU
   obfuscation). **A scheme that (i) overlaps mask-prep / unwrap with GPU compute on the
   single-request critical path, (ii) batches enclave non-linear work across concurrent
   requests, and (iii) coalesces the many small PCIe transfers into fewer large bursts
   would be novel for transformer hybrid inference. No surveyed system does this.**

**Caveat to the "hardware makes it moot" view:** confidential-GPU schemes change the
trust model (trust the silicon vendor + on-die root of trust) and are gated on CC-capable
hardware availability. The obfuscation-rooted hybrid split exists precisely to run on
*commodity* untrusted GPUs (\S7 framing); for that deployment niche the serialization
bottleneck is real and unaddressed.

---

## (e) Verified / unverified / contradicted

**Verified (primary source):**
- PipeLLM: 52.8%/88.2% drop → <19.6% (arXiv:2411.03357, ASPLOS'25).
- H100 CC: <7% typical, PCIe-transfer-bound; per-model Llama-3.1-8B 6.85%, Phi-3-14B
  4.58%, Llama-3.1-70B ≈0 (arXiv:2409.03992); GPU-TEE 4–8%, CPU-TEE <10%/<20%
  (arXiv:2509.18886).
- TwinShield: ~87% offload, 4.0–6.1× / 4.9–7.7×, >60% in-TEE residual (arXiv:2507.03278).
- SecureInfer: 0.2–0.3 ms/tensor GPU→CPU, 4.7× / 2.06× / 8.44 tok/s; **§V.C recommends
  async overlap** as future work (arXiv:2510.19979).
- TensorTEE 4.0× / 2.1% (arXiv:2407.08903); Goten 6.84× / 132.64×; AsymML 7.6×
  (arXiv:2110.01229); Slalom 4–11× / 6–20× (arXiv:1806.03287); SCX 36 ms / +85%
  (DOI 10.1145/3718958.3750509); KV-Cloak 15.41 ms/GB (arXiv:2508.09442).

**Unverified / caveated:**
- *Slalom*: full-text PDF (arXiv:1806.03287, OpenReview) returned unparseable binary to
  the fetchers; blinding/Freivalds mechanism and the 4–20× range are from the abstract +
  secondary pages, not the parsed PDF body. Confidence still high (well-corroborated).
- *TEESlice*: the paper does **not itself** discuss serialization/GPU stalls (medium
  confidence); it is cited here only as TSDP lineage + the 50×/10× overhead framing.
- *KV-Cloak async-OTP*: confirmed to be **future work, not implemented** — do not cite as
  a realized pipelining technique.
- *DarKnight*: only lightly covered in this pass; mechanism stated from secondary recall,
  verify before citing specifics.
- *OpenAlex citation-graph expansion*: intended but the pass leaned on WebSearch/WebFetch;
  a dedicated OpenAlex forward-citation sweep from Slalom/TwinShield could surface more.

**Contradicted (self-correction, 2026-06-08):** the original pass coded SecureInfer's
position on async overlap as a **negative result** ("yields little benefit … many small
secure transfers dominate") and reported `0 of 26 verified claims refuted`. Re-pulling
§V.C of the source (Nayan et al. 2025) shows the paper says the **opposite**: it
measures the 0.2–0.3 ms/tensor GPU→CPU vs <0.01 ms reverse asymmetry and explicitly
recommends "caching or overlapping secure decryption with computation can further
improve overall responsiveness." The "many small transfers dominate" framing was an
extrapolation from the per-tensor number that the source does not draw. §(b), §(c)(2),
and §(d)(3) above have been updated accordingly. **The adversarial-verify-default-refute
pass missed this** — the claim was either paraphrased into the corpus before the
3-vote check, or the verifiers coded a directional misreading as confirmed. The
`0/26 refuted` headline is therefore stale; the accurate figure is `1/26 refuted on
re-check`. Treat the verification credit as load-bearing only after a second pass that
re-pulls primary text for any claim used downstream.

**KV-Cloak threat-model correction (added 2026-06-10):** Part II §5.1 originally
characterized KV-Cloak's threat model as "weaker CPA" and its defense as "orthogonal
cover, fails `WEIGHTS-PUB` membership bar." Re-pulling §III.A of the source shows
KV-Cloak's stated adversary is gray-box weights-known (explicitly covering both
closed-source and open-source-model deployments — matches our `WEIGHTS-PUB`). The
algebraic-attack defense actually lives in the **magnitude-dominant additive mask `A`**
(sampled from `[3θ_K, 4θ_K]` to overwhelm the data signal), not in the orthogonal
`S, M`. The per-token norm-dictionary attack GELO uses against `O_v` has **not been
measured** against KV-Cloak, and the expected outcome under their `A`-magnitude design
is defeat, not failure. Part II §5.1 has been rewritten to reflect this; the cross-
validation work between KV-Cloak and `C_v` is now an open item in §11.

**Round-trip-count derivation for TwinShield (added 2026-06-09):** the paper gives
per-algorithm compute complexity (OutAttnMult `O(mnp)`, OutSoftMax `O(n)`, U-Verify
`O(n²)`) but **does not give a per-layer round-trip count tied to L, H, n**. Counting
RTs from the algorithm definitions (each protocol = one TEE-prep → one GPU-compute →
one TEE-recover cycle) yields:

```
RT_TwinShield(L, H, T) = (2H + 5) · L · (T + 1)
```

where `T` is generated tokens, `H` is attention heads, `L` is layers. The `2H` term is
the **only form the paper proves secure**: OutAttnMult is defined for a single per-head
`(Q, Kᵀ)` pair with one set of masks `(R_Q, R_K, a, b)` and one set of permutations
`(λ_Q, λ_K, λ_1, λ_2)`. The paper acknowledges heads compute independently (§V.A) but
does **not** extend the protocol to a batched-across-heads form and does **not** analyse
the security of shared-vs-per-head masks under batched dispatch. The "batched 7
RTs/layer" form (`7L(T+1)`) is engineering extrapolation; it presumes per-head
independent masks under one kernel launch — almost certainly what TwinShield's A40
prototype implements (the reported wall times are inconsistent with per-head dispatch
overhead), but **not formally proven** in the paper. Treat the per-head form as the
citable baseline; the batched form needs a security extension that does not yet exist
in the literature.

---

## CNN-era TEE↔GPU schemes (decoupled): techniques & transferability to transformers

The CNN-era TEE/untrusted-GPU schemes — pulled out of the findings table (§b) into this
self-contained section. They predate transformers but originated every technique the
hybrid-split family inherits.

**Schemes & ids (all share the *activation×static-weight linear → GPU, non-linear → TEE*
partition that creates the per-layer serial chain):**

| Scheme | Venue / id | Code | Reported |
|---|---|---|---|
| Slalom | ICLR'19 · arXiv:1806.03287 | ftramer/slalom | 4–11× (verif.+private) / 6–20× (verif.) vs TEE-only |
| DarKnight | MICRO'21 · arXiv:2207.00083 | — | 6.5× avg train speedup |
| Goten | AAAI'21 | goten-team/Goten | 6.84× vs CaffeScone; 132.64× vs Falcon (VGG-11) |
| AsymML / 3LegRace | PoPETs'22 · arXiv:2110.01229 | — | 7.6× vs TEE-only (training) |
| ShadowNet | S&P'23 · arXiv:2011.05905 | — | ~90% FLOPs offloaded |
| SOTER | ATC'22 | — | latency ↓ as offload ratio ↑ |
| TEESlice | S&P'24 · arXiv:2310.07152 | — | full-model security at >10× less than whole-model-in-TEE (≤50×) |

**The question:** which of their techniques are useful, and whether/how each ports to the
transformer TEE/untrusted-GPU split. Two structural facts govern the answer.

### Two structural facts that govern translation

1. **The non-linear cost ratio flips.** CNNs: non-linears (ReLU, maxpool) are cheap,
   elementwise, and a *negligible* fraction of compute — this is Slalom's explicit design
   premise ("nonlinear computations are a tiny fraction of total execution time"). For
   transformers this premise breaks: Softmax (exp + row-wise reduction, data-dependent),
   LayerNorm/RMSNorm (mean/variance reductions at *every* sublayer), and GELU are
   collectively expensive; TwinShield measures **>60 % of execution still in-TEE** after
   offloading the linear ops (arXiv:2507.03278). So the per-round TEE stall is
   proportionally *much worse* for transformers — the bottleneck the CNN schemes could
   afford to ignore is now dominant.

2. **Attention is matmul-of-two-activations.** Every CNN offloading trick —
   Slalom's precomputed blinding, ShadowNet's weight-transform, SOTER's associative
   morphing — masks an *activation × static-weight* product, exploiting that `W` is fixed
   so the unblinding term `R·W` (or the transformed weight) can be precomputed offline.
   Transformer **projections and FFN GEMMs are also activation × static-weight, so these
   tricks port directly.** But **attention (`QKᵀ`, `A·V`) is activation × activation** —
   no static weight to pre-mask. This is the single structural reason CNN offloading does
   not cleanly translate, and why transformer-specific mechanisms (TwinShield's
   OutAttnMult, PermLLM's permutations) had to be invented. **CNN techniques translate to
   the FFN/projection two-thirds of a transformer layer and stop at the attention core.**

Two further facts shape *usefulness* (not feasibility):

3. **Autoregressive decode multiplies the round-trip.** A CNN does one forward pass; LLM
   decode runs the per-layer serial chain *once per token*, thousands of times. Per-round
   overheads the CNN schemes tolerated (ShadowNet's "repeated GPU job setup", kernel
   launch) **compound catastrophically in decode** — but the same regular, repetitive
   structure is what makes decode *ideal* for the cross-request pipelining nobody has
   built.

4. **Decode is memory-bandwidth-bound; the GPU is already underutilized.** CNN convs are
   compute-bound, so offloading them to the GPU is a clear win. Decode GEMMs are tiny
   (batch×1×d) and memory-bound — offloading buys less, and the TEE round-trip eats a
   larger share. **CNN offload economics hold for prefill (compute-bound, big GEMMs),
   not decode.**

### Per-technique translation verdict

| CNN-era technique | Mechanism | Ports to transformers? | Useful for the serialization bottleneck? |
|---|---|---|---|
| **Slalom** precomputed blinding (1806.03287) | offline OTP masks `R`, GPU computes `(X+R)·W`, TEE subtracts precomputed `R·W` | **Partly** — to QKV/FFN/output GEMMs (static `W`); **breaks at attention** (no static weight) | Decouples mask-gen from the critical path (already adopted by GELO/ObfuscaTune) — but **orthogonal to the stall**: the GPU still waits for the TEE non-linear. Mild. |
| **ShadowNet** weight-transform (2011.05905) | linear-transform weights offline, restore in TEE | Same as Slalom — projections/FFN yes, attention no | Same: doesn't address overlap. Its own "GPU job setup" overhead **worsens** in decode. Low. |
| **SOTER** associative morphing + partition ratio (ATC'22) | push secret through associative ops, cancel later; tune offload fraction | **Partly** — matmul is associative (projections); Softmax/LayerNorm are **not** | A tunable knob, not an overlap mechanism; transformer security more fragile (static permutations break, see obfuscation-round2). Low. |
| **AsymML / 3LegRace** low-rank split (2110.01229) | sensitive low-rank part in TEE, residual on GPU | **Yes, conceptually** — transformer weights/activations are low-rank (LoRA); SecureInfer already keeps LoRA adapters in-enclave | **Shrinks the TEE-side residual** (the >60 % TwinShield cost) → smaller per-round stall. But the security argument is DP/training-time; porting to *exact-fidelity inference* is non-trivial. Reduces, doesn't overlap. Medium. |
| **Goten** round-optimal comms + overlap (AAAI'21) | minimize TEE↔GPU rounds; overlap memory-swap/communication; dynamic quantization | **Yes** — fuse a layer's offloads into fewer rounds; overlap PCIe transfer with TEE compute | The communication/compute **overlap** idea is a real piece of the missing pipelining. Useful, but Goten did it for training memory-swap, not decode latency. Medium–high. |
| **DarKnight** coded-batch + pipeline (2207.00083) | encode **K inputs → K+M coded queries** (`x̄(i)=Σ αⱼ,ᵢ x(j) + α r`), GPU does `⟨W,x̄(i)⟩`, TEE decodes by `A⁻¹` and does non-linears; **"next virtual batch is encoded under the shadow of GPUs execution time"** | **Yes — best fit.** Batch = K **independent** units → in decode, K **concurrent requests** (composes with vLLM continuous batching). Attention still excluded (static-`W` offload). | **The most useful translation.** It is the only CNN-era scheme that already (a) **batches the offload** to amortize the per-transfer crypto cost SecureInfer found dominant, *and* (b) **pipelines** TEE encode/decode against GPU compute. Directly blueprints the missing cross-request decode pipeline. **High.** |
| **TEESlice** (2310.07152) | partition-before-training; security analysis | n/a | Not a serialization technique. None. |

### Bottom line — what actually translates and is useful

**Most of the lineage is not "lineage only" for performance — three pieces translate:**

1. **DarKnight's coded batching + virtual-batch pipelining is the directly-useful import.**
   It is the CNN-era precedent for the two levers the rest of the field skips (batching to
   amortize the round-trip; overlapping TEE work with GPU compute). The transformer port:
   encode the linear-layer activations of **K concurrent decode requests** into K+M masked
   queries, offload as one GPU batch, decode + run per-request non-linears in the TEE
   **while the next request-batch's GPU GEMMs run** — i.e., request-level pipelining of the
   enclave↔GPU chain, composed with continuous batching. This squarely targets the
   asymmetric per-transfer cost SecureInfer documented (batching coalesces them) and
   exploits decode's repetitive structure.

2. **Goten's round-minimization + communication/compute overlap** is the complementary
   scheduling import: fuse per-layer offloads, and overlap the PCIe transfer of layer
   *i+1*'s operands with the TEE's non-linear on layer *i*.

3. **AsymML's low-rank asymmetric split** is the import for *shrinking* the stall: put a
   low-rank sensitive core in the TEE (LoRA-shaped) so the per-round TEE residual — the
   >60 % TwinShield measured — is smaller.

**What does not translate (the hard limit):** every static-weight masking trick
(Slalom/ShadowNet/SOTER) covers only the projection/FFN GEMMs and **stops at attention's
activation×activation matmuls**. Any transformer system using these for the FFN must pair
them with a transformer-native attention-offload mechanism (TwinShield OutAttnMult,
secret-sharing) — the CNN lineage simply has no analogue, because CNNs have no
activation×activation core. And Slalom's "non-linears are negligible" premise is false for
transformers, so CNN schemes that lean on it under-deliver on the very stall in question.

**Correction to §(c)/(d):** the earlier claim that cross-TEE-GPU pipelining is "entirely
absent" overstated it — **DarKnight pipelines virtual batches in the CNN/training era**
(verified, ar5iv 2207.00083). The accurate gap is narrower and sharper: *no system ports
DarKnight-style coded-batch pipelining to the **autoregressive transformer decode** loop
across concurrent requests.* That — not "pipelining has never been done" — is the open
opportunity, and the CNN lineage already supplies the blueprint.

---

# Part II — GELO design spike

Part I established the literature landscape. Part II asks: **given GELO's threat model
and cover invariant, which of these mechanisms actually help us, and which look promising
but break security?** The aim is to identify design directions worth a measurement spike,
not to commit to one. Each direction below carries an explicit threat-model verdict.

## 1. Threat model (load-bearing — every direction below is gated on this)

Canonical source: `docs/dev/logs/perm-attn-gpu-offload.md` §"Threat model" / `CLAUDE.md`.
Restated here because every optimization is judged against it.

- **`TEE-TRUST`** — the trust boundary is the SEV-SNP enclave. It holds plaintext
  activations, Q·K·V, the secret covers (masks, permutations, rotations) and the noise
  RNG, and performs all cover/un-cover and merge.
- **`GPU-ADV`** — the VFIO-passed GPU is **fully adversarial**. It sees every byte in
  VRAM, every intermediate it chooses to compute (fused or un-fused), dispatch shapes,
  timings, and its own per-step VRAM write locations. A fused kernel is a *performance*
  boundary, never a confidentiality one.
- **`WEIGHTS-PUB`** (conservative default, 2026-05-29) — the adversary knows the model
  weights and the embedding/unembedding tables (we serve open Qwen3). Every
  rotation-/permutation-*invariant* quantity (norm, Gram) is thus a **known bilinear
  form of the secret activations** — an algebraic anchor for reconstruction. The
  negation `WEIGHTS-BLIND` is the optimistic case. Where a result depends on this axis,
  **report the `WEIGHTS-PUB` number as the bar.**
- **`NO-PLAINTEXT`** — the adversary never possesses plaintext activations or tokens.
  This is the definition of the secret, not an attacker capability.

**Secret:** user activations at every layer, Q·K·V, KV-cache contents, prompt/tokens.
**Public:** weights, embedding tables.

**Cover invariant (load-bearing).** **No orthogonal-or-permutation cover, however
refreshed, hides token membership under `WEIGHTS-PUB`** — only a non-orthogonal,
per-session, activation-space cover does. Any new offload site that exposes an
activation to the GPU under only an orthogonal or permutation cover **must clear the
`WEIGHTS-PUB` membership gate** (the norm-dictionary attack measured 1.000 top-1
against orthogonal `O_v`) before it ships.

**One further constraint.** "Reducing round-trips by giving the cloud a usable
KV-cache path" (SCX's lever — see §5) **changes which intermediates the GPU can
independently compute on**. That is a threat-model change, not a free perf knob. Each
proposed optimization below carries a one-line **threat-model verdict**.

## 2. The GELO bottleneck restated in our numbers

From `gpu-offloaded-attention-with-value-cover.md` §1 and `gelo-llm-perf-chronicle*.md`
(B=8, n=2048, Qwen3-4B, dGPU/RTX 5090):

| stage | bucket | wall | share |
|---|---|--:|---|
| prefill | GPU matmul (QKV, FFN, out-proj, lm-head) | ~85 s | ~42% |
| prefill | TEE softmax + RMSNorm + mask apply | ~75 s | ~37% |
| prefill | TEE attention (now offload-capable under `C_v`) | 14.0 s (offloaded) / 44.1 s (in-TEE) | ↓ |
| decode | per-step TEE non-linear chain | ~455 ms / step / 36 layers | dominates |
| decode | offloaded attention | 2.8 s for K=32 steps | small |

The per-layer chain at each decode step is:

```
RMSNorm_pre → QKV_proj(GPU) → attention(GPU, under C_v) →
  out_proj(GPU) → residual+RMSNorm_post(TEE) → FFN_up·gate(GPU) → SiLU(TEE) →
  FFN_down(GPU) → residual(TEE) → next layer
```

That is **5 GPU round-trips per layer** at decode (QKV, attention, out_proj, FFN_up+gate
fused, FFN_down) × 36 layers × K decode steps. For prefill the same chain runs once on
the whole prompt at sequence-length n.

The structural fact: **every TEE-side non-linear creates one round-trip dependency
between consecutive GPU GEMMs**. RMSNorm is the most frequent (twice per layer); SiLU is
once per layer; softmax was once per layer until attention moved to GPU under `C_v`. So
the softmax-on-GPU win we already shipped is exactly the round-trip-reduction lever this
spike generalises.

## 3. Round-trip taxonomy — counting RTs per scheme, per layer

The prior recollection was that "TwinShield requires the same number of round trips."
The correction is **stronger than that**: TwinShield *increases* the per-layer RT count
vs the additive-outsource (Slalom-style) baseline — but each round-trip carries more
work, so wall-time falls.

The 87% offload number is **share of FLOPs offloaded**, not RT-count reduction. These
are orthogonal axes. Reducing RTs is **mostly unexplored**.

| Scheme | RTs / decoder layer (attention path) | What each RT carries | Net wall lever |
|---|--:|---|---|
| **TEE-only** | 0 | — | none |
| **Additive-outsource** (Slalom on Transformer) | ~1–2 | QKV proj, out-proj. Attention QKᵀ/softmax/A·V stay in TEE. | partial |
| **TwinShield** (per-head form, paper-proven) | **2H + 5** per layer (full forward = `(2H + 5)·L·(T+1)`) | Each RT is bigger; non-linears on GPU. | **higher FLOP offload, more RTs** |
| **TwinShield** (batched-H form, paper-unspecified) | 7 per layer (full forward = `7L(T+1)`) | Engineering extrapolation; needs cross-head security argument the paper does not give. | — |
| **GELO (current, decode)** | **2** (QKV+attention fused, out-proj) for attention block, +2 for FFN | Attention runs on GPU under `C_v`+π — softmax stays GPU-side via permutation-equivariance, no return RT for softmax. | **fewer RTs than TwinShield**, equal FLOP offload, stronger cover. |
| **GELO (current, prefill)** | 3 (QKV, attention, out-proj) per attention block, +2 for FFN | Attention under `C_v` + `O_qk` (no permutation; causal mask forbids it). | softmax still rides GPU via the orthogonal score-cancellation, not an OutSoftMax-style additive mask. |

**Surprise finding from this re-count.** GELO is already strictly better than TwinShield
on RT count for attention because permutation-equivariance kills the softmax RT entirely
(decode) and feature-rotation does the same (prefill). The remaining structural RTs are
between attention block and FFN — gated by RMSNorm and SiLU, both element-wise
non-linears that currently stay in TEE.

**This reframes the spike.** The *next* lever is the RMSNorm/SiLU round-trip pair
(twice per layer for RMSNorm, once for SiLU), which together account for the per-layer
TEE bucket that gates the whole serial chain. See §10.

## 4. TwinShield — what to import, what's already done, what doesn't help

| Mechanism | Imports to GELO? | Threat-model verdict | Notes |
|---|---|---|---|
| **OutAttnMult** (QKᵀ via permuted block matrices `[Q+R_Q; aR_Q]·[K+R_K; bR_K]ᵀ`) | **Already subsumed.** GELO's `C_v`-covered attention does QKᵀ on GPU directly; we don't need the secret-shared block dance. | **✓ stronger** — `C_v` is non-orthogonal and per-session, TwinShield's `R_Q`/`R_K` are additive masks on Q,K (i.e. the additive cover *and the orthogonal permutation* of two block matrices). Under `WEIGHTS-PUB` the value-side leaks via norm dictionary unless paired with `C_v` — TwinShield's value path is additive only. | TwinShield's threat model is weaker (no public-weight algebraic attack analysed). |
| **OutSoftMax** (`x' = x − r` → GPU `e^(x')` → TEE recovers `e^x = e^(x')·e^r`) | **Worth investigating for prefill RMSNorm/SiLU**, not for softmax (softmax already moved to GPU via permutation/rotation equivariance). | `x'` is the secret with an additive mask `r` (per-session, per-element). The GPU sees `e^(x − r)` — under `WEIGHTS-PUB` this is **content-blind** because `r` is information-theoretically secret; no invariant matches a public weight matrix. **✓ provisionally clears** — needs measured re-check at production shapes. | This is the lever that could move SiLU (and possibly RMSNorm's `1/√(mean(x²))` step) onto the GPU. See §10. |
| **U-Verify** (integrity for both linear and non-linear via additive-mask hash) | Out of scope for now — GELO trusts SEV-SNP attestation + the C_v key + does not currently verify GPU integrity beyond shape checks. | n/a | Could matter when we add verifiability post-confidentiality. |
| **Permutation embedding of `[Q+R_Q; aR_Q]` block matrices** | **No.** The permutation hides which row is the mask vs the data — but only under `WEIGHTS-BLIND`. Under `WEIGHTS-PUB` the masked block `aR_Q` has predictable norm-distribution from `R_Q ~ uniform`, separable from `(Q+R_Q)` rows by magnitude — block identification is statistical. | **× fails our bar.** TwinShield's threat model does not consider it. | Caveat: GELO need not adopt this — `C_v` makes the row mixture covariant, not just permuted. |

**Net.** TwinShield's RT-reduction story is **not real** (it adds RTs — see §(e)
round-trip derivation in Part I). Its FLOP-offload story is real but GELO with `C_v`
already offloads the same FLOPs with a stronger cover. The one **genuinely transferable
primitive** is the `e^(x+r) → e^x` additive-mask trick of OutSoftMax, adapted to *other*
element-wise non-linears (SiLU, the `exp` inside RMSNorm's reciprocal-sqrt fast-path) —
see §10 direction **B**.

## 5. KV-cache schemes — Shadow-in-the-Cache, KV-Cloak, SCX

Three schemes specifically about KV-cache protection. Each one's adoption is gated on
the cover invariant.

### 5.1 Shadow-in-the-Cache / KV-Cloak (Luo et al., arXiv:2508.09442)

**Mechanism.** Three input-reconstruction attacks on plaintext KV-cache (Inversion via
known `W_K`, Collision via local-model rehearsal, Injection via prompt-driven echo).
Defense: `K' = S·P̂·(K + A)·M`, where `S∈O(b)`, `M∈O(d)` are orthogonal secret
matrices, `P̂` is a block-fresh one-time permutation, and `A` is an additive mask whose
entries are sampled uniformly from `[3θ_K, 4θ_K]` — i.e. **magnitudes designed to
dominate the data signal `K`** (`θ_K` is the maximum absolute value observed in K
during calibration).

**Threat model (verbatim from §III.A).** KV-Cloak's stated adversary is **gray-box,
weights-known**:

> *"We assume the adversary can obtain both the KV-cache and the model weights, but
> does not observe the ephemeral runtime activations within the GPU registers."*

The paper explicitly addresses both deployment cases: closed-source services (CSP owns
the weights) and **open-source models (weights public via licensing or
fingerprinting)**. This is the same axis as our `WEIGHTS-PUB`. CPA in the paper is an
*additional* capability (the adversary can feed known inputs to profile threshold
distributions), layered on top of the weights-known baseline — not weaker than it.

**Threat-model verdict for GELO: ✓ designs for weights-known, but untested against the
specific per-token norm-dictionary attack GELO uses. Cross-validation is open empirical
work, not assumed failure.** (This corrects an earlier characterization in this doc;
see decomposition below.)

**What the defense actually relies on — decomposition.** The earlier framing "swap `M`
for non-orthogonal `C_v` and KV-Cloak becomes `WEIGHTS-PUB`-safe" misread which factor
is doing the load-bearing work. The orthogonal `M` does **not** defeat the
norm-dictionary attack — it provides covariance so attention scores survive. The
algebraic-attack defense lives in `A`:

| Factor | Property | What it actually defeats |
|---|---|---|
| `M ∈ O(d)` (right-mult, orthogonal) | preserves inner products → `QKᵀ` stays correct | **Covariance** for downstream multiplication. Does nothing for membership recovery: `‖row·M‖ = ‖row‖`. |
| `S ∈ O(b)` (left-mult, orthogonal) | mixes rows within a block | Same — covariant, not anti-membership. |
| `P̂` (one-time permutation, block-fresh) | shuffles row order | Defeats positional inference; multiset preserved. |
| **`A` (additive, `‖A‖ ≫ ‖K‖`)** | dominates row norms | **The algebraic-attack defense.** Per-row norms of `K + A` are governed by `‖A‖`, not `‖K‖`, so any public-weight norm-dictionary loses its anchor. |

So KV-Cloak's solution to weights-known adversaries is **magnitude-dominant additive
noise**, not orthogonal-multiplicative cover. The orthogonal factors only ensure the
math remains computable end-to-end.

**Norm-dictionary attack against KV-Cloak's actual design.** Trace through for V:

```
‖V'_i‖ = ‖S·P̂·(V + A_v)·M · e_πi‖
      = ‖(V + A_v)_πi · M‖           (S, M orthogonal — preserve norm)
      = ‖(V + A_v)_πi‖
      ≈ ‖A_v · e_πi‖                 (A's magnitude dominates V's by design)
```

The anchor for the dictionary — the per-token quantity `‖V(t)‖` from layer-0 weights —
is overwritten by `A`'s magnitude. Whether tokens still leak depends on whether `A` is
**per-block random** (norms collapse to a uniform distribution divorced from `V`) or
**structurally fixed** at known positions (adversary could subtract the structural
component). The paper's description is ambiguous: "Magnitude-based Positional
Embedding" suggests fixed positions for the beacons, but the sampling spec
(`uniformly from [3θ_K, 4θ_K]`) suggests per-block random magnitudes within those
positions. The most consistent reading is "per-block random within fixed positions,"
which probably defeats the simple norm-dictionary attack — **but the paper does not
measure this**. The three attacks it does measure (Inversion via `W_K^+`, Collision via
candidate rehearsal, Injection via prompt) are different attacks; the per-token
norm-dictionary is a fourth attack class.

**How this compares to GELO's `C_v`.** Two different mechanisms solving the same
weights-known problem:

| Property | KV-Cloak | GELO `C_v` |
|---|---|---|
| Multiplicative cover | orthogonal `M` (covariance only) | **non-orthogonal `C_v`** (covariance + anti-membership) |
| Additive cover | **magnitude-dominant `A`** (anti-membership) | small σ on Q/K only; no V additive |
| Permutation | per-block `P̂` (decode) | per-block `π` (decode); none on prefill |
| Anti-membership proof structure | quantitative ("`‖A‖` dominates `‖K‖`") — DP-flavoured | information-theoretic per-session (`C_v` uniform in its parameter space, modulo κ) |
| Tested against per-token value-norm dictionary | **No** | Yes — measured 0.000 top-1 at κ=6 |
| Tested against `W_K^+` inversion | Yes | (subsumed — `C_v` makes `W_V^+ V'` non-invertible) |

So they're **parallel designs targeting the same threat axis with different
mechanisms**, neither tested against the other's primary attack. The honest verdict is
not "KV-Cloak is broken" or "GELO has shipped a hardened KV-Cloak" — it's
"cross-validation is owed in both directions." Open empirical work:

1. Run GELO's per-token value-norm-dictionary attack against a KV-Cloak implementation
   on Qwen3-4B. Expected: defeated by `A`-magnitude (high confidence under "A
   per-block random") or reduced to k-anonymity (under "A structurally fixed").
2. Run KV-Cloak's Collision Attack against GELO's `C_v` cover. Expected: defeated by
   `C_v`'s non-orthogonal mixing (the cache's algebraic structure no longer matches a
   candidate's), but worth measuring.

**What to import.** KV-Cloak's *form* — multiply-on-both-sides + additive mask + block
permutation — is structurally close to GELO's existing decode cover (`C_v` + π +
tail-in-TEE) and serves as parallel-design corroboration that the multiply-additive
combination is the right shape for KV-cache protection. The **single
forward-looking lever** named as future work (not implemented): *"latency can be masked
through the asynchronous generation of One-Time Pad (OTP) matrices, decoupling
security operations from the critical inference path."* This is the same lever PipeLLM
realises for AES (§6) and SecureInfer recommends in §V.C — see direction **A** in §10.

### 5.2 SCX (Yuan et al., SIGCOMM '25)

**Mechanism.** *Stateless* KV-cache encoding via user-controlled key: client encodes
the input embedding with a user secret (token permutation + redundant embeddings +
Laplace noise), uploads to the cloud, the cloud runs the model on encoded
representations. KV-cache in intermediate layers is *recoverable by the cloud* because
the user shares per-layer intermediate keys; only the first and last layer's KV-cache
stays user-key-locked. This reduces communication from `O(T·L)` to `O(T + L)` between
user device and cloud.

**Threat-model verdict for GELO: ✗ different threat model entirely; intermediate-layer
keys break our cover invariant.**

SCX's threat model is **client/cloud split** (Apple PCC family) — the cloud cannot
independently complete inference without the user device's keys, but it *does* see all
intermediate KV-cache in a form it can compute on (the cloud has the intermediate-layer
keys). For GELO the "client" is the TEE and the "cloud" is the *same-host* untrusted
GPU across PCIe. The boundary is different; the secrets that need to stay hidden are
stricter (activations at every layer, not just first/last).

If we were to port SCX naively — "share intermediate-layer covers with the GPU" — that
literally hands the GPU layer-by-layer activations under a known transform. The
`WEIGHTS-PUB` adversary then has every layer's activations algebraically. **Hard
reject.**

**What to import anyway.** Two structural ideas, both **threat-model-orthogonal**:

1. **Redundant-token padding** (decoy-row k-anonymity). Already named in the C_v doc §3
   as a *fallback shape*, rejected as the primary lever because it doesn't structurally
   close the leak (priors re-rank decoys). SCX's empirical result is interesting: 100
   redundant tokens drop GPT-4o sequence-recovery cosine similarity 0.87 → 0.46. The
   mechanism only blocks *LLM-assisted-recovery*, not the algebraic norm-dictionary
   attack. **Out of scope** for the algebraic threat model.
2. **KV-cache *delta* coalescing.** SCX's optimization is at the user/cloud boundary;
   the coalescing idea (move many small KV-cache touches as one big transfer) is
   independent of the protection scheme. GELO already does this — the resident-K/V
   design moves only the per-step delta. **Already shipped.**

### 5.3 Direct GELO KV-cache work

The companion `gpu-offloaded-attention-with-value-cover.md` already documents GELO's
KV-cache cover: per-block-fresh permutation `π` + σ-noise on Q/K + `C_v` on V +
tail-in-TEE. Relative to KV-Cloak (§5.1), this is a **different mechanism for the same
weights-known threat model**, with two genuine GELO-specific properties:

- **Anti-membership proof structure differs.** `C_v`'s defense is information-theoretic
  per-session (uniform in its parameter space modulo κ); KV-Cloak's is quantitative
  (`‖A‖` magnitude must dominate `‖K‖`). Different proof shapes; both have not been
  cross-validated against each other's primary attack.
- **Tail-in-TEE closes the per-step write-location side channel.** KV-Cloak does not
  have this — the GPU sees per-token write locations when the cover is applied
  block-at-a-time. For decode this is the per-step side channel direction A in §10
  doesn't address; the tail-in-TEE answer is structural, not scheduling.

Where KV-Cloak's design is **arguably ahead**:

- Its additive `A` is the natural defense for the layer-0 norm-dictionary anchor.
  GELO's `C_v` solves the same problem with a different mechanism, but GELO has no
  *additive* cover on V — relying entirely on the non-orthogonal multiplicative cover
  and the κ margin. Composing both (a GELO `C_v` + KV-Cloak-style `A_v`) is **a real
  defense-in-depth option** that has not been measured.

**Open work on the KV-cache side** (vs the literature):
- The `√N` HNM bound on permutation refresh cadence is still pending its driver (C_v
  doc §6).
- Long-context (2k–16k) covariance-alignment re-check is still owed.
- Cross-session accumulation against `C_v C_vᵀ` is provably safe under per-session
  refresh, but **per-session-but-multi-request** (continuous batching) needs explicit
  analysis.

## 6. PipeLLM — what ports, what doesn't

**Mechanism (Tan et al., ASPLOS '25, arXiv:2411.03357).** Speculative pipelined
*encryption*: on an H100 in CC mode, predict the next memory swap from LLM serving
patterns (FlexGen's layer-by-layer order, vLLM's FIFO/LIFO KV swap policy), pre-encrypt
the predicted ciphertext in parallel with current GPU compute, validate on actual
request via revoked-write-permission + page-fault detection, relinquish-on-mispredict
with IV-arithmetic NOP padding to avoid re-encrypting the whole pipeline.

**Threat-model verdict for GELO: ✓ conceptually clean, but the *encryption* PipeLLM
overlaps is not the GELO bottleneck.**

PipeLLM targets the **AES-GCM PCIe encryption** that H100 CC inserts between
confidential CPU and confidential GPU memory. GELO uses a **commodity untrusted GPU** —
there is no hardware-mandated PCIe AES; the PCIe transfer is plaintext over the masked
activations. So PipeLLM's specific overlap (encryption || GPU compute) doesn't apply
directly.

**What ports — and is the most useful single import from this paper:** the
**predict-ahead + overlap critical-path security work with GPU compute** pattern,
applied to GELO's *mask-generation*. Today, when the TEE constructs the next layer's
`C_v`, the per-call orthogonal activation mask, and the σ-noise tensors, this work runs
**synchronously** between GPU calls — it sits on the critical path. PipeLLM's lesson:
do it **speculatively, in parallel with the previous layer's GPU compute**. This is
exactly KV-Cloak's named-but-unimplemented "async OTP generation" future-work and
SecureInfer's §V.C recommendation in one — three papers in the literature now point at
this single lever.

**Threat-model check.** Pre-generating the next layer's mask is fully TEE-side, uses
only TEE-secret RNG, and writes results to TEE memory. No new GPU exposure. **✓ clears
trivially.** See direction **A** in §10.

## 7. CNN-era lineage — see Part I

The CNN-era analysis (Slalom / DarKnight / Goten / AsymML / ShadowNet / SOTER /
TEESlice) lives in detail under "CNN-era TEE↔GPU schemes" in Part I. For the GELO
spike, the three pieces that translate are unchanged:

1. **DarKnight's coded batching + virtual-batch pipelining** → cross-request decode
   pipelining (direction **C** in §10).
2. **Goten's round-minimization + comm/compute overlap** → operator-reordering at the
   layer boundary (direction **D**).
3. **AsymML's low-rank asymmetric split** → shrinks per-round TEE residual; lower
   priority.

The threat-model gate: each request's `C_v`, `π`, and σ are independent — cross-request
mixing in one GPU call is fine because each row's secret is independent (no
cross-correlation surface). **✓ clears** for direction C. The static-weight masking
tricks (Slalom/ShadowNet/SOTER) are already subsumed by GELO's per-call orthogonal
activation mask.

## 8. Pipelining-technique granularity — single request, single batch, or concurrent?

The PipeLLM and CNN-lineage techniques target serialization, but **at different
granularities of "work in flight."** Because the GELO serial chain (`GPU_GEMM(i) →
TEE_unmask → TEE_nonlinear → TEE_remask → GPU_GEMM(i+1)`) is a true data dependency, no
pipelining technique can break it *within* a single forward pass. The only structural
escape is to find **independent work to occupy the otherwise-idle device while the
dependency-bound work runs**. The granularity at which that independence appears
determines which technique helps.

### 8.1 What the chain forces

For one request progressing through one layer:

- Layer i's GPU output `Y_masked` cannot be touched outside the enclave.
- The TEE must unwrap it, run the non-linear (softmax / RMSNorm / SiLU), and remask
  before layer i+1's GPU GEMM can start.
- Each device idles while the other works on the same request.

Pipelining can only help by (a) moving *non-dependent* TEE work off the critical path
or (b) putting *another request's* work into the idle slot.

### 8.2 Granularity → which technique applies

| Granularity | What "other work" exists? | PipeLLM async mask-prep | Goten comm-compute overlap | DarKnight cross-batch pipelining |
|---|---|:-:|:-:|:-:|
| **Single request (B=1)** | Only TEE-side prep work (mask generation, shield-fill) that depends on TEE-secret RNG, not on data | ✓ helpful — hides mask-gen behind GPU compute | ✓ marginal — hides PCIe transfer behind TEE compute | ✗ does not apply (no second forward pass to overlap) |
| **Single sync batch (B>1, all requests march in lockstep)** | Same as above. The batch dimension widens each call but doesn't create independent layer outputs — all B rows hit layer-i non-linear together. | ✓ same lift fraction as B=1 | ✓ same | ✗ still does not apply (wider batch ≠ multiple independent forward passes) |
| **Concurrent requests, different stages** (request A at layer 17, request B at layer 23, request C at layer 12) | The other in-flight requests at *different layer indices*. TEE can do A's softmax while GPU does B's FFN — both devices busy continuously. | ✓ same | ✓ + can fuse PCIe transfers across requests | **✓ — this is the lever** |

### 8.3 Why DarKnight is the structural break

DarKnight's coded-batch trick presumes K *independent* forward passes. Within one
request's forward pass, layer-i and layer-(i+1) outputs are not independent — they have
the very dependency the chain enforces. So DarKnight cannot coalesce them; the K it
encodes must be K different samples.

In CNN context the K independent samples were one training batch fanned through the
model. The pipelining gain came from "the next virtual batch is encoded under the
shadow of the GPU's execution time" — i.e., **batch N+1 enters the encoding pipeline
while batch N is still mid-flight on the GPU**. Even in DarKnight, the overlap is
*across batches*, not within one.

For transformer decode, "K independent forward passes" is naturally "K concurrent
requests, each at its own decode step." The serial chain still exists within each
request's layer; what changes is that the TEE and GPU **alternate which request they
work on** rather than alternating idle/busy on the same request.

### 8.4 The single-request critical path is bound by TEE non-linear compute

For one isolated request, no scheduling trick removes the per-layer TEE bucket. The
only single-request levers are:

- **PipeLLM-style async mask-prep** — hides the *preparation* portion of the TEE
  bucket (the parts that depend only on RNG and weights, not on the GPU's return).
  Worth 10–20% of the bucket. Already direction A in §10.
- **Goten-style transfer overlap** — hides PCIe latency behind TEE compute. Small on
  our PCIe Gen5x16 path.
- **Moving non-linear compute onto the GPU under cover** (direction B) — *eliminates*
  TEE work rather than hiding it. SiLU and RMSNorm's `x²` step under additive cover
  are the candidates. This is the only structural lift the single-request critical
  path can take.
- **Reducing the TEE residual** (AsymML-style low-rank split) — shrinks the per-layer
  TEE bucket without changing where work runs. Lower priority but additive with the
  above.

### 8.5 Implementation-order bifurcation

- **For single-request latency**: only PipeLLM async-prep + Goten transfer overlap
  apply. Wall-time floor is set by `Σ_layer (TEE_nonlinear_i)` minus the prep portion.
  Direction B (move non-linear to GPU) is the only way below that floor.
- **For aggregate throughput across concurrent users**: DarKnight cross-request
  pipelining composes with PipeLLM async-prep + Goten fusion, and the GPU can be kept
  ~continuously busy. Per-request latency is unchanged, but tokens/sec scale with
  pipeline depth up to TEE memory limits.

This bifurcation should drive the implementation order: **direction A pays off in both
regimes; direction C only pays off if we commit to building a continuous-batching
scheduler.** If the deployment target is single power-user latency, B before C. If the
target is multi-user throughput, C before B.

## 9. Yu et al. — Dual Privacy / CMIF (different design point)

**Mechanism (Yu et al., 2025).** CPU-side TEE holds the embedding layer; subsequent
layers run on plain GPU; input tokens are sanitized via an optimized Report-Noisy-Max
(RNM) mechanism (replace sensitive tokens with semantically-similar substitutes under
DP). Once sanitized, the representation flows through GPU without further TEE
round-trips.

**Verdict: ✗ wrong design axis for GELO.** CMIF gives up exact-fidelity inference for
full GPU pipelining — privacy is DP-bounded at the token-substitution step, then plain
compute follows. GELO's `NO-PLAINTEXT` + `WEIGHTS-PUB` threat model rejects "the GPU
gets a usable representation post-sanitization" by definition: under public weights,
any usable representation is invertible. CMIF works because its threat model is weaker
(semi-honest cloud, no public-weight algebraic attack).

**Usable as a comparison baseline only.** When we write up GELO's trade-offs, CMIF is
the right citation for "what you can do if you accept DP-bounded inference" — a
different point on the privacy-performance curve, not a direction for us.

## 10. Design directions for GELO — ranked by lift, all gated on the threat model

Notation: **lift** = expected wall-time impact at the production shape; **risk** =
engineering + security-validation cost.

### A. Async / speculative mask generation (PipeLLM/KV-Cloak/SecureInfer lesson)

**What.** Pre-generate the next layer's `C_v` correction, the per-call orthogonal
mask, and the σ-noise tensors *in parallel with the current layer's GPU compute*.
Today this work runs synchronously between GPU calls (visible in profiles as
`gelo:mask_apply:*` and `shield:fill` buckets sitting on the critical path).

**Threat-model verdict: ✓ trivially clears.** Pure TEE-side compute on TEE-secret RNG;
no new GPU exposure.

**Lift.** Profiling already shows `gelo:mask_apply` + `shield:fill` are 10–20% of the
per-layer TEE bucket at decode; hiding them behind the GPU's compute claws back most
of that. Engineering: a TEE-side thread pool that runs one layer ahead.

**Risk.** Low. Validation: re-run the AloePri attack suite on `c2_default` to confirm
RNG swap to async doesn't change the distribution.

### B. Move SiLU and the RMSNorm tail-step to GPU via additive-mask cover

**What.** Use TwinShield's `e^(x+r)` insight on *other* element-wise non-linears:
- **SiLU(x) = x · σ(x).** With `σ(x) = 1/(1+e⁻ˣ)`, send `x + r` to GPU, GPU returns
  `σ(x + r)` and `x + r`; TEE recovers `σ(x) = σ(x + r) · (something involving e^r)` —
  *not directly invertible*, σ does not commute with additive shift. **Need a different
  cover.** Multiplicative blinding: send `c·x` (c a per-element TEE secret); SiLU is
  not multiplicatively homomorphic either. **Investigation needed.** A polynomial
  approximation of SiLU (cubic spline, used by some MPC LLM works) **does** linearise
  additively, but costs accuracy.
- **RMSNorm `1/√(mean(x²) + ε)`**: the `x²` step is masking-friendly (Slalom-style
  additive cover commutes with squaring up to a known correction); the reduction
  `mean(...)` is linear and additive-mask-friendly; the `1/√(...)` is the hard step
  (one scalar per row, cheap in TEE anyway). The cheapest win: offload the per-row
  `Σ x²` to GPU under additive mask and keep the `1/√(.)` in TEE — cuts the RMSNorm
  bucket roughly in half.

**Threat-model verdict: ⚠ provisional ✓, needs measured re-check.** Additive mask `r`
per element is information-theoretically blind under `NO-PLAINTEXT`. Under
`WEIGHTS-PUB` the GPU sees `x + r` where `r` is TEE-secret per session; no invariant
of `x + r` (norm, Gram) gives a clean dictionary into the embedding/weight tables
without first recovering `r`. **Need to run the membership-attack suite on
`c2_default + RMSNorm-x²-on-GPU`** before committing.

**Lift.** RMSNorm is **twice per layer** and is the largest remaining non-attention
TEE bucket. Cutting it in half is a real win.

**Risk.** Medium. The polynomial-SiLU path costs accuracy and would need a HumanEval
re-run. The `x²` offload is cheap to spike and threat-model-clean.

### C. Cross-request decode pipelining (DarKnight's transformer port)

**What.** With K concurrent decode requests, batch their per-layer QKV/FFN GEMMs into
a single GPU call. The TEE's softmax/RMSNorm/SiLU on request *batch i*'s outputs runs
**in parallel** with the GPU's GEMMs for request *batch i+1*. This composes with
continuous batching (vLLM-style).

**Threat-model verdict: ✓ clears.** Each request carries independent `C_v`, `π`, σ;
mixing in one GPU call is fine because each row's secret is independent (no
cross-correlation surface).

**Lift.** Big — this is the lever the entire literature has skipped for transformer
decode (§(d) above). At K=4–8 concurrent decode streams, GPU should run near its
compute peak, hiding the TEE non-linear stage entirely. **Only helps aggregate
throughput, not single-request latency** (see §8).

**Risk.** Medium-high. Engineering: requires a proper request scheduler in the TEE.
Validation: AloePri-style attack on a **mixed-request batch** to confirm cross-row
attacks don't exist.

### D. Operator-reordering at the layer boundary (TwinShield-style fusion, stricter)

**What.** Today GELO has `matmul_many` fusing QKV and gate∥up. Extend the fusion
across the attention/FFN boundary: pre-stage out-proj-out into a single GPU GEMM that
includes the post-attention residual-add as part of the matmul (residual is a known
constant on the GPU side under our cover). Saves one round-trip per layer (the
post-attention residual).

**Threat-model verdict: ⚠ depends on cover composition.** The residual stream is
*covered* on the GPU side under the per-call orthogonal mask; combining the
residual-add into the matmul is fine if the residual's cover composes with the
matmul's input cover. Needs the algebra written out.

**Lift.** Small (one RT per layer × 36 layers × K steps); measurable but not headline.

**Risk.** Medium. The composition-of-covers algebra is the bulk of the cost.

### E. Persistent KV-cache delta transfer (already shipped; not a new direction)

Listed here for completeness — GELO's resident K/V design already does this. SCX's
"small delta" lever is structurally identical and confers no new lift.

### Ranking

| direction | lift | risk | threat-model | regime | recommend |
|---|---|---|---|---|---|
| **A**. Async mask generation | medium | **low** | ✓ trivial | single + concurrent | **build first** — best lift/risk, pays in both regimes |
| **C**. Cross-request decode pipelining | **high** | medium-high | ✓ (needs mixed-batch attack run) | concurrent only | **biggest open lever** — second if target is multi-user |
| **B**. `x²`-on-GPU for RMSNorm tail | medium | medium | ⚠ (needs membership-attack re-run) | single + concurrent | **second if target is single-user** — only way below single-request floor |
| **D**. Cross-boundary fusion | small | medium | ⚠ (cover-composition algebra) | single + concurrent | low priority |
| **E**. KV-cache delta | n/a | n/a | n/a | n/a | already done |

## 11. Open questions and gaps before any of this ships

- **Membership attack at production context (2k–16k).** Existing C_v gate was at
  modest context lengths. Any direction that adds GPU-visible intermediates needs the
  gate re-run there.
- **Mixed-batch attack surface for direction C.** No literature analyses cross-request
  attacks when independent `C_v`s are batched in one GPU call. We'd be first.
- **TwinShield OutSoftMax under `WEIGHTS-PUB`.** The paper does not run a
  public-weight norm-dictionary attack on `e^(x − r)`. Before importing the
  additive-mask trick to SiLU or RMSNorm's `x²` step, we owe ourselves that gate.
- **Continuous batching design.** Direction C presumes a request scheduler GELO does
  not yet have. Building this is a non-trivial engineering project; the security
  analysis is the smaller half.
- **Decode profile re-check.** The 5-RT-per-layer count in §2 should be reconfirmed
  against current `gelo_llm_prefill_decode_breakdown` output (some buckets may have
  moved).
- **TwinShield batched-H security argument.** The `7L(T+1)` form of TwinShield's RT
  count needs a multi-head batched-dispatch security argument that does not exist in
  the paper. If anyone cites TwinShield's per-layer RT as 7 (not `2H + 5`), they are
  importing an unproven extension.
- **KV-Cloak ↔ GELO cross-validation (added 2026-06-10).** Both schemes target the same
  weights-known KV-cache threat axis with different mechanisms (magnitude-dominant
  additive `A` vs non-orthogonal multiplicative `C_v`). Neither has been tested against
  the other's primary attack. Two specific measurements owed:
  1. Run GELO's per-token value-norm-dictionary attack against a KV-Cloak
     implementation on Qwen3-4B. Likely outcome: defeated by `A`-magnitude (high
     confidence under "A per-block random") or reduced to k-anonymity (under "A
     structurally fixed at known positions").
  2. Run KV-Cloak's Collision Attack against GELO's `C_v` cover. Likely outcome:
     defeated by `C_v`'s non-orthogonal mixing (cache's algebraic structure no longer
     matches a candidate's), but worth measuring.
- **`C_v + A_v` defense-in-depth.** GELO has no additive cover on V; composing `C_v`
  with a KV-Cloak-style `A_v` is a real option worth measuring before claiming `C_v`
  alone is sufficient at production context (2k–16k).

---

## Pointers for the SoK / writeup

- The bottleneck and its under-addressed status is a clean **open-problem / future-work**
  note for \S7 or \S9 (synthesis): "hybrid-split schemes are architecturally sequential;
  cross-request pipelining of the enclave↔GPU non-linear chain is unexplored for
  transformer decode."
- The **hardware-vs-software fork** (H100 CC / Ascend-CC / TensorTEE remove the round-trip
  vs commodity-GPU obfuscation hybrids that cannot) reinforces the \S7 Tier-2 framing:
  the serialization cost is the price of staying on commodity untrusted GPUs.
- The **GELO-specific framing** (Part II): GELO's `C_v` cover already places it ahead of
  TwinShield on attention round-trips (2 vs `2H + 5` per layer for the attention block);
  the remaining bottleneck is RMSNorm/SiLU, not attention. The unexplored levers ranked
  by lift/risk are direction A (async mask-prep) → C (cross-request pipelining) or B
  (RMSNorm-x² offload), depending on whether the deployment target is throughput or
  latency.
- Primary sources (transformer-era, all in EdgeQuake):
  - TwinShield: Xue et al., arXiv:2507.03278.
  - Shadow-in-the-Cache / KV-Cloak: Luo et al., arXiv:2508.09442.
  - SCX: Yuan et al., SIGCOMM '25, DOI 10.1145/3718958.3750509 (code: yuanmu97/scx).
  - PipeLLM: Tan et al., ASPLOS '25, arXiv:2411.03357 (code: SJTU-IPADS/PipeLLM).
  - SecureInfer: Nayan et al., 2025, arXiv:2510.19979.
  - Dual-Privacy / CMIF: Yu et al., 2025.
  - PipeLLM (2411.03357), TensorTEE (2407.08903), Ascend-CC (2407.11888), H100 CC
    benchmark (2409.03992). None are in `refs.bib` yet.
- CNN lineage primary sources (not all in EdgeQuake; web sources):
  - Slalom: arXiv:1806.03287 (code: ftramer/slalom).
  - DarKnight: arXiv:2207.00083.
  - Goten: AAAI '21 (code: goten-team/Goten).
  - AsymML / 3LegRace: arXiv:2110.01229.
  - ShadowNet: arXiv:2011.05905. SOTER: ATC '22. TEESlice: arXiv:2310.07152.
- GELO companion docs:
  - Threat model canonical source: `CLAUDE.md` §"Threat model",
    `docs/dev/logs/perm-attn-gpu-offload.md`.
  - Current cover design:
    `docs/dev/prototype/gpu-offloaded-attention-with-value-cover.md`.
