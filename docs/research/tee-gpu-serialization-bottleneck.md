---
type: research
status: current
created: 2026-06-05
updated: 2026-06-05
tags: [hybrid-split, tee-gpu, serialization, gpu-utilization, pipelining]
companion: [scheme-categorization, obfuscation-round2-2026-06-05]
---

# The TEE↔GPU serialization bottleneck in hybrid private inference

Research round on the structural bottleneck of hybrid TEE+obfuscation inference: the
GPU stalls waiting for the TEE to return non-linear results, because each layer's GPU
matmul depends on the previous layer's TEE-side non-linearity. Question: *which papers,
schemes, and repositories specifically target this serialization, and with what
techniques?*

Sources: WebSearch + WebFetch on primary arXiv/USENIX/ASPLOS/IEEE sources, GitHub.
Method note: synthesized from a fan-out search+fetch+verify pass — 24 sources had their
claims extracted and adversarially verified (3-vote, "default to refuted"); **0 of 26
verified claims were refuted** (25 high-confidence, 1 medium). Per-claim caveats are in
§(e). This document does **not** modify the manuscript.

---

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
| **KV-Cloak** (Shadow-in-the-Cache) | 2025 · arXiv:2508.09442 | (8) **aspirational** | Async OTP-matrix generation to decouple masking from critical path — **stated as future work, not implemented**. Software obfuscation, **not** TEE+GPU hybrid. | 15.41 ms/GB (0.45% prefill) vs AES 3020.9 | — | Names the async-OTP lever; doesn't realize it |
| **SecureInfer** | 2025 · arXiv:2510.19979 | (4); **negative result** | SGX + GPU; privacy-critical tensors in enclave, linear matmul to GPU under OTP/XOR. **Explicitly finds async overlap yields little benefit — securing many small transfers dominates.** Batching = future work. | 0.2–0.3 ms/tensor GPU→CPU; 4.7× vs TEE-only; 2.06× vs GPU-only; 8.44 tok/s | — | **Key counterpoint to naive pipelining** |
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

**(2) Async / speculative — only for encryption (PipeLLM), and once as a negative
result.** PipeLLM's speculative pipelined encryption is the only mature speculative
overlap, but it targets the *encryption* step on a confidential GPU, not the
non-linear-in-CPU-enclave dependency. SecureInfer tried async overlap of the
enclave↔GPU transfers and reports it **yields little benefit because many small secure
transfers dominate** — an important caution that naive double-buffering may not help when
the cost is per-transfer crypto/setup, not a few large stalls.

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

3. **SecureInfer's negative result reframes the opportunity.** Because the cost is
   per-transfer crypto/setup over *many small tensors*, the productive lever is likely
   **transfer batching/coalescing + request-level interleaving**, not per-layer
   double-buffering. A scheme that (i) batches enclave non-linear work across concurrent
   requests, (ii) coalesces the many small PCIe transfers into fewer large encrypted
   bursts, and (iii) keeps the GPU busy on request B's GEMM while request A's softmax is
   in the enclave — would be novel for transformer hybrid inference. **No surveyed system
   does this.**

**Caveat to the "hardware makes it moot" view:** confidential-GPU schemes change the
trust model (trust the silicon vendor + on-die root of trust) and are gated on CC-capable
hardware availability. The obfuscation-rooted hybrid split exists precisely to run on
*commodity* untrusted GPUs (\S7 framing); for that deployment niche the serialization
bottleneck is real and unaddressed.

---

## (e) Verified / unverified / contradicted

**Verified (primary source, high confidence, 0/26 claims refuted under 3-vote
adversarial check):**
- PipeLLM: 52.8%/88.2% drop → <19.6% (arXiv:2411.03357, ASPLOS'25).
- H100 CC: <7% typical, PCIe-transfer-bound; per-model Llama-3.1-8B 6.85%, Phi-3-14B
  4.58%, Llama-3.1-70B ≈0 (arXiv:2409.03992); GPU-TEE 4–8%, CPU-TEE <10%/<20%
  (arXiv:2509.18886).
- TwinShield: ~87% offload, 4.0–6.1× / 4.9–7.7×, >60% in-TEE residual (arXiv:2507.03278).
- SecureInfer: 0.2–0.3 ms/tensor GPU→CPU, 4.7× / 2.06× / 8.44 tok/s, async-overlap-little-
  benefit (arXiv:2510.19979).
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

**Contradicted:** none. The one tension is interpretive, not factual — SecureInfer's
"async overlap yields little benefit" vs the general intuition that pipelining helps;
both are reconciled by the per-transfer-crypto-dominates explanation in §(d).

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
   enclave↔GPU chain, composed with continuous batching. This squarely targets
   SecureInfer's "many small transfers dominate" finding (batching coalesces them) and
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

## Pointers for the SoK (§7)

- The bottleneck and its under-addressed status is a clean **open-problem / future-work**
  note for \S7 or \S9 (synthesis): "hybrid-split schemes are architecturally sequential;
  cross-request pipelining of the enclave↔GPU non-linear chain is unexplored for
  transformer decode."
- The **hardware-vs-software fork** (H100 CC / Ascend-CC / TensorTEE remove the round-trip
  vs commodity-GPU obfuscation hybrids that cannot) reinforces the \S7 Tier-2 framing:
  the serialization cost is the price of staying on commodity untrusted GPUs.
- New citable entries if pursued: PipeLLM (2411.03357), TensorTEE (2407.08903), Ascend-CC
  (2407.11888), Goten (AAAI'21), AsymML/3LegRace (2110.01229), SecureInfer (2510.19979),
  H100 CC benchmark (2409.03992). None are in `refs.bib` yet.
