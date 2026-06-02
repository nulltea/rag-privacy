---
type: dev-log
status: current
created: 2026-05-29
updated: 2026-06-02
tags: [gelo, dgpu, attention, gpu, persistent-kv, permutation, feature-rotation, security, threat-model, aloepri, covariant-obfuscation, flash-attention]
companion: [2026-05-22-dgpu-attention-revival, gelo-llm-perf-chronicle_dgpu]
---

# GPU attention offload for confidential LLM serving — design exploration & implementation log

**What this document is.** The living design-exploration and
implementation log for moving GELO's in-TEE attention bottleneck onto the
untrusted dGPU (RTX 5090) without breaking the threat model. It began as a
decode implementation plan and grew to cover the whole design journey:
the binding measurements, the standardized threat model, every cover
explored (block-fresh-π permutation, TwinShield additive, feature
rotation), the security gates and their *measured* outcomes, the
fused-kernel decision, and the prefill/decode split — including the dead
ends and why they died.

**How to read it.** Organised by concern, not by file. The load-bearing
parts are: the **threat-model assumptions** (the vocabulary everything
else is stated against), the **gate results** (what was measured, with
numbers and artefact paths), and the **Sequencing** section (what is ✅
done vs ⛔ blocked vs remaining). Decisions are dated; superseded framings
are marked rather than deleted, so the reasoning trail survives. It
supersedes the "Item 1 persistent K/V" sketch in
[`2026-05-22-dgpu-attention-revival.md`](../../handoffs/2026-05-22-dgpu-attention-revival.md);
per-op performance numbers live in the companion perf chronicle
[`gelo-llm-perf-chronicle_dgpu.md`](gelo-llm-perf-chronicle_dgpu.md).

**Current status (2026-06-01).** Decode offload is wired + benched
(perf); the Phase-5 `WEIGHTS-PUB` security spike then showed both offload
covers leak token identity (prefill rotation: token + position; decode
permutation: membership, order held) — so prefill is ⛔ blocked behind
Phase 5b (covariant obfuscation; **re-scoped 2026-06-02** from static AloePri
*weight* obfuscation to a per-session non-orthogonal *activation-space* value
cover `C_v` — see *Adapting covariant obfuscation to the offload*), and decode
ships only if the bag-of-tokens residual is accepted. See *Sequencing* for the
live state.

## Why this exists (the binding measurement)

The Step-0 triage (`amulet_attention_r1_4`, RTX 5090, 2026-05-29 —
`bench-results/amulet-attn-triage-5090-2026-05-29.log`) measured the
naive GPU attention path at the decode shape:

| n_kv (decode, B=8) | in-TEE rayon | GPU full-upload (no mask) | GPU ÷ in-TEE |
|---:|---:|---:|---:|
| 256  | 1.08 ms | 71.6 ms | **66× slower** |
| 1024 | 4.47 ms | 281 ms  | **63× slower** |
| 2048 | 11.35 ms| 510 ms  | **45× slower** |

The GPU time scales linearly with n_kv — an *upload* signature, not a
compute one. `fused_attention_batched` re-uploads **and re-converts**
the entire K/V cache (f32→f16) on every call, then blocks on readback;
that fixed pipeline cost (0.13 GB/s effective, ~200× below PCIe
bandwidth) is the entire 45–66× gap. The 5090's HBM and tensor cores
never get to matter. **Naive GPU attention is non-viable; viability is
gated entirely on persistent K/V** — keeping the cache device-resident
so only the per-step delta moves.

**Setup (holds throughout unless stated).** Model: **Qwen3-4B** — 36
layers, 32 query heads, 8 KV heads (GQA 4:1), `head_dim` 128, hidden
2560, vocab 151,936; activations bf16, GPU engine fp16. Hardware:
Ryzen 9 7900X + **RTX 5090** (32 GB GDDR7, ~1.8 TB/s HBM) over PCIe,
dual-channel DDR5 (~80–90 GB/s); fp16 wgpu **Vulkan** for development,
**CUDA** for production (full substrate + per-op profiles in the
companion perf chronicle [`gelo-llm-perf-chronicle_dgpu.md`](gelo-llm-perf-chronicle_dgpu.md)
§1). Production bench shape unless stated: **B=8, n=2048, K=32** decode
steps.

> **Reading the in-TEE attention baseline (avoids an apparent
> discrepancy).** "In-TEE attention at n_kv=2048, B=8" appears across
> three harnesses and is *not* one figure: **~11 ms** as an isolated
> single-attention microbench (11.35 ms in the triage above; 11.15 ms in
> the resident microbench's in-TEE cell), and **12.65 ms** in-forward
> per-(layer,step) in the full decode bench (`tee:attn_cached_inplace_many`,
> 14 574 ms ÷ 1152 layer-steps; chronicle §9/§11). The spread is
> harness/wrapper overhead, not an inconsistency; each speedup factor
> below is computed against the cell named at its measurement.

## Threat model — standardized assumptions

These named assumptions are fixed for the whole document; every gate,
attack, and cover claim below is stated against them. (They replace the
older "reference-free / reference-equipped" vocabulary, which was
ambiguous — see the note at the end.)

- **`TEE-TRUST`** — The trust boundary is the SEV-SNP enclave. The TEE holds
  the plaintext Q/K/V, the secret permutations/rotations and the noise RNG,
  and performs the un-cover/merge. Everything inside the enclave is trusted.
- **`GPU-ADV`** — The VFIO-passed GPU is **fully adversarial**. The
  adversary observes everything the GPU touches: covered operand bytes in
  VRAM, every intermediate it chooses to compute (it controls the kernel —
  fused *or* un-fused, so e.g. it can always recompute `Q̃·K̃ᵀ = QKᵀ` from
  the operands it holds), dispatch shapes/timing, and its own per-step VRAM
  **write locations**. *Corollary:* a fused kernel is a **performance**
  boundary, never a confidentiality one — security may never rest on the
  kernel staying fused.
- **`WEIGHTS-PUB`** — **(conservative default, DECIDED 2026-05-29.)** The
  adversary knows the model weights `W_Q/W_K/W_V` and the
  embedding/unembedding tables (the deployment serves open Qwen3). This
  turns every rotation-invariant Gram into a *known* bilinear form of the
  secret hidden states `X` (`K·Kᵀ = X(W_K W_Kᵀ)Xᵀ`, …) — an algebraic
  anchor for reconstruction. Its negation **`WEIGHTS-BLIND`** (private
  fine-tune, adversary lacks the weights) is the optimistic case; where a
  gate's result depends on this axis we **measure both** and report the
  `WEIGHTS-PUB` number as the bar.
- **`NO-PLAINTEXT`** — The adversary never possesses the user's plaintext
  activations or tokens. This is the **definition of the secret**, not an
  attacker capability: an "attack" that consumes the plaintext is not an
  attack (it already holds the answer). All attacks in this document
  operate under `NO-PLAINTEXT` by construction; clean activations may be
  used **only for offline scoring** of an attack's success.

> **Why the old "reference" vocabulary is gone.** Earlier decode gates
> labelled attacks "reference-free" vs "reference-equipped," where the
> "reference" was a *clean copy of the user's activations*. A
> "reference-equipped" attack therefore violated `NO-PLAINTEXT` — it
> assumed the secret it was trying to recover — so those results were
> meaningless under our threat model and are **removed**. The only
> legitimate side-knowledge axis is `WEIGHTS-PUB` vs `WEIGHTS-BLIND`; the
> document now uses those names exclusively. ("Reference" survives in this
> document **only** in its unrelated numerical sense — a "CPU reference
> implementation" for fp16-parity checks.)

## Decisions taken (grill, 2026-05-29)

- **Scope:** phased — VRAM-resident K/V first (solves the production
  n_kv≤~16k shape), with an NVMe spill tier designed-in for
  beyond-VRAM context (n_kv ≥ ~16–32k, where the cache exceeds the
  32 GB card). The session-handle API must be spill-ready from day one.
- **Cover scheme:** block-fresh-π (hold the existing Amulet +
  Hidden-No-More permuted-attention cover's permutation fixed across N
  decode steps). **Fallback** if the perf gate fails: TwinShield-Xue
  additive softmax-blinding (arXiv 2507.03278).
- **Perf gate:** a fail-fast persistent-buffer microbench (~1 day) that
  holds `k_t`/`v_t` device-resident across iterations and measures
  steady-state per-step cost **and** the boundary re-permute cost,
  against the 11.35 ms in-TEE baseline — *before* committing to the
  2–3 week session-resident substrate refactor. The σ-vs-N security
  spike runs in parallel.
- **Backend:** **CUDA is the production default** (the deployment target
  is Nvidia dGPU under SEV-SNP — CUDA is the deployment reality, not a
  fork); **Vulkan is the development backend** (portable, cheap SPIR-V
  autotune for fast iteration on the iGPU dev box + the iGPU track). The
  engine already swaps `WgpuRuntime` ↔ `CudaRuntime` behind `Rt`/`Dev`
  aliases, so this is a build-flag choice. CUDA's ~10–16% kernel win
  (chronicle §8/§9) is a bonus on top of the native path; its heavy nvrtc
  cold-autotune (~27 s one-time at B=8) is amortized by a long-running
  prod server but is why dev stays on Vulkan. **Kernel consequence:** the
  production partial-stats kernel must be authored in **cubecl** (compiles
  to both CUDA and SPIR-V) — a hand-rolled WGSL kernel is Vulkan-only and
  cannot serve the CUDA prod path, so it drops to at most a dev-only
  optimization. End-to-end acceptance perf (gate tiers 2–4) is measured
  on the **CUDA** backend (warm).
- **Scope (prefill):** decode-only for v1 — the hybrid targets the
  decode attention bucket (34% B=8 / 52% B=1 of decode wall). Prefill
  attention offload (≈43.7 s, ~21–24% of prefill wall; the ~4 GB
  scores-tensor materialization on dGPU) is a **fast-follow**: it shares
  the deferred FlashAttention-D kernel (prefill tiling) but needs no
  session, so it slots in once the kernel lands.
  **⚠ Superseded (2026-06-01):** prefill offload became a central effort,
  not a casual fast-follow — its cover is feature-rotation (not
  permutation), and the Phase-5 `WEIGHTS-PUB` spike found it **broken**, so
  prefill is now ⛔ blocked behind Phase 5b (covariant weight
  obfuscation). See *Sequencing* and the fused-kernel section.
- **Cache structure:** frozen-prefix / active-tail hybrid (below).
- **Kernel:** phased. The gate-1 microbench uses a minimal
  resident-buffer variant of `fused_attention_batched` (whole cache
  resident, full GPU softmax, no tail-split, no partial stats) — enough
  to measure resident-read per-step cost vs the 11.35 ms baseline. The
  production partial-stats / GQA-aware / tail-merge kernel is **deferred**
  until gates 1–3 clear — chosen with measured numbers in hand. Because
  CUDA is the prod backend, the kernel must be **cubecl-authored**
  (compiles to CudaRuntime + SPIR-V); the choice narrows to a
  **cubecl-custom FlashAttention-D vs an upstream-burn `flash_attention`
  extension**. Raw WGSL is Vulkan-only → off the prod table (at most a
  dev-only optimization). The microbench stub may use the existing
  `fused_attention_batched` (already cubecl) on either backend.
- **Session-resident K/V API:** engine-owned session handle
  (`create_session` / `append` / `attend` / `refresh_block` / `drop`) on
  `GpuOffloadEngine`, with a pluggable `SpillProvider` for the Phase-2
  NVMe cold tier (Phase 1 = VRAM-only null provider). See the
  substrate-refactor section; lands post-gates.
- **V hardening:** on the resident cache, V carries the token-axis
  `perm_kv` *plus* a feature-axis orthogonal rotation `O_v` (with a
  matched `O_qk` on Q,K that leaves Q·Kᵀ invariant). Exactly correctable
  (TEE applies `O_vᵀ` after the online merge); hides V's absolute
  coordinates. **v1 default: apply `O_v` once at prefill (session-fixed)**
  — this keeps the boundary re-permute cheap, since the per-block `O_v`
  rotation is otherwise the dominant re-permute cost (gate 1). But `O_v`
  carries its own recovery clock (covariance / cumulant alignment — see
  threat model), independent of `perm_kv`'s HNM clock; the
  covariance-alignment spike (gate 3) is the **gate on this default** —
  if it shows `O_v` recoverable within a production context, the default
  is demoted to refresh every `M` blocks (with the structured-orthogonal
  O(L·d) signed-permutation trick to keep that affordable). Note `perm_kv`
  and the σ-noise on K still refresh per block regardless — only `O_v`'s
  cadence is relaxed. Hiding V's *geometry* (Gram / pairwise distances,
  rotation-invariant) is **deferred** — no correctable transform on the
  permutation path achieves it; if it becomes a hard requirement it
  re-ranks TwinShield-Xue to primary (fallback section).

## The structure

The resident K/V cache is split at a moving boundary `p`:

```
            absolute positions 0 ───────────────────► n_kv
VRAM-resident:  [ ████████ FROZEN PREFIX [0,p) ████████ │ active tail [p, p+j) ]
                  permuted under perm_kv^(b), K σ-noised   stays IN-TEE, plaintext
                  GPU-resident, read at HBM ~1.8 TB/s      ≤ N rows, never uploaded
```

- **Frozen prefix** — the context committed at the *start* of the current block. Uploaded once, under a single fixed `perm_kv^(b)`, with σ-noise already baked into the K rows. The GPU holds these bytes unchanged for all N steps of the block. This is the bulk (long context) and it's what gets read every step at HBM bandwidth — the ~24× per-step win over the in-TEE DDR5 path (measured, gate 1).
- **Active tail** — the ≤ N tokens generated *during* this block. Small. Lives in TEE enclave memory, plaintext, never uploaded.

## A decode step (the load-bearing mechanism: online-softmax split)

Attention over `prefix ∪ tail` is computed by splitting the key set and merging with FlashAttention's exact running-stats trick — softmax is associative through `(max, sumexp, acc)`:

1. TEE forms the new query `q_t`, permutes it on the q-axis (trivial, n_q=1) and adds σ-noise. Uploads **only `q_t`** — `heads × d_head` ≈ a few KB, microseconds.
2. **GPU** computes the prefix's partial attention state against the resident K/V: `(m_A, l_A, acc_A)` per head — running max, sum-of-exp, and the *unnormalized* value accumulator. It does **not** do the final divide. Reads back `acc_A` (d_head/head) + two scalars — tiny.
3. **TEE** computes the tail's partial `(m_B, l_B, acc_B)` over ≤ N plaintext keys — trivial work.
4. TEE merges the two states exactly (`m=max; l=l_A e^{m_A−m}+l_B e^{m_B−m}; acc=…`), applies `O_vᵀ` to undo the V feature-axis rotation (`acc = acc_true·O_v`), divides, un-permutes via π_q⁻¹. Output.

The expensive O(context) part runs on the GPU over resident data; the per-step PCIe traffic is one query up + one partial-state down. **No tail upload, no per-step re-permute.** That's what gets per-step cost toward sub-millisecond.

Note this *requires* a kernel that returns partial `(m, l, acc)` rather than normalized `probs·V` — which is precisely the FlashAttention-D kernel (revival Item 2+3). So the hybrid and the custom-kernel decision are the same decision; the existing `fused_attention_batched` (full normalized softmax) doesn't expose the stats.

## The block boundary (every N steps)

When the tail fills (N tokens) or a refresh is due:
- TEE folds the tail into the prefix → new prefix `[0, p+N)`.
- Samples a **fresh** `perm_kv^(b+1)` over the new length, re-noises K, uploads. Cost ∝ context length, paid once per N steps. **This is the only heavy transfer, and it's the refresh tax the microbench must measure.**

## Threat model

The trust boundary is the SEV-SNP enclave. **Trusted:** the TEE holds plaintext q/k/v, the secret permutations, the noise RNG, does the merge and un-permute. **Untrusted:** the VFIO-passed GPU — it executes matmul/softmax over whatever it's handed and can read its own VRAM and observe dispatch shapes/timing. The adversary controls the GPU.

Over one block of N steps the GPU sees:
- The frozen prefix: K rows **permuted (`perm_kv^(b)`) + σ-noised**, V rows **permuted, _not_ noised** (the code adds Gaussian noise to Q and K only). Fixed bytes for N steps.
- N query vectors, each q-permuted (trivial) + σ-noised.
- N partial-softmax dispatches against the fixed prefix.

What protects the context:
- **`perm_kv` hides row order** — the adversary sees a *shuffled* set of K/V vectors, not the sequence. Amulet's softmax-permutation equivariance makes this exact: `softmax(PX)·PV` recovers correctly after un-permuting queries, so the GPU never needs to know the order.
- **σ-noise on Q,K** defeats statistical recovery of `perm_kv` across repeated observations (Hidden No More, arXiv 2505.18332).

Where the clock ticks — **and this is the part to grill**:
- HNM recovers a *fixed* permutation at 99%+ given enough observations; σ=0.01 is calibrated for **fresh-per-call** π. Holding `perm_kv^(b)` fixed for N steps gives the adversary N correlated looks; the signal grows ~√N, so σ must scale ~√N to hold resistance. **The block size N _is_ the HNM observation count.** That's the σ-vs-N spike, and it caps N (≈32–64 before the noise degrades model accuracy).
- **The hybrid does _not_ improve this** versus monolithic — same N, same clock. What it buys is purely (a) per-step cost (tail in-TEE, prefix at HBM, no per-step re-permute) and (b) the prefix/tail boundary as the NVMe-spill seam. I want to be explicit so we don't credit it with security it doesn't have.
- **V exposure (and the feature-axis-rotation mitigation):** V carries `perm_kv` *and* a feature-axis orthogonal rotation `O_v` (matched `O_qk` on Q,K leaves the scores invariant). `O_v` is exactly correctable — the GPU returns `acc = acc_true·O_v` and the TEE applies `O_vᵀ` in the merge — so the adversary no longer reads the shuffled value vectors directly; absolute coordinates are hidden. The **accepted residual** is geometry: orthogonal transforms preserve the Gram matrix and pairwise distances `‖v_i − v_j‖`, so the value cloud's configuration leaks regardless of `O_v` / `perm_kv`. **⚠ Superseded under `WEIGHTS-PUB` (2026-06-01):** this "residual" is not merely accepted — it is *exactly* the input to the token-membership dictionary attack that breaks both offload covers (the `O_v`-invariant Gram + public `W_V` → per-token identity). See *Gate-3 @ `WEIGHTS-PUB`*. This bullet's optimism reflects the earlier `WEIGHTS-BLIND` framing. Destroying geometry needs *additive* noise on V, which is uncorrectable on this path (the `probs·V` contraction is over the token axis with softmax weights the TEE never sees — see the token-axis argument in the fallback section). For v1 this residual is gated by the c5 AloePri / σ-vs-N spike; making geometry-hiding mandatory re-ranks TwinShield to primary.
- **Two independent clocks (the cadence tension).** `perm_kv` and `O_v` defend different things and are recovered by different attacks, so they tick independently:
  - `perm_kv` hides *position* (sequence order); recovered by the HNM-class attack on attention/score statistics, **fed by query observations** (N per block), defended by σ-noise on Q,K + per-block refresh. → gate 2.
  - `O_v` hides *content* (the value coordinates, hence token identity via vocabulary matching); recovered by **covariance / cumulant alignment** (Procrustes / ICA — JADE, anchor_ica) against the model's known activation statistics, **fed by the number of distinct token-values observed** (grows with context length). → gate 3. **⚠ "hides content" is `WEIGHTS-BLIND`-only:** under `WEIGHTS-PUB`, covariance-alignment *fails to pin `O_v`* (gate-3 covariance rerun — `O_v` holds), but a *different* attack — matching the `O_v`-invariant value norm/Gram against the public-`W_V` dictionary — recovers token **membership** without ever pinning `O_v` (gate-3 @ `WEIGHTS-PUB`). So `O_v` hides coordinates, not membership.
  - The two don't help each other: `perm_kv` shuffles rows but covariance / Gram are computed over the row *set* (permutation-invariant), so `perm_kv` does **not** slow `O_v` recovery; and σ-noise is on Q,K only, so `V·O_v` is observed *noiselessly*, making `O_v` alignment *easier* than `perm_kv` recovery (and we can't noise `V·O_v` — that's the uncorrectable case). Hence `O_v` needs its own refresh cadence `M`, traded against the per-block rotation cost (gate 1).
- One small *benefit*: the freshest N tokens (often the most sensitive, most-attended) stay in-TEE for the whole block and only ever reach the GPU permuted+noised, after a boundary fold.

The cover's order-hiding is preserved throughout (the `GPU-ADV` boundary holds for *order*): the GPU only ever sees permuted+noised operands, and the softmax it runs is over permuted scores (equivariant) — it never learns π. The TEE-side merge is a small plaintext correction, not a softmax-over-real-positions. (This is *order*-hiding only; *membership*-hiding fails under `WEIGHTS-PUB`, above.)

### Write-location side channel on per-step append (found 2026-05-29)

**The attack.** The resident cache is stored in *permuted* order (the GPU
can't apply the secret π, so the bytes are pre-permuted). A naive
"tail-on-GPU" decode appends each new token's K/V row to its slot via a
per-step write. The GPU is **untrusted (VFIO-passed) and can log its own
VRAM writes**, so it observes *which slot is written each step*. Over a
block it sees `perm⁻¹(p), perm⁻¹(p+1), …` for consecutive positions —
i.e. it **reads off the permutation of the appended tokens directly,
with no plaintext, from the write sequence.** This bypasses the entire cover
(it's a side channel on *writes*, not on the covered *contents* the gates
evaluated). Tail-on-GPU cannot escape it: sequential-slot writes expose
positions outright; permuted-slot writes expose π via the write order.

**Threat-model gate.** The attack requires the adversary to observe
**per-step VRAM write locations/timing** (not just bulk resident
snapshots). A malicious GPU under SEV-SNP + VFIO plausibly can. *If it
can*, the mitigation below is mandatory; if it sees only bulk contents,
tail-on-GPU-covered suffices.

**Mitigation — tail-in-TEE (the Phase-3 partial-stats kernel).** Keep the
frozen prefix fixed-resident (uploaded once per block, **bulk**) and the
newest ≤N tokens **in the TEE** (never written to the GPU). Per step the
GPU returns partial `(m,l,acc)` over the fixed prefix; the TEE attends the
in-TEE tail and merges. Result: **zero observable per-token cache writes**
during the block → the per-step write-timing signal is *eliminated*, not
noised. The only writes are block-boundary bulk re-covers, which are
perm-opaque (sequential buffer fill of already-permuted bytes) and feed
only the cross-block HNM √N channel (gate-2, measured weak, |τ|≈0.1). So
this **converts a direct permutation readout into the statistical channel
the gates already cover.** Per-step residual signals — the covered `q_t`
upload (fixed scratch slot, no cache position) and the `(m,l,acc)`
download (aggregate, no per-key info) — don't reopen it.

**Performance (MEASURED 2026-05-29).** Tail-in-TEE
(`gpu_resident_partial_tailtee_b8`) = **~0.74 ms/step @ n_kv=2048**, i.e.
**15× under in-TEE (11.0 ms)** but **~1.6× slower than tail-on-GPU**
(0.45 ms) — the cost of the security property. Parity verified
(prefix(GPU)+tail(TEE) merge == full attention at fp16 floor,
`kv_session_partial_tail_merge_matches_full`).

**Overhead root cause — it's GPU dispatch count, not the CPU side
(`bench-results/phase3-tailtee-opt-5090-2026-05-29.log`).** I first
guessed the ~0.29 ms was the scalar in-TEE tail + the 3-tensor readback,
and optimised both — **BLAS'd the tail** (`attention_partial` → ndarray
`.dot()`) and **batched the readback** (one `Transaction`). They moved
*nothing* (0.71 → 0.74 ms, within noise). The real cost: `attend_session`
runs one **fused** `softmax`, but the composed `attend_session_partial`
runs `max_dim` + `sub` + `exp` + `sum_dim` as **separate dispatches** (~3
extra launches), and at decode (tiny tensors) launch overhead dominates.
**Recoverable only by fusing** — i.e. the custom single-pass cubecl kernel
(the deferred prefill kernel also serves decode), *not* by CPU-side work.
Kept the two optimisations anyway (cleaner code, parity-preserved,
perf-neutral). At 15× under in-TEE the mandatory fix is cheap in absolute
terms; the dispatch overhead is fused-kernel territory, deferred.

**Alternatives are worse:** random-slot writes still leak via write order;
full-cache rewrite-per-step defeats persistence; ORAM-style decoy writes
are expensive. Tail-in-TEE is the clean fix.

**Consequence for sequencing — DECIDED 2026-05-29.** The deployment
threat model is taken as the **write-observing adversary** (a malicious
VFIO-passed GPU can log its own VRAM writes — the conservative, realistic
assumption). Therefore **Phase 3 (partial-stats kernel → tail-in-TEE) is
MANDATORY, not a prefill/perf fast-follow** — it closes a channel that
otherwise reads the appended permutation directly, at ~zero (favourable)
runtime cost. Phase 3 is promoted from "gated on security" to a required
v1 component.

**Scope of what tail-in-TEE fixes (clarified 2026-06-01).** It closes the
*write-location / order* channel — the per-step append no longer leaks
π. It does **not** address the `WEIGHTS-PUB` *membership* leak (the
`O_v`-invariant value-norm dictionary recovers which tokens are present
regardless of where they were written; gate-3 @ `WEIGHTS-PUB`). The two
are orthogonal channels: tail-in-TEE protects order, membership-hiding
needs covariant weight obfuscation (Phase 5b).

---

## Offloaded-prefill attention: the attack vector (2026-05-29)

The decode threat model above concerns a *resident* cache read one query at
a time. **Offloading prefill** — the single largest in-TEE bucket
(`tee:attn_inplace_many` ≈ 43.7 s, ~21–24 % of prefill wall — 21.2 % in
the chronicle §3.2 baseline run, 23.6 % in the Phase-4 flag-on run; the
absolute is stable, the share moves with the run's total) — is a different
exposure, and it is the reason the offload path cannot use the permutation
cover. This section defines what we are defending against; the cover that
answers it (feature rotation) and the gate are in the fused-kernel section.

### What the adversary wants and what it sees

**Goal.** The untrusted GPU reconstructs the user's **prompt tokens** from
the prefill attention it is asked to run.

**Setting.** To offload prefill, the TEE hands the GPU the prefill operands
`Q, K, V` for all `n` prompt tokens (per layer, per head) under a cover,
and the GPU runs the fused causal self-attention `softmax(QKᵀ/√d + M)·V`.

**Capabilities (conservative, DECIDED 2026-05-29).** The adversary controls
the GPU: it sees every covered operand handed to it, every intermediate it
chooses to compute (fused *or* un-fused — see below), dispatch
shapes/timing, and — the conservative assumption for an **open base model
(Qwen3)** — **the public weights** `W_Q, W_K, W_V` and the embedding table.

**A fused kernel is a *performance* boundary, not a *security* one.** This
is the load-bearing clarification. A feature rotation cancels in the score
for *anyone holding the rotated operands*, not only inside the kernel:
`(Q·O_qk)(K·O_qk)ᵀ = QKᵀ`, so the GPU can reconstruct the true score matrix
`S = QKᵀ` (and `P = softmax(S)`) on its own at any time — the scores are a
*rotation-invariant* of operands it already possesses. So an administrator
who replaces the fused kernel with an un-fused one that spills `S` to HBM
**learns nothing extra**. Security therefore rests entirely on the
rotation-invariant view being insufficient to reconstruct — never on the
kernel staying fused (the fused kernel only buys the no-`[n,n]`-scores-in-HBM
*speed*; cf. TwinShield, where the score *magnitudes* are blinded, so the
accelerator genuinely never holds un-blinded scores).

**The rotation-invariant view (what is and isn't exposed under feature
rotation).** Exposed: the same-rotation Grams `Q·Qᵀ`, `K·Kᵀ`, `V·Vᵀ` (norms +
all pairwise similarities), the cross-Gram `Q·Kᵀ = S` hence the attention
pattern `P = softmax(S)`, and token **order / positions**. Hidden: absolute
operand **coordinates**, the **QK↔V cross-geometry** (`O_v` is independent of
`O_qk`, so `Q̃·Ṽᵀ = Q·O_qk·O_vᵀ·Vᵀ` does not cancel), and the un-rotated
output `P·V`. With **public weights** the exposed Grams are *known bilinear
forms* of the secret hidden states `X` (`Q = X·W_Q`, …):

```
K·Kᵀ = X (W_K W_Kᵀ) Xᵀ,   Q·Kᵀ = X (W_Q W_Kᵀ) Xᵀ,   Q·Qᵀ = X (W_Q W_Qᵀ) Xᵀ
```

— a system of quadratic constraints on `X` that *anchors* the otherwise-free
global rotation. **The bar the offload cover must clear** is therefore:
*can a `WEIGHTS-PUB` adversary (under `NO-PLAINTEXT`) solve `X` (→ tokens, via
the known embedding table) from `{X·M·Xᵀ : M known}` + `P` + positions?* That is
exactly the question Phase 5 measures.

### Why permutation cannot defend offloaded prefill — the causal-mask leak

The block-fresh-π cover defends *decode* (where the mask is trivial) but is
**structurally defeated at prefill by the causal mask** — independent of how
well the contents are covered, because the mask is binary *structure*, not
data.

**Setup.** Prefill is self-attention over `n` tokens with a causal mask: in
original order, query `a` may attend key `b` iff `b ≤ a`. The cover permutes
the sequence axis by secret π (physical slot `s` holds original token
`π(s)`). For the attention to be *correct*, the GPU must apply the mask in
its own (permuted) coordinates, i.e. the **permuted causal mask**:

```
M_phys[i][j] = 1  iff  π(j) ≤ π(i)
```

(cubek's built-in `causal:true` masks in physical-slot order, which over
permuted bytes is the *wrong* math — correctness forces supplying this true
permuted mask, and the untrusted GPU sees it.)

**Recovery — step by step.**
1. Sum any physical row `i`:  `rᵢ = Σⱼ M_phys[i][j] = #{ j : π(j) ≤ π(i) }`.
2. π is a bijection onto `{0,…,n−1}`, so the number of values `≤ π(i)` is
   exactly `π(i)+1`. Hence `rᵢ = π(i)+1`.
3. Therefore `π(i) = rᵢ − 1` — read off π for every slot directly, no
   inversion, no plaintext, **independent of the content cover**.

**Worked example (`n = 4`, secret π = `[2,0,3,1]`):**

```
            j=0  j=1  j=2  j=3   row-sum   π(i)=sum−1
 i=0 (π=2)   1    1    0    1       3          2   ✓
 i=1 (π=0)   0    1    0    0       1          0   ✓
 i=2 (π=3)   1    1    1    1       4          3   ✓
 i=3 (π=1)   0    1    0    1       2          1   ✓
read row-sums [3,1,4,2] − 1 → [2,0,3,1] = π, recovered exactly.
```

**Why it can't be patched.** The leak is information-theoretic: a *correct*
causal mask must encode the total order of the `n` tokens — `log₂(n!)` bits,
exactly π's entropy. Any faithful representation carries it; the row-sum is
merely the cheapest extractor. Permuting columns does not help (the row-sum
is invariant to it), and a *different* column permutation is impossible —
self-attention's softmax-equivariance requires the same π on Q, K, V to
cancel. **Decode is exempt**: with `n_q = 1` the single new token attends
everything, its mask row is all-ones, and there is no pairwise order to leak.

⇒ Permutation rides along on offloaded *decode* but is fatal to offloaded
*prefill*. The offload cover must hide *coordinates without permuting the
token axis* — i.e. feature rotation (next section).

---

## Session-resident K/V API (substrate refactor — gated on the spikes)

The persistence the design needs is an **engine-owned session**, not the
per-call K/V views `offload_attention_permuted_cached` takes today. The
`GpuOffloadEngine` gains a session handle:

- `create_session(prefix_k, prefix_v, perm_kv, O_v) -> SessionId` —
  uploads the rotated + permuted + noised prefix once.
- `append(id, k_row, v_row)` — adds one decode token; the tail is small
  and may stay TEE-side (the online-merge path).
- `attend(id, q) -> partial(m, l, acc)` — the prefix's partial softmax
  state for the TEE-side merge.
- `refresh_block(id, perm_kv', noise)` — re-gather the canonical cache
  under fresh `perm_kv` + re-noise K (every block). `O_v` is **not**
  re-applied (session-fixed default; gate 3).
- `drop_session(id)`.

Residency is engine-owned; the cold tier is a **pluggable
`SpillProvider`** (Phase 1 = null / VRAM-only; Phase 2 = NVMe), so the
spill seam exists from day one and the NVMe tier slots in with no API
change. The TEE retains the canonical rotated cache (`V·O_v` in canonical
order) for the `refresh_block` re-gather — so there are two copies (TEE
canonical + GPU permuted), ~2× cache memory (≤ ~1 GB at 16k B=8 un-
replicated bf16, negligible on the 32 GB card).

This is the load-bearing trait change every engine impl must follow; it
lands only after gates 1–3 clear (it is the 2–3 week substrate refactor,
not part of the microbench).

## Fallback: TwinShield-Xue additive blinding

If the block-fresh-π perf gate fails (the ∝L re-permute tax doesn't
amortize within the security-permitted N), the fallback is the
additive softmax-blinding scheme of TwinShield-Xue (arXiv 2507.03278).
Instead of permutation-equivariance, it blinds the scores additively in
the exponent: `e^{X+R} = e^X · e^R`. The blinding `R` is regenerated
**fresh per call** and the TEE divides it back out, so — unlike a fixed
permutation — it carries no across-step accumulation clock. That lets
K/V persist on the GPU under a *fixed* operand cover while the
fresh-per-call `R` supplies the freshness that defeats HNM-class
recovery, **decoupling persistence from the security clock** that
block-fresh-π pays the re-permute tax to manage.

It is also the one mechanism that could perturb V's *geometry*
correctably. Under our permutation path, additive noise on V is
uncorrectable: `out = probs·V` contracts over the token axis with
softmax weights the TEE never sees, so `Σ_j e^{s_j−m} ε_j` cannot be
subtracted (the same reason the in-tree cover noises Q and K only, and
why feature-axis rotations — which pass through the contraction — are
the only correctable V-hardening on the permutation path, and those
preserve geometry). TwinShield builds the correction into the protocol,
so the additive blinding the permutation path cannot do becomes
available — at the cost of the `R`-rank correction.

### Structural difference for this design

| Axis | Block-fresh-π (primary) | TwinShield additive (fallback) |
|---|---|---|
| What enables persistence | fixed `perm_kv` across N steps | fixed operand cover **+ fresh-per-call `R`** |
| Freshness clock | **N is the HNM observation count** → σ-vs-N gate caps N ≈32–64 | `R` regenerated every step → **no fixed-cover accumulation clock** |
| Refresh / re-permute tax | full re-permute ∝ context, every N steps (the hybrid's one remaining cost) | **none** — `R` is small and per-step |
| Correction cost | ~free (perm un-apply + online merge) | **f(rank R)** — full-rank `R` ⇒ correction ≈ cost of the attention itself ⇒ no win |
| V-geometry hiding | no (orthogonal / permutation only — geometry-preserving) | **yes, if `R` reaches the value path** — the additive thing the permutation path cannot do |
| Maturity | Amulet + HNM, in-tree, fresh-per-call already validated | published 2025; **threat model predates HNM**; needs independent validation |
| Engineering | extends `permuted_attention_cached` | new port + a TEE-side correction pipeline |

The two schemes trade one open risk for a different one. Block-fresh-π
fails on **perf** ("N too small to amortize the ∝L re-permute") — cheaply
falsifiable with the microbench. TwinShield fails on **correction cost**
("R too high-rank to un-blind cheaply") — *not* cheaply falsifiable; it
needs the paper's threat model audited against ours and the `R`-rank
characterized (the 1B security spike). That asymmetry is why the
sequencing runs the cheap in-tree path first and reaches for the
structurally-cleaner-but-higher-unknown path only on a measured perf
failure.

> **Reframing (2026-05-29, corrected after reading the paper).** The
> offloaded-prefill cover is **feature-rotation-only**, not additive. Two
> independent eliminations force this (see the fused-kernel section): (a)
> block-fresh-π leaks π through the causal mask, so permutation cannot run
> on an untrusted causal-prefill kernel; (b) TwinShield additive is
> **architecturally incompatible with a fused kernel** — verified against
> arXiv 2507.03278 v2: it offloads the attention *GEMMs* and runs softmax
> **in the TEE** over a recovered, round-tripped score matrix, which is
> exactly what a fused FlashAttention kernel exists to avoid. The only
> cover that is both causal-mask-safe and transparent to cubek-attention's
> normalised output is orthogonal feature rotation (`O_qk`/`O_v`) — which
> we already have. TwinShield remains a *separate, kernel-incompatible*
> lever (GEMM-offload + TEE softmax), not this lever's cover.

## Fused attention kernel — `cubek-attention` (no hand-roll)

The deferred "custom single-pass cubecl FlashAttention-D" turns out **not
to be custom**. `cubek-attention` (`=0.1.1`, already a dependency; the
spike `cubek_attention_spike.rs` launches + matches a CPU reference at the
fp16 floor on our wgpu stack) is the cubecl-ecosystem fused kernel:
tiled online-softmax (the `[B,H,sq,skv]` scores never touch HBM — this is
what kills the ~4 GB prefill scores tensor), causal + arbitrary supplied
mask, f16, portable `Unit` and tensor-core `BlackboxAccelerated`
strategies, remainder tiling for arbitrary sequence lengths. **We
integrate it; we do not author GPU kernels** (no in-repo `#[cube]`).

cubek-attention is the **prefill** kernel. Decode does **not** route
through it — decode keeps the permutation cover served by the composed
`attend_session_partial` engine path (which already returns the
`(m,l,acc)` stats tail-in-TEE needs); cubek's two gaps are exactly why it
is unfit for decode and therefore prefill-scoped:

| gap | prefill (cubek, the target) | why it rules cubek out of decode |
|---|---|---|
| **no native GQA** — single `num_heads`; K/V shaped `[B,num_heads,skv,d]`, so K/V must be expanded 8→32 heads | modest: ~2.1 GB transient expanded K/V (B=8, n=2048) vs ~0.5 GB un-replicated; the extra HBM reads hide under the n_q=2048 matmul (compute-bound). One `repeat_dim`, once. | costly at decode: memory-bound, so 4× per-step K/V read bandwidth + 4× resident VRAM, breaking the un-replicated-storage / NVMe-spill economics. (Closeable later by a kv-head-broadcast read-index — a fast-follow, **not** a new kernel.) |
| **no `(m,l)`/LSE output** — `launch(...) -> Result<()>` writes only normalized `out` | none: prefill is one-shot, no tail to merge. | decode's permutation cover needs tail-in-TEE, which needs the GPU's partial `(m,l,acc)` stats — cubek can't emit them. Decode therefore stays on the composed `attend_session_partial` path. |

### Why the offload cover is feature-rotation-only (two eliminations)

Two independent arguments rule out the other two covers for a fused,
normalised-output kernel, leaving feature rotation as the only survivor.

**Elimination 1 — permutation leaks π through the causal mask (prefill).**
The permuted `n×n` causal mask reveals π *exactly* via its row-sums
(`rᵢ = π(i)+1`), independent of the content cover — derived step by step,
with a worked example, in *Offloaded-prefill attention: the attack vector*
above. Decode is exempt (`n_q=1`, all-ones mask, no order), so permutation
offloads decode but never prefill.

**Elimination 2 — additive (TwinShield) is incompatible with a fused
kernel.** Verified against arXiv 2507.03278 v2: TwinShield's softmax
offload has the GPU return only the blinded exponentials `e^(xᵢ−rᵢ)` and
the **TEE** multiply back `e^(rᵢ)`, sum, and divide; its attention offload
has the GPU compute the blinded *GEMM* `(Q+R_Q)(Kᵀ+R_Kᵀ)` (a permuted 2×2
block) and the **TEE recover `QKᵀ` and run the softmax**. The accelerator
never returns a normalised attention output. So additive blinding
structurally requires the GPU to stop at the un-normalised stage and hand
intermediates back — which (a) re-materialises and round-trips the
`[sq,skv]` score matrix (the ~4 GB prefill tensor a fused kernel exists to
avoid; ~160 ms/dir/layer over PCIe at n=2048) and (b) keeps the softmax in
the TEE. It cannot run *through* `cubek-attention`. This also matches the
first-principles result: a normalised-output kernel admits a cover only if
the cover preserves `softmax(QKᵀ)` up to a TEE-applicable linear
post-correction — additive on K shifts softmax non-linearly, additive on V
needs the full probability matrix `P` to undo. TwinShield stays a
*separate lever* (GEMM-offload + TEE softmax), evaluated on its own terms,
not this lever's cover.

**The survivor — orthogonal feature rotation (prefill offload).** `O_qk`
on Q and K cancels in the score (`(Q·O_qk)(K·O_qk)ᵀ = QKᵀ` for orthogonal
`O_qk`), so the GPU computes identical scores while never seeing clean Q/K
and **the softmax is untouched**; `O_v` on V passes through the value
contraction, so the GPU returns `O_true·O_v` (normalised) and the TEE
corrects with a single `·O_vᵀ`. Both rotations are on the **head-dim
axis** — the token axis is untouched, so the causal mask is the standard
*public* triangular and leaks nothing. σ-noise is **off** on this path
(additive noise on K is non-correctable through a fused softmax, per
Elimination 2). This is the cover we already have, behind
`PermAttnConfig::feature_rotation` (`O_qk`/`O_v`, `attention.rs:408–508`) —
to be sampled per-head and wired to `cubek-attention`.

### The cover split: feature-rotation prefill, permutation decode

The two phases use **different covers**, because they face different leaks:

| phase | computation | cover | why |
|---|---|---|---|
| **prefill** (offloaded, one-shot) | `softmax(QKᵀ+M)·V` over `n` prompt tokens | **feature rotation** `O_qk`/`O_v`, σ=0, public causal mask | permutation leaks π via the square mask (Elim. 1); additive can't run on a fused kernel (Elim. 2) |
| **decode** (resident, per-step) | one query over the growing cache | **permutation** (block-fresh-π + σ + `O_v`) **+ tail-in-TEE** | decode's mask is trivial (no π leak), so the gate-2/3-validated permutation cover is kept; tail-in-TEE closes the write-location channel |

**The prefill→decode handoff.** The covers protect two *different*
computations and never mix on one buffer. Prefill's offloaded attention
uses a **transient** rotation-covered `Q,K,V` (per head: `K·O_qk`,
`V·O_v`), discarded after the prompt's contextualised output is produced.
The **resident decode cache** is a *separately* permutation-covered `K,V`
that the TEE builds from the clean prompt K/V (which it holds — it computed
the K/V projections) at the prefill→decode boundary. That one-time
re-cover-and-upload is the "re-permute upload" already cost-analysed in
gate-1 / chronicle §10 (≈488 ms current pipeline → modelled ≈5 ms with the
un-replicated + bf16-native substrate). So enabling prefill offload does
**not** change the decode cover; it adds a rotation-covered prefill pass
plus the existing residency upload.

**Security posture differs by phase, deliberately.** Prefill offload runs
the *weaker* rotation-only cover — its residual (the row-Gram, token order,
and the weights-equipped reconstruction of *Offloaded-prefill attention:
the attack vector*) is what the sufficiency spike must clear. Decode keeps
the permutation cover under gates 2/3. Turning prefill offload on accepts
the rotation-only residual **for the prefill computation only**; the
resident decode path is unaffected.

### The offload-cover sufficiency spike (the gate the lever hangs on)

This spike gates the **prefill offload cover only** (decode keeps the
permutation cover under gates 2/3). Rotation-only is the only *correctable,
causal-safe* cover for the fused kernel — but the **weakest**: it preserves
row norms, the full row-Gram, the cross-Gram/scores, and (no permutation)
token order. **The spike is run in both regimes** — `WEIGHTS-PUB` (the
conservative default) and `WEIGHTS-BLIND` (the private-fine-tune case) — to
bracket the outcome and isolate exactly how much the public weights buy the
attacker. It asks: **is the rotation-invariant view — optionally anchored by
public weights — enough to reconstruct the prompt?** Reusing the gate-3
capture harness + the container attack rig, on real Qwen3-4B activations:

1. **`WEIGHTS-PUB` reconstruction (the load-bearing attack).** With public
   `W_Q/W_K/W_V`, the exposed Grams are known bilinear forms `X·M·Xᵀ` (M
   known). Attack: solve the secret hidden states `X` from the system
   `{Q·Qᵀ, K·Kᵀ, V·Vᵀ, Q·Kᵀ}` (+ positions + `P`) — or, the cheap form, match
   the per-token-pair Gram dictionary `D[t_i,t_j]=e_{t_i}(W_V W_Vᵀ)e_{t_j}ᵀ`
   (sharpest at early layers where `X≈E[tokens]`) — then map to tokens via
   the known embedding table; score token-recovery rate vs a chance floor.
   This is the conservative bar the cover must clear — *not* blind ICA.
2. **`WEIGHTS-BLIND` rotation recovery.** The weaker baseline: FastICA /
   JADE on the rotation-only view (`k_rot = K·O_qk`, `v_rot = V·O_v`) — the
   existing gate-3 attacks on the *no-permutation* view — scoring mean
   matched `|corr|` to the clean (position-known) columns, with a
   random-rotation chance control.
3. **Order-is-public residual.** Quantify what preserved order + Gram buys
   the attacker that permutation previously denied (e.g. does `‖Kᵢ‖` / Gram
   structure correlate with positional/semantic structure usefully?) — the
   explicit gap vs the permutation path.
4. **Fixed-rotation accumulation (rotation analog of HNM √N).** `O_qk` is
   fixed for a prefill (and, if a rotated prefix were ever reused, longer),
   so the adversary observes many `Q·O_qk` against fixed `K·O_qk`/`V·O_v`.
   Does sample accumulation pin the rotation that one-shot cannot? Sweep
   observation count; sets any re-rotation cadence.

**Correctness + cost (Rust, alongside the attacks).** Parity test that
cubek's normalised attend over rotated, GQA-expanded K/V + public causal
mask, then in-TEE `·O_vᵀ`, equals clean attention at the fp16 floor (mirror
`kv_session_partial_tail_merge_matches_full`); measure the `O_vᵀ`
correction (one `[B, H·d, d]` right-multiply per layer — `O(B·H·d²)`,
expected trivial).

**Three-way outcome:**
- **Pass under `WEIGHTS-PUB`** ⇒ rotation-only ships as the prefill offload
  cover on `cubek-attention` (decode unchanged).
- **Pass under `WEIGHTS-BLIND` but fail under `WEIGHTS-PUB`** — **this is the
  MEASURED outcome (2026-06-01):** decode covariance-alignment held, but the
  prefill rotation cover leaks every token (top-1 = 1.000) to the
  norm/Gram-dictionary attack; see *Phase-5 spike — prefill rotation cover
  BREAKS*. ⇒ the leak is precisely the
  *public-weight anchor*. Before abandoning, explore **covariant weight
  obfuscation (AloePri)**: statically transform the deployed weights (and
  compensate covariantly so the computation is unchanged) so that the
  attacker's *public* `W` no longer matches the *deployed* `W'` — collapsing
  the known bilinear forms `X·M·Xᵀ` back to *unknown* `M'` and removing the
  dictionary the attack relies on. This converts the open-model
  `WEIGHTS-PUB` deployment into an effective `WEIGHTS-BLIND` one for the
  attacker. Re-run the spike against the obfuscated-weight deployment to
  confirm. (Open cost questions: the obfuscation's correctness/overhead and
  whether it composes with the fused kernel — a sub-spike if we reach here.)
- **Fail under both** ⇒ no fused-kernel prefill offload — the offload would
  need token-order hiding the fused kernel cannot provide (permutation =
  mask leak, additive = kernel-incompatible) — so prefill stays in-TEE, or
  the TwinShield GEMM-offload lever is evaluated separately on its
  score-round-trip cost.

The whole prefill-offload lever is downstream of this gate.

## Open questions (the load-bearing gates)

The kernel / backend / session-handle-API choices below these are
comparatively mechanical; these three gates were framed (2026-05-29) as
deciding whether block-fresh-π ships or we fall to TwinShield-Xue. **That
framing is partly superseded:** TwinShield is *not* a drop-in fallback for
the offload (it can't run on a fused kernel — Elimination 2), and the
dominant later finding is the `WEIGHTS-PUB` membership leak, which neither
gate 1–3 originally targeted. Read gates 1–3 as the perf + `WEIGHTS-BLIND`
security picture; the conservative-bar verdict is in the `WEIGHTS-PUB`
gate-3 results and *Sequencing*.

1. **Prefix re-permute cost (gate 1, perf).** **Per-step half: PASSED
   (2026-05-29).** The `gpu_resident_b8` microbench measures resident
   per-step at **0.30 / 0.373 / 0.465 ms** (n_kv 256 / 1024 / 2048, B=8)
   vs the in-TEE baseline 1.08 / 5.28 / 11.15 ms — **24× faster at n=2048,
   widening with context.** Decomposition confirmed: ~99.9% of the
   full-upload cost is upload+convert+sync (the term persistence deletes).
   See chronicle §10 / `bench-results/amulet-attn-resident-5090-2026-05-29.log`.
   **Re-permute half — conditional.** The microbench also bounds the
   re-permute *upload* directly: `no_mask − resident` = (convert+upload
   K/V) = **~488 ms @ n=2048** on the *current* pipeline (GQA-expanded,
   f32→f16 convert). Amortized over N=16 that is ~30 ms/step — it would
   **lose** to the 11 ms in-TEE baseline. So the re-permute half **only
   wins once the upload is optimized**: un-replicated storage (4× less
   data) + bf16-native K/V (no f32→f16 convert) → modeled ~5 ms (gather
   1.5 + upload 2 + re-noise ~1), ÷16 ≈ 0.3 ms/step. That optimization is
   part of the substrate refactor — **gate-1 is PASS on the per-step read,
   PASS-conditional on the re-permute optimization landing.** The boundary
   re-permute
   decomposes into: gather (`perm_kv`, memory-bound — ~1.5 ms @2048 /
   ~12 ms @16k), the `O_v`/`O_qk` rotation (compute-bound — ~86 ms @2048
   / ~0.7 s @16k *if refreshed*, at an assumed ~100 GFLOP/s CPU), and
   re-noise + bf16 upload (bandwidth-bound — a few ms / ~30 ms,
   un-replicated B=8). The **rotation is the swing term**, so the
   amortized re-permute is governed by the `O_v` cadence `M` (gate 3),
   *not* by `N`: at `O_v` session-fixed it collapses to gather+upload
   (~0.2 ms/step @2048, ~2 ms/step @16k after ÷N — negligible); at
   per-block `O_v` it eats ~half the in-TEE baseline. The fail-fast
   microbench profiles the actual split — its **first deliverable** is
   decomposing the 510 ms triage cost into convert / stage / DMA / sync,
   since the "fixed-overhead-bound" attribution is currently *inferred*,
   not measured. Async double-buffering is **not** a v1 lever — it hides
   transfer, not the CPU rotation that dominates when `O_v` refreshes.

2. **σ-vs-N thresholds (gate 2 — the `perm_kv` clock).** The
   threat-model-valid (`NO-PLAINTEXT`) measurement is pending (Python HNM
   driver + real activations — see gate-status). Settled so far
   (2026-05-29, Rust, `gate2_perm_recovery_vs_sigma_and_n` /
   `bench-results/gate2-perm-recovery-sigma-n-2026-05-29.log`):
   - **σ-noise is not a usable lever (quality ceiling).** Attention quality
     is destroyed by σ≈0.3 (drift 0.08), far below any σ that could blunt
     recovery, so σ cannot be the security knob. (A model-quality
     measurement, independent of any attacker — valid under all
     assumptions.)
   - **Persistence is strictly worse, via √N denoising (HNM principle).**
     Holding π fixed for N steps gives the attacker N correlated looks; the
     signal averages up ~√N, so fixed-π-across-N is the *worst* case for
     `perm_kv`. ⇒ `perm_kv` **refreshes per block** (bounded N); `O_v`
     alone is session-fixed. *(The earlier √N recovery curve was produced
     by a probe that assumed clean Q — it violated `NO-PLAINTEXT` — so the
     curve is **withdrawn**; the √N conclusion stands as the HNM structural
     result, and the realistic driver will quantify the max N.)*
   - **Implication:** the cover's security rests on the **GELO mask
     invariant** (the GPU never sees plaintext positions), not σ-noise.
     The `NO-PLAINTEXT` HNM attack that yields the actual max-N — and, under
     `WEIGHTS-PUB`, the weights-anchored variant (gate 3 below) — is not yet
     run.

3. **Content-recovery thresholds (gate 3 — the `O_v` clock).** Targets the
   value cover. **Two attacker regimes, and the distinction is decisive:**
   - **`WEIGHTS-BLIND`:** feed JADE / anchor_ica the observed `V·O_v` cloud
     and ask whether blind un-mixing recovers `O_v` up to sign/axis flips.
     This is what the 2026-05-29 run measured (`O_v` held).
   - **`WEIGHTS-PUB` (the conservative default — the gate that actually
     matters):** a weights-equipped attacker **does not try to recover
     `O_v` at all** — the value Gram is `O_v`-invariant, so with public
     `W_V` and the embedding table it becomes a *known* per-token-pair
     dictionary `D[t_i,t_j] = e_{t_i}(W_V W_Vᵀ)e_{t_j}ᵀ`, and the attack is
     to match the observed (permuted) Gram / per-row norms against `D` to
     recover token **identity**. `O_v` is irrelevant to this attack, so the
     `WEIGHTS-BLIND` "`O_v` holds" result **does not cover it**. The
     permutation still hides *order* (recovers a multiset, not a sequence)
     and σ is on K not V, so the V Gram is clean — this is the untested
     exposure. **This regime is run as an updated gate 3 (see report
     below).**
   Output: the max `O_v`-fixed observation budget → refresh cadence `M`,
   and — under `WEIGHTS-PUB` — whether token membership leaks via the V
   Gram at all. Determines whether v1 ships `O_v` session-fixed (cheap —
   gate 1) or must refresh every `M` blocks (and whether the
   structured-orthogonal `O(L·d)` signed-permutation trick is needed to make
   a finite `M` affordable).

### Gate-measurement status + environment split

The gates run in two environments, and only part runs on the dGPU box:

- **Rust, on the dGPU box (done 2026-05-29):** the **quality ceiling**
  (`permutation_attention.rs` drift-vs-σ: σ=0.01 drift < 5e-2, tolerable;
  σ≥0.3 destroys output) and the √N-accumulation principle
  (`gate2_perm_recovery_vs_sigma_and_n`). These establish: σ is not the
  security lever, and persistence amplifies recovery via √N — so `perm_kv`
  must refresh per block. *(The recovery curve in that test assumed clean
  Q, violating `NO-PLAINTEXT`; the curve is withdrawn, the √N conclusion
  kept — see gate 2.)*
- **Python AloePri harness, on the eval env / CI (NOT runnable on the
  dGPU box — no pip/ensurepip/apt/sudo; numpy/scipy/sklearn absent):**
  the `NO-PLAINTEXT` attacks that produce the actual cadence numbers —
  HNM statistical permutation recovery (gate 2 proper → max N) and the
  `O_v` / value-content attacks (gate 3 → max T / M), run in **both**
  `WEIGHTS-BLIND` and `WEIGHTS-PUB` regimes.
  Drivers exist (`evals/aloepri-attacks/attack_drivers/run_{jade,anchor_ica}.py`)
  but target the linear-mask channel; the attention-cover scenario
  (fixed `perm_kv`+noise on K across N; fixed `O_v` on the V cloud) is a
  new condition to add. **Gate 3 additionally requires real Qwen3-4B
  activation dumps** — random/isotropic activations make `O_v` recovery
  trivially fail (false pass), so step 0 is capturing real K/V at the
  production shape.

### First real-activation gate run (2026-05-29)

Pipeline now wired end-to-end: `attn_cover_capture.rs` dumps the real
Qwen3-4B adversary view (the **decode permutation cover in its
session-fixed config** — `perm_kv` + σ=0.01 on K + `O_qk`/`O_v`, applied
once at prefill with no per-block refresh; "prefill-only" here means
*cover-applied-once*, **not** the prefill-*offload* rotation cover —
n_kv=545, layers 0/17/35) → `persistent_attn_gate.py`
(in the `gelo-attack` container) attacks it. Artefacts:
`evals/aloepri-attacks/{persistent_attn_gate.py, captures/gate_results.json}`.

**Regime of this run: `WEIGHTS-BLIND` (and `NO-PLAINTEXT`).** All attacks
operate on the cover-applied data alone, with **no use of the model
weights**; ground truth is used **only for scoring**
(benefit-of-the-doubt). So these are robust passes *within the
`WEIGHTS-BLIND` regime* — they do **not** speak to `WEIGHTS-PUB` (see the
caveat below).

| attack (`WEIGHTS-BLIND`) | result (n_kv=545, layers 0/17/35) | reading |
|---|---|---|
| baseline (direct cos-match `v_sent`→`v_clean`) | cos ≈ 0.07, perm ≈ chance | cover defeats naive coordinate matching |
| **gate 3 (content / `O_v`): FastICA** | corr **0.25** | nothing recovered above the coincidental floor |
| **gate 3 (content / `O_v`): JADE (4th-order)** | corr **0.36** ≈ no-attack floor **0.35** | **`O_v` HOLDS** — even 4th-order JADE doesn't un-mix; far from recovery (self-test ref 0.998) |
| **gate 2 (position): K-Gram spectral seriation** | \|Kendall τ\| **0.05 / 0.06 / 0.20** (avg **0.10**) | **position largely HELD** — RoPE's relative-position signal does not survive `perm_kv`+`O_qk`+σ for seriation (mild residual only at the deepest layer) |

JADE validated on a known Laplace mixture (`--self-test`, matched corr
0.998), so the gate-3 negative is trustworthy *as a `WEIGHTS-BLIND`
result*. (An earlier probe suggested position leaked at 0.70–1.00, but it
assumed clean Q — it violated `NO-PLAINTEXT` — and is **removed**.)

> **`WEIGHTS-PUB` caveat — these passes do not clear the conservative
> bar.** The result above measured whether *blind* un-mixing recovers
> `O_v`. A `WEIGHTS-PUB` attacker never targets `O_v`: the value Gram is
> `O_v`-invariant, so with public `W_V` it is a known per-token-pair
> dictionary, and the very **geometry residual** these runs flagged as
> "accepted" (row-norm multiset + Gram) is *exactly* its input. So under
> the conservative default the gate-3 question changes from "does `O_v`
> hold?" (it does, blindly) to "does the `O_v`-invariant Gram leak token
> identity to a weights-equipped matcher?" — **untested here.** The
> permutation still hides *order* (→ multiset, not sequence) and σ is on K
> not V (V Gram clean). The `WEIGHTS-PUB` rerun is the updated gate 3
> **reported below**.

**Breadth — confirmed (still `WEIGHTS-BLIND`).** A 2nd prompt × 5 layers
(0/9/18/27/35, n_kv=479) reproduces both findings: gate-3 ICA 0.25 / JADE
0.37 ≈ no-attack 0.35 (`O_v` holds blindly, stable across prompts); gate-2
seriation |τ| avg 0.15 (per-layer 0.05–0.26, no systematic layer trend).

**Still to do:** the `WEIGHTS-PUB` gate-3 rerun (weights-anchored Gram /
norm dictionary — below); the **HNM score-structure** attack (the √N
channel) as a second `NO-PLAINTEXT` gate-2; wider σ sweep.

**Design implication (`WEIGHTS-BLIND` only — see the `WEIGHTS-PUB` result
two subsections below before drawing conclusions).** The decode permutation
cover (`perm_kv` + `O_qk` + `O_v` + σ) looks **substantially viable under
`WEIGHTS-BLIND`** — content and position both largely hidden. **But this
does not survive the conservative bar:** under `WEIGHTS-PUB` the covariance
form clears (next subsection) yet the token-dictionary attack recovers
**membership** (perfect, *Gate-3 @ `WEIGHTS-PUB` — decode*); only
order-hiding (π) and the K-path (σ) survive `WEIGHTS-PUB`. Read this
paragraph as the `WEIGHTS-BLIND` snapshot, not the final word.

### Gate-3 @ `WEIGHTS-PUB` — covariance-alignment rerun (2026-06-01)

The updated gate-3 (`evals/aloepri-attacks/gate3_weights_anchor.py`, run in
the `gelo-attack` container) gives the `WEIGHTS-PUB` attacker the model's
**population value-covariance** as an anchor and eigen-aligns it to the
observed `Cov(v_sent)` to recover `O_v` — the attack the blind FastICA/JADE
never ran. The anchor is prompt **B**'s clean V (`captures2`, a *different*
prompt — a public-data population proxy, `NO-PLAINTEXT`-faithful); the
target is prompt **A** (`captures`); scored on the shared layers 0 & 35.

| layer | no-attack | covalign (`WEIGHTS-PUB`, pop. anchor) | covalign (self-anchor*) | `O_v` |
|---|--:|--:|--:|---|
| 0  | 0.334 | **0.329** | 1.000 | **HOLDS** |
| 35 | 0.327 | **0.326** | 1.000 | **HOLDS** |

**`O_v` HOLDS under `WEIGHTS-PUB` covariance-alignment.** Knowing the
population covariance does **not** pin the per-session `O_v`: recovery sits
at the no-attack floor. The mechanism is that the value covariance is
**instance-specific** — a different prompt's eigenvectors don't align to the
target's, so `Cov(v_sent)=O_vᵀ·Cov(V)·O_v` can't be solved without the
target's *own* `Cov(V)`. The self-anchor column (target's own clean
covariance) recovers `O_v` at **1.000**, which is the *method validation* —
it proves the attack works given a matching covariance, so the population
result is a real negative, not a broken attack. (*self-anchor uses the
instance's own clean second moments — an upper bound, not a `WEIGHTS-PUB`
capability.)

**Two caveats this rerun does not close:**
1. **It tests the covariance/2nd-moment attack, not the token-identity Gram
   dictionary.** A `WEIGHTS-PUB` attacker could instead match the
   `O_v`-invariant V-Gram against a per-token-pair dictionary
   `D[t_i,t_j]=e_{t_i}(W_V W_Vᵀ)e_{t_j}ᵀ` (needs `W_V`+embeddings+token-ids
   dumped). For **decode** the permutation independently caps this at the
   token *multiset* (order stays hidden); for **prefill** (no permutation)
   it is the sharper, still-open Phase-5 vector.
2. **Measured at n_kv≈500.** As context grows the instance covariance
   approaches the population, so covariance-alignment may strengthen at
   production lengths (2k–16k) — recheck there before relying on this.

### Phase-5 spike — prefill rotation cover BREAKS under `WEIGHTS-PUB` (2026-06-01)

This is the token-identity Gram/norm-dictionary attack caveat #1 named — run
against the **rotation-only prefill cover** (the cover the fused kernel
forces). Capture: `GELO_CAPTURE_COVER=rotation` in `attn_cover_capture.rs`
(feature rotation only, no perm, σ=0) plus a layer-0 value **dictionary**
`V(t)=rms_norm(embed(t))·W_V` built from the public weights (context-free at
layer 0). Attack (`evals/aloepri-attacks/gate3_prefill_dict.py`, in the
container): `O_v` is per-head orthogonal, so the per-head value **norm**
`‖v_sent[p,h]‖=‖V(token_p)[h]‖` is `O_v`-invariant — match each (publicly
ordered) position's `H`-head norm fingerprint to the nearest dictionary
token.

| run | n positions | K dict | dict faithfulness | top-1 | median rank | `O_v` |
|---|--:|--:|--:|--:|--:|---|
| short prompt | 26 | 1 224 | rel-err 0.00 | **1.000** | 0 | **BROKEN** |
| longer prompt | 126 | 8 091 | rel-err 0.00 | **1.000** | 0 | **BROKEN** |

**Every prompt token is recovered perfectly** (top-1 = 1.000, median rank 0
vs chance ~1e-4) from the per-head value norms **alone** — the *cheapest*
`O_v`-invariant; the full Gram is strictly stronger, so this is a lower
bound. The faithfulness check (dictionary `V(t)` vs the captured `v_clean`
on the prompt tokens, rel-err exactly 0) confirms the dictionary is the
model's true layer-0 value map, so the result is valid, not an artifact.

**Conclusion — rotation-only prefill offload does NOT clear the conservative
(`WEIGHTS-PUB`) bar.** It confirms the first-principles prediction: the
feature-rotation cover the fused kernel forces is defeated by a
weights-equipped adversary, because the rotation cancels in every inner
product / norm and the public weights turn those invariants into a per-token
dictionary. Layer 0 suffices (its `V` is the input prompt). Combined with
the decode covariance result, the spike lands at **pass-blind / fail-public**
→ the next move is **covariant weight obfuscation (AloePri)**; absent a
working obfuscation, **prefill stays in-TEE** and Phase 6 does not ship.

*Caveats:* layer-0 only (deeper layers' `V` is contextual — untested, but
layer-0 break already exposes the prompt); candidate pool 8 091, not the
full 152 k vocab (median rank 0 implies it would survive full-vocab, but
top-1 may dip with more collisions). Artefacts: rotation-only mode in
`attn_cover_capture.rs`, `gate3_prefill_dict.py`, `captures_prefill/`.

### Gate-3 @ `WEIGHTS-PUB` — decode permutation cover: membership leaks, order holds (2026-06-01)

The same token-dictionary attack run against the **decode permutation cover**
(default capture: `perm_kv` + σ(K) + `O_qk` + `O_v`; `gate3_prefill_dict.py`
with `GELO_GATE_CAPDIR=captures_decode`), scoring per **physical slot** (the
true token at slot `i` is `input_ids[perm_kv[i]]`):

| cover | n slots | K dict | dict faithfulness | per-slot top-1 | median rank |
|---|--:|--:|--:|--:|--:|
| decode (perm + σ + `O_qk` + `O_v`) | 126 | 8 091 | rel-err 0.00 | **1.000** | 0 |

**Token membership leaks; order does not.** The per-head value norm is
`O_v`-invariant *and* the permutation merely relabels rows, so every slot's
**token identity** is recovered perfectly (top-1 = 1.000) — i.e. the
**bag-of-tokens (multiset)** of the context is exposed under `WEIGHTS-PUB`,
exactly as for prefill. What the permutation **does** protect is **order**:
recovering *which position* a slot maps to is the separate gate-2 seriation
channel, measured weak (|τ| ≈ 0.1). σ on K is irrelevant here (the V path
carries no σ).

This **coexists with the covariance-alignment result** above: `O_v` is not
*pinned* (covariance alignment failed → "O_v holds"), but the dictionary
attack never tries to — it rides the `O_v`-invariant. So the precise decode
statement under the conservative bar is: **`O_v` hides value *coordinates*,
`perm_kv` hides *order*, but neither hides *which tokens are present* once the
weights are public.**

#### Why it leaks — and a minimal recovery attack (step by step)

The resident value cache is covered with three secrets — a row permutation
`π`, a per-head orthogonal rotation `O_v`, and σ-noise — yet **none of them
perturbs the per-head value *norm***, and that norm is a public per-token
quantity. Write the covered cache (per head `h`, physical slot `i`):

```
v_sent[i,h] = V[π(i), h] · O_v[h]            (σ is on K only — the V path is clean)
```

**What each secret protects, and why the norm slips through all three:**
- `O_v[h]` is **orthogonal** ⇒ it preserves length: `‖x·O_v[h]‖ = ‖x‖`. It
  hides the value *direction* (coordinates), not its *norm*.
- `π` only **relabels** slots; the value at a slot is still some token's value,
  so a per-slot quantity remains a per-token quantity. `π` hides *order*, not
  *identity*.
- σ-noise is applied to **K**, never to **V** (the `probs·V` contraction over
  the token axis makes additive V-noise uncorrectable — see the fallback
  section), so the V fingerprint carries no noise at all.

⇒ the H-vector `r_i = (‖v_sent[i,0]‖, …, ‖v_sent[i,H-1]‖)` equals
`(‖V[π(i),0]‖, …)` — a clean fingerprint of the token in slot `i`, free of all
three secrets. The only remaining unknown is the map token → fingerprint, and
`WEIGHTS-PUB` supplies it.

**Minimal recovery attack** (uses only the per-head norm and public `W_V` —
no `π`, no `O_v`, no optimisation):

1. **Fingerprint each slot.** For every physical slot `i`, compute
   `r_i[h] = ‖v_sent[i, h·d:(h+1)·d]‖` (`H` scalars). `O_v`-invariant by the
   orthogonality above; `π`-relabelled only; σ-free.
2. **Build the dictionary from public weights (layer 0, context-free).** For
   every vocabulary token `t`, compute its layer-0 value
   `V(t) = rms_norm(E[t], γ₀)·W_V` and its norm fingerprint
   `D[t] = (‖V(t)[0]‖, …, ‖V(t)[H-1]‖)`. Context-free because at layer 0 the
   V path has no attention/positional dependence (RoPE is Q/K-only), so a
   single value projection over the embedding table suffices — no forward.
3. **Nearest-fingerprint match.** Assign slot `i` the token
   `t*_i = argmin_t ‖r_i − D[t]‖` (per-head-standardised L2).
4. **Read membership.** `{ t*_i }` is the recovered **multiset** of context
   tokens. Order is not recovered (slot↔position is `π`); coordinates are not
   recovered (`O_v`) — but the bag-of-tokens is out.

Measured: per-slot top-1 = 1.000 (n=126, K=8091) — perfect — from `H=8`
scalars per token. **Escalation if fingerprints collide:** the full per-head
Gram `G[i,j] = v_sent[i,h]·v_sent[j,h]ᵀ = V[π(i),h]·V[π(j),h]ᵀ` is also
`O_v`-invariant and disambiguates via a quadratic-assignment match against the
dictionary's cross-Gram — strictly stronger than norms, and unneeded here. The
**only** step that depends on a secret-the-defender-controls is step 2's public
`W_V`; obfuscating it (AloePri, Phase 5b) is the sole lever that closes the
attack.

**Decode finalization (`WEIGHTS-PUB`).** The offloaded decode cover protects
sequence order, not membership. Whether a bag-of-tokens leak is acceptable is
a deployment/policy call: the adversary learns the set of tokens in the
context window (a bag-of-words view of prompt + generation), not their
arrangement. If membership-hiding is required, the only lever that closes it
is the same as prefill — **covariant weight obfuscation (AloePri, Phase 5b)**,
which removes the public-weight dictionary. Absent that, decode offload ships
**only if the bag-of-tokens residual is accepted** (gate it at the c5 AloePri
acceptance condition). Artefact: `captures_decode/`.

#### Alternative mitigation — decoy tokens / k-anonymity (NOT planned)

Recorded as the one cover-independent alternative to covariant obfuscation,
**not on the roadmap.** Since no correctable *cover* can hide membership
without being a covariant representation (the impossibility above), the only
other way to blunt the bag-of-tokens leak is to **obscure rather than
eliminate** it: pad the resident K/V cache with **decoy ("chaff") token
rows**, permuted in among the real ones by `perm_kv`. The norm/Gram-dictionary
attack then recovers the token identity of *every* slot — so the adversary
gets `real ∪ decoy`, and each real token hides among the decoys. This is a
**k-anonymity / plausible-deniability** guarantee (≈ "the prompt is some
size-`m` subset of these `m·(1+r)` tokens"), **not** elimination — the real
tokens are still recovered, just not isolated.

**Why it stays an alternative, not the plan:**
- **Statistical, not structural.** It weakens an attacker's *certainty*, not
  their *recovery*. A determined attacker with content/language priors can
  re-rank real vs decoy (decoys drawn from a flat vocab distribution stand out;
  realistic decoys are themselves expensive to generate).
- **Cost hits exactly what the offload optimises.** Per-step decode attention
  and the resident cache scale with `(1+r)·n` — the chaff ratio `r` directly
  inflates the V-cache reads the offload exists to make cheap, and worsens the
  NVMe-spill economics.
- **Correctly excluding decoys without revealing them is an open ORAM-flavoured
  problem.** Decoys must not perturb the true `P·V`, yet masking their scores
  to `−∞` makes their softmax weight exactly zero — an observable zero/score
  pattern that re-identifies them, defeating the hiding. A construction that
  both neutralises decoys *and* keeps them indistinguishable is non-trivial and
  unbudgeted.

⇒ **a covariant obfuscation remains the planned lever** (it closes the leak
*and* keeps the offload); decoys are noted only as the fallback shape if
obfuscation proves unviable *and* the bag-of-tokens residual is judged
unacceptable. **⚠ Refined (2026-06-02):** the earlier shorthand "covariant
**weight** obfuscation (AloePri)" is *not* the minimal lever — adapting AloePri
to GELO's split-inference architecture narrows it to a **per-session
non-orthogonal value cover**, applied in activation space, not a static weight
transform. See *Adapting covariant obfuscation* immediately below; it supersedes
the weight-space framing wherever the two conflict.

## Adapting covariant obfuscation to the offload: the dynamic↔static spectrum and the minimal lever (2026-06-02)

The Phase-5 spike lands the verdict that *some* covariant obfuscation is
required to neutralise the `WEIGHTS-PUB` token-membership leak. This section
resolves **which** — and corrects the earlier shorthand that the lever is
"covariant **weight** obfuscation (AloePri)." The precise lever is narrower, and
it sits at the **dynamic** end of an obfuscation spectrum rather than importing
AloePri's static construction wholesale.

**The spectrum.** GELO's covers and AloePri's obfuscation are two ends of one
axis — *how often the secret is refreshed, and whether it lives in the
activation or the weight*:

| | Fully dynamic (GELO) | Fully static (AloePri) |
|---|---|---|
| Secret lives in | the per-op **activation** cover | the deployed **weights** |
| Refresh cadence | per session (prefill: per-prefill; decode: session-fixed `O_v`, per-block `perm_kv`) | fixed at deployment, never |
| Runtime cost | a re-cover per refresh | zero (covariant, folds into the served weights) |
| Security rests on | the secret resetting before enough is observed | the obfuscated weights being **hard to invert** (VMA-resistance) |
| Adversary assumption | worst-case `GPU-ADV` (accumulates across sessions) | *constrained* attacker (AloePri's stated scope) |

The question for this junction: **what is the minimal covariant addition, kept
near the dynamic end, that closes the offload leak without (a) deployment
complexity or (b) accuracy loss?** Two findings constrain the answer; one of
them corrects a claim made mid-design (that the per-session orthogonal `O_v`
supplies the accumulation defence — it does not).

### Design constraints (what any solution at this junction must satisfy)

- **C1 — resident-clear weights.** The GPU holds the offloaded projection
  weights **resident, in the clear**: it computes `(A·X)·W` with `W` registered
  once at load (`gelo-gpu-wgpu/src/lib.rs::register_weight{,_bf16}`) and the
  per-call cover masking *only the activation* (`U = A·X`, fresh orthogonal
  row-mask + shield rows; `gelo-protocol/src/mask.rs`). This is a GELO property,
  not an artifact: under `WEIGHTS-PUB` the weights are public, so GELO never hid
  them — its confidentiality protects the *user's activations*, not the model.
  ⇒ any obfuscation baked into a *resident weight* `W' = f(W)` must be
  **non-invertible from public `W`**.
- **C2 — the leak is the un-mixed attention operand, not the projection
  matmul.** The norm/Gram dictionary leak lives **only** where the GPU sees
  per-token-row *un-mixed* value vectors — the `cubek` attention path (`V·O_v`
  uploaded per row). The projection output `A·V` is row-mixed by the orthogonal
  mask (per-token norms are not readable off it; un-mixing needs the fresh
  secret `A`). ⇒ the fix need only act on the attention operands, not the
  weights.
- **C3 — no negative accuracy (user constraint b).** Greedy-token parity must
  hold (today: byte-identical at σ=0). The fix must avoid AloePri's three
  accuracy sinks: RoPE-block permutation (approximate score), embedding/head
  noise, and ill-conditioned key matrices (fp16 cancellation).
- **C4 — no deployment complexity (user constraint a).** No whole-model weight
  rewrite + redeploy, no offline obfuscation pipeline, no secret-vocab-mapping
  I/O layer, no RMSNorm-fusion / router-permutation. A localized change to the
  attention cover, not a new deployment artifact.
- **C5 — correctable through the fused kernel + existing covers.** Composes with
  `cubek`'s normalised output via a TEE-side linear post-correction; never
  touches the token axis / causal mask (no π-leak); composes with
  `O_qk`/`O_v`/`perm_kv`/σ/tail-in-TEE; RoPE-compatible where it touches Q/K.
- **C6 — bounded fp16 conditioning.** A non-orthogonal transform separated from
  its inverse by the fp16 GPU attention incurs cancellation error ∝ `cond`;
  bound it (the κ dial).
- **C7 — accumulation / freshness.** The secret must resist a worst-case
  `GPU-ADV` accumulating observations *across sessions*. A deployment-static
  secret faces unbounded accumulation; a per-session-fresh secret resets.
- **C8 — break the norm *and* the Gram.** The demonstrated attack used per-head
  norms (the cheapest `O_v`-invariant); the per-head Gram is strictly stronger.
  The fix must distort the Gram form `W_v(·)W_vᵀ`, not merely rescale norms (a
  per-head scalar is erased by the attack's per-head standardisation).

### Finding 1 — weight-space obfuscation cannot be *minimal* (the resident-clear-weight constraint)

If we register an obfuscated value weight `W̃_v = W_V·Ũ_vo` resident on the GPU,
a `WEIGHTS-PUB` adversary recovers the secret in one step: `W_V` is public and
full column rank `[d_model, d_head]`, so `W_V⁺·W̃_v = (W_V⁺W_V)·Ũ_vo = Ũ_vo`. A
*bare* covariant weight transform is **trivially invertible when resident** (C1
fails).

The fix is not cheap. Making a resident `W'` non-invertible from public `W`
requires a *left*, cross-layer-entangled key matrix (`W̃_v = Q̂_v·W_v·Ũ_vo`,
with `Q̂_v` constrained by the cross-layer cancellation `P̃·Q̃ = I` of adjacent
layers) and/or the vocabulary permutation `Π`; separating those is exactly
AloePri's VMA problem, which is only defended by the full Algorithm-1 key-matrix
machinery + head/block permutation (AloePri Table 4: bare noise → 40 % recovery;
+KeyMat → 0.82 %; +KeyMat+perm → 0.0 %). **So "safe-when-resident weight
obfuscation" ⇒ (most of) whole-model AloePri** — which violates both C3
(accuracy budget) and C4 (deployment artifact). The resident-clear-weight
property is *precisely why AloePri is a whole-model construction and not a
one-weight patch*; there is no minimal weight-space option.

### Finding 2 — an *orthogonal* refresh does not freshen the attack surface

The attack surface is the per-head value **Gram** `G[i,j] = (V_i C)(V_j C)ᵀ =
V_i (C Cᵀ) V_jᵀ` (and its diagonal, the norm `r_i = √(V_i (CCᵀ) V_iᵀ)`), where
`C` is the value cover. Everything the dictionary attack needs is a function of
**`C Cᵀ`**.

For an orthogonal cover, `C Cᵀ = I` ⇒ `G = V_iV_jᵀ`, the clean Gram, matched by
the public-weight dictionary. **Now refresh the cover with a fresh orthogonal
factor**, `C = C_0·Q` (`Q` Haar per session): `C Cᵀ = C_0 Q Qᵀ C_0ᵀ = C_0 C_0ᵀ`
— **independent of `Q`**. The orthogonal refresh leaves `C Cᵀ` fixed, so it does
**not** change what the attacker sees. *(This corrects a claim made mid-design:
the per-session `O_v` does not supply the accumulation defence, because `O_v` is
orthogonal and the Gram is orthogonal-invariant — it is the very invariance the
attack exploits.)*

Consequences:
- **A static non-orthogonal cover is accumulation-broken.** If `C Cᵀ = M` is
  fixed at deployment, the adversary accumulates `{V_i(session s)·C}` across
  sessions; with public `W_V` each observation is `√(V_i M_h V_iᵀ)` for some
  token, and `M_h` (`d_head(d_head+1)/2` params/head) is massively
  over-determined by thousands of distinct tokens seen across sessions → `M`
  becomes identifiable → reduces back to the public dictionary. (This is the
  risk AloePri accepts under its *constrained-attacker* scope and its
  VMA-hardness; GELO's worst-case `GPU-ADV` does not get to accept it.)
- **Freshness must come from the *non-orthogonal* part.** Only a value cover
  whose `C Cᵀ` **changes per session** denies the accumulation target. A
  per-session-fresh, well-conditioned *invertible* `C_v` (fresh singular values
  **and** vectors) gives a fresh `C_v C_vᵀ` each session ⇒ accumulation-safe at
  session granularity.

### Options considered (against C1–C8)

| # | Option | Verdict |
|---|---|---|
| **A** | **Per-session non-orthogonal value cover `C_v`** — generalise `O_v` from orthogonal to κ-bounded invertible, sampled per session, applied where `O_v` is today, corrected by `C_v⁻¹`. | **✓ all.** The lever — see below. |
| B | Bare static weight `Ũ_vo` (resident `W̃_v = W_V·Ũ_vo`). | ✗ C1 — `W_V⁺W̃_v` inverts it (Finding 1). |
| C | Full AloePri weight obfuscation (Alg.1 key matrices + `Π` + block/head perm + noise). | ✗ C3 (accuracy), ✗ C4 (whole-model redeploy). The static end of the spectrum. |
| D | Additive value blinding (TwinShield-style on V): `V+R` up, subtract after. | ✗ C5 — additive V-noise is uncorrectable through the `probs·V` token-axis contraction (the TEE never sees the softmax weights; cf. the fallback section). |
| E | V-norm flattening — upload unit-norm V + a per-row scale corrected TEE-side. | ✗ C5 — the per-key scale lives *inside* the `Σ_j P_j V_j` contraction; it cannot be factored out post-hoc. |
| F | Cross-head orthogonal mixing (rotate the full `H·d_head` value, scrambling per-head norms). | ✗ C5 — attention is per-head (`P[h]·V[h]`); a cross-head mix breaks the per-head/GQA contraction and cannot be un-mixed after the per-head softmax. |
| G | Static non-orthogonal activation cover `C_v` (fixed at deployment) **under** a fresh per-session `O_v`. | ✗ C7 — by Finding 2 the Gram is `C_v C_vᵀ` (fixed) and `O_v`-invariant; the orthogonal refresh adds no freshness → accumulation-broken. Dominated by A. |
| H | `C_v` refreshed **per block** (finer than per-session). | ✓ security, ✗ minimality — adds the per-block re-rotate (the dominant re-cover term, gate 1) for no gain over per-session, since the protected secret is one session's prompt. Dominated by A. |

### Conclusion — the minimal lever

**The minimal covariant addition is Option A: generalise the value cover from an
orthogonal `O_v` to a per-session, κ-bounded, *non-orthogonal* invertible
`C_v`** — AloePri's value/output covariant pair (`Ũ_vo` / `Ũ_vo⁻¹`), **relocated
from static weight-space to GELO's existing dynamic activation-space cover.**
Concretely:

- **Construction:** `C_v = U·diag(s)·Vᵀ`, Haar `U,V`, singular values in
  `[1/√κ, √κ]` (within-head mixing ⇒ distorts norm *and* Gram, C8;
  `cond ≤ κ` ⇒ bounded fp16 error + no overflow, C6). Sampled **fresh per head
  per session** (fresh `C_v C_vᵀ` ⇒ accumulation-safe, C7/Finding 2).
- **Application:** replaces `O_v` at `rotate_heads`; correction `C_v⁻¹ =
  V·diag(1/s)·Uᵀ` (precomputed once/session) replaces `O_vᵀ` in
  `correct_unfold_into` (prefill) and `acc_uncover_tee` (decode). Decode's
  resident prefix stores `V·C_v`; the in-TEE tail attends plaintext V and the
  merge happens *after* un-covering — unchanged structure (C5).
- **No weight is touched** ⇒ resident weights stay public-and-fine (C1), no
  deployment artifact (C4). **Greedy parity at the chosen κ** (C3, to be
  measured, not assumed — honest "parity at κ", not byte-identical). **Perf
  unchanged** — same op as `O_v`, the only delta is `C_v⁻¹` is not a transpose
  (a one-off per-session inverse; the 2.83× prefill / ~4× decode hold; if κ
  conditioning demands, the in-TEE correction may run in f32 — minor).
- **QK side deferred (the conditional second addition).** `O_qk` must stay
  *orthogonal* (it cancels in the score). The K/Q self-Gram anchor would need
  the `Ĥ_qk` post-qk-norm scaling trick; qk_norm already flattens the cheap
  K-*norm* signal and the self-Gram reconstruction is unmeasured, so `Ĥ_qk`
  stays deferred until a measured attack shows it bites.

**The dynamic↔static hyperparameter, made precise:** it is the **refresh
granularity of the non-orthogonal value transform `C_v`** (equivalently, how
often `C_v C_vᵀ` resets). GELO sits at **per-session** — accumulation-safe and
free of a deployment-static secret. The static (AloePri) end only buys runtime
savings by moving into weight-space, which Finding 1 shows demands the full
VMA-resistant machinery (and forfeits C3/C4). Between them there is no cheaper
midpoint, because the per-session activation re-sample is already ~free (it is
the same op `O_v` already pays). So for the offload, **the minimal-cost secure
point and the dynamic end coincide.**

**Why this is "covariant-obfuscation aligned" and not just "a different cover."**
`C_v` is exactly AloePri's value covariant transform (`V·Ũ_vo` undone by
`Ũ_vo⁻¹` folded into the output path) — the same covariance principle
(computation preserved by an invertible value-basis change), with the single
adaptation that GELO applies it *dynamically in activation space* (where its
secrecy is information-theoretic, C2) instead of *statically in weight space*
(where, per Finding 1, it would be trivially invertible).

### The decisive gate (cheap, unchanged in spirit)

Add a `C_v` capture mode to `attn_cover_capture.rs` (κ-bounded SVD value cover,
`O_qk` orthogonal as today) and re-run `gate3_prefill_dict.py` (container),
**sweeping κ**: accept the smallest κ that drops per-slot top-1 to the chance
floor (≈1/|vocab|) **and** holds greedy parity, run against *both* the per-head
**norm** fingerprint and the strictly-stronger per-head **Gram** (C8). Confirm
at the full-vocab candidate pool (the dev-log's standing caveat) and at a deeper
contextual layer, not only layer 0. **Pass ⇒ flip both offload paths default-on
behind the c5 condition (now read as "`C_v` cover," not "AloePri weights"). Fail
at every κ that holds parity ⇒ escalate to the QK-side `Ĥ_qk` addition, then
re-gate.**

### Stage-1 result — `C_v` breaks the norm dictionary, min κ ≈ 5 (2026-06-02)

Ran the cheap screen. `attn_cover_capture.rs` gained a `cv` cover mode
(`GELO_CAPTURE_COVER=cv`, `GELO_CAPTURE_KAPPA`): prefill structure (identity
perm, σ=0, `O_qk` orthogonal) with the value cover replaced by the κ-bounded
non-orthogonal `C_v = U·diag(s)·Vᵀ` (`sample_kappa_bounded_invertible`, Haar
`U,V`, singular values pinned to give `cond = κ` exactly, geometric mean ≈ 1).
Real Qwen3-4B, layer 0, n_kv=64, CPU reference capture; `gate3_prefill_dict.py`
norm attack, K=8054 dictionary (54 prompt-unique + 8000 distractors), in the
`gelo-attack` container. Hold bar: per-slot top-1 ≤ 0.02.

| κ (=cond `C_v`) | top-1 | top-5 | median rank (/8054) | verdict |
|---:|---:|---:|---:|---|
| 1.0 (orthogonal control) | **1.000** | 1.000 | 0 | LEAKS — reproduces the `O_v` baseline ✓ |
| 1.5 | 0.734 | 0.906 | 0 | LEAKS |
| 2.0 | 0.375 | 0.453 | 18 | LEAKS |
| 3.0 | 0.219 | 0.312 | 252 | LEAKS |
| 4.0 | 0.062 | 0.219 | 531 | LEAKS |
| **5.0** | **0.016** | 0.156 | 870 | **HOLDS** (min κ) |
| 6.0 | 0.000 | 0.156 | 1118 | HOLDS |
| 7.0 | 0.000 | 0.109 | 1258 | HOLDS |
| 8.0 | 0.000 | 0.047 | 1382 | HOLDS |

**Reading.** Recovery falls **monotonically** in κ; the orthogonal control
(κ=1) recovers perfectly, confirming the harness *and* that the leak is exactly
`O_v`'s orthogonality (norm-preservation). The dictionary dies (top-1 ≤ 0.02) at
**κ ≈ 5**; κ=6 gives top-1 = 0.000 at margin (the safe operating point). Dict
faithfulness rel-err = 0 throughout (the dictionary is the true layer-0 value
map — the negative is real, not a broken attack).

**Caveat carried to Stage 2.** top-5 **plateaus at ~0.11–0.16 for κ ∈ [5,8]** —
it does *not* vanish with κ. The per-head norm fingerprint collapses to a small
ambiguity set (~5 candidates) ~15% of the time even when top-1 is zero; raising
κ past ~6 buys little there. So κ is **not** the lever for the residual top-k —
the bare-norm fingerprint saturates. Stage 2 must check whether (a) the full
per-head **Gram** / covariance-Procrustes attacks exploit that residual, and (b)
top-5 falls at the full 152k vocab (the 8k pool inflates collision rates).
**κ ≈ 6 is the Stage-1 recommendation**, pending the Stage-2 + greedy-parity
(fp16-conditioning) checks — `cond=6` ⇒ fp16 round-trip cancellation ≈ 2⁻¹⁰·6 ≈
6e-3 relative, which the parity bench on the real offload path must confirm is
below the greedy-argmax margin.

Artefacts: `GELO_CAPTURE_COVER=cv` mode in `attn_cover_capture.rs`;
`evals/aloepri-attacks/captures_cv_k{1.0…8.0}/`.

### Stage-2 result — full-vocab + covariance/Procrustes: `C_v` HOLDS at κ=6 (2026-06-02)

The conservative-bar confirmation at κ=6.

**(a) Full-vocab norm dictionary** (151 936-token pool via `GELO_CAPTURE_DICT_FULL=1`,
layer 0). The full vocabulary *resolves the Stage-1 top-5 caveat* — it was a
small-pool collision artifact:

| metric | 8k pool (Stage 1) | full vocab (Stage 2) |
|---|--:|--:|
| top-1 | 0.000 | **0.000** |
| top-5 | 0.156 | **0.047** |
| median true-token rank | 1118 / 8 054 | **20 203 / 151 936** (~13th pctile) |

**(b) Covariance/Procrustes recoverability** (`gate3_cv_covalign.py` — the
orthogonal `gate3_weights_anchor` generalised to a non-orthogonal map:
whiten `v_sent`, re-colour to a population anchor's spectrum, 3rd-moment sign;
`D = Vs·diag(√(La/Ls))·Vaᵀ`). Anchor = prompt B (`captures2`), a different
prompt; scored by Hungarian-matched |corr| of de-covered vs clean V. Layers 0 & 35,
n_kv=285, no-attack floor ≈ 0.37:

| target | no_attack | covalign(pub) | covalign(self\*) | verdict |
|---|--:|--:|--:|---|
| **κ=6 (`C_v`)** | 0.370 | **0.332** | **0.359** | **HOLDS** |
| κ=1 (orthogonal control) | 0.372 | 0.344 | 1.000 | HOLDS(pub) |

The κ=1 control's **self\* = 1.000 validates the attack** (it recovers an
orthogonal cover given matching covariance — so the negatives are real) and
reproduces the original `O_v`-holds-under-`WEIGHTS-PUB` result. At κ=6, `C_v`
holds against the population anchor (0.332 ≈ floor) **and at the self-anchor
upper bound (0.359, *not* 1.000)** — *stronger* than `O_v`: a non-orthogonal
`C_v` is not recoverable from 2nd-moment alignment even with the instance's own
covariance, because whitening fixes only the symmetric part `C_vC_vᵀ` and leaves
the orthogonal factor unresolved. The **per-head Gram quadratic-assignment is
subsumed** (covariance is the centered Gram; recovering the cover from 2nd-order
statistics is exactly what failed). Layer 35 holds; deeper layers have no
context-free dictionary regardless.

**Stage-2 verdict: `C_v` at κ=6 clears the conservative bar** — membership
broken (full-vocab top-1=0, top-5=0.047) and the cover non-recoverable
(covariance-alignment at the floor, validated). **Remaining before default-on:**
the **fp16 greedy-parity check** at κ=6 on the *real* offload path (acceptance
tier-3; `cond=6` ⇒ ~6e-3 round-trip cancellation — confirm below the
greedy-argmax margin), then the production wire-up (`O_v`→`C_v` at the
`rotate_heads`/`acc_uncover`/`correct_unfold_into` sites, per *Phase 5b*).
Artefacts: `gate3_cv_covalign.py`, `GELO_CAPTURE_DICT_FULL` mode;
`evals/aloepri-attacks/captures_cv_{k6_fullvocab,covalign_k6.0,covalign_k1.0}/`.

## Offload perf-upside — per-op breakdowns (2026-06-01)

These measure the **performance** of the offloaded attention with its cover
applied end-to-end, *decoupled from the security verdict*: covariant weight
obfuscation (Phase 5b) is a static weight transform that does not change these
per-step costs, so this is "what the offload buys once it is secured." All on
RTX 5090 / Vulkan, Qwen3-4B (Hq=32, Hkv=8, d=128), B=8.

### Decode — permuted-cover, secure (full wire)

`perm_kv` + σ on K + `O_qk` on Q/K + `O_v` on V + **tail-in-TEE** (frozen-prefix
partial-stats on GPU + in-TEE tail + online merge), session-fixed (no per-block
re-permute). Wired end-to-end behind `GELO_GPU_RESIDENT_COVER`; greedy-parity
**byte-identical** at σ=0. Bench: n=2048, K=32 decode steps, σ=0.01.

**Offloaded decode vs full in-TEE attention (B=8, n=2048).** The in-TEE
baseline is `tee:attn_cached_inplace_many` = **14 574 ms** (= 12.65 ms per
(layer,step) → 455 ms/step over 36 layers; chronicle §9). The offload replaces
that single CPU bucket:

| metric (B=8, n=2048) | full in-TEE | pre-O4 | O4(a) | **O5** |
|---|--:|--:|--:|--:|
| decode attn bucket @ K=32 | **14 574 ms** | 13 832 (1.05×) | 9 283 (1.57×) | **3 636 (~4×)** |
| recurring per-step (36 layers) | **455 ms** | 157 (2.9×) | 152 (3.0×) | **~114 (4×)** |
| one-time `build_covered_prefix` | — | 8 765 (decode) | 4 428 (decode) | **→ prefill (5 446)** |
| decode break-even K | — | ≈30 | ≈15 | **none (relocated)** |

**O4(a) — ✅ LANDED (2026-06-01).** The `build_covered_prefix` perm+σ step was a serial
`bh·prefix·dh` (≈16.7 M) scalar loop drawing a `StandardNormal` *per element*;
replaced with a row-level permutation gather (attention is permutation-invariant
over the key set, so σ=0 stays exact) + per-head σ-on-K noise, parallelised over
the B·nkvh heads. `build_covered_prefix` **8 765 → 4 428 ms** (≈halved), bucket
**13.8 → 9.3 s = 1.57×**, break-even **K≈30 → ≈15**. The per-step path is a clean
~3× win. What remains in `build_covered_prefix` is the **dense `O(d²)` rotate**
(`rotate_heads`) + the SIMD convert/upload.

**O5 — ✅ LANDED (2026-06-01).** `build_covered_prefix` is now built for all
GLOBAL layers at the **prefill→decode handoff** (`build_covered_prefix_all_global`
at the end of `run_prefill_batched`, gated on the decode-cover path; the decode
block keeps an idempotent lazy-build fallback), instead of lazily on the first
decode step. The one-time build (5 446 ms ×36 layers) **moves out of the decode
bucket into prefill**: decode `tee:attn_resident_cover` **9.3 → 3.6 s (~4× vs
in-TEE 14.6 s)**, **recurring-only, no break-even K**. This is a *relocation*,
not a speedup — total work is unchanged — but it is the correct structure
(decode steps shouldn't pay a one-time setup) and it clears the ≥30%
decode-wall acceptance tier outright. (The build overlaps the offloaded prefill;
prefill attention itself is the separately-offloaded `tee:attn_prefill_offload`.)
Per-op decomposition (pre-O4 numbers below; build_covered_prefix/bucket per the
table above):

> **Structured-orthogonal cover — tried, reverted (2026-06-01).** Two
> replacements for the dense Haar rotate were considered. **(O4b) signed
> permutation** (`O(L·d)`) was deferred: it collapses the cover's entropy and
> would forfeit the `WEIGHTS-BLIND` content-hiding (gate-3 "`O_v` holds" assumes
> a dense rotation). **HD₃** (FWHT cascade — well-mixed, no entropy collapse) was
> implemented end-to-end (decode + prefill, behind a parity test) but **measured
> a regression and was reverted**: applied per-`d`-block through the existing
> `Hd3Mask::apply_in_place_slice` API, the per-call overhead loses to the BLAS
> dense matmul at our shapes — prefill `rotate_tee` 4.2 s → 6.6 s, `correct_tee`
> 2.6 s → 3.7 s (pure-CPU buckets, so attributable), prefill bucket 2.71× →
> 2.29×; decode was ~neutral (small per-step `q_cover`/`acc_uncover` wins offset
> by a convert/upload-bound `build_covered_prefix` that the rotate no longer
> dominates). Realising HD₃'s `O(d·log d)` advantage needs a **batched
> feature-axis FWHT** (transpose to feature-major + the cols-vectorised kernel),
> whose transposes likely erode the win at `d=128` where BLAS is already
> efficient — left as a deferred spike, not the current lever. The cover stays
> dense Haar.

| op | where | total ms | × executed | per-call | count meaning |
|---|---|--:|--:|--:|---|
| `build_covered_prefix+upload` | TEE+GPU | 8 765 | **36** | 243.5 ms | once per layer (first decode step) — the session-fixed prefix re-cover |
| `prefix_partial_gpu` | **GPU** | 3 399 | **1 152** | 2.95 ms | every layer × step (36×32) — partial-stats attend over the frozen prefix |
| `q_cover_tee` | TEE | 384 | **1 152** | 0.33 ms | every layer × step — `q·O_qk` + σ |
| `acc_uncover_tee` | TEE | 210 | **1 152** | 0.18 ms | every layer × step — `acc·O_vᵀ` |
| `tail_partial_tee` | TEE | 589 | **1 116** | 0.53 ms | 36×31 — skips the create step (tail empty) — in-TEE tail attend |
| `tail_build_tee` | TEE | 436 | **1 116** | 0.39 ms | 36×31 — GQA-expand the tail (scalar) |
| `merge_tee` | TEE | 18 | **1 116** | 0.02 ms | 36×31 — online merge |
| **total** (`tee:attn_resident_cover`) | | **13 832** | **1 152** | 12.0 ms | per-(layer,step) closure |

**Pre-O4 framing (superseded by O4(a)+O5 above — kept for the reasoning
trail).** Before optimisation the build was a one-time per-layer cost
(`build_covered_prefix` = 63% of the K=32 bucket, 8 765 ms / 36 layers) with the
recurring per-step cost only ≈157 ms vs in-TEE ≈455 ms (~2.9×/step) — i.e.
`bucket(K) ≈ 8 765 + 157·K` vs `455·K`, **break-even K≈30**. That motivated O4
(make the build cheaper) and O5 (move it off the decode path): **O4(a) halved
the build → break-even K≈15, then O5 relocated it to prefill → recurring-only,
no break-even** (the comparison table at the top of this subsection). The dense
`O(d²)` rotate that remains in the build is *not* further reduced — signed-perm
is deferred (security) and HD₃ regressed (both in the note above). Wire:
`forward.rs` cover branch + `DecodeCover` (`kv_cache.rs`) +
`TrustedExecutor::resident_kv_attend_partial`.

### Prefill — feature-rotation + cubek-attention (tensor-core)

`O_qk`/`O_v` (σ=0) + `cubek-attention` (`BlackboxAccelerated` / tensor cores),
public causal mask, GQA-expanded K/V. Per-op is per **layer** (prefill is
one-shot; each op runs once per layer per prefill, ×36 layers for the full
prefill — no per-step). Bench measures one layer's worth at B=8;
`crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs`, parity at the fp16 floor.

**Offloaded prefill vs full in-TEE attention (per layer, B=8).** In-TEE is
`causal_gqa_attention` (the bucket the offload replaces); both are measured in
the same bench. **⚠ This is the *unoptimised* analytical baseline** (synthetic
per-layer, scalar f16 convert, pre-O1) — kept because its prep decomposition
below is what *motivated* O1/O2. At this baseline the offload barely wins at
n=2048 (essentially break-even) and only clearly at long context; the
**optimised, production result is 2.83×** at n=2048 — see *Prefill offload —
real-engine wire-up* below. The pre-optimisation per-layer numbers:

| context (per layer, pre-O1) | full in-TEE | offload (rot+cubek+`O_vᵀ`) | ratio |
|---|--:|--:|--:|
| **n=2048** | **1 089 ms** | 1 021 ms | **1.07×** (≈break-even, pre-opt) |
| **n=8192** | **21 946 ms** | 4 653 ms | **4.72×** (pre-opt) |

(Full Qwen3-4B prefill = ×36 layers; the real in-TEE prefill-attention bucket is
`tee:attn_inplace_many` ≈ **43 774 ms**, n=2048, chronicle §3.2 — the target the
offload would replace.) The offload total decomposes as:

| offload component | where | n=2048 ms | n=8192 ms | × / layer | |
|---|---|--:|--:|--:|---|
| rotation + GQA-expand | TEE | 80 | 526 | 1 | `O_qk`/`O_v` apply + 8→32 head expand |
| `cubek` attend (prep+compute) | **GPU** | 918 | 3 986 | 1 | convert+upload+attend+readback (decomposed below) |
| `O_vᵀ` correction | TEE | 23 | 141 | 1 | un-rotate the output |
| **offload total** | | **1 021** | **4 653** | 1 | vs in-TEE 1 089 / 21 946 |

**Preparation breakdown — the `cubek attend` bucket decomposed (measured
2026-06-01, `CUBEK_PROFILE=1`, steady-state after JIT warm-up).** A per-stage
device-sync instrumentation inside `cubek_attention_folded` splits the
previously-opaque bucket. The result **corrects the earlier inferred
"~680 ms fixed + ~204 ms compute, ceiling ~5×"**: the actual tensor-core
compute is far smaller and the path is convert/transfer-bound at *both* sizes.

The `× / layer` column counts how often each stage runs **per layer** (one
cubek call per layer); the full Qwen3-4B prefill runs **×36 layers**, so the
per-prefill count is the listed value ×36. The ms columns are per single call.

| cubek stage | where | × / layer | n=2048 ms | n=8192 ms | note |
|---|---|--:|--:|--:|---|
| f32→f16 convert-in (Q+K+V) | host CPU | 3 (Q,K,V) | **396** | **1 566** | scalar `f16::from_f32` loop, ~2 ns/elem (Q,K,V each ~131 / ~520) |
| upload DMA (host→GPU) | PCIe | 3 (Q,K,V) | **267** | **1 045** | K/V uploaded **4× GQA-expanded** (cubek has no native GQA) |
| **attend (tiled softmax·V)** | **GPU** | 1 | **72** | **628** | the only true compute — tensor cores |
| readback (GPU→host) | PCIe | 1 | 6 | 18 | output only (`[B·Hq, n, d]`) |
| f16→f32 convert-out | host CPU | 1 | **158** | **627** | scalar output convert |

**Finding — compute is ~7% of the path; the bottleneck is the scalar
convert + GQA-expanded upload, at every context length.** At n=2048 the GPU
attend is only **72 ms of the 1 021 ms end-to-end** — the host-side f16
converts (in 396 + out 158 = **554 ms scalar**) plus the **267 ms 4×-expanded
upload** are the cost. At n=8192 the same holds (compute 628 ms of 4 653 ms =
13%): the **4.72× win there is in-TEE being O(n²)-brutal, not the GPU being
saturated**. The cover itself (rotation 80 ms + `O_vᵀ` 23 ms = **10%**) is
cheap, as before.

**What this does *not* mean (ceiling ≠ achievable win).** The compute-only
ratios — 1 089/72 ≈ 15× (n=2048), 21 946/628 ≈ 35× (n=8192) — are *ceilings*
that assume **all** prep → 0. They are **unreachable**: the upload is
irreducible (the operands must reach the GPU; un-replication shrinks it ~½–¼
but not to zero), and the in-TEE rotation + `O_vᵀ` cover stay. The honest
reducible floor at n=2048, with the converts fused/SIMD'd to ~0 and K/V
un-replicated, is

```
  rotation ~50–80  +  upload ~130 (un-repl, at the measured 1.5 GB/s)
  +  compute 72  +  readback/O_vᵀ ~30   ≈  290–310 ms   →  ~3.4× vs in-TEE
```

i.e. **a projected ~3× at n=2048** (not 15×), and **~10–15× at n=8192**. Even
that is contingent on two unknowns: (a) that fusing f16 into the rotation truly
zeroes the convert, and (b) the upload bandwidth — **267 ms for ~400 MB is
1.5 GB/s, ~8× under PCIe5**; if that is a real DMA limit, upload dominates the
floor and caps the win near ~3×; if it is a staging/non-pinned-copy artifact,
the floor drops further (~5–6×). So the offload remains break-even-to-modest at
the production shape until those land — the win is real but projected, not
measured. The prep levers (Phase 5a):
- **SIMD f32→f16 — ✅ LANDED (2026-06-01).** `cubek_attention_folded` now uses
  half's `convert_from_f32_slice` (F16C) for both convert-in and convert-out,
  matching the engine's K/V upload path (`lib.rs:550`) — replacing the scalar
  `to_f16_bytes`/`from_bits` loops. This is the lever behind the real-engine
  1.88× below.
- **un-replicated K/V + on-device GQA broadcast — ✅ LANDED (2026-06-01).**
  `cubek_attention_folded_gqa` converts + uploads K/V un-replicated `[Hkv,n,d]`
  and broadcasts to `Hq` **on-device** via burn `repeat_dim` (bridged to cubek's
  cubecl `TensorHandle` — same client, no host round-trip), mirroring the decode
  `resident_kv_expanded`. forward.rs folds + rotates K/V at `Hkv` (4× less
  rotation). Real-engine effect below.

**Note — SIMD convert (O1) delivered ~2×, not ~10×.** Re-measured post-O1
(`CUBEK_PROFILE`, bh=256, n=2048): convert-in 396 → **192 ms**, convert-out
158 → **99 ms** — half's F16C path is **memory/alloc-bound**, not
arithmetic-bound (matches the earlier engine-side finding). After O1 the
dominant prep term is the **upload (265 ms)**, which is what O2 targets.

**Joint takeaway.** Both offloads are gated by the **convert / upload /
dense-rotate prep pipeline**, not the GPU compute and not the cover: decode by
a one-time prefix re-cover (amortizes → ~2.9× per step), prefill by the scalar
f16 convert + 4× GQA-expanded upload that dwarfs a 72 ms attend at every
context length (the per-stage table above). The GPU compute has ~15× (n=2048)
to ~35× (n=8192) of headroom; Phase 5a's job is to stop wasting it on
host-side prep. The cover (rotation/permutation) is cheap in both; covariant
obfuscation, being a static weight transform, leaves all of these numbers
unchanged.

### Prefill offload — real-engine wire-up (Phase 6 perf wire, 2026-06-01)

The prefill offload is now wired into the **production forward path**
(`decoder_block_batched`), not just the synthetic bench — gated behind
`GELO_GPU_PREFILL_OFFLOAD` (default-off; GLOBAL layers only; SWA stays in-TEE).
Per layer the TEE samples a session-fixed shared cover (`O_qk`/`O_v`), folds +
GQA-expands + rotates each sequence's Q/K/V, hands the rotated operands to the
engine's new `cubek_causal_attend` (`GpuOffloadEngine`/`TrustedExecutor`
delegate → `cubek_attention_folded`), and corrects the output with `·O_vᵀ`. The
GPU only ever sees rotated bytes. **Correctness:** `cover_prefill_matches_in_tee`
(f32, no GPU) pins the fold/expand/unfold + cover cancellation against the
production `causal_gqa_attention` (max_abs < 1e-3); `cubek_folded_causal_parity`
covers the cubek fp16 step.

**Measured on the canonical microbench** (Qwen3-4B, B=8, n=2048, RTX 5090 /
Vulkan, `CUBEK_STRATEGY=blackbox`), full 36-layer prefill attention bucket:

| prefill attention bucket (`tee:attn_prefill_offload`) | full prefill, B=8, n=2048 | ratio vs in-TEE |
|---|--:|--:|
| **in-TEE baseline** (`tee:attn_inplace_many`) | **44 062 ms** | 1.00× |
| offload, O1 (SIMD convert) | 23 490 ms | **1.88×** |
| offload, O1 + O2 (un-replicated K/V, on-device expand) | 16 237 ms | **2.71×** |
| offload, + batched fold/correct (loop-batch + fused `O_vᵀ`) | **15 587 ms** | **2.83×** |

**Loop-batching + fused `O_vᵀ` (2026-06-01).** For uniform-length batches (the
common case — padded prompts share `n_max`) the in-TEE fold + rotate is done
**once** over all B·Hq folded heads, and the `O_vᵀ` correction is **fused with
the unfold** (written straight into `ctx`, no intermediate (B·Hq,n,d) array). The
reliable, low-variance wins: `prefill_cover:rotate_tee` **4 236 → 3 534 ms**,
`prefill_cover:correct_tee` **2 614 → 1 666 ms** (≈1.65 s). (`cubek_gpu` swings
±~1.5× cross-run via autotune/thermal, so the bucket number understates this.)

**cubek stays per-sequence — and that is the *measured* optimum, against the
usual "batch bigger" rule.** A controlled warm A/B
(`cubek_prefill_cover::cubek_dispatch_granularity`, 8 iters, one process):
**one `bh=B·Hq` dispatch 273.8 ms/iter vs B `bh=Hq` dispatches 185.1 ms/iter →
per-seq 1.48× faster.** This inverts best practice because of cubek's
**materialised GQA expand** (`repeat_dim` builds a ~268 MB expanded K/V for the
big dispatch vs small reused per-seq ones). The principled fix — cubek's
kv-head **read-index** (broadcast, no materialisation) — would let the single
dispatch win; until then the prefill folds the TEE work batched but loops cubek
per-sequence.

**Measured, not projected** — the offloaded prefill attention bucket is now
**2.83×** vs in-TEE at the production shape. Still open: the upload-bandwidth
probe (O3 — `queue.write_buffer` staging, ~1.5 GB/s, likely alloc/submit-bound)
and the cubek read-index. **Security unchanged:** default-off *perf* wire; the
rotation cover still fails `WEIGHTS-PUB`, so default-on stays gated on covariant
obfuscation (Phase 5b).

## Acceptance gate (v1)

Layered — failing any tier reopens the TwinShield-Xue fallback:

1. **Fail-fast (microbench).** Resident per-step cost < the in-TEE
   baseline (11.35 ms @ n_kv=2048, B=8). Go/no-go *before* the substrate
   refactor.
2. **Perf (end-to-end).** ≥ 30% decode-wall reduction on top of R3 at
   n=2048, **and** a demonstrably larger reduction at a long-context cell
   (n ≥ 8k — the phased target where HBM's bandwidth edge dominates).
3. **Quality.** Greedy-token parity preserved at the σ chosen by gate 2
   (no extraction-quality regression from the √N-scaled noise). This
   couples acceptance to the gate-2 σ.
4. **Round-trips.** No growth in TEE↔GPU round-trips beyond the design's
   ≤ 1 per decode step, and no growth in mask-offload count (revival
   Step-5 invariants).

## Sequencing — status (✅ done · ⛔ blocked · remaining)

This is the live progress log, not a forward commitment: each phase is
marked with its current state and dated where measured. **Guiding
strategy (as executed):** build the simplest/weakest cover first, attack
it with a real `NO-PLAINTEXT` (and later `WEIGHTS-PUB`) bench as early as
possible, and **gate the expensive kernel + wire-up on the security
result** — the adversary view is just the cover applied to real
activations, so it needs neither the kernel nor the integration. That
ordering is exactly what surfaced the `WEIGHTS-PUB` break before any
kernel was built (TwinShield-Xue remained the parallel-de-risked
fallback throughout).

### Done

0. **Triage + gate-1 microbench** — ✅ **2026-05-29.** `gpu_resident_b8`:
   resident read 0.465 ms @ n=2048 (24× under the 11.15 ms in-TEE baseline);
   ~99.9% of the full-upload cost is upload+convert+sync. `gpu_resident_append_b8`
   (per-step O(1) append, prefill-only): **0.662 ms @ n=2048 → 16.4×**,
   scaling 2.7× → 16.4× with context. **Optimistic case is worth it.**
   Conditional: the re-permute upload (`no_mask − resident` ≈ 488 ms
   current-pipeline) needs the un-replicated + bf16-native optimization
   (~5 ms modeled) to amortize — lands in Phase 2.
   Gate-2 σ-sweep probe (`gate2_perm_recovery_vs_sigma_and_n`): σ is not a
   lever, √N accumulation real ⇒ `perm_kv` refresh per block; but this is a
   *plaintext-assuming* probe (violates `NO-PLAINTEXT`), **not** the
   realistic gate.

9. **Phase 4 — gated decode wire-up + full bench** — ✅ **2026-05-29.**
   GPU-resident attention threaded through `decoder_block_cached_batched`
   for GLOBAL layers behind `GELO_GPU_RESIDENT_ATTN` (in-TEE default).
   Greedy-parity: byte-identical tokens flag-off vs flag-on (Qwen3-4B,
   16 steps) — create/append/attend/stack wire-up correct, fp16 attend
   matches f32-in-TEE on argmax. Full bench (4b, B=8, N=2048, K=32,
   Vulkan, `bench-results/phase4-resident-decode-5090-2026-05-29.log`):

   | B=8 decode, Vulkan | in-TEE baseline | GPU-resident | Δ |
   |---|--:|--:|--:|
   | **attn bucket** | 14 574 ms (`tee:attn_cached_inplace_many`) | **8 724 ms** (`tee:attn_resident_gpu`) | **0.60× (−5.85 s)** |
   | per-call attn | 12.65 ms | 7.57 ms | 1.67× |
   | decode wall | 40.6 s | 28.81 s | 0.71× (*confounded*) |

   **Honest attribution.** The clean, attributable win is the attention
   bucket: **0.60× (−5.85 s)**, the only bucket the flag touches. The
   decode-wall delta (−11.8 s) is **confounded** — `engine:matmul`
   moved 8956→4382 ms and `matmul_many` 9941→8670 ms between the two
   (separate-session) runs, which the flag cannot cause; that is
   cross-run autotune/thermal variance, not the wire-up. A same-session
   off/on wall A/B is the follow-up to claim it cleanly.

   **The 1.67×-not-25× gap is the dispatch tax.** The isolated gate-1
   `attend` was 0.40 ms/step; the wrapped decode path is 7.57 ms/step.
   The delta is per-(layer,step) overhead the microbench omits:
   create/append + `stack_heads` (f32→f16) + GQA-broadcast attend +
   `unstack_heads` (f16→f32) + TEE↔GPU round-trips, ×36 layers ×32
   steps. This is exactly the Phase-3 dispatch-count concern; closing
   it needs the fused FlashAttention-D kernel + fewer per-step
   dispatches (deferred fast-follow). Net: a real, monotonic,
   parity-preserving win on the bucket it replaces — but the headline
   lever for the rest is dispatch count, not raw attend throughput.

### The reordered critical path

```
Phase 1 (cover incl. O_v/O_qk, σ=0 parity)  + real-activation capture
   → HNM/ICA bench on the adversary view          ← cheap, highest-risk, runs FIRST
        ‖ parallel: Phase 2 substrate (cover-AGNOSTIC: session API incl.
        ‖           refresh_block, SpillProvider, un-replicated+bf16 upload)
        ‖ parallel: TwinShield 1B spike (R-rank / correction-cost viability)
   → SECURITY GATE result → THEN commit Phase 3 kernel + Phase 4 wire-up
   → Acceptance (4 tiers) → Flip default behind c5 AloePri condition
```

1. **Phase 1 — cover math + parity** (`gelo-protocol/attention.rs`) —
   ✅ **DONE 2026-05-29.** `O_qk` (Q,K feature rotation — cancels in
   scores) + `O_v` (V feature rotation — corrected by `O_vᵀ`) added to
   `permuted_attention_cached`, behind `PermAttnConfig::feature_rotation`
   (default false → production unchanged) with `sample_orthogonal`
   (Gram-Schmidt). Verified: `feature_rotation_sigma_zero_matches_plain`
   (bit-exact at σ=0, d=32 & 128) and `feature_rotation_preserves_scores`
   (ON ≡ OFF). The lever gate-2 showed σ-noise can't provide. *Cover
   sampled fresh per call here; the session fixes it at prefill.*
2. **Real-activation capture** (Rust, this box): dump real Qwen3-4B
   attention Q/K/V at the production decode shape + apply the cover →
   the adversary view the HNM/ICA bench consumes.
3. **Security bench** (Python eval env — *not* this box): `NO-PLAINTEXT`
   HNM (gate 2 → max N) + JADE/anchor_ica covariance-alignment
   (gate 3 → max T / `O_v` cadence M) against the captured adversary
   view. **Prefill-only first; if recovery is bad, flip on per-block /
   per-N-block `perm_kv` refresh and re-attack.** This is the gate on
   Phases 4+.
4. **Phase 2 — session substrate (cover-agnostic, parallel to 1–3)** —
   🟡 **first increment landed 2026-05-29.** Done: `GpuOffloadEngine`
   session API (`kv_create_session` / `kv_append` / `kv_attend` /
   `kv_refresh_block` / `kv_drop_session`, default-unsupported so existing
   engines don't break), `WgpuVulkanEngine` impl (engine-owned session map
   shared across `clone_shared`, reusing the `ResidentKvSession` logic),
   `SpillProvider` trait + `NullSpillProvider` seam, all exported from
   `gelo-protocol`. Verified: `tests/kv_session.rs` — create→append→attend
   matches direct `fused_attention_batched` at fp16 floor; refresh/drop
   correct; f32 engine rejects. `refresh_block` is in the API (per the
   build-it-now decision).

   **Un-replicated GQA storage — done 2026-05-29.** `kv_attend` now
   broadcasts un-replicated K/V (`num_kv_heads`) up to `num_q_heads` with
   a device-side interleaved expand (`reshape → repeat_dim(group) →
   reshape`), so the upload/storage is 4× smaller (the gate-1 re-permute
   win) while the compute sees the expanded view. Verified
   (`kv_session_gqa_broadcast_matches_expanded`: 8-head storage + 32-head
   Q matches a manually-expanded reference at fp16 floor). The Phase-3
   kernel later folds the broadcast in-shader (no re-materialisation).

   **Upload-cost decomposition (gate-1 deliverable, done 2026-05-29;
   `bench-results/upload-decomposition-5090-2026-05-29.log`).** At
   n_kv=2048, B=8, the per-call K/V cost (`no_mask − resident` ≈ 490 ms)
   splits **convert 368 ms (~75%) / DMA+sync ~122 ms (~25%)** — and the
   convert is a *scalar* f32→f16 loop (2.7 ns/elem). **Implication:** the
   dominant upload cost is killable **without roadmap §4.E.3** via a **vectorised
   convert** (half's F16C `convert_from_f32_slice`, ~10×) — the cheap,
   unblocked next lever; full **bf16-native** (no convert at all) needs
   the roadmap §4.E.3 activation pipeline and is now *less urgent* since the
   vectorised convert + un-replicated (4×) capture most of the win.

   **Vectorised convert — done, but under-delivered (2026-05-29;
   `bench-results/simd-convert-5090-2026-05-29.log`).** Swapped the scalar
   `f16::from_f32` map in `array3/array2_to_tensor_f16` for half's
   `convert_from_f32_slice` (F16C `vcvtps2ph`, runtime-detected); parity
   holds. Measured **only 1.7× (130 vs 223 ms convert; `no_mask` 490→232 ms)**,
   *not* the hoped ~10× — at ~1 ns/elem the convert is **memory/allocation-
   bound** (read 536 MB f32, alloc+write 268 MB f16/tensor), not
   arithmetic-bound, so F16C buys little. Kept (free), but it is *not* the
   answer.

   **Does it beat in-TEE? Regime-dependent:**
   - **Prefill-only** (no decode re-permute — the case the `NO-PLAINTEXT`
     gates leaned toward): per-step decode is the resident read only
     (0.40 ms) → **25× under in-TEE; the convert is moot** (one-time
     prefill cost).
   - **Per-block refresh** (if gate-2 σ-sweep, *deferred*, requires it):
     amortised per-step ≈ resident 0.40 + (un-replicated `no_mask` 232/4)/N.
     At N=16 ≈ **4 ms vs in-TEE 10 ms → ~2.5×** (beats it, modest margin;
     the convert is still ~half the refresh).

   **⇒ bf16-native is promoted (per the decision rule).** Since the
   vectorised convert under-delivered and the convert still dominates the
   *refresh* path, **bf16-native upload — which eliminates the convert
   entirely (resident bf16 cache, memcpy not convert) — is the priority
   refresh lever** (refresh → DMA-only ≈ 25 ms /N → ~2 ms/step → ~5×).
   **Contingency:** this only matters if per-block refresh is actually
   required — which is the deferred gate-2 σ-sweep. If prefill-only holds,
   refresh (and the whole convert/bf16 question) is moot for decode.

   **Remaining Phase-2 sub-steps:** bf16-native upload (now priority, but
   gated on gate-2 confirming refresh is needed; full version needs
   roadmap §4.E.3); `/code-review` of the trait change. `kv_attend` still returns
   the full normalised context (partial-stats `(m,l,acc)` is the Phase-3
   kernel); not yet wired into `TrustedExecutor`/forward (Phase 4).
5. **Phase 3 — partial-stats attend + tail-in-TEE merge** — **MANDATORY**
   (closes the write-location channel; grill 2026-05-29). Two parts,
   decided:
   - **Decode (now): compose partial `(m,l,acc)` from burn ops** —
     `attend` returns `(m = scores.max, l = Σexp(s−m), acc = exp(s−m)·V)`
     over the frozen prefix; the TEE attends the in-TEE tail (≤N tokens)
     and merges (online softmax). **No custom kernel, no `cubek` fork** —
     upstream-burn `flash_attention` returns normalised output only and
     can't expose the stats; at decode (n_q=1, scores tiny) the composed
     ~5-dispatch form is fine (~0.5 ms/step, ≈ the gate-1 resident attend).
     Neutral-to-faster than tail-on-GPU (drops the append round-trip).
   - **Prefill: `cubek-attention`, rotation-only cover** —
     *superseded — see the fused-kernel section.* It is **not** a custom
     kernel (`cubek-attention =0.1.1`) and there is **no GQA in-shader**
     (K/V expanded 8→32). cubek's tiling is what avoids the ~4 GB scores
     tensor. Offloaded *causal* prefill rules out permutation (mask leak)
     and additive (kernel-incompatible, verified vs the paper), so the
     offload cover is **feature-rotation-only**, gated on the sufficiency
     spike — not a free fast-follow.
6. **Phase 4 — decode wire-up (perf)** — ✅ **2026-05-29** (see Done #9).
   Gated GPU-resident `attend_session` (full normalised), in-TEE default.
   *Note:* this shipped the bare **perf** path (no cover). The secure
   decode path stays on the **permutation cover** (block-fresh-π + σ +
   `O_v` + tail-in-TEE via the composed `attend_session_partial`) — decode
   is **not** affected by the prefill-offload cover decision below.
7. **Phase 5 — prefill-offload cover sufficiency spike** (the gate the
   prefill-offload lever hangs on). Run in **both** `WEIGHTS-PUB` and
   `WEIGHTS-BLIND`: `WEIGHTS-PUB` reconstruction (the load-bearing attack) +
   `WEIGHTS-BLIND` rotation recovery + order-is-public residual +
   fixed-rotation accumulation, plus a cubek-parity / `O_vᵀ`-cost check —
   see *The offload-cover sufficiency spike*. **Pass (`WEIGHTS-PUB`) ⇒
   rotation-only ships (decode unchanged); pass blind / fail public ⇒
   explore covariant weight obfuscation (AloePri); fail both ⇒ in-TEE
   prefill (or the separate TwinShield GEMM-offload lever).**
   **⛔ BLOCKED by the Phase-5 result (2026-06-01):** the rotation cover
   leaks every prompt token under `WEIGHTS-PUB` (top-1 = 1.000), so Phase 6
   does **not** proceed as-is. It is gated behind **Phase 5b** below.
7b. **Phase 5b — covariant obfuscation to neutralise `WEIGHTS-PUB`.**
   **⚠ Re-scoped 2026-06-02** (see *Adapting covariant obfuscation to the
   offload*): the lever is **not** static weight obfuscation. Two findings rule
   that out as the *minimal* path — GELO holds the offloaded weights
   resident-in-the-clear, so a bare covariant weight transform is trivially
   invertible from public `W` (a VMA-safe resident `W'` would require
   ~whole-model AloePri, forfeiting accuracy + adding a deployment artifact);
   and an orthogonal refresh of the cover does not freshen the `O_v`-invariant
   Gram the attack reads. The minimal lever is instead a **per-session,
   κ-bounded, non-orthogonal value cover `C_v`** (AloePri's value covariant pair
   relocated from static weight-space into GELO's existing dynamic
   activation-space cover): generalise `O_v` from orthogonal to invertible,
   correct with `C_v⁻¹`. No weight touched, no deployment complexity, greedy
   parity at the chosen κ, perf unchanged, accumulation-safe at session
   granularity. Gate: re-run `gate3_prefill_dict.py` against a `C_v` capture
   (κ-sweep, norm + Gram, full vocab, deeper layer). QK-side `Ĥ_qk` is a
   conditional second addition, deferred until the K/Q self-Gram is measured to
   bite. If no κ both breaks the dictionary and holds parity, escalate to
   `Ĥ_qk`; if that also fails, **prefill stays in-TEE**.
8. **Phase 6 — prefill-attention offload.** **🟡 PERF WIRE LANDED, default-on
   still ⛔ gated on Phase 5b.** The perf wire is in the production forward
   path (`decoder_block_batched`, `GELO_GPU_PREFILL_OFFLOAD`, default-off):
   per-layer shared cover (`O_qk`/`O_v`, σ=0) → fold+GQA-expand+rotate →
   `cubek_causal_attend` (engine/executor delegate to
   `cubek_attention_folded{,_gqa}`) then `·O_vᵀ`. O1 (SIMD convert) + O2
   (un-replicated K/V, on-device GQA broadcast) + loop-batching (one fold/rotate
   over B·Hq) + fused `O_vᵀ`+unfold folded in. Verified:
   `cover_prefill_matches_in_tee` / `cover_prefill_batched_matches_in_tee`
   (f32 floor) + `cubek_folded_causal_parity` (fp16). **Measured 2.83×**
   (44.1 s → 15.6 s) on the real `tee:attn_inplace_many` bucket — see *Prefill
   offload — real-engine wire-up*. cubek dispatched **per-sequence** (a
   controlled warm A/B settled it as 1.48× faster than one big dispatch — the
   materialised-GQA-expand artifact). Remaining perf: the **cubek kv-head
   read-index** (kills the materialised expand → lets the single dispatch win +
   cuts `cubek_gpu`) and the upload-bandwidth lever (O3, `write_buffer` staging).
   **Default-on remains blocked** — the rotation cover fails `WEIGHTS-PUB`;
   flipping requires covariant obfuscation (Phase 5b).
9. **Acceptance + flip** — the 4-tier gate, then default-on behind the
   c5 AloePri condition (mirrors R3).
10. **Fast-follows** — cubek kv-head read-index (broadcast K/V in-shader,
    recover the GQA materialisation + flip the dispatch-granularity result);
    decode O6 fused partial-stats kernel; NVMe `SpillProvider`.

### TwinShield reuse (what a pivot costs)

If the cover fails HNM at every cadence, fall back to TwinShield-Xue.
Most of Phases 2–4 is cover-agnostic and survives the pivot:

| Component | Permutation path | TwinShield | Reused on pivot |
|---|---|---|---|
| Session substrate (API, resident K/V, SpillProvider, un-replicated+bf16) | ✓ | ✓ (also persists K/V) | **100%** |
| Decode wire-up (prefill builds / decode appends+attends) | ✓ | ✓ | **100%** |
| Kernel skeleton (resident reads, GQA broadcast, matmuls, partial-stats/merge) | ✓ | ✓ | **mostly** |
| Cover math (perm + σ + `O_v`, πᵀ/`O_vᵀ` recovery) | ✓ | ✗ — `e^(X+R)` blinding + correction | **no** |
| Kernel softmax stage | standard over permuted scores | blinded-exponent + correction | **differs** |
| Block-refresh machinery | needed (√N) | not needed (fresh per-call R) | TwinShield *removes* |

⇒ **~60–70% of Phases 2–4 survives a pivot.** Keep the session API
**cover-agnostic** (cover is a swappable strategy) so the cost of a pivot
is the cover layer + the softmax stage, not the substrate. De-risk
TwinShield's R-rank in parallel (the 1B spike) so the fallback is
*known-viable* before it's needed.

## References

- [`2026-05-22-dgpu-attention-revival.md`](../../handoffs/2026-05-22-dgpu-attention-revival.md) — the Item 1/2/3 design this concretizes; σ-vs-N table, 1A vs 1B trade
- [`2026-05-29-dgpu-attention-offload.md`](../../handoffs/2026-05-29-dgpu-attention-offload.md) — dGPU bring-up handoff; the §2 headline that set up this task
- [`gelo-llm-perf-roadmap.md`](../../plans/gelo-llm-perf-roadmap.md) §4.C.2 — the EV/engineering table for these levers
- `docs/dev/logs/gelo-llm-perf-chronicle_dgpu.md` §8 — the per-call-readback correction (the backend-invariant bottleneck)
- `bench-results/amulet-attn-triage-5090-2026-05-29.log` — the triage that gates viability on persistent K/V
- `crates/gelo-protocol/src/attention.rs::permuted_attention_cached` — the existing fresh-per-call cover this extends
- `crates/gelo-protocol/src/substrate.rs::offload_attention_permuted_cached` — the trait seam
- `crates/gelo-gpu-wgpu/src/lib.rs::fused_attention_batched` — the current re-upload-every-call engine path
- Amulet softmax-permutation equivariance (arXiv 2512.07495); Hidden No More (arXiv 2505.18332); TwinShield-Xue additive blinding (arXiv 2507.03278)
