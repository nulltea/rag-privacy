---
type: prototype-note
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, attention, gpu-offload, permutation, feature-rotation, value-cover, aloepri, threat-model, humaneval]
companion: [perm-attn-gpu-offload]
---

# GPU-offloaded confidential attention with a value-basis cover

How GELO moves its attention bottleneck onto an untrusted GPU without
revealing the user's prompt — using a permutation cover for decode, a
feature-rotation cover for prefill, and a per-session non-orthogonal
**value-basis cover** (adapted from AloePri's covariant obfuscation) to
close the leak that rotation and permutation alone cannot.

The full design-exploration log, with the measurement trail, lives in
[`perm-attn-gpu-offload.md`](../logs/perm-attn-gpu-offload.md); this note is
the distilled description of *what was built and why*.

---

## 1. Background — the in-TEE attention bottleneck

GELO serves an open model (Qwen3-4B: 36 layers, 32 query / 8 KV heads,
GQA 4:1, head-dim 128) inside an SEV-SNP enclave on a host with an
untrusted, VFIO-passed RTX 5090 (32 GB GDDR7, ~1.8 TB/s HBM) over PCIe.
The matmuls already run on the GPU; **attention is the largest bucket that
stays on the CPU inside the enclave.** At the production shape (batch 8,
context 2048):

| phase | in-TEE attention bucket | cost | share |
|---|---|--:|---|
| prefill | `tee:attn_inplace_many` | **44.1 s** | ~21–24% of prefill wall |
| decode | `tee:attn_cached_inplace_many` | **14.6 s** (K=32 steps) | 455 ms/step over 36 layers |

The obvious fix — run attention on the GPU — fails naively: a full
re-upload + f32→f16 re-convert of the K/V cache on every call is
**45–66× slower** than the in-TEE path (an upload signature, not a compute
one). Viability is therefore gated on **persistent device-resident K/V**
(only the per-step delta moves) and a confidentiality cover the untrusted
GPU cannot strip.

---

## 2. The two base covers, and why they are not enough

Decode and prefill face different leaks, so they use different covers.

**Decode — the block-fresh permutation cover.** The resident K/V cache is
stored *permuted* by a secret per-block permutation `π` (`perm_kv`), with
σ-noise added to Q and K, and the value rows feature-rotated by an
orthogonal `O_v`. Softmax-permutation equivariance (Amulet) makes this
exact: the GPU runs softmax over permuted scores and never needs to know
`π`; the TEE un-permutes the query axis and corrects `O_v` after an
online-softmax merge. The newest ≤N tokens stay **in the TEE** (never
written to the GPU), which also closes a per-step write-location side
channel. `π` hides *order*; σ defeats statistical recovery of `π` across a
block; `O_v` hides the value *coordinates*.

**Prefill — the feature-rotation cover.** Prefill is one-shot
self-attention over the whole prompt under a causal mask. Here permutation
is **fatal**: a correct causal mask in permuted coordinates leaks `π`
exactly through its row-sums (row `i` sums to `π(i)+1`), independent of any
content cover — it must encode the token total order (`log₂ n!` bits = `π`'s
entropy). So prefill uses **feature rotation only**: an orthogonal `O_qk`
on Q and K that cancels in the score (`(Q·O_qk)(K·O_qk)ᵀ = QKᵀ`), and `O_v`
on V corrected by `O_vᵀ` after the value contraction. σ is off (additive
noise on K cannot be undone through a fused softmax). The token axis is
untouched, so the causal mask stays the standard public triangular one.

**Why this is not enough — the public-weight norm-dictionary attack.**
The deployment serves *open* weights, so the conservative adversary knows
`W_Q/W_K/W_V` and the embedding table (the `WEIGHTS-PUB` assumption). Under
that assumption both covers leak the prompt's **token membership**:

- A feature rotation `O_v` is *orthogonal*, so it preserves the per-head
  value **norm** `‖V_i·O_v‖ = ‖V_i‖`.
- A permutation `π` only *relabels* rows — a per-slot quantity is still a
  per-token quantity.
- σ-noise is on K, never on V — so the value path carries no noise.

⇒ the per-head value-norm fingerprint of each slot is a clean, public,
per-token quantity, free of all three secrets. With public `W_V` it matches
a context-free per-token **dictionary** (`V(t) = rmsnorm(E[t])·W_V` at
layer 0), and every token is recovered.

**Security-gate evaluation.** Attacks were run on real Qwen3-4B activations
with the cover applied (the adversary never sees plaintext; ground truth is
used only for scoring):

| attack | regime | result | reading |
|---|---|--:|---|
| content / coordinate recovery (JADE, FastICA) | `WEIGHTS-BLIND` | corr 0.36 ≈ floor 0.35 | `O_v` **holds** |
| position recovery (K-Gram seriation) | `WEIGHTS-BLIND` | \|Kendall τ\| ≈ 0.10 | order largely **held** |
| coordinate recovery via covariance-alignment | `WEIGHTS-PUB` | 0.33 ≈ floor (self-anchor 1.0 validates) | `O_v` **holds** |
| **membership via norm-dictionary** | **`WEIGHTS-PUB`** | **top-1 = 1.000** | **BREAKS** — every prompt token recovered |

The blind/coordinate gates pass; the **membership** gate fails. For prefill
the rotation cover leaks the whole prompt; for decode the permutation still
hides order (a multiset, not a sequence), but the **bag of tokens** is
exposed. Rotation and permutation are orthogonal-and-relabel operations, and
the attack rides their invariants — so no orthogonal/permutation cover, however
refreshed, can close it.

---

## 3. Design choices to close the membership leak

Closing a leak that lives in an *invariant* requires a **covariant**
obfuscation — a change of value basis that the model's computation cancels
but that the public weights no longer match. Four candidates, by where the
secret lives and how it refreshes:

- **Baked-weight obfuscation** *(static, in the deployed weights — full
  AloePri).* Fold a covariant transform into `W_V`. Rejected: the GPU holds
  the projection weights **resident in the clear**, so a bare transform is
  trivially invertible from public `W` (`W_V⁺·W̃_v` recovers it); making it
  safe-when-resident needs AloePri's whole-model key-matrix + permutation
  machinery — forfeiting greedy accuracy and adding a deployment artifact.
- **Additive value blinding** *(TwinShield-style `V+R`).* Rejected: the
  correction lives *inside* the `Σ_j P_j V_j` token-axis contraction, whose
  softmax weights the TEE never sees, so additive V-noise is uncorrectable
  through the fused kernel.
- **Decoy-row k-anonymity** *(chaff tokens permuted into the cache).* The
  attack then recovers `real ∪ decoy`, hiding each real token among decoys.
  Rejected as the primary lever: statistical not structural (priors re-rank
  decoys), inflates exactly the resident-cache reads the offload optimises,
  and neutralising decoys without an observable score pattern is an open
  ORAM-flavoured problem. Noted only as a fallback shape.
- **Per-session value-basis cover (`C_v`)** *(dynamic, in the activation
  cover).* Generalise `O_v` from an orthogonal rotation to a κ-bounded
  **non-orthogonal invertible** transform, resampled per session. **Chosen.**

A key negative result narrows the field: an *orthogonal* refresh buys
nothing. The attack reads the value Gram `G = V(C Cᵀ)Vᵀ`, and for any
orthogonal factor `Q`, `(C_0Q)(C_0Q)ᵀ = C_0C_0ᵀ` — the Gram is
orthogonal-invariant. Freshness must come from the **non-orthogonal** part.

---

## 4. The chosen defense — the per-session value-basis cover `C_v`

`C_v` is AloePri's value covariant transform, **relocated from static
weight-space into GELO's existing dynamic activation-space cover**: the same
principle (computation preserved by an invertible value-basis change), applied
where its secrecy is information-theoretic rather than weight-inversion-bounded.

- **Construction.** `C_v = U·diag(s)·Vᵀ` with Haar-random `U, V` and singular
  values pinned to `cond = κ` (geometric mean ≈ 1, so `det ≈ 1`). Sampled
  **fresh per head, per session.**
- **Application.** `C_v` replaces `O_v` at the cover sites (prefill rotate;
  decode resident-prefix build); the correction `C_v⁻¹ = V·diag(1/s)·Uᵀ`
  (computed once per session) replaces `O_vᵀ` in the TEE-side merge/unfold.
  Decode keeps `π` + σ + tail-in-TEE; prefill keeps `O_qk` orthogonal
  (it must cancel in the score).

**Why this is the minimal lever.**

- The leak lives **only** at the un-mixed attention operand (`V·O_v` uploaded
  per row); the projection output is already row-mixed by the per-call
  orthogonal activation mask. So the fix need only touch the *attention
  operands*, not the weights.
- Because the resident weights are public-and-clear by design, any
  weight-space transform is invertible — the cover **must** be in activation
  space, where its per-session secret is information-theoretically hidden.
- A per-session-fresh **non-orthogonal** `C_v` gives a fresh `C_v C_vᵀ` each
  session, so a worst-case adversary accumulating observations across sessions
  has no fixed target — accumulation-safe at session granularity.
- Within-head mixing distorts the value **norm *and* the Gram** (a per-head
  scalar would be erased by the attack's per-head standardisation), and
  `cond = κ` bounds the fp16 round-trip cancellation.
- **No weight is touched** → resident weights stay public-and-fine, no
  deployment artifact, and the runtime op is the same dense matmul as `O_v`
  plus a one-off per-session inverse — so securing the offload is ~free.

The dynamic↔static axis is exactly the **refresh granularity of the
non-orthogonal value transform**. GELO sits at the per-session (dynamic) end;
the static (AloePri) end only saves runtime by moving into weight-space, which
the resident-clear-weight property shows demands the full whole-model
machinery. For the offload, the minimal-cost secure point and the dynamic end
coincide.

---

## 5. Results

**Security — `C_v` at κ=6 clears the conservative `WEIGHTS-PUB` bar.**

| gate (`WEIGHTS-PUB`, real Qwen3-4B) | `O_v` (κ=1) | `C_v` (κ=6) |
|---|--:|--:|
| membership, norm-dictionary, 8k pool | top-1 1.000 | top-1 **0.000** |
| membership, full vocab (151 936) | — | top-1 **0.000**, top-5 0.047, median rank ~20 200 (13th pctile) |
| cover recovery, covariance/Procrustes (floor 0.37) | 0.34 (self-anchor 1.000) | 0.33 (self-anchor **0.359**, not 1.000) |

Recovery falls monotonically with κ; the dictionary dies (top-1 ≤ 0.02) at
κ≈5, κ=6 is the margin point. The covariance/Procrustes attack — which
subsumes the per-head Gram quadratic-assignment — fails even at its
self-anchor upper bound, so a non-orthogonal `C_v` is **strictly stronger**
than the orthogonal `O_v`: whitening fixes only `C_v C_vᵀ` and leaves the
orthogonal factor unresolved.

**Performance — offload vs in-TEE (CUDA, RTX 5090, B=8, n=2048, K=32,
warmed, κ=6 secure cover):**

| phase | in-TEE | offload (portable kernel) | offload (tensor-core kernel) |
|---|--:|--:|--:|
| **prefill** attention | 44.1 s | 24.7 s (**1.79×**) | 14.0 s (**3.16×**) |
| **decode** attention | 14.6 s | 3.0 s (**4.85×**) | 2.8 s (**5.25×**) |

The tensor-core kernel makes the GPU attend 2.45× faster than the portable
one (Amdahl-capped to ~6% of prefill wall, since attention is ~12% of
prefill — the GPU matmuls dominate). **Securing the offload is perf-free:**
`C_v` touches only the strategy-invariant in-TEE cover buckets, flat to ~1%
between κ=1 and κ=6; the GPU attend buckets are `C_v`-independent by
construction. Decode does not route through the tensor-core kernel (it uses
the resident partial-stats path), so its speedup is a backend effect, not a
kernel one.

**Accuracy — HumanEval pass@1, 20 prompts, secure offload path:**

| configuration | pass@1 |
|---|--:|
| in-TEE GELO (baseline) | 7/20 |
| **offload + tensor-core kernel + `C_v` (κ=6)** | **7/20** (parity) |
| plain Qwen3-4B via llama.cpp (reference) | 6/20 |

The secured, GPU-offloaded path matches in-TEE accuracy exactly — **no
regression from offloading or from securing it** — and beats the plain-model
reference. All 20 prompts (38–391 tokens, ragged) ran on the tensor-core
kernel; the handful of misses are model-quality artifacts (markdown/prose
contamination, logic errors) of the same class as in-TEE.

---

## 6. Caveats and remaining gaps

- **Security validation depth.** The `C_v` gate was measured against the
  context-free layer-0 value dictionary (and layer 35). Deeper contextual
  layers have no context-free dictionary, but the layer-0 break alone exposes
  the prompt, so layer 0 is the right bar; still, the gate has not been swept
  over **production context lengths (2k–16k)** — covariance-alignment may
  strengthen as the instance covariance approaches the population at long
  context, and should be re-checked there.
- **Residual ambiguity.** At full vocab the norm fingerprint still collapses
  to a ~5-candidate set ~5% of the time (top-5 ≈ 0.047) — a k-anonymity-like
  residual, not full elimination. Accepting it is a deployment/policy call.
- **Numerical margin.** `cond = κ = 6` implies an fp16 round-trip cancellation
  ≈ 6e-3 relative; greedy parity is *parity at κ* (the 7/20 result holds), not
  byte-identical. Larger κ trades margin for conditioning.
- **QK side deferred.** `O_qk` stays orthogonal (it must cancel in the score);
  a non-orthogonal Q/K hardening (`Ĥ_qk`) is held back until a measured K/Q
  self-Gram attack shows it bites.
- **Position clock not finalised.** The √N HNM bound on the permutation refresh
  cadence (the position-recovery gate, run under `NO-PLAINTEXT`) is still
  pending its driver; the structural conclusion (refresh `perm_kv` per block)
  stands.
- **Engineering.** Performance is single-sample (±~7% cross-run variance); the
  tensor-core kernel has a small-KV floor (degenerate few-token prompts fall to
  the portable kernel); open perf levers (in-shader GQA broadcast, upload
  bandwidth) do not affect security or accuracy.
- **Status.** Correctness and accuracy are cleared on the production CUDA
  backend; turning the prefill offload on by default awaits only the policy
  **acceptance** of the `WEIGHTS-PUB`/`C_v` security result above.
