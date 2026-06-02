---
type: handoff
status: current
created: 2026-06-02
updated: 2026-06-02
tags: [gelo, dgpu, attention, gpu-offload, cubek, cuda, sm120, vulkan, blackbox, accuracy, humaneval, diagnosis]
companion: [perm-attn-gpu-offload, 2026-06-02-attn-offload-cv-cover-gate]
supersedes: [2026-06-02-bf16-attention-kernel-fix]
---

# Handoff — offload-attention accuracy collapse: root-caused; bf16 diagnosis FALSIFIED

**Read first:** this **supersedes
[`2026-06-02-bf16-attention-kernel-fix`](2026-06-02-bf16-attention-kernel-fix.md)**,
whose diagnosis (f16-storage overflow → fix = BF16 cubek kernel) is **wrong on
every load-bearing claim** (details below). The offload design / `C_v` security
gate are unchanged — see the dev-log
[`perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md) and the
`C_v` gate handoff [`2026-06-02-attn-offload-cv-cover-gate`](2026-06-02-attn-offload-cv-cover-gate.md).
This doc records only what changed this session: the **real** root cause of the
HumanEval 0/20 collapse, the falsification of the bf16 story, and the CUDA/sm_120
finding.

## TL;DR

- **The 0/20 collapse = cubek's `BlackboxAccelerated` (tensor-core) kernel is broken
  on the cubecl-wgpu / Vulkan backend** — produces NaN for *all* n≥3 on *random*
  data (data-independent), corroborated by the archived
  `2026-05-21-attn-offload-spike.md:107` note ("`CUBEK_STRATEGY=blackbox` … SIGSEGV
  via cubecl-wgpu"). The `Unit` kernel is clean on Vulkan.
- **The original 0/20 ran with `CUBEK_STRATEGY=blackbox` leaked into the env** (from
  perf benches, which all export it). The offload *defaults to `Unit`* and the gate
  never sets the var, so every later gate run used `Unit` → clean → couldn't
  reproduce until blackbox was tested explicitly.
- **bf16 was doubly irrelevant.** The handoff's mechanism is falsified: cubek's
  softmax/accumulator are **F32** (`Strict(F32)` default; `AttentionElems::from_global_types`
  sets `softmax = accumulator`), softmax is **max-subtracted** (`exp(score−max)∈[0,1]`),
  so there is **no f16 overflow** (operands measured ≪ 65 504; f16 handles `exp=7e20`).
  bf16 only rounds operands *more* coarsely (7 vs 10 mantissa bits) and is **8× worse**
  numerically.
- **CUDA (the real prod backend, RTX 5090 / sm_120): the cubek prefill-attention
  offload does NOT run *on our pinned versions*** — `Unit` hits
  `assert!(num_cols % line_size == 0)`, `blackbox` fails nvrtc compile
  (`incomplete type nvcuda::wmma::fragment<…float…>`). **These are CLOSED upstream
  bugs** (cubek #55 = the Unit assert, #102/#81 = the WMMA types, burn #4622 = the
  RTX 50-series arch/nvrtc flag) **fixed in cubek 0.2.0 + burn 0.21.0**. **✅ BUMP
  LANDED 2026-06-02** (cubek 0.1.1→0.2.0, burn-cubecl/tensor/backend 0.20.1→0.21.0,
  cubecl* 0.9.0→0.10.0; API churn resolved): on CUDA/sm_120 the **Unit kernel now runs
  clean for all n** (mini-gate 0/88) and **blackbox runs clean at tile-aligned n**
  (16…2048); blackbox on *non*-aligned n returns a clean config error ("Stage seq_q
  must divide problem seq_q"), not a crash. The offload now works on the production
  CUDA backend. (cubecl's `tensor_cores_per_sm` table still omitting 120 was a red
  herring — not the blocker.)
- **IMPORTANT — this is scoped to the cubek attention kernel ONLY.** burn-cubecl's
  matmul path (everything in the GELO main forward) *does* run on CUDA/sm_120 — that
  is what the perf chronicle `gelo-llm-perf-chronicle_dgpu.md` §8/§9 measured
  (provision/prefill/decode/generation all complete, CUDA ~10–16% faster). **Whether
  burn matmul uses real sm_120 tensor cores or silently falls back to an unoptimised
  kernel is UNVERIFIED — investigate (§ Open).**
- **Consequence: the dev-log's headline offload perf (2.83× prefill / 3.6× decode)
  is INVALID** — measured on Vulkan + blackbox = a NaN-producing kernel. The **only
  correct config that has ever existed is Vulkan + `Unit`, which has never been
  benchmarked.**

## The correctness matrix (measured this session)

| | `Unit` (portable) | `blackbox` (tensor-core) |
|---|---|---|
| **Vulkan** (dev) | ✅ correct (0/88 n) | ❌ NaN at all n≥3 (86/88) |
| **CUDA** (prod, sm_120) | ❌ `num_cols % line_size` assert | ❌ nvrtc WMMA "incomplete type" |

Fast gate that reproduces deterministically (model-free, ~16 s):
`cubek_gqa_nan_nsweep` (NEW, in `tests/cubek_prefill_cover.rs`) +
`CUBEK_STRATEGY=blackbox`. Under `Unit` it is 0/88; under `blackbox` it is 86/88 NaN.

## The grounded numerical decomposition (the *regression*, not the collapse)

Distinct from the collapse: the per-layer offload attention error vs an f32 CPU
reference, via the deterministic `cv_cover_fp16_drift_sweep` (same-process, no model)
with the operand dtype toggled by `GELO_CUBEK_DTYPE`:

| operand dtype | kernel floor vs CPU-f32 | C_v conditioning @κ=6 (f32-isolated) |
|---|---|---|
| **f32** | 1.5e-7 (kernel exact) | 4.1e-5 (C_v negligible) |
| **f16** | 4.4e-4 (= f16 ulp 2⁻¹¹) | (1.2e-3 total) |
| **bf16** | 4.2e-3 (8× worse; trips the test's `<5e-3` lock at κ=2) | — |

⇒ there is **no meaningful accuracy *regression*** — the offload matches f32 at the
fp16 floor, dominated by unavoidable f16 operand rounding; `C_v` conditioning is
negligible; the cubek kernel is numerically exact at f32. Token-level greedy
divergence between in-TEE and offload is **expected fp16-greedy noise**, not a cover
regression (see `cover_greedy_parity`'s own docstring). in-TEE pass@1 = **7/20**
(measured; vs llama.cpp ref 6/20). Offload pass@1 is **unmeasured** (do not claim
parity until measured on a *correct* kernel).

## What landed (committed 2026-06-02 on `dgpu-nvidia-bringup`)

The bump + CUDA-default + cubek API-churn fixes + the n-sweep gate were committed.
Contents:
- `crates/gelo-gpu-wgpu/src/lib.rs` — `CubekDtype` enum + `GELO_CUBEK_DTYPE`
  (f16 default / bf16 / f32) plumbed through `cubek_attention_folded` and
  `cubek_attention_folded_gqa`; `CubeWgpuBf16` backend + `array3_to_tensor_bf16`;
  `[cubek-tile]` operand/score probe under `GELO_DEBUG_OFFLOAD`; un-gated the
  `cubecl_common::future` import so the `cuda` build compiles (was a pre-existing
  `cuda`-feature break). **The bf16 path is dead weight given the diagnosis — keep
  only the `GELO_CUBEK_DTYPE` toggle if useful for f32-fidelity experiments; bf16
  should NOT be pursued as the fix.**
- `crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs` — `cubek_gqa_nan_nsweep`
  (the fast collapse gate).
- `crates/gelo-gpu-wgpu/Cargo.toml` — **CUDA set as a DEFAULT feature**
  (`default = ["blas", "cuda"]`) per user instruction (this is the Nvidia box);
  non-CUDA hosts must build `--no-default-features --features blas`. ⚠ With CUDA
  default, any `GELO_GPU_PREFILL_OFFLOAD=1` gate **panics** (cubek broken on sm_120);
  offload-off gates are fine.
- `2603.01499v2.pdf` deletion is unrelated pre-existing.

## Why I couldn't reproduce for most of the session (so you don't repeat it)

- cubek uses a *deterministic* `Inferred` blueprint and is **not** in the autotune
  cache (only burn matmul/reduce are). The autotune cache **persists to disk**
  (`target/autotune-cache`; resolved to `/home/timo/repos/private-rag/target/...`
  on this box) and is `load()`ed every process — so "cross-process nondeterminism"
  doesn't manifest once cached.
- Levers tried that were all CLEAN (on Vulkan+Unit): n-sweep on random data; 4×
  clean-cache fresh-autotune real-prompt runs; the pristine `7cf74b7` at the exact
  κ=6 secure config. The collapse only appeared once `CUBEK_STRATEGY=blackbox` was
  set. **The variable was the kernel strategy, not the commit / cache / data / dtype.**

## Open questions / next steps (in priority order)

1. **Investigate burn-cubecl matmul on sm_120** — does it use real Blackwell tensor
   cores or fall back to an unoptimised kernel? The chronicle §8/§9 prove it *runs*
   and is ~1.2–1.9× faster than Vulkan, but cubecl-cuda 0.9.0's arch table tops at
   100, so sm_120 may be hitting a generic/slow path. Check
   `cubecl-cuda-0.9.0/src/runtime.rs` (`tensor_cores_per_sm`, `supported_wmma_combinations`,
   `contiguous_elements`) + `cubecl-cpp` mma codegen for how 120 is handled in the
   *matmul* path (vs the cubek attention path that hard-fails).
2. **✅ DONE — the version bump was applied and works on CUDA/sm_120** (committed
   2026-06-02). Outcome: Unit clean for all n; blackbox clean at tile-aligned n (incl.
   production n=2048), with a clean config error on non-aligned n. Remaining choice for
   the production prefill path: **pad `n_q` to a tile multiple to use blackbox (tensor
   cores)** vs **use Unit for arbitrary/ragged prompt lengths** (Unit handles any n). The
   historical analysis that led here (corrects an earlier "bump won't help" mis-steer): (this corrects an earlier in-session
   steer toward "bypass / patch" that was based on cubecl's `tensor_cores_per_sm`
   occupancy table omitting 120 — that table is NOT the blocker; the real fixes are
   cubek's tile/line-size hardcode and burn's arch flag, both fixed upstream):
   - **★ Version bump (cubek 0.1.1→0.2.0 + burn-cubecl 0.20.1→0.21.0 + cubecl
     0.9.0→0.10.0) — RECOMMENDED.** Our two exact CUDA failures are *closed upstream
     bugs* shipped in these releases:
     - **cubek #55** (in 0.2.0): the `UnitRoutine` `Inferred` blueprint hardcoded
       `AttentionTileSize{seq_q:4,head_dim:4,seq_kv:4,val_dim:4}` — fine for f32
       (line_size 4) but f16/bf16 need line_size 8 → our exact
       `assert!(num_cols % line_size == 0)`. Fixed.
     - **cubek #81/#102** (in 0.2.0): "early cuda support" + "fix cuda accelerated
       types" → our `blackbox` WMMA `incomplete type` failure.
     - **burn #4622** (closed 2026-03-14, in burn 0.21.0): RTX 50-series/Blackwell
       CUDA compile failures spanning *exactly* burn-cubecl 0.20.1 (PTX JIT) and
       0.21.0-pre.2 (nvrtc `--gpu-architecture`).
     Cost: cubek 0.1→0.2 + burn 0.20→0.21 API churn in our engine
     (`cubek_attention_folded{,_gqa}` use `AttentionGlobalTypes`/`launch` +
     `CubeBackend`/`TensorHandle` bridge — assess churn). **Verify burn #4163 (OPEN —
     "Attention panics with Cuda backend and specific tensor shape [1,30,4096,128]")
     does NOT bite at our per-seq shape [Hq=32, n=2048, d=128] before relying on it.**
   - **Backport patch (only if the bump's API churn is prohibitive):** vendor cubek
     0.1.1 via `[patch.crates-io]` and backport #55 (line-size-aware tile dims, ×8 for
     f16/bf16) + #102 (accelerated WMMA types) + the burn #4622 nvrtc `--gpu-architecture`
     fix for sm_120. Concrete and bounded, but you're re-implementing closed PRs.
   - **Bypass cubek on CUDA (fallback):** route prefill-attention offload through burn
     `fused_attention_batched` (works on CUDA today). Loses cubek's tiled-flash
     `[n,n]`-scores-free memory advantage. Use only if the bump+backport both stall.
3. **Re-measure offload perf on a CORRECT kernel** (Vulkan+Unit, and/or whatever CUDA
   path is chosen) — all existing offload perf numbers are invalid (Vulkan+blackbox).
   Use the canonical bench; expect Unit to be slower on the ~7%-of-offload compute,
   but upload/convert dominates so end-to-end may move little.
4. **Make `cubek_gqa_nan_nsweep` a regression lock** asserting clean under BOTH
   strategies on whatever backend ships (so a broken-kernel config can't silently
   return).
5. **Correct the docs** (task owed): the dev-log `perm-attn-gpu-offload.md`
   ("Cover wired in" subsection) and the superseded bf16 handoff both state the
   wrong f16-overflow mechanism and cite blackbox perf as real. Per
   `feedback-no-debugging-in-paper-devlog`, keep the dev-log to conclusions
   (collapse = blackbox-on-Vulkan kernel bug; perf invalid pending re-measure; bf16
   not the fix) and leave the diagnosis trail here.

## Artefacts (this session, `bench-results/`)

- Collapse repro: `diag-blackbox-nsweep-20260602-104848.log` (86/88 NaN, blackbox/Vulkan)
  vs `diag-nsweep-20260602-103150.log` (0/88, Unit/Vulkan).
- CUDA failures: `diag-cuda-nsweep-20260602-105631.log` + `diag-cuda-n24-20260602-105819.log`
  (Unit line_size assert; blackbox wmma incomplete-type).
- Numerical decomposition: `diag-dtype-floor-20260602-100957.log`, `diag-cv-drift-20260602-100844.log`.
- Non-repro under clean cache / pristine commit: `diag-cleancache-loop-…`, `diag-prevcommit-…`.
- in-TEE pass@1 = 7/20: `bf16-pass1-gate2-20260602-092548.log`.

## Suggested skills

- **`diagnose`** — for step 1 (burn matmul sm_120 path) and validating whichever
  CUDA fix is chosen; the fast gate (`cubek_gqa_nan_nsweep` + `CUBEK_STRATEGY`) is
  already built — reproduce → fix → prove against it, before any slow HumanEval.
- **`grill-me`** (selectors) — to settle the CUDA-fix decision (bypass vs patch vs
  bump) with the corrected facts.
- **`code-review`** — before committing the working-tree changes (decide what to keep:
  the `cuda` default + n-sweep gate yes; the bf16 plumbing probably drop/trim).

## Gotchas

- `--release` mandatory (cubecl shader compile). With `cuda` default, the build needs
  the CUDA toolkit/nvrtc (present: CUDA 13.0, driver 595.71.05, sm_120).
- Don't trust "it ran" as "it's correct" — perf benches time NaN-producing kernels
  silently. Always pair perf with a finiteness/parity check.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit only when asked.
