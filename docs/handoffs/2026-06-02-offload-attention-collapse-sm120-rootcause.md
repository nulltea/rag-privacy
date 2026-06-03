---
type: handoff
status: current
created: 2026-06-02
updated: 2026-06-03
tags: [gelo, dgpu, attention, gpu-offload, cubek, cuda, sm120, vulkan, blackbox, accuracy, humaneval, diagnosis]
companion: [perm-attn-gpu-offload, 2026-06-02-attn-offload-cv-cover-gate]
supersedes: [2026-06-02-bf16-attention-kernel-fix]
---

# Handoff — offload attention on CUDA/sm_120: collapse root-caused, stack bumped, kernel running

## Status (2026-06-03) — what landed and what's next

- **✅ CUDA support for the offloaded attention kernel is implemented.** The stack
  bump (cubek 0.2.0 / burn 0.21.0 / cubecl 0.10.0; commit `8189d17`) makes the cubek
  prefill-attention offload run correctly on the RTX 5090 (sm_120), which it never did
  before. CUDA is a default build feature on this box.
  - **Unit kernel: correct at any n** (incl. ragged prompt lengths). This is the
    no-caveat production kernel today.
  - **blackbox (tensor-core) kernel: correct at tile-aligned n** (incl. n=2048), 2.45×
    faster on the GPU attend than Unit — but **does NOT support ragged inputs**: it
    rejects non-tile-aligned n_q with a clean config error ("Stage seq_q must divide
    problem seq_q"), a cubek-0.2.0 accelerated-path limitation (not a crash, not ours).
- **⏭ Next steps:** investigate proper fixes for the **ragged blackbox kernel** —
  either (a) caller-side `n_q` padding to a tile multiple (+ slice-back; sound under
  causal masking) in the prefill-offload path, or (b) an upstream cubek fix (the
  `check_bounds` machinery to mask partial query tiles is already half-present in the
  blackbox blueprint — its `validate()` just guards seq_q-divisibility conservatively).
  Mature FlashAttention kernels and cubek's own Unit kernel handle ragged n, so this is
  liftable. Plus: run the end-to-end HumanEval pass@1 on CUDA, and decide Unit (ragged-
  safe) vs blackbox+padding (tensor-core perf) for the production prefill path.

**Corrected perf** (CUDA, RTX 5090, B=8 n=2048 K=32, warmed, κ=6; vs in-TEE prefill
44 062 ms / decode 14 574 ms). These **replace the documented Vulkan figures, which
were measured on `CUBEK_STRATEGY=blackbox` = the broken NaN kernel and are invalid**:

| phase | CUDA Unit | CUDA blackbox | documented Vulkan (invalid) |
|---|--:|--:|--:|
| prefill attn (`tee:attn_prefill_offload`) | 24 659 ms (1.79×) | 13 952 ms (3.16×) | 15 090 ms ("2.92×", NaN kernel) |
| └ `cubek_gpu` (GPU attend) | 18 034 ms | 7 352 ms (2.45× faster) | 8 248 ms (NaN) |
| decode attn (`tee:attn_resident_cover`) | 3 008 ms (4.85×) | 2 775 ms (5.25×) | 4 042 ms ("3.6×") |

CPU cover controls (`rotate_tee` ~3.46 s, `correct_tee` ~1.63 s) match the documented
Vulkan run → same machine, clean comparison. Decode is strategy-invariant (no cubek);
its CUDA gain over Vulkan is the faster burn-ops attend (`prefix_partial_gpu` ~1.2 s vs
2.4 s). blackbox's 2.45× attend is Amdahl-capped to ~6% prefill wall (attention ≈ 12%
of prefill; matmuls dominate) — its case strengthens at long context (n=8192, untested).
Single-sample; ±~7% cross-run variance on wall/decode.

The full diagnosis is below; the dev-log
[`perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md) carries the
conclusions.

## Ragged blackbox kernel — confirmed an unaddressed upstream gap (research 2026-06-03)

Before committing to a fix, the cubek upstream + fork landscape was swept
(authenticated GitHub API + source read of `main`). **Conclusion: the
ragged-`seq_q` limitation of the tensor-core (`BlackboxAccelerated`) kernel is a
genuine, still-open upstream gap — not a missing config and not an in-flight fix.**

- **Not an algorithmic limit of tensor cores.** MMA instructions are fixed-tile
  (m16n8k16 / WMMA 16³ / Blackwell `tcgen05`), but ragged `n` is handled
  industry-wide by `ceil_div`-ing the tile count and *predicating* the boundary
  tile (FA2/3/4, CUTLASS, cuDNN, Triton all do this). cubek's own **`Unit`
  routine already runs ragged `n`** — same FlashAttention algorithm, no guard.
- **Root cause = one unchecked reader.** The accelerated query reader
  (`components/global/simple/reader/query.rs::get_tile`) slices the query tile
  straight from gmem via the *unchecked* `to_source_pos` path, so an out-of-range
  row is UB. **PR #150** (merged 2026-03-30) chose to **reject** non-divisible
  `seq_q` (`routines/.../blackbox_accelerated.rs::validate` → `"Stage seq_q must
  divide problem seq_q"`) rather than predicate. The predication primitive —
  `AttentionGlobalLayout::is_in_bounds` / `to_source_pos_checked`, gated by the
  comptime `check_bounds.seq_q` flag — **already exists and is used by every other
  reader and by the `Unit` query path**; the accelerated query read is the sole
  consumer that ignores it (the flag is computed in `blueprint.rs` and plumbed in
  `setup.rs`, just not consulted on this load).
- **No upstream or fork fix exists (authenticated API sweep, 2026-06-03):**
  - Latest release **0.2.0**; dev head **`0.3.0-pre.1`** (`main`) — both the
    reject and the unchecked read survive verbatim (confirmed by reading `main`;
    the crate was reorganised into `forward/`+`backward/` but the logic is
    unchanged).
  - **Code search:** the reject string exists in exactly one file; `is_in_bounds`
    appears in the attention crate only in `global/layout.rs` (the primitive) and
    `tile/mask.rs` — never in a query reader.
  - **Issues (open+closed):** 0 mentioning seq_q / ragged / predicate /
    partial-tile / variable-length.
  - **PRs (open+closed):** only #150 (*added* the reject), #55 (Unit line-sizes,
    already in 0.2.0), #141 (swizzle) — none fix it. The 4 open PRs are all
    unrelated/stale (matmul launch-on-tile, sort, tracing ×2).
  - **Branches (20):** none on attention seq_q. **Forks (all 34):** only 3 ahead
    of upstream, all trivial (cubecl bump; gemv-layout validate; matmul m-size +
    version bumps) — none touch attention predication.
  - The team is *actively* developing attention (a full backward pass landed,
    #281/#286; tile/softmax refactors) — so it is not abandoned — but this forward
    ragged-`seq_q` gap is simply not on their board. Likely because `Unit` covers
    ragged and their accelerated-path consumers use fixed/padded shapes; our
    confidential-serving prefill is an unusual caller that needs tensor cores *and*
    ragged prompts at once.
- **⇒ The fix is ours to write** (and worth upstreaming, since the primitive
  already exists). Two routes — see the dev-log:
  - **(a) caller-side `n_q` padding** to a multiple of `elements_in_stage_seq_q()`
    + slice-back. Sound under causal masking (phantom query rows attend real keys,
    discarded). Overhead ≈ `(E−1)/n` ≤ ~0.7% at n=2048 (E = stage seq_q extent).
    Zero upstream change — **ships now**, keeps the ~2.45× blackbox win.
  - **(b) vendored `[patch.crates-io]` cubek fork**: route the accelerated query
    load through `to_source_pos_checked`/`is_in_bounds` and relax `validate()`.
    Near-zero overhead; carry a fork until upstreamed.
  - Validate either against the existing **`cubek_gqa_nan_nsweep`** gate under
    `CUBEK_STRATEGY=blackbox` at ragged `n`.
- **⏳ Open before committing:** evaluate alternative GPU kernel libraries vs a
  cubek patch (burn-cubecl native attention; candle-flash-attn / Dao CUDA; cuDNN
  via cudarc; the Rust-CUDA ecosystem) — in progress, see the live chat / a
  follow-up note.

---

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
