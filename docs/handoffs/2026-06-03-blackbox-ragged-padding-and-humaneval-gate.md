---
type: handoff
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, dgpu, attention, gpu-offload, cubek, blackbox, ragged, humaneval, accuracy, perf, cv-cover]
companion: [perm-attn-gpu-offload, gpu-offloaded-attention-with-value-cover]
supersedes: [2026-06-03-candle-fa-spike-and-cubek-ragged-patch]
---

# Handoff — blackbox ragged-prompt offload (caller-side padding) + HumanEval acceptance gate PASS

## TL;DR

- **Ragged-prompt support for the blackbox (tensor-core) prefill offload shipped**
  via **caller-side `n_q` padding** (not a cubek patch — see *Decision* below).
- **HumanEval accuracy acceptance gate PASSES: pass@1 = 7/20 = in-TEE parity**
  (offload + blackbox + `C_v` κ=6, Qwen3-4B; llama.cpp plain = 6/20). No regression
  from offloading or from securing it.
- **The secure offloaded path is now the DEFAULT inference path** —
  capability-gated: offload + resident cover + `C_v` κ=6 + σ=0.01 engage whenever
  the executor supports the fused kernel, with a transparent in-TEE fallback
  otherwise. Verified to engage with **no env set** (committed `9f6ec0a`).
- **CUDA stays the default backend; a new `vulkan` feature overrides it**
  (committed `58c7567`) — `--features vulkan` (override on a CUDA box) or
  `--no-default-features --features blas,vulkan` (lean, no CUDA toolkit).
- Everything this session is **committed** on `dgpu-nvidia-bringup` (`9f6ec0a`,
  `58c7567`; **not pushed**). The `WEIGHTS-PUB`/`C_v` security acceptance is
  **enacted** with documented caveats (see *Remaining work* #1).

Design write-up: [`gpu-offloaded-attention-with-value-cover.md`](../dev/prototype/gpu-offloaded-attention-with-value-cover.md).
Living log: [`perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md).
cubek-gap diagnosis + upstream/fork research: [`2026-06-02-offload-attention-collapse-sm120-rootcause.md`](2026-06-02-offload-attention-collapse-sm120-rootcause.md).

## Decision (this session) — padding, not a cubek patch

The cubek ragged-`seq_q` limit is a **genuine, unaddressed upstream gap** (confirmed
by an authenticated GitHub API sweep — no release/issue/PR/branch/fork fix; recorded
in the rootcause handoff). Two fixes were on the table: **(a) caller-side `n_q`
padding** (our code, no fork) vs **(b) an in-kernel cubek predication patch**.

Investigating (b) showed it is **real kernel work, not a 2-line change**: the
blackbox query reader (`cubek-attention/.../reader/query.rs::get_tile`) reads the
query tile straight from gmem (`to_linear_slice`, unchecked), distinct from the
K/V `FullStageGlobalReader` which staged-reads with bounds — so a correct fix needs
smem-staged/predicated query loads (or a new checked-read primitive), with
GPU-loop iteration. Given that, and that padding is <1% overhead at n=2048, the
user chose **(a)**. (b) is parked (see *Parked alternatives*).

## What landed (committed on `dgpu-nvidia-bringup` — `9f6ec0a`, `58c7567`)

1. **Caller-side `n_q` padding** — `crates/gelo-gpu-wgpu/src/lib.rs`:
   `pad_seq_q_zeros()` to a multiple of `BLACKBOX_SEQ_Q_ALIGN` (=16; the stage
   extent for the `num_planes=1`/`partition=1` hint × fp16 MMA m=16), wired into
   `cubek_attention_folded` and `cubek_attention_folded_gqa` via `CowArray`
   (borrowed when aligned, owned when padded), slicing phantom rows off the
   output. Padding is keyed off the resolved kernel (see #5's `cubek_strategy`),
   not a standalone env check. K/V left ragged (cubek's staged reader masks
   `seq_kv`). Correct because cubek masks causal on **absolute** positions, so
   real rows are unaffected and zero phantom rows attend real keys then get
   discarded.
   - **Validated** (`crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs`,
     `CUBEK_STRATEGY=blackbox`): `cubek_gqa_nan_nsweep` clean (0 NaN) across ragged
     n = 8…2048; `cubek_folded_causal_parity` (ragged n=24) max_abs **2.7e-5** vs
     CPU. Unit path unaffected. **Small-KV floor:** n_kv below the tensor-core tile
     (a handful of tokens) has no valid MMA tile — degenerate only.
2. **HumanEval gate instrumentation** — `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`
   `humaneval_gate_generate`: per-prompt `profile` capture accumulated into a
   cumulative per-op table (prefill `tee:attn_prefill_offload`/`prefill_cover:*` +
   decode `tee:attn_resident_cover`/`cover:*` + shared matmul/mask), with
   prompt-length stats. New env knobs: `GELO_HE_LIMIT` (cap prompts; smoke=1),
   `GELO_HE_PROFILE_EACH` (per-prompt dump). ⚠ `GELO_HE_OUT` must be **absolute**
   (cargo-test CWD is the crate dir, not the repo root).
3. **Dev-log update** (`perm-attn-gpu-offload.md`): compressed the falsified-fp16
   debugging narrative to conclusions + handoff refs (paper-devlog rule), fixed the
   stale "blackbox refuses ragged n" claims → "landed via padding", recorded the
   7/20 pass, added `---` section separators, renamed the duplicate `## Threat
   model` header.
4. **Prototype-note created**: `docs/dev/prototype/gpu-offloaded-attention-with-value-cover.md`
   (6-section design+results description).
5. **Secure offload made the DEFAULT (`9f6ec0a`)** — capability-gated, not a hard
   flag flip:
   - New capability predicate `supports_offloaded_attention()` on the
     `GpuOffloadEngine` + `TrustedExecutor` traits (`substrate.rs`), overridden by
     `WgpuVulkanEngine` (= `self.fp16`, `lib.rs`) and delegated by
     `InProcessTrustedExecutor` (`sim.rs`). Default false → CPU/in-TEE executors
     keep the in-TEE path (so the CPU test suite is unaffected; a hard flip would
     have errored since `cubek_causal_attend` returns `Err` without a fused engine).
   - `forward.rs`: `GELO_GPU_PREFILL_OFFLOAD` / `GELO_GPU_RESIDENT_COVER` are now
     `Option` overrides; the default resolves to
     `exec.supports_offloaded_attention()`. Secure cover defaults flipped: σ
     `0.0→0.01`, κ `1.0→6.0` (env still overrides; `GELO_COVER_KAPPA=1` =
     legacy/insecure orthogonal `O_v`).
   - `lib.rs`: `cubek_strategy(n_kv)` — **blackbox by default on CUDA**, Unit on
     Vulkan/non-CUDA and for **tiny `n_kv` < `BLACKBOX_MIN_NKV` (16)** (the
     production guard — blackbox can no longer panic on degenerate prompts).
     Replaces the old `cubek_strategy_from_env` / `cubek_blackbox_seq_q_align`.
   - **Verified**: a zero-env HumanEval smoke engages offload+cover+blackbox+κ=6 by
     default (`tee:attn_prefill_offload` + `prefill_cover:cubek_gpu` +
     `tee:attn_resident_cover` all fire), coherent output on a ragged prompt, no
     NaN. Workspace libs + the two key test targets compile.
6. **`vulkan` feature to override the CUDA default (`58c7567`)** — `Cargo.toml`:
   `default = ["blas", "cuda"]` unchanged; added `vulkan = []`. cfgs are exact
   complements (CUDA = `all(cuda, not(vulkan))`, Vulkan = `any(vulkan, not(cuda))`),
   so `vulkan` wins when set; `cubek_strategy`'s blackbox default tracks the same
   cfg (Vulkan → Unit, since blackbox NaNs there). `gpu_ctx`'s device-type match
   now keys on the Debug string so the cuda+vulkan combined build (two
   `wgpu_types` versions) compiles. All three configs check clean.

## Acceptance gate result (2026-06-03)

`GELO_GPU_PREFILL_OFFLOAD=1 GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 GELO_COVER_KAPPA=6 GELO_BENCH_VARIANT=4b CUBEK_STRATEGY=blackbox`,
20 prompts, `GELO_HE_MAXTOK=384` → **`completions.jsonl`** → `score.py`:

- **pass@1 = 7/20 = in-TEE GELO baseline (7/20)**; plain Qwen3-4B/llama.cpp = 6/20.
  `B ≥ A−ε` (ε=0) → cell C (κ=1) not needed.
- All 20 prompts (38–391 tok, ragged) generated cleanly on blackbox via padding,
  no NaN/launch failure. The ~6 misses are model-quality artifacts (markdown/prose
  contamination, logic errors) of the same class as in-TEE — not offload defects.
- Production-shape perf (from the dev-log, B=8 n=2048, κ=6): prefill **1.79× (Unit) /
  3.16× (blackbox)**, decode **4.85× / 5.25×** vs in-TEE; securing with `C_v` is
  perf-free. (HumanEval prompts are short/decode-dominated; the prefill-heavy
  production shape is where blackbox's 3.16× lands.)

## Remaining work (priority order)

1. **Deepen the `WEIGHTS-PUB`/`C_v` security validation.** Acceptance is **enacted**
   (default-on shipped), but the gate was measured at layers 0/35 only. Owed: sweep
   **production context lengths (2k–16k)** — covariance-alignment may strengthen as
   instance→population covariance at long context — and a policy call on whether the
   full-vocab top-5 ≈ 0.047 residual is acceptable.
2. **Push** `dgpu-nvidia-bringup` (commits are local only) — when ready.
3. **Make `cubek_gqa_nan_nsweep` a dual-strategy regression lock** — start the
   blackbox sweep at `BLACKBOX_MIN_NKV`; it asserts no-NaN at n=1, which now routes
   to Unit anyway, but the test still forces raw blackbox, so align it with the guard.
4. **Pre-existing stale GPU bench/spike tests don't compile** against the bumped
   burn/cubecl/cubek 0.2.0 API (`q4_kernel_spike`, `parity`, `qwen3_overhead_*`,
   `gelo_overhead_bench`, `cubek_attention_spike` — `GpuContext` fields, private
   `Shape.dims`, `cubek::launch::launch`). Unrelated to this work and broken before
   it; a cleanup pass would green the full `--tests` build.
5. **Open perf levers (no security/accuracy impact)**: cubek kv-head read-index
   (kills the materialised GQA expand), upload-bandwidth probe.

*(Done this session: caller-side ragged padding; the HumanEval gate + per-op
instrumentation; default-on enablement with the tiny-`n_kv`→Unit guard; the `vulkan`
feature; all docs — committed `9f6ec0a` + `58c7567`. The KV-floor guard and the
commit, previously listed here as TODO, are done.)*

## Parked alternatives (not pursued; revisit only if needed)

- **In-kernel cubek patch (b).** Fork cloned at `/home/timo/cubek`, branch
  `fix/blackbox-ragged-seqq` off `release/0.2` (= our pinned cubek 0.2.0, cubecl
  0.10.0 — compatible; `main`/0.3.0-pre pins a git-rev cubecl and is **not**
  compatible). No commits. The two edits would be: predicate the query load in
  `reader/query.rs::get_tile` (route through `to_source_pos_checked`/`is_in_bounds`)
  + relax `routines/blackbox_accelerated.rs::validate()`; prove on
  `cubek_gqa_nan_nsweep`. Worth upstreaming if pursued.
- **candle-flash-attn / cuDNN / Rust-CUDA.** Evaluated, **not** spiked: burn's
  attention *is* cubek (no escape); candle-flash-attn (Dao) is the strongest kernel
  but a different runtime (cudarc, not cubecl) → buffer-bridging out of cubecl +
  CUDA-only + sm_120 support unverified (FA2 tops at sm90); only a fallback if we
  ever leave cubecl. Details in the rootcause handoff's alternatives note.

## Gotchas

- `--release` mandatory; CUDA is the default backend (override: `--features vulkan`);
  `GELO_HE_OUT` must be absolute. Perf is single-sample (±~7%).
- `BLACKBOX_SEQ_Q_ALIGN` / `BLACKBOX_MIN_NKV` (both 16, `lib.rs`) track
  `cubek_strategy`'s blackbox partition hint (`num_planes=1`/`partition=1` × fp16 MMA
  m=16); update them together if that hint changes.
- The backend cfgs are exact complements — keep any new `feature = "cuda"` /
  `not(feature = "cuda")` cfg in the `all(cuda, not(vulkan))` / `any(vulkan, not(cuda))`
  form so `vulkan` keeps overriding.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Suggested skills

- **`code-review`** — before committing the working tree (padding + instrumentation
  + dev-log/prototype docs).
- **`verify`** — re-confirm the 7/20 + blackbox-ragged behaviour if any of the
  offload path changes before commit.
