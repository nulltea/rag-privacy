---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-01
tags: [gelo, dgpu, attention, prefill, decode, gpu-offload, perf, phase-5a]
companion: [perm-attn-gpu-offload, 2026-06-01-feature-rotated-prefill-offload-weights-pub-break]
archive_reason: >
  Supersedes the PERF framing of 2026-06-01-feature-rotated-prefill-offload-weights-pub-break
  (prefill is now wired + O1/O2-optimised, not "blocked behind perf-opt"); that
  doc's SECURITY content (the WEIGHTS-PUB break math + AloePri Phase-5b tasks)
  remains the live reference — hence companion, not supersedes.
---

# Handoff — prefill offload (2.71×) + decode O4(a)/O5 (~4×); next = O6 / AloePri

**One-liner.** Phase-5a perf, largely done. **Prefill** attention offload wired
into the production path (`decoder_block_batched`, default-off): O1 (SIMD
convert) + O2 (un-replicated K/V + on-device GQA expand) → **2.71×** (44.1 →
16.2 s). **Decode** permuted-cover: O4(a) (vectorised `build_covered_prefix`
perm+σ) + O5 (build the cover at the prefill→decode handoff) → decode attention
bucket **9.3 → 3.6 s (~4× vs in-TEE), recurring-only, no break-even K**. All
committed (`a6a17ac`→`b25f0c3`), parity-verified, default-off (security gated on
AloePri). **Two negative results recorded** (dev-log): the upload-bandwidth
anomaly micro-levers are deferred, and the **HD₃ structured-orthogonal cover was
implemented + reverted** (per-`d`-block FWHT loses to BLAS at d=128; needs a
batched feature-axis FWHT — a deferred spike). **Next:** decode **O6** (fused
partial-stats kernel) is the only remaining perf lever; otherwise the real
default-on gate is **AloePri (Phase 5b)**.

The design source of truth is **`docs/dev/logs/perm-attn-gpu-offload.md`** — read
the *Offload perf-upside* + *Sequencing* sections first. This handoff only
captures session deltas + what to do next; do not re-derive the plan.

## What this session did (commits on `dgpu-nvidia-bringup`)

- `a6a17ac` — prefill offload **engine wire-up** + **O1 SIMD convert** + cubek
  prep **instrumentation** (`CUBEK_PROFILE=1`).
- `1c84b2e` — **O2**: un-replicated K/V + on-device GQA broadcast.

New surface (see commits for detail): `cubek_causal_attend` on
`GpuOffloadEngine` + `TrustedExecutor` (takes un-replicated K/V + `group`);
`cubek_attention_folded_gqa` (`gelo-gpu-wgpu/src/lib.rs`, burn `repeat_dim`
expand bridged to cubek's cubecl `TensorHandle` via `into_primitive().tensor()`);
`forward.rs` prefill branch behind `GELO_GPU_PREFILL_OFFLOAD` (GLOBAL only) +
helpers `fold_heads_2d`/`unfold_heads_2d` + `gpu_prefill_offload_enabled`.

**Measured (real engine, Qwen3-4B, B=8, n=2048, RTX 5090/Vulkan, blackbox):**
prefill attention `tee:attn_inplace_many` 44.1 s (in-TEE) → 23.5 s (O1, 1.88×)
→ **16.2 s (O1+O2, 2.71×)**. Full per-op table in the dev-log
(*Prefill offload — real-engine wire-up*).

**Correctness:** `cover_prefill_matches_in_tee` (lib unit test, f32 floor, no GPU)
+ `cubek_folded_causal_parity` (fp16). Both green.

**Two honest caveats (recorded in dev-log):** O1 SIMD convert delivered **~2×,
not ~10×** (half's F16C path is memory-bound); the 15×/35× "ceiling" numbers are
*compute-only ceilings*, not achievable (upload is irreducible) — realized win is
the measured 2.71×. See memory `feedback-ceiling-vs-achievable`.

## Deferred (do NOT pursue now, per this session's scope call)

Remaining prefill micro-levers, in rough EV order if revisited:
1. **Batch the per-sequence loop** — 288 cubek calls/prefill (36 layers × 8 seq);
   one fold per layer would drop per-call launch/alloc overhead.
2. **`O_vᵀ` correction** (`prefill_cover:correct_tee` ≈ 2.6 s) — now the largest
   in-TEE term; fuse the f16→f32 readback into it, or rotate on-device-adjacent.
3. **O3 — upload bandwidth probe** — 267 ms / ~400 MB ≈ **1.5 GB/s, ~8× under
   PCIe5**; likely a staging/non-pinned-copy artifact. If real, un-replication
   already halved it; if artifact, fixing it dwarfs everything.

These are diminishing returns on a default-off, security-blocked path. Stop here.

## Decode optimizations — O4(a) + O5 DONE; O6 remaining

Decode permuted-cover tail-in-TEE wired (`GELO_GPU_RESIDENT_COVER`, σ=0 parity).
Done this session (committed):

- **O4(a) ✅** — vectorised the `build_covered_prefix` perm+σ (row-gather + per-head
  ChaCha σ-on-K, parallel over heads; was a 16.7 M-element per-element-RNG scalar
  loop). `build_covered_prefix` 8.8 → 4.4 s.
- **O5 ✅** — build the cover for all GLOBAL layers at the prefill→decode handoff
  (`build_covered_prefix_all_global` in `run_prefill_batched`; idempotent lazy
  fallback kept). Decode attn bucket **9.3 → 3.6 s (~4× vs in-TEE), no break-even
  K**; the build (5.4 s) relocates to prefill.
- **O4(b) signed-perm — deferred** (entropy collapse forfeits `WEIGHTS-BLIND`).
- **HD₃ — tried + reverted** (regressed: per-`d`-block FWHT loses to BLAS at
  d=128; needs a batched feature-axis FWHT — transpose-based, uncertain at d=128).
  See dev-log *Structured-orthogonal cover — tried, reverted*.

**Remaining: O6 — fused partial-stats kernel.** Collapse the 5-dispatch
`prefix_partial_gpu` (≈3 ms × 1152, the composed `attend_session_partial`:
matmul→max_dim→sub+exp→sum_dim→matmul + per-call `repeat_dim`) into one
cubek/FlashAttention-D kernel emitting `(m,l,acc)`. The one genuinely new kernel;
attacks the recurring per-step term (now the dominant decode cost after O5).
Largest effort — grill scope first.

## And the real default-on gate (not perf): AloePri (Phase 5b)

Both offload covers leak token **membership** under `WEIGHTS-PUB` (norm/Gram
dictionary; prefill top-1 = 1.000). Perf wires are default-off until **covariant
weight obfuscation** removes the public-weight dictionary. Tasks in the dev-log
*Phase 5b*. This is what flips either offload default-on; the prior handoff
`2026-06-01-feature-rotated-prefill-offload-weights-pub-break.md` (now superseded
for the perf parts) has the attack math + reproduce commands.

## Reproduce / bench

Canonical invocations are in the dev-log *Reproduce* block and `CLAUDE.md`.
Quick refs:
- Prefill offload bench: prepend `GELO_GPU_PREFILL_OFFLOAD=1 CUBEK_STRATEGY=blackbox`
  to the `gelo_llm_prefill_decode_breakdown` invocation (CLAUDE.md). Compare
  `tee:attn_prefill_offload` vs the flag-off `tee:attn_inplace_many` (≈44 s).
- Prefill prep per-stage split: `CUBEK_PROFILE=1 CUBEK_STRATEGY=blackbox` on
  `cubek_prefill_cover` `prefill_attention_breakdown`.
- Prefill parity (fast, no GPU): `cargo test -p gelo-embedder --lib cover_prefill_matches_in_tee`.
- Decode cover bench: `GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 …`
  (dev-log Reproduce).

## Gotchas / environment

- `--release` mandatory for anything touching cubecl (debug = minute-scale shader
  compile).
- **`crates/gelo-gpu-wgpu/tests/parity.rs` has a PRE-EXISTING breakage** on this
  branch (`GpuContext` field drift: `vendor`/`device`/`driver`) — unrelated to
  attention; don't be alarmed, not introduced here.
- Host box is **Rust-only** (no pip/numpy/sudo); Python attacks run in the
  `gelo-attack` Podman container.
- Commit identity `Timo <timofey.luin@gmail.com>`; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit
  only when asked. Branch `dgpu-nvidia-bringup` (not master).
- The on-device-expand bridge (`into_primitive().tensor()` → `TensorHandle`)
  relies on the burn tensor + cubek launch sharing the **same client/device**
  (both `Dev::default()`); keep that invariant if refactoring.

## Suggested skills

- **`grill-me`** — for the O4 signed-permutation / cover-cadence design forks and
  the O5 prefill-overlap shape, before implementing (user prefers being grilled
  on design forks; explain mechanism in prose before structured options — memory
  `feedback-elaborate-before-options`).
- **`code-review`** — before committing the decode `build_covered_prefix` changes
  (touches the parity-critical cover path).
- **`diagnose`** — if an O4 change breaks the σ=0 greedy byte-parity, for a tight
  loop to find why.
