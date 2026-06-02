---
type: handoff
status: stale
created: 2026-06-01
updated: 2026-06-01
tags: [gelo, dgpu, attention, prefill, decode, gpu-offload, perf, phase-5a]
companion: [perm-attn-gpu-offload]
superseded_by: 2026-06-02-attn-offload-cv-cover-gate
archive_reason: >
  Unified into 2026-06-02-attn-offload-cv-cover-gate (which folds in the
  later loop-batching + fused-O_vᵀ + dispatch-granularity findings). Superseded
  in full.
---

# Handoff — prefill 2.83× + decode ~4×; next = cubek read-index / O6 / AloePri

**One-liner.** Phase-5a perf, largely done — all default-off, parity-verified,
committed (`a6a17ac`→`4499898`). **Prefill** attention offload wired into the
production path (`decoder_block_batched`, `GELO_GPU_PREFILL_OFFLOAD`): O1 (SIMD
convert) + O2 (un-replicated K/V + on-device GQA expand) + loop-batching (one
fold/rotate over B·Hq) + fused `O_vᵀ` → **2.83×** (44.1 → 15.6 s). **Decode**
permuted-cover: O4(a) (vectorised `build_covered_prefix` perm+σ) + O5 (build the
cover at the prefill→decode handoff) → decode attention bucket **9.3 → 3.6 s
(~4× vs in-TEE), recurring-only, no break-even K**. **Three findings recorded
(dev-log):** HD₃ cover implemented + reverted (per-`d`-block FWHT loses to BLAS
at d=128); cubek dispatch settled per-sequence via a controlled warm A/B
(per-seq **1.48× faster** than one big dispatch — a materialised-GQA-expand
artifact, not "small is better"); O3 upload is `queue.write_buffer` staging,
alloc/submit-bound. **Next perf levers:** prefill **cubek kv-head read-index**
(kills the materialised expand → lets the single dispatch win + cuts `cubek_gpu`;
the biggest remaining prefill upside, touches cubek) and decode **O6** (fused
partial-stats kernel). The real default-on gate is **AloePri (Phase 5b)**.

The design source of truth is **`docs/dev/logs/perm-attn-gpu-offload.md`** — read
the *Offload perf-upside* + *Sequencing* sections first. This handoff only
captures session deltas + what to do next; do not re-derive the plan.

## What this session did (commits on `dgpu-nvidia-bringup`)

- `a6a17ac` — prefill engine **wire-up** + **O1 SIMD convert** + cubek prep
  **instrumentation** (`CUBEK_PROFILE=1`).
- `1c84b2e` — **O2**: un-replicated K/V + on-device GQA broadcast (→ 2.71×).
- `8606bd0` — decode **O4(a)**: vectorised `build_covered_prefix` perm+σ.
- `03c525a` — rename `create_build`→`build_covered_prefix`; HD₃ negative result.
- `b25f0c3` — decode **O5**: build the cover at the prefill→decode handoff.
- `a35e041` — handoff refresh.
- `4499898` — prefill **loop-batching + fused `O_vᵀ`** (→ 2.83×); cubek dispatch
  granularity settled (per-seq); O3 probe findings.

Key new surface: `cubek_causal_attend` (un-replicated K/V + `group`) on
`GpuOffloadEngine`/`TrustedExecutor`; `cubek_attention_folded_gqa`
(`gelo-gpu-wgpu/src/lib.rs`, burn `repeat_dim` expand bridged to cubek's cubecl
`TensorHandle` via `into_primitive().tensor()`); `forward.rs` prefill branch
(`GELO_GPU_PREFILL_OFFLOAD`, GLOBAL only) with uniform-batched fold/rotate +
`correct_unfold_into` (fused `O_vᵀ`+unfold) + per-seq cubek + ragged fallback;
`build_covered_prefix_session`/`_all_global` (decode O5). Tests:
`cover_prefill_matches_in_tee`, `cover_prefill_batched_matches_in_tee`,
`hd3_cover_roundtrips_and_cancels` (in `forward.rs`), `cubek_dispatch_granularity`
(in `cubek_prefill_cover.rs`).

**Honest caveats (dev-log):** O1 SIMD convert delivered **~2×, not ~10×**
(memory-bound); the 15×/35× figures are *compute-only ceilings*, not achievable
(upload irreducible) — realised prefill win is **2.83×**, decode **~4×**. See
memory `feedback-ceiling-vs-achievable`.

## Deferred prefill levers (in rough EV order)

1. **cubek kv-head read-index** *(biggest upside, touches cubek)* — broadcast K/V
   in cubek's loader instead of the materialised `repeat_dim` expand. The
   controlled A/B (`cubek_dispatch_granularity`) showed the materialised expand
   makes one big `bh=B·Hq` dispatch 1.48× *slower* than B per-seq ones; the
   read-index removes that, would let the single batched dispatch win, and cuts
   `cubek_gpu` (the largest prefill bucket, ~6–9 s). Currently worked around by
   dispatching cubek per-sequence.
2. **O3 — upload via pinned/persistent buffers.** Probed: cubecl-wgpu uses
   `queue.write_buffer` (staging belt); ~1.5 GB/s is **alloc/submit-bound**, not
   a double-convert. A persistent/pinned operand buffer would help but is a
   deeper cubecl-side change; un-replication (O2) already captured the easy win.

The loop-batching + fused-`O_vᵀ` levers from the prior handoff are **DONE**
(commit `4499898`).

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
