---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-01
tags: [gelo, dgpu, attention, gpu-offload, prefill, decode, perf, phase-5a, weights-pub, aloepri]
companion: [perm-attn-gpu-offload]
supersedes: [2026-06-01-feature-rotated-prefill-offload-weights-pub-break, 2026-06-01-prefill-offload-o1-o2-decode-next]
---

# Handoff — GPU attention offload: Phase-5a perf landed (prefill 2.83×, decode ~4×); next gate = AloePri

**One-liner.** Both GPU attention offload paths are now **wired into the
production forward path, perf-optimised, parity-verified, and committed** — but
**default-OFF**, because the covers leak token membership under `WEIGHTS-PUB`.
The remaining work splits cleanly into **more perf** (a few levers, optional)
and **the one thing that actually ships it: AloePri** (covariant weight
obfuscation, the security gate that flips default-on).

**Design source of truth: `docs/dev/logs/perm-attn-gpu-offload.md`** — read its
*Offload perf-upside*, *Gate-3 @ `WEIGHTS-PUB`*, and *Sequencing* sections. This
handoff is the current-state + next-step index; it does not re-derive the
attack math or the design (those live in the dev-log).

## State — what's done (commits `a6a17ac`→`4499898` on `dgpu-nvidia-bringup`)

All default-off, parity-verified at the fp16/f32 floor. Measured on Qwen3-4B,
B=8, n=2048, RTX 5090 / Vulkan, `CUBEK_STRATEGY=blackbox`.

**Prefill offload** (`GELO_GPU_PREFILL_OFFLOAD`, GLOBAL layers, in
`decoder_block_batched`): feature-rotation cover (`O_qk`/`O_v`, dense Haar) →
`cubek-attention` (tensor cores) → in-TEE `O_vᵀ`. Levers landed:
O1 SIMD f16 convert · O2 un-replicated K/V + on-device GQA expand · loop-batching
(one fold/rotate over B·Hq) · fused `O_vᵀ`+unfold. **Prefill attention bucket
`tee:attn_inplace_many` 44.1 s (in-TEE) → `tee:attn_prefill_offload` 15.6 s =
2.83×.**

**Decode offload** (`GELO_GPU_RESIDENT_COVER`, GLOBAL layers): permutation cover
(`perm_kv` + σ-on-K + `O_qk`/`O_v`) + **tail-in-TEE** (frozen-prefix partial-stats
on GPU + in-TEE tail + online merge). Levers landed: O4(a) vectorised
`build_covered_prefix` perm+σ · O5 build the covered prefix at the
prefill→decode handoff (off the decode critical path). **Decode attention bucket
`tee:attn_resident_cover` 14.6 s (in-TEE) → 3.6 s = ~4×, recurring-only, no
break-even K** (the one-time build now lands in prefill).

## Why both are default-OFF — the security gate (`WEIGHTS-PUB`)

Both covers are exactly correctable (orthogonal rotation / permutation undone
in-TEE) and cheap, **but neither hides token membership under `WEIGHTS-PUB`**
(adversary knows the public Qwen3 weights). The per-head value **norm** is
invariant to `O_v`, permutation, and σ, so matching it against a public-weight
per-token dictionary recovers the prompt's bag-of-tokens (measured **top-1 =
1.000**, prefill; decode leaks membership, order held). Full math + gate-3
results: dev-log *Gate-3 @ `WEIGHTS-PUB`* and *Phase-5 spike*. ⇒ **the perf wires
stay default-off until `WEIGHTS-PUB` is neutralised — that is AloePri.**

## Deferred — perf (optional, diminishing on a default-off path)

1. **Prefill `cubek` kv-head read-index** — *biggest perf upside; touches cubek.*
   A controlled warm A/B (`cubek_prefill_cover::cubek_dispatch_granularity`)
   showed cubek's **materialised** GQA expand makes one big `bh=B·Hq` dispatch
   **1.48× slower** than B per-seq ones (~268 MB `repeat_dim` blow-up), so prefill
   currently loops cubek per-sequence. A broadcast **read-index** (no
   materialisation) removes that, lets the single batched dispatch win, and cuts
   `cubek_gpu` (the largest prefill bucket, ~6–9 s).
2. **O3 — upload via pinned/persistent buffers.** Probed: cubecl-wgpu uploads via
   `queue.write_buffer` (staging belt); the ~1.5 GB/s is **alloc/submit-bound**,
   not a double-convert. Pinned/persistent operand buffers would help — deeper
   cubecl-side change; O2's un-replication already took the easy win.
3. **Decode O6 — fused partial-stats kernel.** Collapse the 5-dispatch
   `prefix_partial_gpu` (≈3 ms × 1152: matmul→max→exp→sum→matmul + per-call
   `repeat_dim`) into one cubek/FlashAttention-D kernel emitting `(m,l,acc)`. The
   one genuinely new kernel; attacks the recurring per-step term (now dominant
   after O5). Largest effort — grill scope first.

## Deferred — negative results (recorded so they're not re-tried blindly)

- **HD₃ structured-orthogonal cover** — implemented end-to-end + parity-tested,
  then **reverted**: applied per-`d`-block via `Hd3Mask::apply_in_place_slice`, it
  loses to the BLAS dense rotate at `d=128` (prefill `rotate_tee`/`correct_tee`
  regressed). A *batched feature-axis* FWHT (transpose-based) might realise the
  `O(d·log d)` win but transposes likely erode it at d=128 — a deferred spike,
  not the lever.
- **O4(b) signed-permutation cover** — deferred: entropy collapse forfeits the
  `WEIGHTS-BLIND` content-hiding the dense rotation provides.

## Deferred — the real default-on gate: AloePri (Phase 5b)

**Covariant weight obfuscation.** Statically transform deployed weights `W → W'`
with a covariant compensation so the computation is unchanged but the attacker's
*public* `W` no longer matches `W'`, collapsing the per-token value-norm/Gram
dictionary to *unknown* — converting the open-model `WEIGHTS-PUB` deployment into
an effective `WEIGHTS-BLIND` one. Tasks (dev-log *Phase 5b*): read the
`evals/aloepri-attacks/` material; pin whether a static `W'` destroys the
`‖V(t)‖` dictionary while staying exactly correctable **and** fused-kernel
compatible; add an obfuscation mode to `attn_cover_capture.rs` and re-run
`gate3_prefill_dict.py` (cheap, decisive). **If it clears → flip default-on
behind the c5 AloePri condition. If not → prefill stays in-TEE; decode ships only
if the bag-of-tokens residual is accepted.**

## Deferred — scale

- **NVMe `SpillProvider`** for beyond-VRAM context (n_kv ≳ 16–32k). Designed-in
  (null provider today); out of the current production shape.

## Key code surface (this work)

- Engine/executor: `cubek_causal_attend` (un-replicated K/V + `group`) on
  `GpuOffloadEngine`/`TrustedExecutor`; `cubek_attention_folded{,_gqa}`
  (`gelo-gpu-wgpu/src/lib.rs`; the `_gqa` variant bridges burn `repeat_dim` →
  cubek cubecl `TensorHandle` via `into_primitive().tensor()`).
- Prefill (`forward.rs`): branch behind `GELO_GPU_PREFILL_OFFLOAD`;
  `fold_heads_2d{,_batched}` / `unfold_heads_2d` / `correct_unfold_into` (fused
  `O_vᵀ`+unfold); uniform-batched fold/rotate + per-seq cubek + ragged fallback.
- Decode (`forward.rs`/`kv_cache.rs`): `build_covered_prefix_session` +
  `build_covered_prefix_all_global` (O5 handoff build); `DecodeCover`.
- Tests: `cover_prefill_matches_in_tee`, `cover_prefill_batched_matches_in_tee`,
  `hd3_cover_roundtrips_and_cancels` (`forward.rs` lib tests);
  `cubek_dispatch_granularity`, `cubek_folded_causal_parity`
  (`tests/cubek_prefill_cover.rs`).

## Reproduce

Canonical invocations in `CLAUDE.md` + dev-log *Reproduce*. Quick refs:
- Prefill offload bench: prepend `GELO_GPU_PREFILL_OFFLOAD=1 CUBEK_STRATEGY=blackbox`
  to `gelo_llm_prefill_decode_breakdown`; compare `tee:attn_prefill_offload` vs
  flag-off `tee:attn_inplace_many`.
- Decode offload bench: `GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 …`.
- Prefill prep per-stage split: add `CUBEK_PROFILE=1` (device-sync barriers).
- Dispatch granularity A/B (warm, low-variance): `cubek_dispatch_granularity` in
  `cubek_prefill_cover`.
- Parity (fast, no GPU): `cargo test -p gelo-embedder --lib prefill_offload_tests`.

## Gotchas / environment

- `--release` mandatory for anything touching cubecl (debug = minute-scale shader
  compile).
- **Measure with the right instrument.** `cubek_gpu` swings ±~1.5× cross-run
  (autotune/thermal) in the full-model bench — use the **warm, same-process**
  microbench for A/Bs (this is how the dispatch-granularity question was settled
  after a noisy full-bench reading misled an earlier call). See memories
  `feedback-ceiling-vs-achievable`, `feedback-no-polling-background-tasks`.
- **`crates/gelo-gpu-wgpu/tests/parity.rs`** has a **pre-existing** breakage on
  this branch (`GpuContext` field drift) — unrelated to attention.
- Host box is **Rust-only** (no pip/numpy/sudo); Python `WEIGHTS-PUB` attacks run
  in the `gelo-attack` Podman container.
- Commit identity `Timo <timofey.luin@gmail.com>`; trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit
  only when asked. Branch `dgpu-nvidia-bringup` (not master).

## Suggested skills

- **`grill-me`** — for the AloePri obfuscation math + threat-model fork (Phase 5b)
  and the cubek read-index / O6 kernel scope; the user prefers being grilled on
  design forks, mechanism-in-prose before options (memory
  `feedback-elaborate-before-options`).
- **`code-review`** — before committing the AloePri capture-harness mode or the
  O6 kernel.
- **`diagnose`** — only if an AloePri re-run gives a surprising pass/fail and you
  need a tight loop.
