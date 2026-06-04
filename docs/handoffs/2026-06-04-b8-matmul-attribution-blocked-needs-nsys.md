---
type: handoff
status: current
created: 2026-06-04
updated: 2026-06-04
tags: [gelo, dgpu, cuda, perf, b8-scaling, engine-matmul, readback, pinned-memory, cubecl, nsys, blocked]
companion: [gelo-llm-perf-chronicle_dgpu, 2026-06-04-ffn-unmask-spike-killed-nonlinearity-offload-research]
supersedes: []
---

# Handoff — B=8 `engine:matmul*` attribution is BLOCKED on `nsys`; what landed and everything tried

## TL;DR / current blocker (read this first)

**`engine:matmul*` is ~45% of B=8 prefill wall (~57 s) and scales super-linearly.
Its true cause is UNRESOLVED and is BLOCKED on an external GPU profiler.**

In-process timing (`Instant` around `into_data` / `tx.execute`) is **confounded by
cubecl's lazy-execution + sync model**: the same `matmul().into_data()` measured
**10 ms in one harness and 237 ms in another**, because `into_data` on a matmul
output vs an uploaded tensor drains queued GPU work at different points. I cannot
separate compute / transfer / sync-drain from inside the process. **The next step
is `nsys` (Nsight Systems) on the real B=8 prefill** — it traces the actual CUDA
API + kernel/copy timeline, immune to the lazy/sync ambiguity. `nsys` is **not
available in this environment** — that is the blocker.

**Do not attempt another fix from in-process numbers** — three already failed
(below). Get the `nsys` timeline first.

## What LANDED this session (branch `dgpu-nvidia-bringup`, NOT pushed)

Commits `6ef121b..1c9efd0` (4 commits on top of the prior session):
- `6ef121b` docs: threat model in `CLAUDE.md` + FFN-unmask spike handoff.
- `3620d65` cleanup: **f16 offload readback** — drop the bf16-narrowing detour
  (raw f16 readback + fused f16 unapply). **Parity-clean, NO perf win.**
  ⚠ **Owed: HumanEval-20 gate** — numerics shifted bf16→f16 readback (a precision
  *improvement*; sim-level dct4 round-trip tests pass, but the end-to-end greedy
  gate on the wgpu/CUDA engine was not run). Run before pushing.
- `d7fa8f2` perf: **tier-1 — parallelize the f32→f16 upload convert** (rayon-chunk
  into an uninit buffer; drops the redundant `vec![f16::ZERO]` zerofill). **The
  one CONFIRMED win: B=8 prefill 126.4 → 119.9 s (−6.5 s, ~5%); `:up_cvt` 9.6 →
  2.4 s.** `rayon` added to `gelo-gpu-wgpu` deps.
- `1c9efd0` perf: strip the sync-split diagnostic instrumentation, keep tier-1.

Tree is clean at `1c9efd0`. All diagnostic micro-bench tests were reverted (they
gave contradictory numbers — see below).

## ELIMINATED hypotheses (with the measurement that killed each)

| hypothesis | verdict | evidence |
|---|---|---|
| wgpu unpinned-staging map penalty | **dead** | the bench runs **CUDA** (`default = ["blas","cuda"]`), not wgpu |
| host f16→bf16 narrow is the cost | **dead** | f16 cleanup removed it → B=8 `matmul*` flat (29.6→30.0 s) |
| burn `TensorData` marshalling (`into_vec`) | **dead** | tier-2a `read_one_unchecked` (client bypass) = equal-or-worse than `into_vec` |
| cubecl-cuda 100 MB pinned threshold → pageable | **dead** | the `size>100*MB` check is **bypassed on reads** (`read_async` passes `marked_pinned=true`); and a resident-tensor readback **sweep is flat 45 GB/s, no cliff** at 100 MB (16 MB→512 MB) |
| PCIe transfer | **dead** | resident readback = 45 GB/s ≈ PCIe-5 peak |
| GPU compute (slow kernel) | **dead (isolated)** | isolated gate∥up matmul = **228 TFLOP/s** (tensor cores) |
| VRAM pressure | **dead** | 19 GB resident ballast → **1.0×**, no slowdown on isolated matmul+readback |
| burn `Transaction` overhead | **dead** | Transaction vs direct per-weight `into_data` = **1.0×** (both slow in the prod-path harness) |

**Net:** every *per-op* cost is fast on an empty/pressured GPU (compute ~3.6 ms,
readback ~6.5 ms for a 319 MB gate∥up output). Yet the **production path
replication** (`array2_to_tensor_f16` upload + `execute_registered_many_f16_raw`)
is **~470–528 ms/call**, matching production's ~47× gap. The cost is real and in
the production path, but **the in-process harness cannot attribute it** (the
10 ms-vs-237 ms contradiction).

## Everything tried, in order (so it isn't re-attempted)

1. **f16 readback cleanup** (`3620d65`) — replace bf16-narrow readback with raw
   f16 + fused f16 unapply. Parity-clean, **0 perf change** at B=8.
2. **Forced-sync sub-bucket instrumentation** — split `engine:matmul*` into
   `:up_cvt`/`:up_dev`/`:submit`/`:gpusync`/`:readback` with a `Backend::sync`
   after `tx.execute()`. Showed `:gpusync` ≈ 8 ms (GPU "done") and `:readback`
   ~18.6 s — which I **mis-read** as transfer; it was lazy-execution artifact.
3. **tier-2a `read_one_unchecked`** (cubecl client readback bypass, drop
   `Transaction`) — **regressed** (lost batched sync; 18.6→22.0 s). Reverted.
4. **Chunked readback ≤96 MB** (slice output into pinned-eligible row-chunks) —
   **regressed +9 s** (38.7 s `matmul_many`). Reverted.
5. **tier-1 parallel convert** (`d7fa8f2`) — **WORKED, −6.5 s.** Kept.
6. **Readback throughput sweep micro-bench** (resident f16 tensor, sync, time
   `into_data`) — **45 GB/s flat, no 100 MB cliff.** Killed pinned/transfer.
7. **Isolated matmul compute+readback** — 228 TFLOP/s + 49 GB/s. Both fast.
8. **VRAM-pressure variant** (19 GB ballast) — 1.0×. Killed pressure.
9. **Production-path replication** — 528 ms/call (matches prod). Localized the
   cost to the path.
10. **Transaction vs direct into_data** — 1.0× (both ~470 ms with a resident
    lhs). Killed Transaction-as-cause.
11. **Contradiction surfaced** — same `matmul().into_data()` = 10 ms (test 7) vs
    237 ms/weight (test 10). → in-process timing is **unreliable** for cubecl
    lazy ops. Stopped; reverted all micro-bench tests.

## Code map (for the next attempt / `nsys` run)

- Engine: `crates/gelo-gpu-wgpu/src/lib.rs` — `matmul_many_f16_out` (F16 arm),
  `array2_to_tensor_f16` (upload convert, now rayon-parallel),
  `execute_registered_many_f16_raw` (`Transaction` + `into_vec` readback),
  `tensor_data_to_array_f16_raw`. Backend = CUDA (`CudaRuntime`) by default.
- cubecl-cuda pinned logic: `~/.cargo/registry/.../cubecl-cuda-0.10.0/src/compute/
  command.rs::reserve_cpu` (the 100 MB heuristic; `read_async` line ~196 passes
  `marked_pinned=true`). `RuntimeOptions` (runtime.rs:39) only exposes
  `memory_config`; the 100 MB is a hardcoded literal (not a runtime knob).
- The diagnostic micro-bench tests (readback sweep, matmul compute/readback,
  VRAM-pressure, prod-path replication, Transaction-vs-direct) were **reverted**
  but are reconstructable from this handoff if needed — they belong in a
  `#[cfg(test)] mod` in `lib.rs` (need private types `CubeWgpu16`/`Dev`).

## Next session

1. **Run `nsys` on the canonical B=8 prefill** (CLAUDE.md invocation) and read the
   CUDA timeline: is the ~470 ms/call a kernel, a DtoH stall, or sync
   serialization? That decides the lever. Nothing else should be attempted first.
2. **HumanEval-20 gate** on the f16 readback (`3620d65`) before pushing the branch.
3. The chronicle **§21 "B=8 prefill is GPU-matmul-bound" is contradicted** — GPU
   compute is fast in isolation (228 TFLOP/s); record the eliminations + tier-1.
4. Carry-over from the prior next-levers handoff still open: decode tail-fold,
   U-Verify-vs-16-bit-GPU precision, push the branch after `code-review`.

## Meta-lesson (candidate memory)

In-process `Instant` timing of burn/cubecl **lazy** GPU ops is unreliable —
`into_data`/`tx.execute` drain queued work at non-obvious points, producing
contradictory per-op numbers. Use an **external profiler (`nsys`)** for GPU
attribution on this stack; don't ship perf fixes off in-process micro-bench
deltas (three failed here). Worth saving as a `reference` memory if it recurs.

## Gotchas

- `--release` mandatory; CUDA default; `target-cpu=native` via `.cargo/config.toml`.
- B=1 needs `GELO_BENCH_FORCE_BATCHED=1`; B=8 uses the batched path natively.
- Artifacts this session: `bench-results/gelo-b8-{subbucket,f16out,syncsplit,
  upsplit,parconvert,tier2a-readback,pinned-chunked-readback}-native-n2048-2026-06-04.log`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Suggested skills

- **`diagnose`** — resume here once `nsys` output exists (real feedback loop).
- **`verify`** — HumanEval-20 gate on the f16 readback before push.
- **`code-review`** — before pushing the branch.
