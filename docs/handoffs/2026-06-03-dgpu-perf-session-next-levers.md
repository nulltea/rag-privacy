---
type: handoff
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, dgpu, cuda, perf, mask, ffn-permutation, b8-scaling, decode-tail-fold, security-spike]
companion: [gelo-llm-perf-chronicle_dgpu]
supersedes: [2026-06-03-deferred-dgpu-cuda-optimizations]
---

# Handoff — dGPU perf session wrap; next: FFN-unmask elimination (security spike), B=8 matmul scaling, decode tail-fold

## State (all committed on `dgpu-nvidia-bringup`, **not pushed**; 13 commits `fa7c7ad..c23affa`)

Everything measured/decided this session is in the dGPU chronicle
[`gelo-llm-perf-chronicle_dgpu.md`](../dev/logs/gelo-llm-perf-chronicle_dgpu.md)
**§12–§22** — read §20–§22 first. Headline (Qwen3-4B, n=2048, native,
bf16-on, secure covers):

| | session start | now |
|---|--:|--:|
| prefill B=1 | 18.6 s (SSE2 f32) | **10.0–10.3 s** (199 tok/s) |
| decode B=1 | 6.6 s / 207 ms/step | **3.2 s / ~100 ms/step** |
| prefill B=8 | 197.6 s (§9 anchor) | **122.9 s** |
| decode B=8 | 34.8 s | **11.3 s** (44 ms/tok/seq) |

What landed (chronicle § in parens): bf16 read-back default-on, DCT-IV
gate (§13); `target-cpu=native` in `.cargo/config.toml` — F16C was the
real lever (§14); HD₃ rayon-misfire fix, decode −45% (§15); smooth-size
`stacked_n` via `dct4::next_fast_n` (§16); fused bf16 unmask +
`mallopt` mmap fix (§17); FFTW batched-DCT refuted (§18); R4 async
struck — serial-dependency, bus-independent (§19); concurrency/SIMD
audit, prefill −15% (§20); B=8 re-measured (§21); decode-length scaling
(§22). Security: offload covers now derive from the session `MaskSeed`
(`TrustedExecutor::cover_seed`, `fixed-cover-seed` feature reverts for
dev) — commit `723c6c2`.

## Next-session focus

### 1. Eliminate the FFN-intermediate unmask (biggest mask lever; **security spike first**)

gate∥up outputs are 19 456 of 30 720 unapply columns (~63%) + the 9 728-col
down-input apply. At current numbers ≈ **−2.5 s ≈ −20%+ of B=1 prefill**
(and ~5% decode). Mechanism sketch: a per-session feature-axis
permutation π baked into the registered weights (W_gate/W_up columns by
π, W_down rows by π⁻¹) lets the intermediate stay covered through
SwiGLU — `silu(g)_π ⊙ u_π = (silu(g) ⊙ u)_π`.

**Hard constraints found when scoping (do the spike before any code):**
- SiLU commutes with **permutation only** — NOT with ±1 signs
  (`silu(−x) ≠ −silu(x)`) and not with the row-axis orthogonal mask. So
  the row-mask round-trip on gate∥up/down cannot simply "stay on"; the
  design replaces it for the FFN chain.
- Security gate: the GPU would see the FFN intermediate as
  **feature-permuted plaintext** (silu inputs/outputs). Per-column
  statistics are permutation-invariant → per-feature value
  distributions leak. Quantify with an AloePri-style attack (column
  moments / matching against public-weight predictions) before
  accepting; compare against the current orthogonal row-mask baseline.
  This is a real weakening — the spike may kill the lever.
- If it survives: weights are public (`WEIGHTS-PUB`), so π must be
  per-session ⇒ per-session weight re-registration of W_gate/W_up/W_down
  (provisioning cost; measure).

### 2. B=8 prefill GPU-matmul scaling (§21 finding)

`engine:matmul*` = 45% of B=8 prefill and scales **25–32× vs B=1 for 8×
rows** (B=1: 2.0 s → B=8: 56.3 s). CPU mask side scales sub-linearly —
this is GPU/transfer-side. Suspects: VRAM pressure (~640 MB f32
operands, ~1 GB read-backs per layer), burn `Transaction` behaviour at
big tensors, autotune at large shapes, PCIe. Approach:
- Add sub-buckets inside the engine matmul path (upload-convert /
  dispatch+GPU / read-back-narrow) and re-run B=8 — one warm cell is
  ~4 min (`bench-results/gelo-b8-current-native-n2048-2026-06-03.log`
  is the reference).
- Watch `nvidia-smi` memory during the run (32 GB card; weights + 36
  resident K/V sessions + transients).
- A/B: split the B=8 operand into 2×4 or 4×2 chunked dispatches.

### 3. Decode periodic tail-fold (§22 finding)

Decode/step = ~100 ms flat + **0.26 ms × tokens-generated** (the in-TEE
tail grows; frozen GPU prefix is flat). Tail overtakes the fixed cost at
~380 tokens; K=512 already 6.1 tok/s. Implement fold-every-N: rebuild
the covered prefix including the tail (machinery exists:
`build_covered_prefix_cpu/all_global` in
`crates/gelo-embedder/src/decoder/forward.rs`; drop the old GPU session
via `resident_kv_drop`, update `DecodeCover.prefix_len`, reset the tail
origin). Full 36-layer rebuild ≈ 322 ms ⇒ N=128 amortises ~2.5 ms/step,
caps tail at ~33 ms ⇒ flat ~120 ms/step at any K. Folding re-permutes
*more* often than prefill-only ⇒ security-neutral-or-better. Validate
with the K=512 cell (`gelo-b1-decscale-*` logs are the baseline).

## Owed / open (unchanged this session)

1. **HumanEval gate re-run** — numerics legitimately shifted (bf16
   read-back accepted at 6/20 == Plain ref; σ reassociation ~1e-7;
   session-secret covers). The gate generator is now **B=8
   length-sorted** (`GELO_HE_BATCH`); expect only ~1.3× faster than the
   old ~25 min B=1 run (decode is per-row-bound — chronicle §13.5 note).
2. Decode **bimodality** (~2.8 vs ~3.2 s across runs, thread-count
   independent) — needs repeats + core affinity.
3. **Push** the branch when ready (`code-review` first — 13 commits).
4. Larger-eval re-test of the bf16 6/20-vs-7/20 question (HumanEval-164
   / MBPP) remains owed from §13.5.

## Gotchas

- `--release` mandatory; CUDA default; **the bench binary embeds
  `target-cpu=native`** via `.cargo/config.toml` — setting `RUSTFLAGS`
  env replaces the config array incl. the libblis rpath (binary then
  needs `LD_LIBRARY_PATH=vendor/aocl-install/lib`).
- B=1 must use `GELO_BENCH_FORCE_BATCHED=1` to exercise the offload
  (single-stream path runs attention in-TEE).
- Canonical bench invocation + env knobs: `CLAUDE.md`. Per-op tables in
  reports must include the **calls** column.
- Perf is single-sample ±~7%; transform-only microbenches overstate
  bucket wins (§16 lesson); rayon gates: measure the fork-join
  crossover, don't assume (§20 counter-finding).
- Stale GPU spike tests (`q4_kernel_spike`, `parity`,
  `qwen3_overhead_*`, `cubek_attention_spike`) don't compile against
  cubek 0.2.0 — pre-existing, unrelated.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Suggested skills

- **`grill-with-docs` / `grill-me`** — for the FFN-permutation security
  spike design (focus #1): grill the leakage model before building.
- **`diagnose`** — for the B=8 matmul scaling investigation (focus #2):
  reproduce → instrument sub-buckets → attribute.
- **`code-review`** — before pushing the 13-commit branch.
- **`verify`** — after the tail-fold lands (greedy coherence + K=512
  re-bench).
