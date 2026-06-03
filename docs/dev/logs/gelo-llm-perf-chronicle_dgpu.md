---
type: dev-log
status: current
created: 2026-05-29
updated: 2026-06-03
tags: [gelo, perf, dgpu, nvidia, rtx5090, vulkan, cuda, attention, offload, cubek, mask, chronicle]
companion: [gelo-llm-perf-chronicle]
---

# GELO-LLM perf chronicle — dGPU (Nvidia) track

> Companion to [`gelo-llm-perf-chronicle.md`](gelo-llm-perf-chronicle.md)
> (the iGPU / Strix Halo track). That document is the spine for the
> protocol cost model, mask-family history, and the dated optimisation
> chronicle. **This document holds the discrete-GPU measurements** —
> the hardware the iGPU roadmap repeatedly deferred to as the
> "dGPU substrate (M5.9), hardware-gated."
>
> First entry: 2026-05-29, RTX 5090 bring-up. The headline finding is
> that **raw dGPU hardware does not lift throughput on the current
> code** — prefill is materially *slower* than the Strix Halo iGPU, and
> the win the dGPU was supposed to deliver is gated on the per-call
> upload-pipeline replacement the iGPU chronicle's dGPU-revival design
> already specified (persistent K/V, end-to-end bf16 activations,
> GQA-aware kernel). The measurements below pin exactly which buckets
> regress and why.

## 1. Hardware substrate

| | iGPU track (reference) | dGPU track (this doc) |
|---|---|---|
| Box | Strix Halo (Ryzen AI Max+ 395) | Ryzen 9 7900X + RTX 5090 |
| CPU | 16 Zen5 cores | **12 Zen4 cores** (24 threads), 5.7 GHz max |
| System RAM | quad-channel LPDDR5X, **~256 GB/s** (UMA, shared with iGPU) | dual-channel DDR5, **~80–90 GB/s** (not shared) |
| GPU compute | Radeon 8060S iGPU (RDNA 3.5, gfx1151), fp16 wgpu Vulkan | **RTX 5090** (Blackwell, 32 GB GDDR7 ~1.8 TB/s), fp16 wgpu Vulkan |
| GPU memory bus | shared DDR5 (UMA) | dedicated GDDR7 over **PCIe** |
| BLAS | AOCL-BLIS, `GELO_BLIS_THREADS=16` | AOCL-BLIS, threads unset (default) |

Both run the same engine: fp16 wgpu **Vulkan**, R3 LM-head GPU offload
on, mask = Auto. The wgpu adapter auto-selects
`NVIDIA GeForce RTX 5090 (DiscreteGpu)` here — confirmed in the bench
banner and by GPU-memory movement under load.

The two boxes differ on **three** axes at once — GPU (5090 vs iGPU),
CPU core count (12 vs 16), and **system-memory bandwidth (~3× in the
iGPU box's favour)**. The last one matters most: GELO runs attention
and the mask GEMM on the CPU, and both are memory-bandwidth-bound (see
the iGPU chronicle §2 / §7). So this is not a clean GPU-vs-GPU
comparison — it is a whole-machine comparison, and the bandwidth gap is
the dominant confound to keep in mind throughout.

## 2. Benchmark

`gelo_llm_prefill_decode_breakdown` (the Gelo-LLM main bench, see
`CLAUDE.md`), Qwen3-4B, fp16, R3 on, mask=Auto, K=32 decode tokens,
single sample per cell. Release build. First-run cubecl autotune is
included in the GPU buckets (no warm-up pass) — see §5 confound 1.

## 3. Results — RTX 5090

### 3.1 B=1 n=2048 (single-stream)

Prefill 25.3 s (80.9 tok/s); decode 17.5 s (1.8 tok/s, 546 ms/step).

| PREFILL op | ms | share | DECODE op | ms | share |
|---|--:|--:|---|--:|--:|
| `engine:matmul_many` ◆ | 6760 | 28.5% | `tee:attn_cached` | 9179 | 51.6% |
| `tee:attn_cached` | 5015 | 21.2% | `engine:matmul_many` ◆ | 2236 | 12.6% |
| `engine:matmul` ◆ | 4094 | 17.3% | `engine:matmul` ◆ | 2233 | 12.6% |
| `gelo:mask_unapply:dct4` | 2567 | 10.8% | `gelo:mask_unapply:hd3` | 1837 | 10.3% |
| `gelo:strip_shield` | 2384 | 10.1% | `gelo:mask_apply:hd3` | 1041 | 5.9% |
| `gelo:mask_apply:dct4` | 1603 | 6.8% | `gelo:shield_stack` | 862 | 4.8% |
| (rest) | <400 ea | | `tee:compute_logits` | 340 | 1.9% |

◆ = GPU (Vulkan). All other buckets are CPU, in-TEE.

### 3.2 B=8 n=2048 (production shape — the iGPU comparison anchor)

Prefill 216.9 s (75.5 tok/s agg, 9.44 tok/s/seq); decode 40.6 s
(6.30 tok/s agg, 0.79 tok/s/seq, 1269 ms/step).

| PREFILL op | ms | share | DECODE op | ms | share |
|---|--:|--:|---|--:|--:|
| `engine:matmul_many` ◆ | 68790 | 33.4% | `tee:attn_cached_inplace_many` | 14574 | 34.0% |
| `tee:attn_inplace_many` | 43774 | 21.2% | `engine:matmul_many` ◆ | 9941 | 23.2% |
| `engine:matmul` ◆ | 39986 | 19.4% | `engine:matmul` ◆ | 8956 | 20.9% |
| `gelo:mask_unapply:dct4` | 27367 | 13.3% | `gelo:mask_unapply:hd3` | 3445 | 8.0% |
| `gelo:mask_apply:dct4` | 11759 | 5.7% | `tee:compute_logits` | 2526 | 5.9% |
| `tee:residual` | 4648 | 2.3% | `gelo:shield_stack` | 1564 | 3.6% |
| `gelo:shield_stack` | 4151 | 2.0% | `gelo:mask_apply:hd3` | 1533 | 3.6% |
| (rest) | <2200 ea | | (rest) | <200 ea | |

**GPU matmul total**: prefill 108.8 s (52.8%), decode 18.9 s (44.1%).

## 4. Comparison vs iGPU (B=8 n=2048, the documented production cell)

iGPU reference: post-DCT-cascade prefill **135.13 s** (121.3 tok/s agg),
decode **55.08 s** (0.58 tok/s/seq); decode shares attn 53.9% /
matmul 37.7% / mask 4.2% (iGPU chronicle §4 + roadmap §1.3). iGPU
prefill bucket absolutes are *derived* from the pre-cascade per-op
shares (matmul and attention are untouched by the cascade; mask is the
post-cascade figure).

### 4.1 Wall

| Phase | RTX 5090 | iGPU (Strix Halo) | 5090 ÷ iGPU |
|---|--:|--:|--:|
| **Prefill** | 216.9 s | 135.1 s | **1.61× (slower)** |
| **Decode** | 40.6 s | 55.1 s | **0.74× (faster)** |

So we are **slower on prefill, faster on decode**. The "we're slower"
impression is real and lives entirely in prefill.

### 4.2 Where prefill regresses (absolute bucket time)

| Prefill bucket | RTX 5090 | iGPU (approx) | 5090 ÷ iGPU |
|---|--:|--:|--:|
| GPU matmul (both) ◆ | 108.8 s | ~70.8 s | ~1.5× slower |
| in-TEE attention | 43.8 s | ~22.0 s | ~2.0× slower |
| mask DCT-IV (apply+unapply) | 39.1 s | ~27.6 s | ~1.4× slower |

**Every prefill bucket is slower on the 5090 box** — including the GPU
matmul. Two distinct causes:

- **GPU matmul slower (◆) — the PCIe upload tax.** GELO uploads the
  *masked* f32 activation `U = A·H` to the GPU on every projection (it
  is fresh per forward and cannot be cached). At B=8 n=2048 that is
  ~80 MB/dispatch × 72 dispatches, plus the f32→f16 conversion and the
  output read-back. On the iGPU this rides a **mapped UMA buffer**
  (DDR5, no transit); on the dGPU it is a **PCIe DMA round-trip per
  call**. The 5090's HBM compute finishes in microseconds but the
  per-call upload dominates, so the matmul bucket *grows* rather than
  shrinks. This is precisely the failure mode the iGPU chronicle's
  dGPU-revival design flagged — "the upload-pipeline tax scales worse
  on PCIe, not better" (roadmap §4.E.3) — now measured directly.
- **CPU buckets slower — memory bandwidth.** Attention and the mask
  GEMM are memory-bandwidth-bound. Strix Halo's quad-channel LPDDR5X
  (~256 GB/s) is ~3× this box's dual-channel DDR5 (~80–90 GB/s), and it
  also has 16 cores vs 12. The prefill attention (`tee:attn_inplace_many`,
  a streaming O(n²) pass) is the most bandwidth-sensitive and regresses
  the hardest (~2×).

### 4.3 Why decode is faster despite all that

| Decode bucket | RTX 5090 | iGPU | 5090 ÷ iGPU |
|---|--:|--:|--:|
| in-TEE attention | 14.6 s | ~29.7 s | **0.49× (2× faster)** |
| GPU matmul (both) ◆ | 18.9 s | ~20.8 s | ~0.91× |
| mask HD₃ | 5.0 s | ~2.3 s | ~2.2× slower |

Decode flips because its working set is tiny — `n_q=1`, so each masked
activation upload is a handful of rows, and the **PCIe upload tax
vanishes** (GPU matmul becomes competitive). The big swing is
attention: decode attention is **2× faster** on the 5090 box. The
likely reason is **UMA bus contention** — on Strix Halo the CPU
attention and the GPU matmul fight over the *same* DDR5 bus every step;
here the CPU attention has system DDR5 to itself (the GPU lives on
GDDR7), and the 7900X's higher single-thread clocks help the
latency-bound per-step kernel. The decode working set is too small to
be bandwidth-bound, so the iGPU's bandwidth edge doesn't apply — only
its contention penalty does.

## 5. Confounds (read the deltas against these)

1. **Cold autotune.** Single run, no warm-up; first-touch cubecl
   autotune is counted inside `engine:matmul*`. Part of the prefill
   GPU-matmul regression is one-time. A warm re-run is needed to split
   autotune from the genuine PCIe upload tax.
2. **BLIS threads unset** here vs `GELO_BLIS_THREADS=16` on the iGPU
   reference — affects the mask-bucket comparison.
3. **Whole-machine, not GPU-isolated.** CPU core count and memory
   bandwidth both differ; the CPU buckets are a different-machine
   comparison, not a GPU one.
4. **iGPU prefill bucket absolutes are derived** from pre-cascade
   shares; treat ±10%.
5. **Single sample**, ~7% variance floor (iGPU chronicle §7).

## 6. Takeaways + next levers

- **The dGPU does not help the current code at the production shape.**
  Net prefill is 1.6× slower; decode is 1.35× faster; the binding
  prefill bottleneck moved from CPU mask (iGPU) to **PCIe activation
  upload** (dGPU GPU-matmul bucket) stacked on **lower CPU memory
  bandwidth**.
- **This validates the dGPU-revival sequencing.** The card's HBM is
  wasted until the per-call upload pipeline is replaced. The relevant
  levers, all already scoped on the iGPU track, are now the critical
  path here:
  - **End-to-end bf16 activations** (roadmap §4.E.3) — halves the
    bytes-on-wire per upload; prerequisite for any dGPU attention move.
  - **Persistent K/V on GPU** (§4.C.2 / dGPU-revival Item 1) — kills
    the per-step K/V upload; the decode attention bucket is the target.
  - **GQA-aware single-pass WGSL kernel** (Item 2+3) — 4× less K/V
    motion + fused FlashAttention.
  - **R4 async overlap** — on PCIe the CPU mask / GPU matmul overlap
    that vanished on UMA reappears (roadmap §4.D disposition).
- **Immediate measurement debt:** (a) warm re-run to isolate autotune
  from the upload tax in the GPU-matmul bucket; (b) sweep
  `GELO_BLIS_THREADS` (12 physical cores here); (c) re-profile after
  bf16 activations land — that is the first lever expected to move the
  prefill GPU bucket on this hardware.

## 7. Artefacts

- Bench logs: `/tmp/gelo_b8_n2048.log`, `/tmp/gelo_perop_split.log`
  (B=1) — capture to `bench-results/` on the next run.
- Bench: `gelo_llm_prefill_decode_breakdown` in
  `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`.
- Trace fix this session: GPU offload now emits `engine:matmul` (single
  weight: O, FfnDown, R3 LM-head) vs `engine:matmul_many` (fused QKV,
  gate∥up) instead of the merged `engine:registered_linear` bucket
  (`crates/gelo-protocol/src/substrate.rs::run_registered_linear`).

## 8. CUDA backend A/B (2026-05-29) — modest, not transformative

A compile-time `cuda` feature on `gelo-gpu-wgpu` swaps the cubecl runtime
(`WgpuRuntime`→`CudaRuntime`) behind `Rt`/`Dev` aliases; Vulkan stays the
default. cubecl-cuda 0.9.0 + cudarc 0.18.2 build against CUDA 13.0.3 and
**execute correctly on the Blackwell RTX 5090 (sm_120)** — provision,
prefill, decode, token generation all complete.

**Protocol:** pure runtime swap, **warm** (one discarded forward via
`GELO_BENCH_WARMUP=1` populates the autotune cache, then the measured
forward), per-call readback held identical. Vulkan warm ≈ Vulkan cold
(SPIR-V autotune is cheap); CUDA cold was 2–4× *slower* (nvrtc autotune)
— so only the warm numbers are meaningful.

Warm A/B, Qwen3-4B B=1 n=2048 K=32, R3:

| Bucket | Vulkan warm | CUDA warm | CUDA advantage |
|---|--:|--:|--:|
| prefill `engine:matmul_many` | 6865 ms | 5042 ms | 1.36× |
| prefill `engine:matmul` | 4369 ms | 2574 ms | 1.70× |
| **prefill GPU total** | 11.2 s | 7.6 s | **1.47×** |
| **decode GPU total** | 4.3 s | 2.4 s | **1.84×** |
| CPU attn / mask | ~5.0 / ~4.2 s | ~5.1 / ~4.2 s | backend-invariant |
| **prefill wall** | 25.8 s | 22.5 s | **1.15×** |
| **decode wall** | 17.4 s | 15.1 s | **1.15×** (1.84→2.12 tok/s) |

**Finding — and a correction to §4.2.** §4.2 attributed the slow GPU
matmul to a "PCIe upload tax." That was wrong: the per-call data is ~1 s
of transfer over prefill, and CUDA (which would not change PCIe
bandwidth) makes the matmul only ~1.5× faster. The real picture:

- GPU matmul on the Vulkan/SPIR-V path is **not catastrophically
  tensor-core-starved** — CUDA's kernels are only ~1.5× (prefill) to
  ~1.8× (decode) faster, not the 50–300× an absent-vs-present tensor-core
  gap implies. Effective throughput is ~1.3 TFLOP/s (Vulkan) → ~2 TFLOP/s
  (CUDA) at B=1, both far below the card.
- The dominant cost is **backend-invariant**: the per-call blocking
  readback (every masked matmul round-trips to the TEE to unmask,
  serialising the dispatches) plus the CPU attention/mask buckets. GPU
  matmul is only ~45 % of prefill / ~25 % of decode, so a 1.5× kernel
  win caps at ~13 % wall.

**Disposition (research phase).** CUDA delivers a real but modest ~13 %
wall win at B=1 — far below the bar that would justify a production
backend fork. The high-value lever is the **per-call readback/sync**
(batched/streamed unmask, R4 async, persistent K/V) — backend-agnostic,
helps Vulkan too. The `cuda` feature is retained as a measurement tool +
opt-in Nvidia path; it is not promoted. Next probes: (a) same warm A/B at
B=8; (b) whether enabling cubecl-wgpu SPIR-V cooperative-matrix closes
even the ~1.5× matmul gap on the portable path.

**Artefacts:** `/tmp/warm_vulkan_b1.log`, `/tmp/warm_cuda_b1.log`,
`/tmp/gelo_cuda_b1.log` (cold) — capture to `bench-results/` next run.
Engine `cuda` feature: `crates/gelo-gpu-wgpu/{Cargo.toml,src/lib.rs}`;
warmup knob `GELO_BENCH_WARMUP` in the main bench.

## 9. Per-op A/B tables: CUDA vs Vulkan (B=1 and B=8)

Qwen3-4B n=2048 K=32, R3, RTX 5090. ◆ = GPU-executed (`engine:*`); all
other buckets run on the CPU in-TEE and are backend-invariant. B=1 is
warm-vs-warm. B=8 is **Vulkan cold vs CUDA warm** — fair because Vulkan
warm≈cold (SPIR-V autotune is cheap; verified at B=1: 25.8 vs 25.3 s),
whereas CUDA must be warm (nvrtc autotune is heavy: the B=8 CUDA warmup
prefill was 225 s vs the measured 197.6 s — a ~27 s one-time cost, vs
~1 s at B=1). Single sample each; ~7 % variance floor.

### B=1 — PREFILL (wall 25.8 s → 22.5 s, 0.87×)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| ◆ `engine:matmul_many` | 6865.0 | 5041.9 | −1823.0 | 0.73× |
| `tee:attn_cached` | 5036.0 | 5061.6 | +25.6 | 1.01× |
| ◆ `engine:matmul` | 4368.6 | 2574.4 | −1794.2 | 0.59× |
| `gelo:mask_unapply:dct4` | 2562.9 | 2606.4 | +43.6 | 1.02× |
| `gelo:strip_shield` | 2423.8 | 2589.8 | +166.0 | 1.07× |
| `gelo:mask_apply:dct4` | 1605.3 | 1602.1 | −3.1 | 1.00× |
| `gelo:shield_stack` | 380.0 | 417.1 | +37.2 | 1.10× |
| `tee:swiglu_activate` | 305.6 | 307.1 | +1.5 | 1.00× |
| `tee:qk_norm` | 254.3 | 254.2 | −0.1 | 1.00× |
| `tee:residual` | 158.1 | 159.8 | +1.6 | 1.01× |
| `tee:rmsnorm` | 120.9 | 127.8 | +6.8 | 1.06× |
| `tee:rope` | 67.7 | 65.3 | −2.4 | 0.96× |
| **TOTAL (Σ)** | **24151.7** | **20811.0** | **−3340.7** | **0.86×** |

### B=1 — DECODE (wall 17.4 s → 15.1 s, 0.87×; 1.84→2.12 tok/s)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| `tee:attn_cached` | 9148.3 | 8943.7 | −204.6 | 0.98× |
| ◆ `engine:matmul` | 2205.6 | 1247.3 | −958.3 | 0.57× |
| ◆ `engine:matmul_many` | 2115.2 | 1106.2 | −1009.1 | 0.52× |
| `gelo:mask_unapply:hd3` | 1886.0 | 1797.1 | −88.9 | 0.95× |
| `gelo:mask_apply:hd3` | 1067.9 | 1016.8 | −51.1 | 0.95× |
| `gelo:shield_stack` | 866.8 | 854.1 | −12.6 | 0.99× |
| `tee:compute_logits` (LM-head◆) | 327.1 | 214.1 | −113.1 | 0.65× |
| **TOTAL (Σ)** | **17686.3** | **15243.4** | **−2442.8** | **0.86×** |

### B=8 — PREFILL (wall 216.9 s → 197.6 s, 0.91×)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| ◆ `engine:matmul_many` | 68789.7 | 57617.2 | −11172.5 | 0.84× |
| `tee:attn_inplace_many` | 43773.7 | 43475.2 | −298.6 | 0.99× |
| ◆ `engine:matmul` | 39986.1 | 32031.7 | −7954.4 | 0.80× |
| `gelo:mask_unapply:dct4` | 27366.6 | 27312.2 | −54.4 | 1.00× |
| `gelo:mask_apply:dct4` | 11758.6 | 11589.1 | −169.5 | 0.99× |
| `tee:residual` | 4647.5 | 4812.4 | +164.9 | 1.04× |
| `gelo:shield_stack` | 4151.3 | 4112.7 | −38.5 | 0.99× |
| `tee:swiglu_activate` | 2196.7 | 2186.6 | −10.1 | 1.00× |
| `tee:qk_norm` | 1997.8 | 1998.5 | +0.7 | 1.00× |
| `tee:rmsnorm` | 894.9 | 900.5 | +5.6 | 1.01× |
| `tee:rope` | 547.2 | 543.3 | −3.9 | 0.99× |
| **TOTAL (Σ)** | **206181.7** | **186654.1** | **−19527.6** | **0.91×** |

### B=8 — DECODE (wall 40.6 s → 34.8 s, 0.86×; 0.79→0.92 tok/s/seq, agg 6.3→7.4)

| op | Vulkan ms | CUDA ms | Δ ms | CUDA/Vk |
|---|--:|--:|--:|--:|
| `tee:attn_cached_inplace_many` | 14574.4 | 14769.4 | +195.0 | 1.01× |
| ◆ `engine:matmul_many` | 9941.1 | 6377.6 | −3563.6 | 0.64× |
| ◆ `engine:matmul` | 8955.9 | 6860.6 | −2095.3 | 0.77× |
| `gelo:mask_unapply:hd3` | 3445.1 | 3173.3 | −271.8 | 0.92× |
| `tee:compute_logits` (LM-head◆) | 2525.5 | 1599.8 | −925.7 | 0.63× |
| `gelo:shield_stack` | 1563.9 | 1521.4 | −42.5 | 0.97× |
| `gelo:mask_apply:hd3` | 1533.3 | 1509.7 | −23.6 | 0.98× |
| **TOTAL (Σ)** | **42925.5** | **36184.0** | **−6741.4** | **0.84×** |

**Reading.** Identical conclusion at both batch sizes: CUDA moves only the
`engine:*` rows (prefill matmul ~0.73–0.84×, decode matmul ~0.52–0.77×,
i.e. ~1.2–1.9× faster kernels) and the R3 LM-head (`compute_logits`
~0.63–0.65×). **Every CPU bucket is ~1.00×** — `tee:attn*` is literally
within 1 % across backends at both B. Net wall lands at **0.84–0.91×**
(CUDA ~10–16 % faster), Amdahl-capped by the backend-invariant CPU
attention + mask buckets. The matmul advantage is slightly *smaller* at
B=8 (0.80–0.84× vs 0.59–0.73× prefill) — larger matmuls are more
compute-bound, so the Vulkan/SPIR-V kernels close some of the gap. None
of this is the step-change a tensor-core-absent Vulkan path would show;
the lever remains the backend-invariant in-TEE attention + per-call
round-trip, not the GPU backend.

**Artefacts (B=8):** `/tmp/gelo_b8_n2048.log` (Vulkan cold),
`/tmp/warm_cuda_b8.log` (CUDA warm).

## 10. Gate-1 persistent-K/V microbench (2026-05-29) — the upload tax is ~100% of it

Per the persistent-attention plan
([`perm-attn-gpu-offload.md`](../../dev/logs/perm-attn-gpu-offload.md)),
gate 1 asks: does **device-resident K/V** (upload once; per-step upload
only Q) beat the in-TEE baseline? Added a `gpu_resident_b8` cell to
`amulet_attention_r1_4` (engine `upload_resident_kv` / `attend_resident`,
fp16, Vulkan) alongside the existing in-TEE and full-upload cells.

| n_kv (decode, B=8) | in-TEE rayon | full-upload (no_mask) | **resident** | resident vs in-TEE | resident vs full-upload |
|---:|---:|---:|---:|---:|---:|
| 256  | 1.08 ms | 69.8 ms  | **0.30 ms**  | 3.6× faster | 233× |
| 1024 | 5.28 ms | 271.8 ms | **0.373 ms** | 14× faster  | 728× |
| 2048 | 11.15 ms| 489 ms   | **0.465 ms** | **24× faster** | 1052× |

**Gate 1 passes by 24× at the production shape.** Two measured findings:

- **The "fixed-overhead-bound" hypothesis is confirmed, not inferred.** At
  n_kv=2048, full-upload 489 ms − resident 0.465 ms ≈ **99.9%** of the
  no-mask cost is the per-call K/V upload + f32→f16 convert + staging +
  blocking sync — the exact term persistent K/V deletes. The 5090's HBM
  read + kernel is ~0.5 ms; the upload pipeline was ~100% of the §3 triage
  cost, ~0% the compute. (Corrects §4.2's bandwidth framing definitively.)
- **The win grows with context** (3.6× → 14× → 24× as n_kv 256 → 2048):
  resident reads at HBM ~1.8 TB/s while in-TEE scales with DDR5 ~85 GB/s.
  Resident is sub-linear in n_kv (0.30 → 0.47 ms for 8× the keys) — per-step
  cost is fixed dispatch / Q-upload / readback, not the HBM read — so it
  stays cheap into the long-context regime the design targets.

### 10.1 Representative decode — per-step append, prefill-only re-permute

The `gpu_resident_append_b8` cell adds the realistic growing-cache cost:
the cover is applied **once at prefill** (`create_kv_session`); each step
*appends* the new token's K/V row (`append_kv`, O(1) `slice_assign`) and
attends over `[0..len]` (`attend_session`) — **no per-block re-permute**.
This is the **optimistic prefill-only case** (N = ∞) the security gate
will later test: we benchmark it first to decide if the approach is worth
building at all.

| n_kv (decode, B=8) | in-TEE | resident (attend only) | **append + attend** | append overhead | vs in-TEE |
|---:|---:|---:|---:|---:|---:|
| 256  | 1.35 ms  | 0.298 ms | **0.500 ms** | +0.20 ms | 2.7× |
| 1024 | 4.44 ms  | 0.379 ms | **0.572 ms** | +0.19 ms | 7.8× |
| 2048 | 10.88 ms | 0.463 ms | **0.662 ms** | +0.20 ms | **16.4×** |

- **Append is O(1):** ~0.20 ms *constant* across n_kv → `slice_assign`
  mutates the resident buffer in place (no O(n) recopy). The growing-cache
  per-step cost is flat.
- **Worth it, decisively:** the optimistic prefill-only case, *with* the
  per-step append, is **16.4× faster than in-TEE at production n=2048**,
  scaling 2.7× → 16.4× with context. Even the simplest persistent design
  (zero decode re-permute) is a large win → the approach is worth building.
  The open question collapses to a security one: does the gate permit
  prefill-only (this best case), or force periodic re-permute (still a
  likely win per the §10 upload-optimization model)?

**The re-permute half is conditional.** `no_mask − resident` isolates the
K/V convert+upload exactly = **~488 ms @ n=2048** on the *current* pipeline
(GQA-expanded, f32→f16). That is the per-block re-permute *upload* tax;
amortized over N=16 it is ~30 ms/step — it would **lose** to the 11 ms
in-TEE baseline. So persistence wins the per-step read unconditionally
(0.465 ms) but wins the re-permute half **only after** the upload is
optimized: un-replicated storage (4× less) + bf16-native K/V (no convert)
→ modeled ~5 ms ÷16 ≈ 0.3 ms/step. That optimization is part of the
substrate refactor.

**Remaining gaps (after §10.1).** Per-step append is now measured (O(1),
~0.20 ms). What's left unmodelled is **by design**: the §10.1 cell is the
optimistic *prefill-only* case (no decode re-permute) — whether that's
security-achievable is the gate's job, and the re-permute fallback cost is
the §10 upload-optimization model. Also: no permutation / σ-noise / `O_v`
in the per-step path (they're prefill-one-time, trivial for Q); full
softmax over the whole active slice (no prefix/tail merge — a wash);
GQA-**expanded** K/V (un-replicated is 4× less → faster); fp16 Vulkan
(CUDA prod kernel ~1.5–1.9× faster, §9).

**Artefacts:** `bench-results/amulet-attn-resident-5090-2026-05-29.log`
(§10) + `bench-results/amulet-attn-append-5090-2026-05-29.log` (§10.1);
cells `gpu_resident_b8` / `gpu_resident_append_b8` in
`crates/gelo-gpu-wgpu/benches/amulet_attention.rs`; engine seams
`ResidentKvF16` (`upload_resident_kv` / `attend_resident`) and
`ResidentKvSession` (`create_kv_session` / `append_kv` / `attend_session`)
in `crates/gelo-gpu-wgpu/src/lib.rs`.

## 11. Wired decode path (2026-05-29) — the microbench-to-forward gap is dispatch, not attend

The §10.1 cell is an *isolated* attend. §11 is the same engine seam
threaded through the real forward (`decoder_block_cached_batched`,
GLOBAL layers, behind `GELO_GPU_RESIDENT_ATTN`; in-TEE default). Same
production bench as §9 (`gelo_llm_prefill_decode_breakdown`, 4b, B=8,
N=2048, **K=32**, Vulkan). Greedy-parity holds byte-for-byte off-vs-on.

| B=8 decode, Vulkan | in-TEE (§9 baseline) | GPU-resident | Δ |
|---|--:|--:|--:|
| **attn bucket** | 14 574 ms (`tee:attn_cached_inplace_many`) | **8 724 ms** (`tee:attn_resident_gpu`, 1152 calls) | **0.60×** |
| per-call attn | 12.65 ms | 7.57 ms | 1.67× |
| decode wall | 40.6 s | 28.81 s | 0.71× **(confounded — see below)** |
| tok/s/seq | 0.79 | 1.11 | 1.40× |

**Read the bucket, not the wall.** The attention bucket — the only one
the flag touches — is the clean number: **0.60× (−5.85 s)**. The decode
wall fell 40.6→28.81 s, but `engine:matmul` (8956→4382) and
`matmul_many` (9941→8670) also moved between these *separate-session*
runs, and the flag cannot touch the projection/LM-head matmuls. That is
cross-run autotune/thermal variance (§5 confounds), not the wire-up.
**Do not attribute the −11.8 s wall to this change** without a
same-session off/on A/B.

**Why 1.67×, not the §10.1 16.4×.** §10.1 measured a lone `attend`
(0.662 ms/step incl. append). The wired path is 7.57 ms per *(layer,
step)* — the gap is integration overhead the microbench omits, paid
×36 layers ×32 steps: `stack_heads` (f32→f16 of the full Q block per
step), GQA-broadcast attend over the growing slice, `unstack_heads`
(f16→f32 of ctx), and a TEE↔GPU round-trip *serialized per layer*. The
decode loop is **dispatch-bound**, exactly the Phase-3 concern. The
next lever is the fused FlashAttention-D kernel + collapsing per-step
dispatches — not raw attend throughput, which §10.1 already showed is
ample.

**Artefact:** `bench-results/phase4-resident-decode-5090-2026-05-29.log`;
wire-up in `crates/gelo-embedder/src/decoder/forward.rs`
(`decoder_block_cached_batched`, `stack_heads`/`unstack_heads`),
`KvCache::gpu_sessions`, `TrustedExecutor::resident_kv_*`.

## 12. Re-baseline on the default secure offload (2026-06-03) — attention is no longer the bottleneck

First measurement since the **secure GPU-offloaded attention became the
default inference path** (`9f6ec0a`; capability-gated cubek blackbox
prefill offload + resident-cover decode + per-session `C_v` κ=6, σ=0.01,
CUDA backend). Profiles a **single sequence (B=1)** *through* the offload
path — the single-stream forward (`run_prefill`/`run_decode_step`) runs
Global attention in-TEE, so the bench was extended with
`GELO_BENCH_FORCE_BATCHED=1` to drive `run_prefill_batched` /
`run_decode_step_batched` at B=1 (a batch of one through the offload
machinery). Adapter banner confirms `CUDA device 0 (cubecl-cuda)
(Discrete)`; the `tee:attn_prefill_offload` + `prefill_cover:cubek_gpu` +
`tee:attn_resident_cover` + `cover:prefix_partial_gpu` buckets all fire.

Qwen3-4B, fp16, **warmed** (`GELO_BENCH_WARMUP=1`; nvrtc autotune
cached before the measured pass), K=32 decode, σ=0.01, κ=6. Single
sample, ~7% variance floor. Three prompt lengths: **n=2048** (tile
aligned), **n=2047** (ragged → caller-side `n_q` padding to 2048, one
phantom row; chosen for direct comparability to n=2048), **n=8192**
(long-context).

### 12.0 Wall summary

| Cell | Prefill wall | Prefill tok/s | Decode wall | ms/step | Decode tok/s |
|---|--:|--:|--:|--:|--:|
| n=2048 (aligned) | 18.30 s | 111.9 | 6.56 s | 205 | 4.88 |
| n=2047 (ragged)  | 20.12 s | 101.7 | 6.60 s | 206 | 4.85 |
| n=8192 (long)    | 121.36 s | 67.5 | 7.01 s | 219 | 4.57 |

Decode per-step is **near context-independent** (205 → 219 ms over a 4×
context jump) — the resident-K/V cover keeps decode attention flat. The
ragged n=2047 matches aligned n=2048 within the variance floor (the
+1.8 s prefill delta is single-sample noise in the CPU mask bucket, not
padding — padding is one phantom row, <0.05%); ran clean, no NaN. The
caller-side `n_q` padding path is validated at ~zero cost.

### 12.1 Full per-op profiles (ms · share · **calls**)

◆ = GPU-executed (CUDA). All other buckets run on the CPU in-TEE.
**Nesting:** `tee:attn_prefill_offload` is a wrapper whose time
*contains* the indented `↳ prefill_cover:*` rows (+ per-layer cover
sampling); likewise `tee:attn_resident_cover` contains the `↳ cover:*`
rows. The `TOTAL (Σ)` therefore double-counts the wrapper vs its
children — read the de-nested grouping in §12.2 for wall-share.

#### n=2048 — PREFILL (wall 18.30 s)

| op | ms | share | calls |
|---|--:|--:|--:|
| `gelo:mask_unapply:dct4` | 4967.89 | 26.1% | 252 |
| ◆ `engine:matmul_many` | 4663.01 | 24.5% | 72 |
| `tee:attn_prefill_offload` (wrapper) | 2128.72 | 11.2% | 36 |
| `gelo:mask_apply:dct4` | 1516.68 | 8.0% | 144 |
| ◆ `engine:matmul` | 1205.92 | 6.3% | 72 |
| `gelo:shield_stack` | 1170.37 | 6.1% | 144 |
| ↳ `prefill_cover:correct_tee` | 842.66 | 4.4% | 36 |
| `cover:build_covered_prefix+upload` | 592.09 | 3.1% | 36 |
| ↳ ◆ `prefill_cover:cubek_gpu` | 583.45 | 3.1% | 36 |
| ↳ `prefill_cover:rotate_tee` | 468.36 | 2.5% | 36 |
| `tee:swiglu_activate` | 326.59 | 1.7% | 36 |
| `tee:qk_norm` | 256.80 | 1.3% | 36 |
| `tee:residual` | 157.89 | 0.8% | 72 |
| `tee:rmsnorm` | 116.78 | 0.6% | 73 |
| `tee:rope` | 64.20 | 0.3% | 36 |
| `tee:embed_lookup` | 3.46 | 0.0% | 1 |
| `gelo:mask_sample` | 0.05 | 0.0% | 1 |
| **TOTAL (Σ)** | **19064.91** | | |

#### n=2048 — DECODE (wall 6.56 s, K=32)

| op | ms | share | calls |
|---|--:|--:|--:|
| `gelo:mask_unapply:hd3` | 1722.32 | 22.7% | 8096 |
| ◆ `engine:matmul_many` | 1227.67 | 16.2% | 2304 |
| `gelo:mask_apply:hd3` | 937.76 | 12.4% | 4640 |
| ◆ `engine:matmul` | 934.90 | 12.3% | 2336 |
| `tee:attn_resident_cover` (wrapper) | 840.81 | 11.1% | 1152 |
| `gelo:shield_stack` | 825.39 | 10.9% | 4640 |
| ↳ ◆ `cover:prefix_partial_gpu` | 442.65 | 5.8% | 1152 |
| ◆ `tee:compute_logits` (LM-head) | 213.54 | 2.8% | 32 |
| ↳ `cover:q_cover_tee` | 141.77 | 1.9% | 1152 |
| ↳ `cover:acc_uncover_tee` | 111.86 | 1.5% | 1152 |
| ↳ `cover:tail_partial_tee` | 72.10 | 1.0% | 1152 |
| ↳ `cover:tail_build_tee` | 57.31 | 0.8% | 1152 |
| `tee:swiglu_activate` | 26.10 | 0.3% | 1152 |
| `tee:rmsnorm` | 9.61 | 0.1% | 2336 |
| `tee:qk_norm` | 5.21 | 0.1% | 1152 |
| `tee:residual` | 2.83 | 0.0% | 2304 |
| ↳ `cover:merge_tee` | 2.34 | 0.0% | 1152 |
| `tee:rope` | 1.39 | 0.0% | 1152 |
| `gelo:strip_shield` | 1.06 | 0.0% | 32 |
| `tee:embed_lookup` | 0.09 | 0.0% | 32 |
| `gelo:mask_sample` | 0.04 | 0.0% | 64 |
| **TOTAL (Σ)** | **7576.77** | | |

#### n=2047 (ragged) — PREFILL (wall 20.12 s)

| op | ms | share | calls |
|---|--:|--:|--:|
| `gelo:mask_unapply:dct4` | 6060.23 | 28.8% | 252 |
| ◆ `engine:matmul_many` | 4548.03 | 21.6% | 72 |
| `tee:attn_prefill_offload` (wrapper) | 2318.87 | 11.0% | 36 |
| `gelo:mask_apply:dct4` | 2159.05 | 10.2% | 144 |
| ◆ `engine:matmul` | 1214.32 | 5.8% | 72 |
| `gelo:shield_stack` | 1180.36 | 5.6% | 144 |
| ↳ `prefill_cover:correct_tee` | 830.54 | 3.9% | 36 |
| ↳ ◆ `prefill_cover:cubek_gpu` | 778.19 | 3.7% | 36 |
| `cover:build_covered_prefix+upload` | 588.23 | 2.8% | 36 |
| ↳ `prefill_cover:rotate_tee` | 473.62 | 2.2% | 36 |
| `tee:swiglu_activate` | 323.72 | 1.5% | 36 |
| `tee:qk_norm` | 258.91 | 1.2% | 36 |
| `tee:residual` | 158.40 | 0.8% | 72 |
| `tee:rmsnorm` | 115.00 | 0.5% | 73 |
| `tee:rope` | 63.99 | 0.3% | 36 |
| `tee:embed_lookup` | 3.48 | 0.0% | 1 |
| `gelo:mask_sample` | 0.05 | 0.0% | 1 |
| **TOTAL (Σ)** | **21074.98** | | |

#### n=2047 (ragged) — DECODE (wall 6.60 s, K=32)

| op | ms | share | calls |
|---|--:|--:|--:|
| `gelo:mask_unapply:hd3` | 1745.82 | 22.9% | 8096 |
| ◆ `engine:matmul_many` | 1227.39 | 16.1% | 2304 |
| `gelo:mask_apply:hd3` | 956.40 | 12.5% | 4640 |
| ◆ `engine:matmul` | 934.28 | 12.3% | 2336 |
| `tee:attn_resident_cover` (wrapper) | 834.56 | 11.0% | 1152 |
| `gelo:shield_stack` | 826.03 | 10.8% | 4640 |
| ↳ ◆ `cover:prefix_partial_gpu` | 435.55 | 5.7% | 1152 |
| ◆ `tee:compute_logits` (LM-head) | 228.42 | 3.0% | 32 |
| ↳ `cover:q_cover_tee` | 143.70 | 1.9% | 1152 |
| ↳ `cover:acc_uncover_tee` | 114.42 | 1.5% | 1152 |
| ↳ `cover:tail_partial_tee` | 70.04 | 0.9% | 1152 |
| ↳ `cover:tail_build_tee` | 55.71 | 0.7% | 1152 |
| `tee:swiglu_activate` | 26.45 | 0.3% | 1152 |
| `tee:rmsnorm` | 9.38 | 0.1% | 2336 |
| `tee:qk_norm` | 5.26 | 0.1% | 1152 |
| `tee:residual` | 2.61 | 0.0% | 2304 |
| ↳ `cover:merge_tee` | 2.35 | 0.0% | 1152 |
| `tee:rope` | 1.59 | 0.0% | 1152 |
| `gelo:strip_shield` | 1.07 | 0.0% | 32 |
| `tee:embed_lookup` | 0.10 | 0.0% | 32 |
| `gelo:mask_sample` | 0.04 | 0.0% | 64 |
| **TOTAL (Σ)** | **7621.16** | | |

#### n=8192 — PREFILL (wall 121.36 s)

| op | ms | share | calls |
|---|--:|--:|--:|
| ◆ `engine:matmul_many` | 31468.60 | 24.1% | 72 |
| `gelo:mask_unapply:dct4` | 26756.64 | 20.5% | 252 |
| ◆ `engine:matmul` | 17489.95 | 13.4% | 72 |
| `tee:attn_prefill_offload` (wrapper) | 17399.49 | 13.3% | 36 |
| ↳ ◆ `prefill_cover:cubek_gpu` | 9042.58 | 6.9% | 36 |
| `gelo:mask_apply:dct4` | 7710.09 | 5.9% | 144 |
| ↳ `prefill_cover:correct_tee` | 4849.55 | 3.7% | 36 |
| `gelo:shield_stack` | 4774.90 | 3.6% | 144 |
| `cover:build_covered_prefix+upload` | 3131.86 | 2.4% | 36 |
| `tee:residual` | 2666.92 | 2.0% | 72 |
| ↳ `prefill_cover:rotate_tee` | 2570.48 | 2.0% | 36 |
| `tee:swiglu_activate` | 1168.36 | 0.9% | 36 |
| `tee:qk_norm` | 998.02 | 0.8% | 36 |
| `tee:rmsnorm` | 482.68 | 0.4% | 73 |
| `tee:rope` | 278.70 | 0.2% | 36 |
| `tee:embed_lookup` | 36.93 | 0.0% | 1 |
| `gelo:mask_sample` | 0.13 | 0.0% | 1 |
| **TOTAL (Σ)** | **130825.87** | | |

#### n=8192 — DECODE (wall 7.01 s, K=32)

| op | ms | share | calls |
|---|--:|--:|--:|
| `gelo:mask_unapply:hd3` | 1732.85 | 20.5% | 8096 |
| `tee:attn_resident_cover` (wrapper) | 1264.58 | 15.0% | 1152 |
| ◆ `engine:matmul_many` | 1223.45 | 14.5% | 2304 |
| ◆ `engine:matmul` | 943.65 | 11.2% | 2336 |
| `gelo:mask_apply:hd3` | 936.54 | 11.1% | 4640 |
| ↳ ◆ `cover:prefix_partial_gpu` | 869.08 | 10.3% | 1152 |
| `gelo:shield_stack` | 829.16 | 9.8% | 4640 |
| ◆ `tee:compute_logits` (LM-head) | 219.42 | 2.6% | 32 |
| ↳ `cover:q_cover_tee` | 142.13 | 1.7% | 1152 |
| ↳ `cover:acc_uncover_tee` | 108.83 | 1.3% | 1152 |
| ↳ `cover:tail_partial_tee` | 72.53 | 0.9% | 1152 |
| ↳ `cover:tail_build_tee` | 57.25 | 0.7% | 1152 |
| `tee:swiglu_activate` | 26.53 | 0.3% | 1152 |
| `tee:rmsnorm` | 9.60 | 0.1% | 2336 |
| `tee:qk_norm` | 5.23 | 0.1% | 1152 |
| `tee:residual` | 2.72 | 0.0% | 2304 |
| ↳ `cover:merge_tee` | 2.33 | 0.0% | 1152 |
| `tee:rope` | 1.41 | 0.0% | 1152 |
| `gelo:strip_shield` | 1.03 | 0.0% | 32 |
| `tee:embed_lookup` | 0.09 | 0.0% | 32 |
| `gelo:mask_sample` | 0.04 | 0.0% | 64 |
| **TOTAL (Σ)** | **8448.49** | | |

### 12.2 De-nested wall-share — where the time goes

Grouping the buckets above (wrapper rows replace their `↳` children; the
attention group is the wrapper, GPU-matmul is `engine:matmul*`, the cover
on the *matmul* offload is mask + shield):

| Group | Prefill n=2048 | Prefill n=8192 | Decode (both n) |
|---|--:|--:|--:|
| **Mask + shield** (DCT-IV prefill / HD₃ decode + `shield_stack`) | **41.8%** | 32.3% | **~50–53%** |
| **GPU matmul** (`engine:matmul` + `matmul_many`) | 32.1% | **40.3%** | ~31–33% |
| **Attention offload** (`tee:attn_*`, incl. cover prep) | 11.6% | 14.3% | 13–18% |
| Decode K/V prep hoisted into prefill (`build_covered_prefix`) | 3.2% | 2.6% | — |
| Other in-TEE (rmsnorm, rope, qk_norm, swiglu, residual) + LM-head | ~8% | ~7% | ~3% |

### 12.3 Findings

- **Attention is no longer the bottleneck.** Having offloaded it, the
  binding cost is the **per-matmul activation cover** — the DCT-IV
  (prefill) / HD₃ (decode) mask apply/unapply + `shield_stack` around
  every registered-linear GPU matmul (the `WEIGHTS-PUB` round-trip:
  apply→upload→matmul→readback→unapply). It is **42% of prefill / ~50%
  of decode** at n=2048. The mask is a CPU transform, DDR5-bandwidth-
  bound on this box (~85 GB/s).
- **`mask_unapply` ≫ `mask_apply`** (252 vs 144 calls at prefill; 8096 vs
  4640 at decode): each fused matmul's 2–3 outputs are unmasked
  separately, so the unapply direction carries more calls and more time.
- **GPU matmul overtakes the mask only at long context** (40% vs 32% at
  n=8192). It grows **super-linearly** — `matmul_many` 4663 → 31469 ms is
  6.75× for a 4× token jump — i.e. increasingly compute/occupancy-bound,
  not transfer-bound. Consistent with §8's ~2 TFLOP/s finding: tensor
  cores look under-utilised, and the gap widens with n.
- **The attention offload is prep-dominated.** In-TEE cover bookkeeping
  (`rotate_tee` + `correct_tee` + O-sampling + `build_covered_prefix`)
  exceeds the GPU attention it protects (`cubek_gpu`) by ~3.6× at n=2048
  (≈2.1 s vs 0.58 s); only at n=8192 does the O(n²) GPU term (9.0 s)
  approach the prep (≈11.5 s). Decode attention grows **only** in the GPU
  prefix term (`prefix_partial_gpu` 443 → 869 ms); cover prep is
  context-flat (~390 ms).

### 12.4 Re-ranked next levers (against these numbers)

The binding bottleneck has **moved off attention onto the matmul cover**;
the catalog levers re-rank accordingly (detail + threat-model/accuracy
trade-offs in the deferred-optimizations handoff):

1. **End-to-end bf16 activations (X1).** Halves the bytes the DCT-IV/HD₃
   mask touches and uploads → projected ~20–25% off prefill, ~25% off
   decode. Threat-model-neutral (cover unchanged); the GPU leg is already
   fp16 so accuracy impact is within the offload's existing fp16 floor
   (re-gate HumanEval). The single biggest lever; the documented
   prerequisite (forward-wire 3b/3c unbuilt).
2. **R4 async overlap (X2).** On PCIe the CPU mask of matmul *k+1* can
   overlap the GPU matmul of *k* (the engines are independent, unlike
   UMA). Mask (~42%) and matmul (~32%) are comparable → projected
   ~10–20% wall, exact (pure scheduling), compounds with #1.
3. **Blackwell sm_120 CMMA verification (P4).** The super-linear matmul
   growth + §8's ~2 TFLOP/s say tensor cores are under-used; if CMMA
   engages, matmul 2–5× → ~15–25% prefill at long context (the win widens
   with n). Investigate first.

Attention levers (P1 cubek read-index, D1 fused decode kernel) drop in
priority — attention is 12–18% today — but matter at 16–32k context
where `cubek_gpu` (O(n²)) and `prefix_partial_gpu` climb.

**Artefacts:** `bench-results/gelo-b1-offload-n{2048,2047,8192}-2026-06-03.log`.
Bench knob `GELO_BENCH_FORCE_BATCHED` (routes B=1 through the offload
path) in `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`.

## 13. bf16 offload read-back (2026-06-03) — +9% prefill, and a correction to §12's bandwidth premise

Implemented the first bf16-activation lever (X1) on the registered-linear
offload: the matmul outputs are read back as **bf16** (host narrowing
f16→bf16 instead of f16→f32) and the mask unapply runs on bf16 storage.
The masked operand / apply side stays f32 (apply was the smaller bucket
and HD₃'s bf16 apply is widen-narrow). Engine seam:
`matmul_many_bf16_out` + `run_registered_linear_bf16_out` +
`tensor_data_to_array_bf16_*` (`gelo-gpu-wgpu/src/lib.rs`); wiring +
`unmask_per_sequence_bf16` + the `dispatch_unmask_per_sequence` router
(`gelo-protocol/src/sim.rs`). Capability-gated: engages only when the
engine is fp16 (`prefers_bf16_output`) **and** the mask is DCT-IV — so
CPU/sim/f32 executors keep exact-f32 parity, and HD₃/Haar stay f32.
Escape hatch `GELO_BF16_OFFLOAD=0`.

Same-process A/B, B=1 forced-batched, warmed, K=32, Qwen3-4B, CUDA.

### 13.1 Prefill A/B (bf16 OFF → ON)

**n=2048 (wall 18.62 → 16.83 s, −9.6%)**

| op | OFF ms | ON ms | Δ | calls |
|---|--:|--:|--:|--:|
| `gelo:mask_unapply:dct4` | 4971.12 | 4809.60 | −3.2% | 252 |
| ◆ `engine:matmul_many` | 4932.32 | 3599.04 | **−27.0%** | 72 |
| `tee:attn_prefill_offload` | 2128.74 | 2131.10 | ~0 | 36 |
| `gelo:mask_apply:dct4` | 1549.71 | 1544.18 | ~0 | 144 |
| ◆ `engine:matmul` | 1221.73 | 1311.58 | +7.4% | 72 |
| `gelo:shield_stack` | 1165.42 | 1165.34 | ~0 | 144 |

**n=8192 (wall 119.64 → 108.91 s, −9.0%)**

| op | OFF ms | ON ms | Δ | calls |
|---|--:|--:|--:|--:|
| ◆ `engine:matmul_many` | 31006.12 | 23931.81 | **−22.8%** | 72 |
| `gelo:mask_unapply:dct4` | 26492.75 | 25377.25 | −4.2% | 252 |
| ◆ `engine:matmul` | 17094.79 | 15662.94 | −8.4% | 72 |
| `tee:attn_prefill_offload` | 17318.41 | 17296.55 | ~0 | 36 |
| `gelo:mask_apply:dct4` | 7758.58 | 7740.58 | ~0 | 144 |

### 13.2 Decode A/B — why bf16 is gated to DCT-IV (prefill) only

bf16 applied to the HD₃ decode mask **regressed** (the bf16 HD₃ slice is
bulk widen-narrow — no DRAM win — and the decode outputs are tiny
(n_q=1), so there is no read-back win to offset the extra conversion):

| op | OFF ms | ON (HD₃ bf16) ms | Δ | calls |
|---|--:|--:|--:|--:|
| `gelo:mask_unapply:hd3` (n=2048) | 1734.25 | 1963.32 | **+13.2%** | 8096 |
| decode wall (n=2048) | 6.61 s | 6.92 s | **+4.7%** | — |
| `gelo:mask_unapply:hd3` (n=8192) | 1729.34 | 1952.14 | **+12.9%** | 8096 |
| decode wall (n=8192) | 7.02 s | 7.34 s | **+4.6%** | — |

With the **DCT-IV gate**, decode reverts to the exact f32 path — confirmed
identical (n=2048: decode wall 6.61 s, `mask_unapply:hd3` 1734 ms = the
OFF numbers) — so the shipped config is **prefill −9.6% / decode flat**.

### 13.3 Finding — the mask transform is compute-bound, not bandwidth-bound

§12 (and the X1 plan) projected ~20–25% from bf16 by assuming the DCT-IV /
HD₃ mask transforms were DDR5-bandwidth-bound. **The A/B refutes that.**
The bf16 win lands almost entirely in `engine:matmul_many` (−23–27%), not
in `mask_unapply` (−3–4%, ~flat):

- **`mask_unapply:dct4` barely moved** even though its input is now bf16
  (half the bytes). The tiled DCT-IV cascade widens to f32 and runs the
  **same f32 cosine FLOPs** — it is **compute-bound**, so halving the
  storage traffic does almost nothing. (Revises §12.3's "DDR5-bandwidth-
  bound" attribution for the mask buckets.)
- **The real lever is the read-back narrowing.** `run_registered_linear_bf16_out`
  narrows the GPU f16 result to a bf16 host array instead of f32 — half
  the host write, concentrated in the wide gate∥up/QKV outputs — and that
  cost is timed under `engine:matmul*`. That is the −1.3 s (n=2048) /
  −7.1 s (n=8192) on `matmul_many`, ≈ the whole −9% wall.

Implication for the remaining bf16 work: the **apply-side operand bf16**
(bf16 shield-stack + `apply_in_place_slice_bf16`) is **not** worth it —
its transform is the same compute-bound cascade, and the upload narrowing
is a smaller surface than the read-back. The bf16-input engine path
(`matmul_many_bf16_input`) is therefore deprioritised. The mask-transform
cost itself only falls with a **cheaper transform** or **fewer columns**,
not with precision — see the vectorise spike below.

### 13.4 Cascade vectorise spike — FFT-bound, our glue is 8%

Follow-up to §13.3 ("the cascade is compute-bound"): is vectorising the
cascade kernel worth it? Per-tile attribution
(`dct4::tests::dct4_cascade_vectorize_spike`, release, n=2064):

| cost centre | share |
|---|--:|
| FFT — 3× DCT-IV (rustdct/rustfft, **already SIMD**) | **91.9%** |
| diagonal multiplies (D₁/D₂/D₃, our glue) | 6.3% |
| tile transpose (`copy_tile_in/out`, our glue) | 1.8% |

The cascade is **92% the FFT**, which rustfft already vectorises (AVX/SSE).
Our non-vectorised glue (the branchy sign-diag + scalar transpose) is only
**8.1%** of the cascade; since the cascade (`mask_unapply`) is ~25% of
prefill wall, vectorising the glue to *zero* caps at ~2% of prefill, and
realistically ~1%. **Not worth it.** Full-unapply rate for reference:
~1.5–1.8 ns/elem (30.5 ms at n=2064 × d=9728).

So the cascade only gets cheaper via (a) a **cheaper transform** — HD₃/FWHT
is pure ±1 add/sub butterflies vs DCT-IV's Bluestein-FFT cosine at non-pow2
n — or (b) **fewer columns** — eliminate the FFN-intermediate unmask
(gate∥up = 63% of unapply columns) via a SiLU-commuting feature-axis
permutation. Both are security-gated (mask-family strength / permutation
vs orthogonal cover). The earlier "merge fused-output unapplies" idea is
**also dead** (compute-bound → identical FLOPs; would only add concat
copies; the transform is already column-parallel).

**SIMD A/B + the real lever is the build target.** Implementing the
branchless diag (multiply by ±1, no branch) and A/B-ing it on the spike
microbench surfaced something bigger:

| build | diag baseline (branched) | diag branchless | full unapply (d=9728) | cascade |
|---|--:|--:|--:|--:|
| **default `--release` (x86-64 / SSE2)** | 38.4 µs | **5.95 µs (6.45×)** | 1.52 ns/elem | — |
| **`target-cpu=native` (Zen4 AVX-512)** | **3.5 µs** | 3.5 µs (1.00×) | 1.32 ns/elem | **~13% faster** |

- On the **default build** the branched diag is slow (the data-dependent
  negate doesn't vectorise); branchless gives 6.45×, but the diag is ~7%
  of the cascade → only **~1.5% prefill**.
- On **`target-cpu=native`** the compiler vectorises the *branched*
  baseline too (38→3.5 µs, 11×), so branchless gives nothing — **and the
  whole cascade is ~13% faster** (rustfft + glue both pick up AVX-512).
- **The repo's `.cargo/config.toml` sets no `target-cpu`** — production
  (and every §12/§13 bench here) builds at the x86-64 SSE2 baseline. So
  the SIMD headroom isn't in hand-vectorising one kernel; it's a
  **build-flag**: adding `target-cpu=native` (fixed dGPU box) or
  `x86-64-v3` (portable AVX2) to the config. The cascade gets ~13% free,
  and this almost certainly speeds **every autovectorised CPU bucket**
  (mask apply/unapply, shield, cover rotates, in-TEE attention) — which
  are ~50%+ of prefill/decode. **Next: measure `target-cpu=native` on the
  full prefill/decode bench** (RUSTFLAGS must re-include the rpath
  link-args, or add `target-cpu` to the config) — likely the largest
  free CPU-side win surfaced so far. (Branchless-diag is kept as a
  spike-only `#[cfg(test)]` helper; it's subsumed by the build flag.)

### 13.5 Disposition

bf16 read-back gives net **−9% prefill wall** at both n=2048 and n=8192,
decode unchanged. **Accuracy gate (HumanEval pass@1, secure offload +
`C_v` κ=6, B=1): pass@1 = 6/20** — a **1-problem drop** from the prior
f16-offload baseline (7/20), landing exactly at the Plain Qwen3-4B
reference (6/20). On a 20-prompt gate with documented fp16-cliff
sensitivity (greedy autoregression amplifies ~1e-3 deviations on
cliff-adjacent prompts) this is within the noise floor; the bf16 read-back
carries 3 fewer mantissa bits than the f16 result it replaces, the
plausible cause of the flipped completion.

**Decision: bf16 read-back ships default-on** for the DCT-IV prefill
offload (fp16 GPU), accepting the Plain-Qwen3 reference (6/20) as the bar
for the ~9% prefill win; escape hatch `GELO_BF16_OFFLOAD=0`. A
larger-eval re-test (full HumanEval-164 / MBPP) is owed to confirm the
6-vs-7 is gate noise, not real degradation.

**Updated lever ranking (§12.4):** bf16's realised win is ~9% (read-back),
not the projected ~20–25% (transform) — and the transform itself is
FFT-bound (§13.4), so vectorising it is out. The top remaining levers are
**R4 async overlap (X2)** (overlap CPU mask with GPU matmul), then the
**FLOP-reducing mask levers** (cheaper HD₃-at-prefill transform; eliminate
the FFN-intermediate unmask) — not further bf16 or kernel-vectorisation work.

**HumanEval gate batching:** the gate generator was switched from B=1
per-prompt to **B=8, length-sorted** (`GELO_HE_BATCH`, default 8) to cut
wall. Expectation is modest — GELO decode is per-row (mask-cover) bound,
not weight-bound, so batching amortises only the ~55 ms/step fixed cost
(~1.3× throughput ceiling), and `generate_batched` has no continuous
batching (a batch runs to its longest generator), so length-sorting is
needed to limit idle-slot waste. Not the order-of-magnitude a weight-bound
server would see.

**Artefacts:** `bench-results/gelo-b1-bf16{on,off}-n{2048,8192}-2026-06-03.log`,
`bench-results/gelo-b1-bf16dct4gate-n2048-2026-06-03.log`,
`bench-results/humaneval-bf16on-2026-06-03.log`. Escape hatch
`GELO_BF16_OFFLOAD=0`.

## 14. Build target (2026-06-03) — the no-F16C baseline was the bigger lever

Following the §13.4 spike (which found `target-cpu=native` ~13% on the
isolated cascade), the full prefill/decode bench was re-run with
`target-cpu=native` (Zen4 AVX-512/F16C) and compared to the **documented
SSE2-baseline bf16-on numbers** (no SSE2 re-run). B=1 forced-batched,
warmed, bf16-on, CUDA. The repo `.cargo/config.toml` sets **no
`target-cpu`**, so production — and every measurement in §3–§13 — built at
the plain `x86-64` baseline (SSE2, **no F16C, no AVX2**).

### 14.1 Wall (SSE2 baseline → native)

| cell | SSE2 | native | Δ | tok/s |
|---|--:|--:|--:|--:|
| prefill n=2048 | 16.83 s | 13.86 s | **−17.6%** | 121→148 |
| prefill n=8192 | 108.91 s | 97.69 s | **−10.3%** | 75→84 |
| decode n=2048 | 6.61 s | 5.77 s | **−12.7%** | 4.8→5.5 |
| decode n=8192 | 7.02 s | 6.17 s | **−12.1%** | 4.6→5.2 |

### 14.2 Where it comes from (◆ = GPU; rest CPU in-TEE)

| bucket (prefill n=2048) | SSE2 | native | Δ |
|---|--:|--:|--:|
| ◆ `engine:matmul_many` | 3599 | 1159 | **−68%** |
| ◆ `engine:matmul` | 1312 | 806 | **−39%** |
| `gelo:mask_apply:dct4` | 1544 | 1416 | −8% |
| `gelo:mask_unapply:dct4` | 4810 | 4602 | −4% |
| `gelo:shield_stack` (decode bucket) | 825 | 620 | **−25%** |

The win is **not** in the mask transform (−2 to −8%; rustfft already
runtime-detects AVX, so it was vectorised even at the SSE2 build). It is
overwhelmingly in the **`engine:matmul*` buckets (−34 to −68% across both
phases)** — which wrap the GPU offload's host-side **f32↔f16/bf16
conversions** (upload `array2_to_tensor_f16`, read-back
`tensor_data_to_array_bf16_from_f16`). The baseline `x86-64` target lacks
**F16C**, so those conversions ran **scalar**; native enables
`vcvtps2ph`/`vcvtph2ps` and they vectorise. `shield_stack`'s Gaussian fill
also picks up AVX/FMA (−25%). (Single-sample; ±~7% — but the
`engine:matmul*` deltas and the walls are far above the floor.
`prefill_cover:correct_tee` *rose* ~8–35%, a codegen/variance quirk worth a
confirm re-run.)

### 14.3 Finding + disposition

**The §3–§13 `engine:matmul*` buckets were partly inflated by scalar
half-precision conversions on a no-F16C build.** Enabling F16C+AVX2 is a
one-line build-config change worth **~10–18% prefill / ~12% decode** —
broader than the bf16 read-back lever (~9% prefill) it sits on top of, and
**backend-invariant** (it speeds the host side of every GPU offload, not a
single kernel).

Recommended: add to `.cargo/config.toml [build] rustflags` either
`target-cpu=x86-64-v3` (**portable** — Haswell+/Zen+; includes F16C, AVX2,
FMA — captures the conversion win) or `target-cpu=native` (this box's
AVX-512, **non-portable**, slightly more on the wide loops). Either pins
the deployment ISA, so it's a portability call. ⚠ When set via the
`RUSTFLAGS` *env* it **replaces** the config's rpath link-args (libblis
won't load at runtime) — add `target-cpu` *inside* the config's `rustflags`
array (keeping the rpath args), or carry the rpath args in `RUSTFLAGS` too.

This now outranks the other levers: **build target (≈12–18%, free) >
R4 async overlap > FLOP-reducing mask levers**; bf16 read-back stays as a
landed ~9% on top.

**Artefacts:** `bench-results/gelo-b1-bf16on-native-n{2048,8192}-2026-06-03.log`.

## 15. HD₃ rayon misfire fix (2026-06-03) — decode −45%, and the DCT-IV size finding

Investigating "what did we miss in the mask transforms" surfaced two
shape-sensitivity bugs/levers, one now fixed:

### 15.1 The decode HD₃ cost was fork-join overhead, not transform

The decode mask is HD₃ at **stacked_n=16** (1+k, pow2). The FWHT and the
diagonal passes gated rayon on **total elements** (≥65 536), tuned for
tall prefill-era shapes — but at n=16 the first radix-8 stage splits into
**2 chunks** and the radix-2 tail into **one**, and a diag pass is ≤16
trivial row-negates. So the wide decode outputs (gate/up 16×9728, Q
16×4096) paid ~6 fork-joins per unapply for ≤2-way parallelism on ~µs of
work. Measured discontinuity (`hd3_decode_shape_spike`, n=16):

| d | before | after fix | path before |
|---|--:|--:|---|
| 4095 | 22.1 µs (0.34 ns/elem) | 23.7 µs | serial |
| **4096** | **347.3 µs (5.30 ns/elem)** | **22.7 µs** | rayon (**15.7× cliff**) |
| 9728 | 506.7 µs | **53.4 µs** | rayon (~10×) |

**Fix:** per-stage chunk-parallelism gate (`FWHT_MIN_PAR_CHUNKS = 4`) on
the FWHT stages + a row floor (`FWHT_MIN_PAR_ROWS = 64`) on the diag
passes. Tall shapes (≥64 rows) keep the parallel path bit-for-bit; the
change is dispatch-only (no numerics). All HD₃ round-trip/parity tests
pass unchanged.

**Production impact (n=2048 B=1, native, bf16-on):**

| | before | after | Δ |
|---|--:|--:|--:|
| `gelo:mask_unapply:hd3` (8096 calls) | 1710 ms | **206 ms** | **−88%** |
| `gelo:mask_apply:hd3` (4640 calls) | 947 ms | **108 ms** | **−89%** |
| **decode wall** | 5.77 s | **3.18 s** | **−45%** |
| decode tok/s | 5.5 | **10.1** | 1.8× |
| prefill wall | 13.86 s | 13.88 s | unchanged ✓ |

Cumulative decode this session: 6.61 s (SSE2) → 5.77 s (native) →
**3.18 s** (FWHT fix) = **2.1×**. The decode bottleneck is now the
resident-cover attention (21%) + GPU matmul round-trip (34%) + shield
(13%); the HD₃ mask is down to ~8%.

**Verdict on bf16-native HD₃ (§4.E.1):** retired. The HD₃ cost was
neither compute- nor memory-bound but **overhead-bound**; after the fix
the FWHT is an L2-resident streaming add/sub kernel at ~0.34 ns/elem
where bf16 storage would trade halved bytes for per-stage widen/narrow
ops — at best a wash on a now-~300 ms bucket, with extra per-stage
rounding. Not worth pursuing.

### 15.2 DCT-IV is size-pathological — smooth-size padding is worth ~−18% prefill

`stacked_n = n_data + k` lands on whatever factorisation chance gives.
The production prefill size **2064 = 2⁴·3·43** is the worst in its
neighbourhood (rustdct hits the 43-prime path). Size sweep
(`DCT4_BENCH_N`, full unapply, native, ns/elem at d=9728):

| stacked_n | factors | ns/elem | vs 2064 |
|---|---|--:|--:|
| 2048 | 2¹¹ | 0.974 | 1.32× |
| **2064 (current)** | 2⁴·3·**43** | **1.285** | — |
| 2080 | 2⁵·5·13 | 1.050 | 1.22× |
| **2112** | 2⁶·3·11 | **0.748** | **1.72×** |
| 2160 | 2⁴·3³·5 | 0.733 | 1.75× |
| 2304 | 2⁸·3² | 0.701 | 1.83× |

Padding 2064 → 2112 (+2.3% rows) is **1.68× faster per call including
the extra rows**. The DCT-IV buckets are ~43% of native prefill →
**≈ −18% prefill wall**, exact and security-neutral (pad rows are extra
shield/zero cover, sliced off after unapply). **Not yet implemented** —
needs smooth-size rounding in the stacked_n sizing + pad-row fill on the
DCT-IV path (the HD₃ path already has the pad machinery). This is now the
top prefill lever, ahead of R4 async overlap. Deeper variants if more is
needed: batched DCT across columns (FFTW/AOCL-FFT `REDFT11` + `howmany` —
attacks the 95%-FFT share, we already vendor AOCL) and a security-gated
3→2 cascade-stage reduction (−33% FFT).

**Artefacts:** `bench-results/gelo-b1-fwhtfix-native-n2048-2026-06-03.log`;
spikes `hd3_decode_shape_spike` (hd3.rs), `dct4_cascade_vectorize_spike`
(`DCT4_BENCH_N` sweep).

## 16. DCT-IV smooth-size padding (2026-06-03) — implemented; −5% prefill (vs −18% projected)

Implements §15.2. `dct4::next_fast_n(n)` rounds the DCT-IV `stacked_n` up
to the smallest `m ≥ n` with `v₂(m) ≥ 4` and largest prime factor ≤ 11 —
the criterion the candidate sweep favoured (high pow2 content beats pure
smoothness: 2112 = 2⁶·3·11 at 0.747 ns/elem outruns 7-smooth 2100 at
0.854; 13+ is slow). Production shapes: 2064 → **2112**, 8208 → **8400**
(both +2.3% rows). Wired at the five `Dct4Mask::fresh` sites + the
single-path `stacked_n` computation; pad rows are zero cover through the
orthogonal round-trip (data rows exact — all 90 protocol tests pass
unchanged). `n < 32` passes through.

Measured (B=1, native, bf16-on, vs the §15 build):

| | before | after | Δ |
|---|--:|--:|--:|
| prefill n=2048 | 13.88 s | **13.20 s** | **−4.9%** |
| prefill n=8192 | 97.69 s | **93.98 s** | **−3.8%** |
| `gelo:mask_apply:dct4` (n=2048, f32) | 1416 ms | 1113 ms | **−21%** |
| `gelo:mask_unapply:dct4` (n=2048, bf16) | 4602 ms | 4138 ms | −10% |
| decode | 3.18 s | 3.28 s | untouched (noise) |

**Why only ~⅓ of the §15.2 projection (−18%):** the projection scaled the
whole mask bucket by the microbench transform ratio. The **f32 apply**
tracked it (−21%); the **bf16 unapply** did not — its per-tile bf16⇄f32
widen/narrow + copy-out/alloc overhead is a constant ~0.7–1.0 ns/elem
that doesn't shrink with the FFT and now *dominates* the unapply bucket
(production 1.77 ns/elem vs 0.747 microbench transform-only). Lesson
repeated from §13: transform-only microbenches overstate bucket-level
wins when a fixed conversion overhead rides the same profile label.

Cumulative this session (n=2048, B=1): prefill **16.83 → 13.20 s
(−21.6%)**, decode **6.61 → ~3.2 s (2.05×, 4.8 → ~10 tok/s)**. Next
levers: the unapply's now-dominant **conversion/copy overhead** (fuse the
bf16 widen into the first cascade tile load — partially exists; audit the
copy-out), **R4 async overlap**, and the **FLOP-reducing mask levers**
(FFN-intermediate unmask elimination; batched DCT via AOCL-FFT REDFT11).

**Artefacts:** `bench-results/gelo-b1-smoothpad-native-n{2048,8192}-2026-06-03.log`;
`dct4::next_fast_n` + `next_fast_n_picks_fast_sizes` (dct4.rs).

## 17. Fused bf16 unmask + glibc mmap-churn fix (2026-06-03) — prefill −7.7%

§16 left the unapply bucket dominated by conversion/copy overhead. The
overhead-split spike (`dct4_bf16_unapply_overhead_spike`, n=2112,
gate∥up width) decomposed it:

| component (per call, d=9728) | ms |
|---|--:|
| f32 transform floor | 14.99 |
| + bf16 per-tile widen/narrow | +0.49 |
| + output alloc + data-row widen copy (production tail) | **+33.36** |
| — of which: warm widen-copy alone | 7.06 |
| — of which: **fresh-mmap page-fault churn** | **~26** |

The bf16 tile conversion is noise; the cost is (a) **page-fault churn** —
glibc serves blocks above its mmap threshold (dynamic cap 32 MiB) by
fresh `mmap` and frees by `munmap`, so every wide unmask re-pays ~20k
minor faults + kernel zeroing on its ~80 MB output — and (b) a separate
**single-threaded widen-copy** pass.

Two fixes, both exact:

1. **Fused unapply** — `Dct4Mask::unapply_bf16_into_f32_rows`: the
   cascade tiles write the data rows **directly into the f32 output**
   (read-only bf16 source; no whole-buffer bf16 narrow store, no separate
   copy pass, pad/shield rows never stored, one bf16 rounding *fewer* per
   element — accuracy equal-or-better, parity-tested).
2. **`mallopt(M_MMAP_THRESHOLD/M_TRIM_THRESHOLD, 1 GiB)`** at executor
   init (linux-gnu only): large transient buffers are arena-reused,
   faulted once. Process-wide — it also covers the engine's upload Vecs.
   Explicit `malloc_trim` (the bench's `glibc_release_freed`) still
   reclaims.

Measured (B=1, native, bf16-on, vs §16):

| | before | after | Δ |
|---|--:|--:|--:|
| prefill n=2048 | 13.20 s | **12.19 s** | **−7.7%** |
| `gelo:mask_unapply:dct4` (n=2048) | 4138 ms | **3169 ms** | **−23%** |
| prefill n=8192 | 93.98 s | **88.80 s** | **−5.5%** |
| ◆ `engine:matmul_many` (n=8192) | 15334 ms | **12729 ms** | **−17%** |
| ◆ `engine:matmul` (n=8192) | 14622 ms | **12515 ms** | **−14%** |
| decode (both n) | 3.28 / 3.61 s | 3.20 / 3.57 s | flat ✓ |

At n=8192 the win lands in the **matmul buckets** — the mallopt relieved
the mmap churn on the ~86 MB *upload* allocations (process-wide effect),
while the unapply bucket there sat within variance. At n=2048 the unapply
took the −23%.

**Cumulative (n=2048, B=1, this session):** prefill **16.83 → 12.19 s
(−27.6%)** [vs the pre-bf16 f32 SSE2 build: 18.62 → 12.19 = −34.5%],
decode **6.61 → 3.20 s (2.07×, ~10 tok/s)**. The prefill bucket order is
now: unapply 23% · attention-offload prep 18% · matmul+shield+correct+apply
~8% each — the FFT transform itself is finally the unapply's dominant
share, so further mask-side gains need the FLOP-reducing levers
(batched/AOCL DCT, FFN-intermediate unmask elimination) or R4 overlap.

**Artefacts:** `bench-results/gelo-b1-fusedunmask-native-n{2048,8192}-2026-06-03.log`;
spike `dct4_bf16_unapply_overhead_spike`, parity
`dct4_fused_bf16_unapply_parity` (dct4.rs).

## 18. Batched-DCT library spike (2026-06-03) — FFTW/AOCL-FFTW route refuted

Measure-only spike of the "batched DCT via FFTW `REDFT11` + `plan_many`"
lever (§17 next-levers): a standalone C harness against the system
`libfftw3f` (FFTW_MEASURE plans; manual declarations — no dev header, no
shipped dependency), at the cascade tile shape (16 columns × n),
single-threaded per-tile vs rustdct's per-column structure.

Per single DCT-IV pass over a 16-column tile:

| n=2112 | µs/tile-pass | ns/elem |
|---|--:|--:|
| **rustdct (current)** | **76.2** | **2.26** |
| FFTW per-column ×16 | 122.7 | 3.63 |
| FFTW `howmany=16` contiguous | 115.4 | 3.42 |
| FFTW `howmany=16` interleaved | 121.8 | 3.61 |

(n=8400: rustdct 341 µs vs FFTW 389–432 µs — same picture.)

**Refuted on two counts:** (1) FFTW's r2r/REDFT11 codelets show **no
across-batch SIMD win** — `howmany=16` ≈ per-column, the interleaved lane
layout is *worse*; FFTW's SIMD strength is its complex-DFT codelets, not
the DCT paths, so the 2–4× batch-FFT expectation does not materialise
through this library. (2) rustdct at our `next_fast_n` sizes is already
**1.14–1.5× faster** than system FFTW. AOCL-FFTW shares the r2r codelet
architecture — closing some absolute gap is plausible, flipping a 0%
batching structure into 2–4× is not. (The GPL question dissolves with it.)

What survives: a **custom lane-parallel DCT-IV** (vertical SIMD, 16
columns in AVX-512 lanes — the argument FFTW doesn't implement for r2r).
First-principles ceiling is still several-× over rustdct's 2.26 ns/elem,
but it is a from-scratch, week-scale kernel with no library shortcut —
**deprioritised below R4 async overlap** (exact, hides the ~31% mask
behind the GPU matmul) and the security-gated FFN-intermediate unmask
elimination.

## 19. Correction — R4 async overlap is dead on dGPU too; concurrency re-evaluated

This chronicle (§6, §8, §14, §17) repeatedly ranked "R4 async overlap"
as a top lever on the theory that *"on PCIe the CPU-mask/GPU-matmul
overlap that died on UMA reappears."* **That framing misattributes the
documented failure.** The roadmap §4.D record (tested 2026-05-26,
`feat/r4-async-overlap`):

- The green-light spike measured 58% overlap on a mask buffer
  **unrelated** to the in-flight matmul — the dependency removed.
- The full implementation measured **flat to +2.8% regression**
  (`r4_async_minibench`: 1808.7→1802.7 ms noise; 1698.0→1745.6 ms
  regression), with **~+2 ms/dispatch** of token/submit bookkeeping.
- Root cause: a **protocol-inherent serial chain** — `apply M_{i+1}`
  consumes the *unapplied output* of matmul `M_i`, within layers and
  across them. Nothing to hide behind the in-flight matmul.

PCIe changes neither the data dependency nor the dispatch overhead, so
the "dGPU revival" hope is unfounded. **R4-as-designed is struck from
the lever list.** Its overhead number also sets the bar for any future
async: at 252 offloads/prefill and 1152+/decode, per-call overlap
machinery must cost ~µs or it loses outright (the same granularity
lesson as the §15 FWHT fork-join cliff).

**What true-independence concurrency remains (post-§17 profile):**

1. **Decode prefix-GPU ∥ tail-CPU** — independent given `q` (disjoint
   K/V, online-merged): overlap ceiling ≈ min(`prefix_partial_gpu`
   ~437 ms, tail ~150 ms) ≈ **−4–5% decode**, only with a ~zero-cost
   submit/compute/read restructure.
2. **Covered-prefix build** (600 ms, per-layer independent): parallelise
   the CPU permute/rotate across layers ≈ **−2–3% prefill**.
3. **`shield_stack` probe first** — 8.2 ms/call vs a ~3–5 ms
   copy+norm+RNG back-of-envelope suggests a plain inefficiency; fix
   serially before considering noise-pregeneration overlap.

Conclusion: scheduling is nearly tapped out by the protocol's serial
structure — the remaining large prefill levers are **FLOP reduction**
(FFN-intermediate unmask elimination, security-gated; custom
lane-parallel DCT-IV), not concurrency. §18's "deprioritised below R4"
ranking is superseded accordingly.
