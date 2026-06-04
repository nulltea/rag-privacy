---
type: handoff
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, dgpu, cuda, perf, optimization, deferred, roadmap, prefill, decode, backlog]
companion: [gelo-llm-perf-roadmap, perm-attn-gpu-offload, gelo-llm-perf-chronicle_dgpu]
superseded_by: 2026-06-03-dgpu-perf-session-next-levers
---

# Handoff — deferred / unimplemented dGPU + CUDA optimizations (catalog + re-evaluation plan)

## Next-session focus

1. **Run the canonical bench** to re-baseline now that the secure GPU-offloaded
   path is default-on (capability-gated, this session — see *Baseline state*).
2. **Re-evaluate** each deferred optimization below against the fresh numbers —
   several were scoped on stale (Vulkan / pre-offload / iGPU) measurements and the
   binding bottleneck may have moved.
3. **Identify new optimizations** we have missed so far (hints in *Where to hunt*).

This doc is the consolidated **catalog** of what's documented-but-not-built; it
references the source docs rather than restating them.

## Baseline state (what's already implemented — don't re-scope these)

Committed this session on `dgpu-nvidia-bringup` (`9f6ec0a`, `58c7567`; not pushed):
secure GPU-offloaded attention is now the **default** inference path
(capability-gated), CUDA is the default backend with a `vulkan` override. Already
landed: persistent K/V + resident decode cover; cubek prefill offload (feature
rotation + per-session `C_v`); SIMD f32→f16 convert (O1); un-replicated K/V +
on-device GQA broadcast (O2); caller-side ragged-`n_q` padding; blackbox-by-default
with a tiny-`n_kv`→Unit guard. Full state:
[`2026-06-03-blackbox-ragged-padding-and-humaneval-gate.md`](2026-06-03-blackbox-ragged-padding-and-humaneval-gate.md),
[`perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md).

## The catalog — deferred / unimplemented, by phase

Status key: **DEFERRED** (scoped, not built) · **OPEN** (question/unverified) ·
**REGRESSED** (tried, reverted) · **OOS** (explicitly out of scope).

### Prefill (dGPU/CUDA)
| # | Lever | Status | Source |
|---|---|---|---|
| P1 | **cubek kv-head read-index** — in-shader GQA broadcast; kills the materialised `repeat_dim` 8→32 expand (~268 MB transient), lets one dispatch beat the per-seq loop, cuts `cubek_gpu`. | DEFERRED | `perm-attn-gpu-offload.md` — Fast-follows #10 + "cubek stays per-sequence" |
| P2 | **Upload-bandwidth lever (O3)** — `queue.write_buffer` staging / pinned DMA; measured ~1.5 GB/s upload is ~8× under PCIe5, likely alloc/submit-bound. | OPEN | `perm-attn-gpu-offload.md` — "Still open" + prep-floor analysis |
| P3 | **Q4 weight quantization on dGPU** — GPU-matmul memory ÷4, ~2–3× throughput (~4 wk). | DEFERRED | roadmap §4.B.2 → `docs/plans/q4-gpu-weights.md` |
| P4 | **burn-cubecl matmul on sm_120 — tensor-core verification** — is Blackwell CMMA actually used or a generic path? + cubecl-wgpu SPIR-V cooperative-matrix to close the ~1.5× gap. | OPEN | `2026-06-02-offload-attention-collapse-sm120-rootcause.md` Open #1; chronicle §8 "Next probes" |

### Decode (dGPU/CUDA)
| # | Lever | Status | Source |
|---|---|---|---|
| D1 | **Fused single-pass partial-stats kernel ("FlashAttention-D" / O6)** — emit `(m,l,acc)` in one dispatch, collapse the ~5-dispatch `attend_session_partial` decode path (the dispatch tax → why decode is 1.67×, not 25×). | DEFERRED | `perm-attn-gpu-offload.md` — Fast-follows #10 + "overhead root cause"; chronicle §9 |
| D2 | **bf16-native resident K/V cache** — resident bf16 (memcpy, no convert) → per-block refresh becomes DMA-only. Gated on whether per-block `perm_kv` refresh is actually required (σ-vs-N gate). | DEFERRED | `perm-attn-gpu-offload.md` — Phase-2 sub-steps |

### Cross-cutting (both phases)
| # | Lever | Status | Source |
|---|---|---|---|
| X1 | **End-to-end bf16 activations (§4.E.3 / 3b–3c)** — halves bytes-on-wire/upload; documented **prerequisite** for the offload to stop wasting GPU headroom. Infra (phases 1–3a) landed; forward-wire (3b/3c) **not done**. | DEFERRED | roadmap §4.E.2/§4.E.3 → `docs/plans/m1-12-bf16-activation-pipeline.md`; chronicle §6 |
| X2 | **Per-call readback/sync batching + R4 async overlap** — the dominant *backend-invariant* cost is the blocking per-matmul readback that serialises dispatches; batched/streamed unmask + R4 async (PCIe revives the CPU-mask/GPU-matmul overlap that died on UMA). R4 substrate exists on `feat/r4-async-overlap`, never cut over for dGPU. | DEFERRED | roadmap §4.D disposition; chronicle §8 "high-value lever" |
| X3 | **NVMe `SpillProvider` (Phase-2 cold tier)** — for `n_kv` > 32 GB VRAM (~16–32k+ ctx); session API has the seam, only `NullSpillProvider`. | DEFERRED | `perm-attn-gpu-offload.md` — Session-resident K/V API + Fast-follows #10 |

### Perf-adjacent / structural
| # | Lever | Status | Source |
|---|---|---|---|
| S1 | **HD₃ batched feature-axis FWHT** — `O(d·log d)` structured-orthogonal cover to replace the dense Haar rotate; needs feature-major transpose + cols-vectorised kernel. Per-`d`-block version regressed (lost to BLAS). | REGRESSED / DEFERRED | `perm-attn-gpu-offload.md` — "Structured-orthogonal cover — tried, reverted" |

### Out of scope (noted, not backlog)
Continuous batching / PagedAttention (dGPU evaluates at M5.9+), confidential GPU
(H100 CC / B200 TEE-I/O), encrypted KV on GPU, speculative decoding. — roadmap §7–§8.

**Dependency chains to respect when re-evaluating:** X1 (bf16-native) is the stated
prerequisite for P2 (upload) to matter; P1 (read-index) is what makes the
single-dispatch prefill win real; D1 (fused kernel) is what converts decode's
dispatch-bound 1.67× toward the bandwidth ceiling.

## Next steps in detail

### 1. Run the canonical bench (re-baseline)
Per `CLAUDE.md` (production shape, CUDA default, `--release` mandatory):
```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=32 \
  cargo test --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench \
  gelo_llm_prefill_decode_breakdown -- --ignored --nocapture
```
- ⚠ **The default now offloads** (capability-gated). At B=8 this exercises the
  offloaded+covered path by default — that *is* the new production baseline. For the
  in-TEE comparison set `GELO_GPU_PREFILL_OFFLOAD=0 GELO_GPU_RESIDENT_COVER=0`.
  (Note the B=1 canonical path historically ran in-TEE; confirm B=8 routes through
  `run_prefill_batched` so the offload engages.)
- Warm first (`GELO_BENCH_WARMUP=1` if available) — CUDA nvrtc autotune is heavy and
  inflates a cold first run 2–4× (chronicle §8). Capture logs to `bench-results/`.

### 2. Re-evaluate the catalog against fresh numbers
For each lever, recompute its share-of-wall on the *current* offloaded baseline and
re-rank — the binding bottleneck has likely moved since these were scoped (most
predate the offload landing and several are Vulkan/iGPU-era). Specifically:
- Is GPU compute (matmul + cubek attend) now the bottleneck, or still the
  convert/upload/readback prep (P2, X1, X2)? The dev-log's "compute is ~7% of the
  path" finding was Vulkan, pre-O1/O2 — re-measure on CUDA with `CUBEK_PROFILE=1`.
- Does blackbox's 3.16× prefill attend still hold at B=8 warmed, and what's the new
  prefill-wall share of attention vs matmul?
- Is decode now dispatch-bound (→ D1) or bandwidth-bound at n=2048/8k?

### 3. Hunt for new / missed optimizations — where to look
- **CUBEK_PROFILE=1 stage split** on the offloaded prefill (convert / upload /
  attend / readback / O_vᵀ) — re-derive which stage dominates now (O1/O2 changed it).
- **Readback/sync coalescing** across the per-layer `cubek_causal_attend` loop and the
  registered-linear unmask round-trips (chronicle §8 names this the high-value lever).
- **Long-context (n=8k/16k)** shapes — the offload's HBM edge widens there; the
  catalog is mostly measured at n=2048.
- **B=8 vs B=1** dispatch amortization and the cubek per-seq-vs-batched crossover
  (the read-index P1 flips it).
- Anything the roadmap §5 per-bucket EV table now mis-ranks given the offload landed.

## Gotchas
- `--release` mandatory; CUDA default (override `--features vulkan`); perf is
  single-sample (±~7% cross-run variance) — warm + repeat before trusting a delta.
- Pre-existing stale GPU bench/spike tests don't compile against the bumped
  burn/cubecl/cubek 0.2.0 API (`q4_kernel_spike`, `parity`, `qwen3_overhead_*`,
  `cubek_attention_spike`) — unrelated; don't let them block `--test`-target builds.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Suggested skills
- **`Explore`** / general research — to re-read the roadmap §4–§5 + chronicle and
  cross-check the catalog against current code, and to surface missed levers.
- **`diagnose`** — for the `CUBEK_PROFILE=1` stage-split re-measurement and isolating
  the current binding bottleneck (reproduce → instrument → attribute).
- **`Plan`** — to turn the re-ranked top lever(s) into an implementation plan once
  the fresh bench identifies the highest-EV next move.
