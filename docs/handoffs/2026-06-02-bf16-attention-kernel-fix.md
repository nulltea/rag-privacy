---
type: handoff
status: stale
created: 2026-06-02
updated: 2026-06-02
tags: [gelo, dgpu, attention, gpu-offload, cubek, fp16, bf16, accuracy, humaneval, numerical]
companion: [perm-attn-gpu-offload, 2026-06-02-attn-offload-cv-cover-gate]
superseded_by: 2026-06-02-offload-attention-collapse-sm120-rootcause
archive_reason: >
  Diagnosis FALSIFIED (2026-06-02). The 0/20 collapse is NOT an f16-storage
  overflow and BF16 is NOT the fix. Real cause: cubek's BlackboxAccelerated kernel
  is broken on cubecl-wgpu/Vulkan (NaN ∀ n≥3), and the original run had
  CUBEK_STRATEGY=blackbox leaked into the env. cubek's softmax/accumulator are F32
  (max-subtracted) so no overflow exists; bf16 is 8× worse. See the superseding
  handoff for the root cause, the CUDA/sm_120 finding, and next steps.
---

# Handoff — offload attention fp16 NaN bug; fix = BF16 cubek kernel (in progress)

> **⚠ SUPERSEDED / FALSIFIED (2026-06-02).** Everything below — the "f16-storage
> overflow" diagnosis and the BF16-kernel fix — is **wrong**. Read
> [`2026-06-02-offload-attention-collapse-sm120-rootcause`](2026-06-02-offload-attention-collapse-sm120-rootcause.md)
> instead. Kept only for the reasoning trail.

**Focus for the next session: implement a BF16 cubek attention path** to fix a
pre-existing fp16-storage overflow that makes the offloaded prefill attention
produce NaN → degenerate output on real Qwen3-4B activations. Diagnosis is
closed; the fix is specified; the production GQA path is the blocker.

**Background (do NOT re-derive — read these):**
- Dev-log `docs/dev/logs/perm-attn-gpu-offload.md` — the offload design, the
  `C_v` security gates, the 3-way perf table, the acceptance tiers.
- Handoff `docs/handoffs/2026-06-02-attn-offload-cv-cover-gate.md` — the `C_v`
  value-cover (security PASS, perf-free) + the HumanEval gate definition.
- Commits on `dgpu-nvidia-bringup`: `8b4b7ee` `5e2464b` `1fa2464` `89dca21`
  (`C_v` re-scope, Stage-1/2 gates, wire-up + diagnosis, perf table + gate scaffold).

## What's new this session (not yet in the above): the accuracy gate FAILED

Ran the HumanEval gate (`evals/humaneval-gate/`, scaffold committed in `89dca21`)
on the secure offload (κ=6): **0/20**, because the model emitted degenerate
`!!!!` on every prompt. A `/diagnose` pass localised it precisely:

- **Root cause: cubek's F16 *storage* dtype overflows on real activations.** The
  offloaded prefill attention generates NaN/Inf from *finite* inputs at layer 0
  (probe: `ctx_nonfinite=16640`, `ctx_max=1.86e4`); the residual stream is then
  NaN for all 36 layers → first token `!` → greedy loop. cubek's intermediate
  score/prob/value tiles are stored in the global dtype (F16, max 65 504); real
  Qwen3-4B activations (qk-norm×γ Q/K, **un-normalized large V**) exceed it.
- **NOT accumulation** — cubek's `accumulator_precision` already defaults to
  `Strict(F32)` (`lib.rs:110`). The reductions are f32-stable; only *storage* range fails.
- **NOT `C_v`** — fails identically at κ=1 (orthogonal) and κ=6. `C_v` fully exonerated.
- **NOT batch** — the kernel is called *per-sequence* (never sees batch size);
  the earlier "B=1 bug" framing was wrong (B=2 happened to use a non-overflowing prompt).
- **NOT n / not random magnitude** — `cubek_gqa_nan_sweep` (random unit data, all
  n, amp up to 16) is clean; only real activation *structure* trips it. Prompt-dependent.
- **This subsumes the earlier "degenerate collapse, non-monotonic in κ,
  non-deterministic" saga** — that was this same fp16 overflow, not cliff/autotune/cond.

in-TEE (f32) is immune → it produces coherent code on the same prompts.

## The fix: BF16 storage (cubek 0.1.1 supports it natively)

`cubecl` has `FloatKind::BF16` (`half::bf16`); bf16's fp32-range exponent removes
the overflow, matches the bf16-native model, and composes with the existing F32
accumulator. **No version bump, no burn-cube, no hand-rolled kernel.** Two entry
points in `crates/gelo-gpu-wgpu/src/lib.rs`:
- `cubek_attention_folded` (~1184; non-production): raw-bytes path
  (`to_f16`/`as_bytes`/`read_one` reinterpret) → **trivial** `f16`→`bf16` swap
  (`half::bf16` has the same `HalfFloatSliceExt`, also `repr(transparent)` u16).
- **`cubek_attention_folded_gqa` (~1398; PRODUCTION — the blocker)**: builds
  operands as **burn tensors** via `array3_to_tensor_f16` + `repeat_dim` (on-device
  GQA broadcast), then bridges to cubek. The burn backend is **f16-typed**
  (`type CubeWgpu16 = CubeBackend<Rt, f16, i32, u8>`, `lib.rs:70`), so bf16 is
  *not* a constant flip. Needs **(a)** a bf16 burn backend + `array3_to_tensor_bf16`
  (keeps O2's un-replicated upload — recommended), or **(b)** rewrite to
  raw-bf16-bytes + manual GQA expand (reintroduces the materialised expand O2 removed).

The flip is `let f16_dtype = …FloatKind::F16` → `BF16`, the host convert
`half::f16`→`half::bf16`, and the readback `&[f16]`→`&[bf16]`. Leave
`accumulator_precision: default()` (F32).

## Feedback loops already built (reuse — don't rebuild)

- **Deterministic repro / verify loop** (fast, model-loaded ~3min): the b=1/n=23
  prose case reproduces the NaN. `humaneval_gate_generate` test
  (`crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`) with
  `GELO_HE_SUBSET=/tmp/cv-bench/probe_prose.jsonl GELO_HE_MAXTOK=12
  GELO_GPU_PREFILL_OFFLOAD=1 GELO_GPU_RESIDENT_COVER=0 GELO_COVER_KAPPA=1
  GELO_DEBUG_OFFLOAD=1` → fix is good when L0 `ctx_nonfinite=0` and the completion
  is coherent (not `!!!!`). (`probe_prose.jsonl` is one prose line; regenerate via the
  `python3 -c` in the transcript if `/tmp` was cleared.)
- **Kernel NaN sweep** `cubek_gqa_nan_sweep` (`cubek_prefill_cover.rs`) — clean on
  random data (NaN is data-dependent), so it is NOT the regression seam; the real
  seam is the prose probe / HumanEval gate.
- **Full accuracy gate**: `evals/humaneval-gate/README.md` (prepare_subset.py →
  humaneval_gate_generate → score.py vs **6/20** Plain reference).

## Verification plan for the fix
1. Prose probe → `ctx_nonfinite=0`, coherent token.
2. HumanEval gate (B=1, secure κ=6) → pass@1 vs 6/20; if regressed, ablate κ=1
   (offload − defense) per the README. **Coherence-pre-gate first** (`cover_greedy_parity`).

## Cleanup / corrections owed (Phase 6)
- **Remove** the `[DEBUG-off]` instrumentation in `forward.rs` (tagged; grep `DEBUG-off`).
- I already **reverted** the wrong Q-prescale in `cubek_causal_attend` (cubek
  *ignores* the passed `scale` — derives `1/√d_head` internally; do NOT pre-scale).
- **Correct the docs** — dev-log/handoff currently imply the offload is faithful.
  It is NOT: record the fp16-storage-overflow bug, the bf16 fix, that **default-on
  is blocked on this fix (not just security)**, and that `C_v` is exonerated. Keep
  the paper dev-log to conclusions; debugging detail stays in handoffs (memory
  `feedback-no-debugging-in-paper-devlog`).

## Gotchas
- `--release` mandatory (cubecl shader compile). cubek autotune is cross-process
  non-deterministic — use the deterministic same-process tests, not free-running token match.
- Accuracy gate is **B=1** (matches the llama.cpp `-np 1` 6/20 reference); perf bench is B=8.
- Host is Rust-only; `datasets` for `prepare_subset.py` is absent in the gelo-attack
  container — `prepare_subset.py` has a stdlib urllib/gzip fallback (works on host);
  `subset.jsonl` is committed-pinned to the same 20 task_ids as the reference.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit only when asked.

## Suggested skills
- **`diagnose`** — continue the fix loop (implement bf16, run the prose probe);
  the loop is built, the hypothesis is confirmed, this is the fix+regression phase.
- **`code-review`** — before committing the bf16 kernel change.
- **`verify`** — drive the HumanEval gate end-to-end once bf16 lands.
