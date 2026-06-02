---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-02
tags: [gelo, dgpu, attention, gpu-offload, prefill, decode, perf, covariant-obfuscation, cv-cover, weights-pub]
companion: [perm-attn-gpu-offload]
supersedes: [2026-06-01-attn-offload-phase5a-aloepri-gate, 2026-06-01-feature-rotated-prefill-offload-weights-pub-break, 2026-06-01-prefill-offload-o1-o2-decode-next]
---

# Handoff — GPU attention offload: perf landed (prefill 2.83×, decode ~4×); next gate = the C_v cover (re-scoped from AloePri weights)

**One-liner.** Both GPU attention offload paths are now **wired into the
production forward path, perf-optimised, parity-verified, and committed** — but
**default-OFF**, because the covers leak token membership under `WEIGHTS-PUB`.
The remaining work splits into **more perf** (a few levers, optional) and **the
one thing that ships it: neutralising `WEIGHTS-PUB`**. That gate has been
**re-scoped (2026-06-02)** after a design grill: *not* static AloePri weight
obfuscation, but a **per-session non-orthogonal value cover `C_v`** applied in
GELO's existing activation-space cover — the minimal covariant-obfuscation
addition. Next concrete step: the **two-tier `C_v` gate** (cheap κ-sweep screen
→ full confirm).

**Design source of truth: `docs/dev/logs/perm-attn-gpu-offload.md`** — read
*Adapting covariant obfuscation to the offload* (the junction + the resolved
lever), then *Offload perf-upside*, *Gate-3 @ `WEIGHTS-PUB`*, and *Sequencing*.
This handoff is the current-state + next-step index; it does not re-derive the
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
stay default-off until `WEIGHTS-PUB` is neutralised — see the re-scoped lever
below.**

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

## The default-on gate — RE-SCOPED 2026-06-02: the C_v cover, not AloePri weights

A design grill resolved *which* covariant obfuscation neutralises `WEIGHTS-PUB`.
The full derivation (spectrum, 8 constraints, two findings, options A–H) is in
the dev-log section *Adapting covariant obfuscation to the offload*; the verdict:

**The lever is a per-session, κ-bounded, non-orthogonal value cover `C_v`** —
AloePri's value covariant pair (`Ũ_vo`/`Ũ_vo⁻¹`) **relocated from static
weight-space into GELO's existing dynamic activation-space cover.** Generalise
the value cover from orthogonal `O_v` to invertible `C_v = U·diag(s)·Vᵀ` (Haar
`U,V`, `s ∈ [1/√κ,√κ]`); correct with `C_v⁻¹` where `O_vᵀ` is applied today.
Non-orthogonal ⇒ distorts the per-head norm **and** Gram the dictionary reads;
per-session-fresh ⇒ accumulation-safe.

**Why *not* static AloePri weight obfuscation (the two findings):**
1. **Resident-clear weights** — GELO holds the offloaded projection weights
   resident *in the clear* (`gelo-gpu-wgpu/src/lib.rs::register_weight`; the
   matmul cover masks only the activation, `mask.rs`). So a bare covariant weight
   transform `W̃_v = W_V·Ũ_vo` is **trivially inverted**: `W_V⁺W̃_v = Ũ_vo`. A
   VMA-safe resident `W'` would need ~whole-model AloePri (cross-layer key
   matrices) → forfeits accuracy (constraint b) + adds a deployment artifact
   (constraint a).
2. **Orthogonal refresh doesn't freshen the attack surface** — the Gram is
   `V(C Cᵀ)Vᵀ`; an orthogonal factor cancels (`C Cᵀ` invariant), so a *static*
   non-orthogonal cover is accumulation-broken and freshness must come from
   refreshing the *non-orthogonal* part. (Corrects a mid-grill claim that the
   per-session `O_v` carried the accumulation defence — it cannot; `O_v` is
   orthogonal.)

**Forced construction details** (minimal swap of `O_v`→`C_v`):
- Cadence **inherits `O_v`** — prefill per-prefill; decode session-fixed, built
  at the prefill→decode handoff (O5). Per-layer-independent `C_v` (GLOBAL
  layers). Single global κ band, swept at the gate. New sampler
  `sample_kappa_bounded_invertible` alongside `sample_orthogonal`.
- **f32 correction fallback** if the chosen κ's conditioning bites greedy parity
  (run `C_v⁻¹` in `correct_unfold_into`/`acc_uncover_tee` at f32). Measured
  contingency, not a fork.
- QK-side `Ĥ_qk` (post-qk-norm scaling to break the K/Q self-Gram) is a
  **conditional second addition**, deferred until that anchor is measured to
  bite (qk_norm already flattens the cheap K-*norm* signal).

**The gate — two-tier (no production wiring until it passes):**
- *Stage 1 — cheap screen.* Add a `C_v` capture mode to `attn_cover_capture.rs`
  (κ-bounded SVD value cover; `O_qk` orthogonal as today). Reuse the existing
  harness (layer 0, 8091 pool, **norm** attack) and **sweep κ** for the minimum
  that drops top-1 toward chance while greedy parity holds. Go/no-go.
- *Stage 2 — confirm (only if Stage 1 passes).* Full 152k vocab + ≥1 deep
  contextual layer + **all three** attacks at that κ: norm-dictionary,
  Gram-dictionary (quadratic assignment), and covariance/Procrustes alignment
  (the real test of whether `C_v` is recoverable — analog of the gate-3
  covariance result that showed orthogonal `O_v` holds).
- **Pass ⇒** flip both offload paths default-on behind the c5 condition (now read
  as "`C_v` cover," not "AloePri weights"). **Fail at every parity-holding κ ⇒**
  escalate to the QK-side `Ĥ_qk`, re-gate; if that also fails, prefill stays
  in-TEE and decode ships only if the bag-of-tokens residual is accepted.
- *Follow-up (not v1 gate):* a multi-session capture (fresh `C_v` each)
  confirming no cross-session recovery — accumulation is defeated by construction
  (finding 2), this would measure it.

## Gate status + tier-3 diagnosis (2026-06-02) — `C_v` ships at κ=6

**Stage 1 (norm dictionary) — PASS.** `cv` capture mode in `attn_cover_capture.rs`
(κ-bounded SVD value cover). Real Qwen3-4B layer 0, `gate3_prefill_dict.py` norm
attack: κ=1 (orthogonal control) top-1=1.000 (reproduces the `O_v` leak), falling
monotonically to top-1≤0.02 at **κ≈5**, =0.000 at κ=6.

**Stage 2 (conservative bar) — PASS.** Full-vocab (151 936): κ=6 top-1=0.000,
top-5=0.047 (the 8k-pool top-5 plateau was a collision artifact). Covariance/
Procrustes recoverability (`gate3_cv_covalign.py`, layers 0/35): covalign(pub)
0.332 ≈ no-attack 0.370, self\*=0.359 — `C_v` holds *more* strongly than `O_v`
(non-orthogonal ⇒ not recoverable from 2nd moments; κ=1 control self\*=1.000
validates the attack).

**Wire-up.** `O_v`→`C_v` at the cover sites (prefill `rotate_heads` +
`correct_unfold_into`; decode `build_covered_prefix_session` + `acc_uncover_tee`)
behind `GELO_COVER_KAPPA` (default 1.0 = orthogonal `O_v`, byte-identical rng
draw). `O_qk` stays orthogonal. `sample_value_cover` uses **log-uniform** `s`
(det≈1; the first linear-uniform draw had a benign det-drift — cosmetic).

**Tier-3 (fp16 parity) — diagnosed; `C_v` exonerated.** Free-running greedy
generation through the offload collapsed into degenerate loops (`99491…`),
**non-monotonically in κ** (κ=3 collapsed, κ=5/6 coherent) and **non-deterministic
across processes** — initially mistaken for "κ=6 fails." A `/diagnose` pass with a
**deterministic, same-process** loop (`cv_cover_fp16_drift_sweep` in
`cubek_prefill_cover.rs` — real cubek fp16, autotune cached) measured the cover
round-trip rel-error:

| κ | cover-vs-uncov | cover-vs-CPU |
|--:|--:|--:|
| 1 | 6.4e-4 | 5.2e-4 |
| 6 | 1.2e-3 | 1.2e-3 |
| 16 | 1.6e-3 | 1.5e-3 |

(fp16 floor: cubek-vs-CPU 4.4e-4.) **Sub-linear in κ, fp16-floor-level — `C_v`
adds no κ-dependent accuracy cost.** A CPU/numpy isolation
(`analysis_cv_fp16_error.py`) agreed (2e-4→4e-4 over κ=1→8).

**Root cause of the collapse — not a code bug, not the cover (CONFIRMED 2026-06-02):**
1. The offload is **fp16-on-GPU, never bit-exact to f32 in-TEE** (~1e-3 deviation
   present at **κ=1** / the pre-existing orthogonal cover).
2. The stress prompt (n=64, repetitive) has near-tied next-token logits.
3. Free-running greedy amplifies the ~1e-3 deviation autoregressively → degenerate
   attractor on the cliff prompt; which side it lands on is set by cubek
   **autotune** (per-process kernel choice), not κ → non-monotonic, non-deterministic.

**Confirming runs (`cover_greedy_parity`, B=2, σ=0):**
- **Realistic prompt (n=24): in-TEE == κ=1 == κ=6, byte-for-byte identical.** The
  offload at κ=6 reproduces the in-TEE greedy tokens exactly → no accuracy issue
  on normal input; κ is not the problem.
- **Stress prompt (n=64): in-TEE is DETERMINISTIC across processes** (run A == run
  B). ⇒ the nondeterminism is **specific to the cubek offload** (fp16 + autotune),
  *not* a general GPU property (corrects an earlier wrong prediction that in-TEE
  would also vary). It only *manifests* as token divergence on margin-adjacent
  prompts; realistic prompts are bit-identical.

**Architectural finding.** Bit-identical greedy parity on **all** prompts is not
an achievable invariant for an fp16-GPU attention offload — it holds where the
logit margin exceeds the ~1e-3 floor (realistic prompts: confirmed identical;
pathological cliff prompts: can flip). The false alarm was the **acceptance
instrument** (bit-exact free-running token match on a cliff prompt across
non-deterministic offload processes), not the cover. Pre-existing, independent of
`C_v`. Optional hardening for the pathological-input edge (separable from
security): **pin cubek autotune** (determinism) + **f32 accumulation in the value
contraction** (tighten the ~1e-3).

**Fixes applied:** (1) faithfulness gate = the deterministic
`cv_cover_fp16_drift_sweep` (asserts cover rel-error < 5e-3 for κ≤8); (2)
log-uniform `s`; (3) `cover_greedy_parity` demoted to an informational coherence
smoke-check (doc updated), never a gate.

**Open / separate (NOT from this work):** the offload's *intrinsic* ~1e-3 fp16
divergence + cubek autotune nondeterminism can flip greedy tokens **only on
margin-adjacent (pathological/repetitive) prompts** — realistic prompts are
bit-identical (confirmed above). To eliminate the edge case, the levers are
**pin cubek autotune** (determinism) and **f32 accumulation in the value
contraction** (tighten the ~1e-3) — a distinct, non-security task. **Residual
check:** realistic parity was confirmed at n=24; re-confirm at production length
(n=2048) when convenient (expected fine — real-text logit margins are wide).

## Final accuracy gate (HumanEval) — defined; to run after full default-on wiring

The acceptance methodology (dev-log *Acceptance gate (v1)*, tier 5): a HumanEval
pass@1 **ablation** — A=in-TEE (baseline), B=offloaded+`C_v` (accept iff
`B ≥ A−ε`), C=offloaded−cover (only if B regresses, to attribute offload vs
cover). Always **coherence-pre-gated** (1–3 prompts; `cover_greedy_parity`
covers this on a realistic prompt) and used **sparingly** — debug any regression
with microbenches + theory (`cv_cover_fp16_drift_sweep`, `analysis_cv_fp16_error`),
not by re-running HumanEval.

**Harness requirement (not yet built).** `python/aloepri-llm/evals/tasks/humaneval.py`
(50-problem subset, pass@1, subprocess-scored) drives an inference **endpoint**
via `evals/runner.complete(Endpoint, …)` — it does **not** drive the Rust GELO
offload. To run the gate on the Rust path, either (a) expose Rust GELO
`generate_batched` (with the offload flags) as an OpenAI-style endpoint the
existing runner can hit, or (b) add a thin Rust HumanEval runner. Gate runs
**after** the offload is wired default-on, not before.

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
- **Where `C_v` lands** (the re-scoped lever): the cover sampler (`sample_orthogonal`
  → add `sample_kappa_bounded_invertible`) and the value-cover application/inversion
  sites — prefill `rotate_heads(v_f, O_v)` + `correct_unfold_into(…, O_vᵀ)`; decode
  `rotate_heads(vp, O_v)` in `build_covered_prefix_session` + `acc_uncover_tee`
  (`rotate_heads(acc_a_cov, O_v.t())`). `O_qk` stays orthogonal (cancels in score);
  only the V cover becomes invertible.
- Tests: `cover_prefill_matches_in_tee`, `cover_prefill_batched_matches_in_tee`
  (`forward.rs` lib tests); `cubek_dispatch_granularity`,
  `cubek_folded_causal_parity` (`tests/cubek_prefill_cover.rs`). (The HD₃
  round-trip test was removed with the reverted HD₃ cover.)

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
- Host box is **Rust-only** (no pip/numpy/sudo); Python `WEIGHTS-PUB` attacks
  (`gate3_prefill_dict.py` + the new Gram / covariance-alignment variants) run in
  the `gelo-attack` Podman container.
- Commit identity `Timo <timofey.luin@gmail.com>`; trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit
  only when asked. Branch `dgpu-nvidia-bringup` (not master).

## Suggested skills

- **`grill-me`** — the `C_v` lever + two-tier gate are *resolved* (this handoff);
  reach for it on the remaining open forks: the covariance-Procrustes attack
  design, the c5 default-on wiring, or the cubek read-index / O6 kernel scope. The
  user prefers being grilled on design forks, mechanism-in-prose before options
  (memory `feedback-elaborate-before-options`).
- **`code-review`** — before committing the `C_v` sampler + capture-harness mode
  or the O6 kernel.
- **`diagnose`** — only if a `C_v` gate re-run gives a surprising pass/fail and
  you need a tight loop.
