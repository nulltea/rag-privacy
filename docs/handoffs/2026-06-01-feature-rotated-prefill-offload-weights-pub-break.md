---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-01
tags: [gelo, dgpu, attention, gpu, prefill, feature-rotation, threat-model, aloepri]
companion: [perm-attn-gpu-offload]
---

# Handoff — offload covers break under public weights; both paths now wired + benched; next is perf-opt then AloePri

**One-liner.** The permutation cover can't defend *offloaded prefill* (causal-mask
leak), so prefill pivoted to a **feature-rotation cover**; the Phase-5
`WEIGHTS-PUB` spike then showed *both* covers leak token identity (prefill rotation:
token + position; decode permutation: membership). Both offload paths are now
**wired with their cover applied end-to-end and benched** (see *Perf-upside
measured*): the GPU compute is fast, but a **convert/upload/dense-rotate
bottleneck** keeps them from cleanly beating in-TEE at production shape. Two open
threads for the next session, **in order**: **Phase 5a** optimize that bottleneck
on both paths (pure perf), then **Phase 5b** verify + implement **covariant weight
obfuscation (AloePri)** to make the rotation cover secure.

Everything design-level lives in **`docs/dev/logs/perm-attn-gpu-offload.md`** — read it
first. This handoff only captures the load-bearing math, the current artifact
state, and what to do next. Do not re-derive the plan here.

## Background and the pivot (why feature rotation at all)

- The lever: offload the in-TEE attention to the untrusted GPU over **persistent
  K/V**. Decode wire-up is done and benched (chronicle `…_dgpu.md` §11; commit
  `e074a32`): attention bucket 0.60×, the wall delta confounded — read that section
  rather than trusting a single wall number.
- **Prefill is the prize** (the in-TEE `tee:attn_inplace_many` ≈ 43.7 s bucket), and
  it needs a *fused* kernel (`cubek-attention`, already a dep) so the n×n scores
  never materialise.
- **Permutation cover is incompatible with offloaded *causal prefill*:** the n×n
  causal mask over a permuted buffer leaks π exactly (row-sum = rank). Decode is
  exempt (n_q=1, trivial mask). See plan §"Offloaded-prefill attention: the attack
  vector".
- **Additive (TwinShield) is incompatible with a fused kernel** (verified vs
  arXiv 2507.03278: it does TEE-side softmax + score round-trip). So the *only*
  cover that survives a fused, normalized-output softmax is **orthogonal feature
  rotation** (`O_qk` on Q/K, `O_v` on V). Hence the pivot.
- Architecture decided: **prefill offload = feature rotation; decode = permutation
  (+ tail-in-TEE)**; the TEE builds the permuted resident cache at the handoff.

## The failed gate — exact mathematics

Threat model is standardized in the plan (`TEE-TRUST`, `GPU-ADV`, **`WEIGHTS-PUB`**
= adversary knows public Qwen3 weights/embeddings — the conservative default,
`WEIGHTS-BLIND` its negation, `NO-PLAINTEXT`). The break is intrinsic to
correctable rotation:

The adversary holds `v_sent = V·O_v`. `O_v` is **orthogonal** — which is *required*
for correctability (the TEE undoes it via `·O_vᵀ`, since `O_v·O_vᵀ = I`). That same
identity makes every inner product / norm an `O_v`-invariant:

```
‖ v_sent[p,h] ‖² = (V[p,h]·O_v[h])(V[p,h]·O_v[h])ᵀ
                 = V[p,h]·(O_v[h] O_v[h]ᵀ)·V[p,h]ᵀ
                 = V[p,h]·V[p,h]ᵀ = ‖ V[p,h] ‖²        (O_v O_vᵀ = I)
```

and likewise the Gram `v_sent·v_sentᵀ = V·O_v·O_vᵀ·Vᵀ = V·Vᵀ`. (Same on Q/K:
`(Q·O_qk)(K·O_qk)ᵀ = QKᵀ` — the cancellation that keeps the fused softmax correct
hands the adversary the true scores.)

`WEIGHTS-PUB` then turns the invariant into a **token oracle** via the value
projection. At layer 0, `V` is context-free:

```
V = X·W_V ,  X = rms_norm(embed(t))   ⇒   V(t) = rms_norm(embed(t))·W_V
```

so `‖V(t)[h]‖` is a *known* per-token, per-head scalar. The observed fingerprint of
position `p` equals dictionary entry `V(token_p)` → match → recover `token_p`.

**Measured (2026-06-01, `gate3_prefill_dict.py` in the `gelo-attack` container):**

| run | positions | dict K | dict faithfulness | top-1 | median rank |
|---|--:|--:|--:|--:|--:|
| short prompt | 26 | 1,224 | rel-err 0.00 | **1.000** | 0 |
| longer prompt | 126 | 8,091 | rel-err 0.00 | **1.000** | 0 |

Perfect token recovery from per-head **value norms alone** (the cheapest
`O_v`-invariant → this is a *lower bound*; full Gram is stronger). Faithfulness
rel-err 0 confirms the dictionary is the model's true layer-0 value map, so the
result is valid. Layer-0 break = the prompt is exposed.

**Bracketing result (decode, `gate3_weights_anchor.py`):** the *covariance-alignment*
form of the weights attack HELD (`O_v` corr ≈ no-attack; population covariance is
instance-specific so it can't pin per-session `O_v`; self-anchor=1.000 validated the
method). So the spike lands at **pass-`WEIGHTS-BLIND` / fail-`WEIGHTS-PUB`**, and
the decisive `WEIGHTS-PUB` failure is the token-dictionary attack above, not
covariance.

## Current artifact state

**The design log is the source of truth:** `docs/dev/logs/perm-attn-gpu-offload.md`
(promoted from a plan; standardized threat model, attack vector + causal-mask leak,
cubek/cover-split, both `WEIGHTS-PUB` gate-3 results, the *Offload perf-upside —
per-op breakdowns* tables, sequencing with **Phase 6 ⛔ BLOCKED** behind Phase 5a
(perf) → Phase 5b (AloePri)).

Committed:
- `6f5b31c` — threat-model standardization + cover-split + `gate3_weights_anchor.py`.
- `9935d34` — Phase-5 spike (rotation cover capture mode, `gate3_prefill_dict.py`, prefill-break + decode-membership results).
- `b33409f` — promote plan → dev-log (moved to `docs/dev/logs/`) + paper-review fixes.

**UNCOMMITTED (this session — the two offload wires + the perf section):**
- *Permuted-decode secure wire* (greedy-parity byte-identical at σ=0): `forward.rs`
  (cover branch + `gpu_resident_cover_enabled`/`rotate_heads`/`stack_tail_expanded`/
  `sample_orthogonal`), `kv_cache.rs` (`DecodeCover` + accessors),
  `substrate.rs`/`sim.rs` (`resident_kv_attend_partial`), `gelo-protocol/lib.rs`
  (re-export `attention_partial`/`merge_attention_partials`), `gelo-embedder/Cargo.toml`
  (`rand_distr`). Flags: `GELO_GPU_RESIDENT_COVER`, `GELO_RESIDENT_SIGMA`.
- *Prefill cubek bench*: `gelo-gpu-wgpu/src/lib.rs` (`cubek_attention_folded` +
  `CUBEK_STRATEGY` env: `unit`|`blackbox`), `tests/cubek_prefill_cover.rs`.
- *Dev-log* perf-upside per-op breakdowns section.
- Capture/bench data (`evals/aloepri-attacks/captures_*`) is **gitignored**.

## Reproduce

```bash
# --- Phase-5 WEIGHTS-PUB spike (the break) ---
# rotation-only prefill capture + layer-0 dictionary (CPU, ~40s; GPU not needed)
GELO_CAPTURE_COVER=rotation GELO_CAPTURE_DICT=1 GELO_CAPTURE_DICT_N=8000 \
GELO_CAPTURE_LAYERS=0 GELO_CAPTURE_PROMPT="<any sensitive-looking text>" \
GELO_CAPTURE_DIR="$PWD/evals/aloepri-attacks/captures_prefill" \
  cargo test -p gelo-embedder --test attn_cover_capture --release \
  capture_attn_cover_adversary_view -- --ignored --nocapture
evals/aloepri-attacks/run-in-container.sh \
  python3 evals/aloepri-attacks/gate3_prefill_dict.py   # numpy → in-container

# --- Perf-upside benches (re-run after Phase 5a optimizations) ---
# Prefill: in-TEE vs cubek+cover, per-op breakdown (n=2048 & 8192), tensor cores
CUBEK_STRATEGY=blackbox cargo test --release -p gelo-gpu-wgpu \
  --test cubek_prefill_cover prefill_attention_breakdown -- --ignored --nocapture
# Decode: permuted-cover secure, end-to-end per-op breakdown (σ=0.01)
GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 \
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=32 \
  cargo test --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench \
  gelo_llm_prefill_decode_breakdown -- --ignored --nocapture   # cover:* buckets
# Decode greedy-parity (σ=0 must be byte-identical to flag-off):
GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0 GELO_BENCH_VARIANT=4b \
GELO_BENCH_MAX_TOKENS=16 cargo test --release -p gelo-gpu-wgpu \
  --test qwen3_m1_12_r1_q1_microbench m1_12_r1_q1_microbench -- --ignored --nocapture
```

## Perf-upside measured — what the offload buys, and the bottlenecks (2026-06-01)

Both paths are now wired with their cover applied end-to-end and benched. Full
per-op tables **with execution counts** are in the dev-log §*Offload perf-upside
— per-op breakdowns (2026-06-01)*; headlines (B=8, Qwen3-4B, RTX 5090 / Vulkan):

- **Decode — permuted-cover, secure** (perm + σ + `O_qk` + `O_v` + tail-in-TEE,
  session-fixed; `GELO_GPU_RESIDENT_COVER`, greedy-parity byte-identical at σ=0):
  attention bucket **13.8 s** vs in-TEE 14.6 s (bare-resident 8.7 s) at n=2048,
  K=32. **Per step it is ~2.9× faster than in-TEE** (157 vs 455 ms/step over 36
  layers); the one-time prefix re-cover `build_covered_prefix+upload` (8.8 s = 36 layers ×
  243 ms — dense-`O` rotate + permute + upload of the 2048-row prefix) is **63%
  of the K=32 bucket**, so it is break-even at **K≈30** and a growing win beyond
  (≈2.4× at K=256, →2.9×).
- **Prefill — feature-rotation + `cubek` (tensor cores)** (per layer): n=2048
  **1.09×**, n=8192 **4.70×**. The cover is cheap (**10–14%**); cubek's attend at
  n=2048 is ~680 ms fixed overhead (scalar f32→f16 convert + 4× GQA-expanded
  upload + sync) + only ~204 ms compute → **compute-only ceiling ~5×**.

### Bottlenecks that prevent beating full (in-TEE) attention

The GPU attention **compute is not the bottleneck** (tensor cores; ~5× headroom).
Both paths are gated by the **f32→f16 convert + upload / dense-rotate pipeline** —
the same "upload tax" the original triage flagged, not the attention math:

- **Prefill:** per-call **f32→f16 convert + 4× GQA-expanded K/V upload** (cubek
  has no native GQA), a ~680 ms fixed cost that only amortizes at long context →
  marginal at n=2048, strong at n≥8k.
- **Decode:** the **one-time dense-`O` prefix re-cover + upload** (`build_covered_prefix`,
  ~243 ms/layer), dominating at short K; the per-step path is already a 2.9× win.

Common root: the cover/operand **convert + upload + dense rotation**. Covariant
obfuscation does **not** touch any of these (it is a static weight transform), so
optimizing them is independent of — and should precede — the security work.

## Next steps

### Phase 5a — optimize both offload paths (perf; before security)

The bottlenecks above are convert/upload/rotate, not compute, and are reducible.
This is pure perf — cover and security are unchanged — so it lands and measures
independently, and it sets the real go/no-go for each offload.

- **Prefill:** swap the scalar f32→f16 convert for the SIMD/bf16-native path; add
  a **kv-head-broadcast** read-index to cubek's K/V loader to drop the 4×
  GQA-expanded upload. Target: lift n=2048 from 1.09× toward the ~5× compute
  ceiling. Re-run `crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs`.
- **Decode:** cut the one-time `build_covered_prefix` — use the **structured
  signed-permutation `O(L·d)`** cover instead of the dense rotate (the bulk of
  243 ms/layer), and/or build the cover **at prefill** (overlap), and/or
  bf16/un-replicated upload. Target: break-even K well below 30. Secondary: the
  **fused partial-stats FlashAttention-D kernel** collapses the decode
  partial-stats dispatches (`prefix_partial_gpu`, 2.95 ms/call).

### Phase 5b — verify + implement covariant obfuscation (AloePri)

The general statement (plan): any correctable-through-fused-softmax cover preserves
the bilinear forms `X·M·Xᵀ` that `WEIGHTS-PUB` reads. The **only** escape is to make
`M` (i.e. `W_V`, `W_Q W_Kᵀ`) **unknown** to the attacker — i.e. **covariant weight
obfuscation (AloePri)**: statically transform the deployed weights `W → W'` with a
covariant compensation so the computation is unchanged but the attacker's *public*
`W` no longer matches the *deployed* `W'`, collapsing the per-token dictionary.

Concrete tasks:
1. **Read the AloePri paper / the `evals/aloepri-attacks/` material** for the exact
   covariant-obfuscation construction and what guarantees it claims.
2. **Pin down the obfuscation math for the attention path:** does a static `W'`
   actually destroy the `‖V(t)‖` / `V·Vᵀ` dictionary while staying (a) exactly
   correctable in-TEE and (b) compatible with the fused `cubek-attention` kernel?
   Watch for the same trap: if the obfuscation is itself an orthogonal/correctable
   transform, the invariants may survive again.
3. **Re-run `gate3_prefill_dict.py` against an obfuscated-weight capture** (add an
   obfuscation mode to `attn_cover_capture.rs`) — the spike is now cheap and decisive.
4. If it clears: unblock Phase 6 (cubek integration). If not: **prefill stays
   in-TEE** — record and move on (decode offload remains the shipped win).
5. Untested escalations to keep in mind: the attack used layer-0 norms only;
   deeper-layer (contextual `V`) and full-Gram/QAP variants are stronger and unrun.

## Gotchas / environment

- Host box is **Rust-only**: no pip/numpy/sudo/apt. Python attacks run in the
  **`gelo-attack` Podman container** (`run-in-container.sh`); CUDA image,
  numpy/scipy/sklearn/torch present.
- The capture test uses the CPU `ReferenceCpuEngine` by design (exact f32 ground
  truth for a one-time offline capture) — the GPU is not needed. The dictionary is a
  single layer-0 value projection (do **not** reintroduce a full 36-layer forward).
- `--release` is mandatory for anything touching cubecl (debug pays minute-scale
  shader compile).
- Commit identity: `Timo <timofey.luin@gmail.com>`; end messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Commit
  only when asked. A PAT pasted in an earlier session is **compromised — never
  reuse**. `bench-results/*.log` and `captures*/` are gitignored.
- Branch: `dgpu-nvidia-bringup` (not master).

## Suggested skills

- **`grill-me`** — for step 2 (pinning the AloePri obfuscation math + threat-model
  reasoning); the user prefers being grilled on design forks, with the
  mechanism/threat-model explained in prose before structured options.
- **`diagnose`** — only if the obfuscation re-run produces a surprising
  pass/fail and you need a tight feedback loop to find why.
- **`code-review`** — before committing the capture-harness obfuscation mode.
