---
type: handoff
status: current
created: 2026-06-01
updated: 2026-06-01
tags: [gelo, dgpu, attention, gpu, prefill, feature-rotation, threat-model, aloepri]
companion: [perm-attn-gpu-offload]
---

# Handoff — feature-rotated prefill offload breaks under public weights; next is AloePri obfuscation

**One-liner.** The permutation cover can't defend *offloaded prefill* (causal-mask
leak), so the prefill-offload path pivoted to a **feature-rotation cover**. A
Phase-5 security spike then showed feature rotation is **broken under the
public-weights threat model** — perfect token recovery. The open question for the
next session: is feature-rotated prefill offload viable if we add **covariant
weight obfuscation (AloePri)**?

Everything design-level lives in **`docs/plans/perm-attn-gpu-offload.md`** — read it
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

- **Plan** `docs/plans/perm-attn-gpu-offload.md` — fully refactored: standardized
  assumptions, attack vector + causal-mask leak, cubek/cover-split, the two gate-3
  `WEIGHTS-PUB` results, and sequencing with **Phase 6 ⛔ BLOCKED** behind a new
  **Phase 5b (AloePri weight obfuscation)**.
- **Commit `6f5b31c`** = plan (threat-model standardization, cover-split) +
  `gate3_weights_anchor.py` (decode covariance rerun).
- **UNCOMMITTED** (this session's prefill spike): `crates/gelo-embedder/tests/attn_cover_capture.rs`
  (new `GELO_CAPTURE_COVER=rotation` mode + faithful layer-0 value dictionary via a
  single projection), `evals/aloepri-attacks/gate3_prefill_dict.py`, and the plan's
  Phase-5 prefill-break subsection. Commit these first.
- Capture data `evals/aloepri-attacks/captures_prefill/` is **gitignored** (regenerate
  via the command below).

## Reproduce the spike

```bash
# rotation-only prefill capture + layer-0 dictionary (CPU, ~40s; GPU not needed)
GELO_CAPTURE_COVER=rotation GELO_CAPTURE_DICT=1 GELO_CAPTURE_DICT_N=8000 \
GELO_CAPTURE_LAYERS=0 GELO_CAPTURE_PROMPT="<any sensitive-looking text>" \
GELO_CAPTURE_DIR="$PWD/evals/aloepri-attacks/captures_prefill" \
  cargo test -p gelo-embedder --test attn_cover_capture --release \
  capture_attn_cover_adversary_view -- --ignored --nocapture
# attack (numpy only; runs in the container — host has no numpy)
evals/aloepri-attacks/run-in-container.sh \
  python3 evals/aloepri-attacks/gate3_prefill_dict.py
```

## Next steps — Phase 5b: viability of feature-rotated prefill offload under AloePri

The general statement (plan): any correctable-through-fused-softmax cover preserves
the bilinear forms `X·M·Xᵀ` that `WEIGHTS-PUB` reads. The **only** escape is to make
`M` (i.e. `W_V`, `W_Q W_Kᵀ`) **unknown** to the attacker — i.e. **covariant weight
obfuscation (AloePri)**: statically transform the deployed weights `W → W'` with a
covariant compensation so the computation is unchanged but the attacker's *public*
`W` no longer matches the *deployed* `W'`, collapsing the per-token dictionary.

Concrete tasks for the next session:
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
