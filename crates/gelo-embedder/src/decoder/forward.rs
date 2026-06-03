use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};

use gelo_protocol::profile;
use gelo_protocol::tee_matmul_bf16;
use gelo_protocol::{
    ForwardSessionShape, TrustedExecutor, WeightHandle, WeightKind, attention_partial,
    merge_attention_partials,
};

use super::attention::{
    causal_gqa_attention, causal_gqa_attention_cached, causal_gqa_attention_permuted,
    causal_gqa_attention_permuted_cached, causal_gqa_attention_swa_cached,
    causal_gqa_attention_with_offload,
};
use super::config::{AttentionClass, DecoderConfig};
use super::kv_cache::{DecodeCover, KvCache};
use super::rms_norm::{apply_qk_norm, rms_norm};
use super::rope::RopeTables;
use super::swiglu::swiglu;
use super::weights::{DecoderLayerWeights, DecoderWeights};

/// Run a Qwen3-style decoder embedder forward pass under the GELO protocol.
///
/// `input_ids` is a flat `[seq_len]` slice. Returns the per-token hidden
/// state matrix `(seq_len, hidden_size)` after the final RMSNorm. The caller
/// applies last-token pooling + L2 normalize.
pub fn run(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
) -> Result<Array2<f32>> {
    run_with_hook(cfg, weights, rope, exec, input_ids, |_, _| {})
}

/// Same as [`run`] but invokes `after_layer(layer_idx, &mut h)` after the
/// residual stream output of each transformer block (before the next
/// layer's input). The hook is a general per-layer instrumentation point.
pub fn run_with_hook<F: FnMut(usize, &mut Array2<f32>)>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    mut after_layer: F,
) -> Result<Array2<f32>> {
    let n = input_ids.len();
    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, input_ids)
    });

    // GELO paper §3.2 forward-pass session: see bert/forward.rs for the
    // rationale. Paper-parity executors sample one mask here; per-offload
    // executors and PlaintextExecutor treat this as a no-op.
    with_forward_session(exec, ForwardSessionShape::Single { n }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
            )?;
            after_layer(li, &mut h);
        }
        Ok(profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        }))
    })
}

fn embedding_lookup(cfg: &DecoderConfig, w: &DecoderWeights, ids: &[u32]) -> Array2<f32> {
    let n = ids.len();
    let d = cfg.hidden_size;
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &id) in ids.iter().enumerate() {
        // bf16 → f32 widening per element. No intermediate row alloc.
        let row = w.token_embedding.row(id as usize);
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v.to_f32();
        }
    }
    out
}

fn with_forward_session<X, T, F>(exec: &mut X, shape: ForwardSessionShape, body: F) -> Result<T>
where
    X: TrustedExecutor,
    F: FnOnce(&mut X) -> Result<T>,
{
    exec.begin_forward_session(shape)?;
    let result = body(exec);
    let end = exec.end_forward_pass();
    match (result, end) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Err(err), Err(_end_err)) => Err(err),
    }
}

/// Prefill — run a full prompt forward and populate the KV cache.
///
/// Returns the per-token hidden state matrix `(n_prompt, hidden_size)`
/// after the final RMSNorm. Caller takes the last row for next-token
/// sampling and re-uses the populated `kv_cache` for subsequent
/// [`run_decode_step`] calls.
///
/// Equivalent to [`run`] for one-shot embedding, except K and V are
/// preserved in `kv_cache` for autoregressive continuation. The
/// protocol-level forward-pass bracket (one fresh Haar `A`) covers the
/// full prefill in a single call — same property as [`run`].
pub fn run_prefill(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Array2<f32>> {
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let n = input_ids.len();
    let q_pos_offset = kv_cache.len();
    assert!(
        q_pos_offset + n <= kv_cache.capacity(),
        "prefill would overflow kv_cache: {} + {} > {}",
        q_pos_offset,
        n,
        kv_cache.capacity(),
    );

    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, input_ids)
    });

    with_forward_session(exec, ForwardSessionShape::Single { n }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block_cached(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
                kv_cache,
                q_pos_offset,
            )?;
        }
        Ok(profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        }))
    })
}

/// Decode one token — append its K/V to the cache, return the
/// resulting last-layer hidden state row `(hidden_size,)`.
///
/// `token_id` is the token whose embedding becomes the single-row input
/// to this step. The caller is responsible for the prefill phase having
/// populated `kv_cache` for positions `0..kv_cache.len()`; this
/// function appends one position at `kv_cache.len()` to every layer's
/// cache. The protocol-level forward-pass bracket fires once per
/// decode step — one fresh Haar `A` per token, per the locked design
/// decision.
pub fn run_decode_step(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    token_id: u32,
    kv_cache: &mut KvCache,
) -> Result<Array1<f32>> {
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let q_pos_offset = kv_cache.len();
    assert!(
        q_pos_offset + 1 <= kv_cache.capacity(),
        "decode would overflow kv_cache: {} + 1 > {}",
        q_pos_offset,
        kv_cache.capacity(),
    );

    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, &[token_id])
    });

    with_forward_session(exec, ForwardSessionShape::Single { n: 1 }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block_cached(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
                kv_cache,
                q_pos_offset,
            )?;
        }
        let normed = profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        });
        Ok(normed.row(0).to_owned())
    })
}

/// **M1.11 D1.3** — Batched decode-step forward over B sequences.
///
/// `token_ids` has length B — sequence `b` contributes `token_ids[b]`
/// as its next input row. Each sequence's KV cache is appended one
/// position at the sequence's current `kv_cache.len_b(li, b)`. Returns
/// the final hidden states `(B, hidden)` — caller gathers per-sequence
/// rows for logit computation.
///
/// Prefill must already have populated `kv_cache` per-sequence via
/// `forward::run_batched` (or a future `run_prefill_batched` if we
/// split it out). Each call here advances every sequence by exactly
/// one token.
///
/// Per the M1.11 plan §3.4: under `BATCHED_DECODE_SHARED_A=1` the
/// shielded operand at each layer goes through one shared dense mask
/// `(B+k, B+k)`; default is per-sequence `A_b` of size `(1+k, 1+k)`.
/// The executor's `begin_decode_pass(B)` handles the topology switch.
pub fn run_decode_step_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    token_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Array2<f32>> {
    let batch_size = token_ids.len();
    assert!(batch_size > 0, "run_decode_step_batched: token_ids empty");
    assert_eq!(
        batch_size,
        kv_cache.batch_size(),
        "token_ids len {} != kv_cache batch_size {}",
        batch_size,
        kv_cache.batch_size(),
    );
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    // Per-sequence position offset: each sequence's row sits at its
    // own absolute position `kv_cache.len_b(0, b)`. Layer 0 is the
    // representative — debug-asserts in `lens()` catch cross-layer
    // divergence.
    let q_pos_offsets: Vec<usize> = (0..batch_size).map(|b| kv_cache.len_b(0, b)).collect();
    for (b, off) in q_pos_offsets.iter().enumerate() {
        assert!(
            off + 1 <= kv_cache.capacity(),
            "decode would overflow kv_cache: sequence {b} cur {off} + 1 > {}",
            kv_cache.capacity(),
        );
    }

    // (B, hidden) — one embedding row per sequence.
    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, token_ids)
    });

    with_forward_session(
        exec,
        ForwardSessionShape::DecodeBatch { batch_size },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h = decoder_block_cached_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h.view(),
                    cfg.offload_layer(li),
                    kv_cache,
                    &q_pos_offsets,
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )
}

/// **M1.11 D1.3** — Batched cache-aware decoder block. Mirror of
/// [`decoder_block_cached`] for `B` parallel sequences each with their
/// own KV cache prefix length. Differences from the single-sequence
/// version:
///
/// 1. `hidden` shape is `(B, hidden)` — one new token row per sequence.
/// 2. `q_pos_offsets[b]` is per-sequence; RoPE applies row `b`'s
///    rotation at its own absolute position.
/// 3. KV cache append uses `append_decode` (one row per sequence,
///    placed at each sequence's current `lens[b]`).
/// 4. Attention runs per-sequence in-TEE over each sequence's full
///    cached prefix (`tee:attn_cached_inplace_many` — same stopgap
///    pattern as `decoder_block_batched`'s prefill attention). R1.4
///    would replace this with a single batched-kernel dispatch.
#[allow(clippy::too_many_arguments)]
/// Phase-4 perf wire-up flag (perm-attn-gpu-offload): route GLOBAL-layer
/// decode attention through the GPU-resident K/V session. Default off
/// (`GELO_GPU_RESIDENT_ATTN` ∈ {1, true}). Read once.
fn gpu_resident_attn_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("GELO_GPU_RESIDENT_ATTN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Explicit override for the prefill-attention GPU offload
/// (perm-attn-gpu-offload Phase 6): route GLOBAL-layer prefill self-attention
/// through the fused `cubek` kernel under the value cover (`O_qk` + the
/// per-session non-orthogonal `C_v`), instead of the in-TEE
/// `causal_gqa_attention` B-loop. `Some(true/false)` from
/// `GELO_GPU_PREFILL_OFFLOAD`; `None` (unset) → the caller uses the
/// **capability default** (on when the executor supports the fused offload —
/// `exec.supports_offloaded_attention()` — and in-TEE otherwise). SWA layers
/// always stay in-TEE.
fn gpu_prefill_offload_override() -> Option<bool> {
    static EN: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("GELO_GPU_PREFILL_OFFLOAD")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// `(n, H·d)` → `(H, n, d)` — fold a per-sequence projection into stacked
/// per-head shape (the prefill offload layout; one folded head per attention
/// problem).
fn fold_heads_2d(x: ArrayView2<'_, f32>, h: usize, d: usize) -> Array3<f32> {
    let n = x.nrows();
    let mut out = Array3::<f32>::zeros((h, n, d));
    for hi in 0..h {
        for j in 0..n {
            for c in 0..d {
                out[(hi, j, c)] = x[(j, hi * d + c)];
            }
        }
    }
    out
}

/// `(H, n, d)` → `(n, H·d)`, inverse of [`fold_heads_2d`].
fn unfold_heads_2d(x: ArrayView3<'_, f32>, h: usize, d: usize) -> Array2<f32> {
    let n = x.shape()[1];
    let mut out = Array2::<f32>::zeros((n, h * d));
    for hi in 0..h {
        for j in 0..n {
            for c in 0..d {
                out[(j, hi * d + c)] = x[(hi, j, c)];
            }
        }
    }
    out
}

/// Batched fold `(B·n, H·d)` → `(B·H, n, d)` (b-major: folded head `b*H+h`) —
/// the single-dispatch prefill fold for uniform-length batches, so all B
/// sequences attend in **one** `cubek` call instead of B (prefill offload
/// loop-batching). Parallelised over the B·H folded heads.
fn fold_heads_2d_batched(
    x: ArrayView2<'_, f32>,
    b: usize,
    n: usize,
    h: usize,
    d: usize,
) -> Array3<f32> {
    use ndarray::parallel::prelude::*;
    let mut out = Array3::<f32>::zeros((b * h, n, d));
    out.outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(fh, mut head)| {
            let bi = fh / h;
            let hi = fh % h;
            for j in 0..n {
                // Contiguous-slice memcpy per row (the per-element
                // indexed copy paid an ndarray bounds check per f32).
                let fast = if let (Some(src_row), Some(dst)) =
                    (x.row(bi * n + j).to_slice(), head.row_mut(j).into_slice())
                {
                    dst.copy_from_slice(&src_row[hi * d..(hi + 1) * d]);
                    true
                } else {
                    false
                };
                if !fast {
                    for c in 0..d {
                        head[(j, c)] = x[(bi * n + j, hi * d + c)];
                    }
                }
            }
        });
    out
}

/// Fused `O_vᵀ` correction **+ unfold**: write `ctx_raw[b·Hq+qh] · O_vᵀ`
/// directly into `ctx[b·n + .., qh·d ..]`, skipping the intermediate
/// `(B·Hq, n, d)` array and the separate unfold copy (prefill offload, fuse
/// lever). Parallelised over **(sequence, head)** — the per-head column
/// chunks of each block are disjoint, so rayon fans out across heads
/// inside a block. (Block-only parallelism was degenerate at B=1: one
/// chunk → the Hq per-head GEMMs ran fully serial, the dominant share of
/// the 1.15 s `prefill_cover:correct_tee` bucket at B=1 n=2048.) The
/// per-head `(n,d)·(d,d)` stays a BLAS `dot`. `o_vt` is `O_vᵀ`.
fn correct_unfold_into(
    ctx: &mut Array2<f32>,
    ctx_raw: ArrayView3<'_, f32>,
    o_vt: ArrayView2<'_, f32>,
    n: usize,
    hq: usize,
    d: usize,
) {
    use ndarray::parallel::prelude::*;
    ctx.axis_chunks_iter_mut(Axis(0), n)
        .into_par_iter()
        .enumerate()
        .for_each(|(bi, mut block)| {
            block
                .axis_chunks_iter_mut(Axis(1), d)
                .into_par_iter()
                .enumerate()
                .for_each(|(qh, mut col)| {
                    let head = ctx_raw.index_axis(Axis(0), bi * hq + qh); // (n, d)
                    col.assign(&head.dot(&o_vt)); // (n, d), BLAS
                });
        });
}

/// `(B, H·d)` → `(B·H, 1, d)` (b-major), the stacked per-head shape the
/// resident session expects.
fn stack_heads(x: ArrayView2<'_, f32>, b: usize, h: usize, d: usize) -> Array3<f32> {
    let mut out = Array3::<f32>::zeros((b * h, 1, d));
    for bi in 0..b {
        for hi in 0..h {
            for c in 0..d {
                out[(bi * h + hi, 0, c)] = x[(bi, hi * d + c)];
            }
        }
    }
    out
}

/// `(B·H, 1, d)` → `(B, H·d)`, inverse of [`stack_heads`].
fn unstack_heads(x: ArrayView3<'_, f32>, b: usize, h: usize, d: usize) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((b, h * d));
    for bi in 0..b {
        for hi in 0..h {
            for c in 0..d {
                out[(bi, hi * d + c)] = x[(bi * h + hi, 0, c)];
            }
        }
    }
    out
}

/// Stack per-`(layer, b)` cache views `(n_kv, kv_heads·d)` into the
/// un-replicated session shape `(B·kv_heads, n_kv, d)` (b-major).
fn stack_cache(
    views: &[(ArrayView2<'_, f32>, ArrayView2<'_, f32>)],
    b: usize,
    kvh: usize,
    d: usize,
) -> (Array3<f32>, Array3<f32>) {
    let n_kv = views[0].0.nrows();
    let mut k_st = Array3::<f32>::zeros((b * kvh, n_kv, d));
    let mut v_st = Array3::<f32>::zeros((b * kvh, n_kv, d));
    for bi in 0..b {
        let (kb, vb) = views[bi];
        for hi in 0..kvh {
            for j in 0..n_kv {
                for c in 0..d {
                    k_st[(bi * kvh + hi, j, c)] = kb[(j, hi * d + c)];
                    v_st[(bi * kvh + hi, j, c)] = vb[(j, hi * d + c)];
                }
            }
        }
    }
    (k_st, v_st)
}

/// Explicit override for the permuted-cover tail-in-TEE decode path
/// (perm-attn-gpu-offload). When on, the resident decode attention runs under
/// the full feature-rotation + permutation + σ + per-session `C_v` cover
/// (session-fixed, no per-block re-permute), with the newest tokens held
/// in-TEE (partial-stats prefix attend on GPU + in-TEE tail + online merge).
/// `Some(..)` from `GELO_GPU_RESIDENT_COVER`; `None` (unset) → the caller uses
/// the capability default (on when the executor supports the offload).
/// Requires `Global`.
fn gpu_resident_cover_override() -> Option<bool> {
    static EN: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("GELO_GPU_RESIDENT_COVER")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// σ for the cover's K/q Hidden-No-More noise. **Default 0.01** (the secure
/// decode config; only read on the cover path). Set `GELO_RESIDENT_SIGMA=0`
/// for an exact, byte-identical-parity run.
fn resident_cover_sigma() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("GELO_RESIDENT_SIGMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.01)
    })
}

/// A `d×d` orthogonal matrix via modified Gram-Schmidt on a Gaussian —
/// the feature-rotation cover operand (`O_qk`/`O_v`). Deterministic in
/// `rng`, so a fixed per-layer seed re-derives the same cover each step
/// (session-fixed) without storing the matrices.
fn sample_orthogonal<R: rand::Rng>(d: usize, rng: &mut R) -> Array2<f32> {
    use rand_distr::{Distribution, StandardNormal};
    let mut a = Array2::<f32>::from_shape_fn((d, d), |_| StandardNormal.sample(rng));
    for j in 0..d {
        for i in 0..j {
            let ci = a.column(i).to_owned();
            let proj = ci.dot(&a.column(j));
            let cj = a.column(j).to_owned();
            let mut cjm = a.column_mut(j);
            for k in 0..d {
                cjm[k] = cj[k] - proj * ci[k];
            }
        }
        let norm = a.column(j).dot(&a.column(j)).sqrt();
        a.column_mut(j).mapv_inplace(|x| x / norm.max(1e-12));
    }
    a
}

/// Cover condition-number κ for the value cover `C_v` (`GELO_COVER_KAPPA`).
/// **Default 6.0** — the secure non-orthogonal `C_v` that breaks the
/// `WEIGHTS-PUB` value norm/Gram dictionary (Stage-1/2 gate: min κ≈5,
/// recommended κ≈6). Set `GELO_COVER_KAPPA=1` for the legacy orthogonal `O_v`
/// (insecure under `WEIGHTS-PUB`).
fn cover_kappa() -> f32 {
    std::env::var("GELO_COVER_KAPPA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6.0)
}

/// Value-cover operands `(C_v, C_v⁻¹)`. At `kappa ≤ 1` this is the orthogonal
/// `O_v` (inverse = transpose) — the legacy cover, with a byte-identical rng
/// draw. At `kappa > 1` it is the κ-bounded non-orthogonal cover
/// `C_v = U·diag(s)·Vᵀ` (`cond = kappa`, singular values pinned so the
/// geometric mean ≈ 1 → magnitude-neutral), with exact inverse
/// `C_v⁻¹ = V·diag(1/s)·Uᵀ`. Non-orthogonal ⇒ distorts the per-head value
/// norm/Gram (breaks the public-weight dictionary) while staying exactly
/// correctable. See docs/dev/logs/perm-attn-gpu-offload.md
/// (*Adapting covariant obfuscation to the offload*).
fn sample_value_cover<R: rand::Rng>(d: usize, kappa: f32, rng: &mut R) -> (Array2<f32>, Array2<f32>) {
    if kappa <= 1.0 {
        let o = sample_orthogonal(d, rng);
        let oi = o.t().to_owned();
        return (o, oi);
    }
    let u = sample_orthogonal(d, rng);
    let vt = sample_orthogonal(d, rng); // right factor (acts as Vᵀ)
    let hi = kappa.sqrt();
    let lo = 1.0 / hi;
    let (lhi, llo) = (hi.ln(), lo.ln());
    let mut s = vec![1.0f32; d];
    s[0] = hi;
    if d > 1 {
        s[d - 1] = lo;
    }
    // Log-uniform singular values ⇒ geometric mean exactly 1 (magnitude-neutral,
    // det(C_v)≈1) so κ is a clean conditioning dial with no volume drift. (A
    // *linear*-uniform draw biases the geometric mean >1 — det-drift — which is
    // cosmetically wrong though immaterial to the fp16 error; diagnosed
    // 2026-06-02, see the C_v fp16-drift sweep.)
    for sj in s.iter_mut().take(d.saturating_sub(1)).skip(1) {
        let t: f32 = rng.random();
        *sj = (llo + t * (lhi - llo)).exp();
    }
    // C = (U·diag(s))·Vᵀ ;  C⁻¹ = (V·diag(1/s))·Uᵀ
    let mut us = u.clone();
    let mut vi = vt.t().to_owned(); // V = (Vᵀ)ᵀ
    for j in 0..d {
        for i in 0..d {
            us[(i, j)] *= s[j];
            vi[(i, j)] /= s[j];
        }
    }
    (us.dot(&vt), vi.dot(&u.t()))
}

/// Per-head right-multiply `out[h] = x[h] · o` for `x (H, n, d)`,
/// `o (d, d)` — applies a shared feature rotation to every stacked head.
///
/// Parallelises over heads when the call carries real total work. (A
/// per-head gate was tried and **measured worse** at the decode shape —
/// `cover:acc_uncover_tee` 122→246 ms — the 32-way tiny-GEMM fan-out
/// still beats serial there; only truly tiny calls go serial.)
fn rotate_heads(x: ArrayView3<'_, f32>, o: ArrayView2<'_, f32>) -> Array3<f32> {
    let (h, n, d) = x.dim();
    let mut out = Array3::<f32>::zeros((h, n, d));
    // Total flops ≈ h·n·d·d; fork when ≥ ~256K.
    if h >= 2
        && h.saturating_mul(n)
            .saturating_mul(d)
            .saturating_mul(d)
            >= (1 << 18)
    {
        use ndarray::parallel::prelude::*;
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(hi, mut row)| {
                row.assign(&x.index_axis(Axis(0), hi).dot(&o));
            });
    } else {
        for (hi, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
            row.assign(&x.index_axis(Axis(0), hi).dot(&o));
        }
    }
    out
}

/// Stack the in-TEE active tail `[prefix_len..len)` of each sequence's
/// cache into the GQA-expanded per-q-head shape `(B·nqh, n_tail, d)`
/// (plaintext — the tail never reaches the GPU).
fn stack_tail_expanded(
    views: &[(ArrayView2<'_, f32>, ArrayView2<'_, f32>)],
    prefix_len: usize,
    b: usize,
    nqh: usize,
    nkvh: usize,
    d: usize,
) -> (Array3<f32>, Array3<f32>) {
    let group = nqh / nkvh;
    let total = views[0].0.nrows();
    let n_tail = total.saturating_sub(prefix_len);
    let mut k = Array3::<f32>::zeros((b * nqh, n_tail, d));
    let mut v = Array3::<f32>::zeros((b * nqh, n_tail, d));
    for bi in 0..b {
        let (kb, vb) = views[bi];
        for qh in 0..nqh {
            let kvh = qh / group;
            let idx = bi * nqh + qh;
            for j in 0..n_tail {
                for c in 0..d {
                    k[(idx, j, c)] = kb[(prefix_len + j, kvh * d + c)];
                    v[(idx, j, c)] = vb[(prefix_len + j, kvh * d + c)];
                }
            }
        }
    }
    (k, v)
}

/// Build the session-fixed **covered resident prefix** for one GLOBAL layer
/// (perm-attn-gpu-offload): permute the frozen-prefix rows + add σ-noise to K +
/// feature-rotate K/V (shared per-layer `O_qk`/`O_v`), then upload to a GPU
/// resident K/V session. One-time per layer; idempotent at the call sites
/// (callers guard on `kv_cache.gpu_session(..).is_none()`). Hoisted out of the
/// decode block so it can run either lazily on the first decode step **or** at
/// the prefill→decode handoff (O5 — moves the cost off the decode critical
/// path). σ-on-K uses per-head ChaCha streams (O4(a); skipped at σ=0).
fn build_covered_prefix_session(
    exec: &mut impl TrustedExecutor,
    layer_idx: usize,
    kv_cache: &mut KvCache,
    batch_size: usize,
    nkvh: usize,
    dh: usize,
    sigma: f32,
) -> Result<()> {
    let (k_cov, v_cov, cover) =
        build_covered_prefix_cpu(layer_idx, kv_cache, batch_size, nkvh, dh, sigma)?;
    let cap = kv_cache.capacity();
    let id = exec.resident_kv_create(k_cov.view(), v_cov.view(), cap)?;
    kv_cache.set_gpu_session(layer_idx, id);
    kv_cache.set_gpu_cover(layer_idx, cover);
    Ok(())
}

/// CPU half of the covered-prefix build for one layer: sample the
/// per-layer cover (seeded by `layer_idx` — layer-independent), stack +
/// permute + σ-noise the frozen prefix, feature-rotate K/V. Reads
/// `kv_cache` immutably so multiple layers can build **in parallel**
/// (the prefill→decode handoff fans this out across layers); the upload
/// half (`resident_kv_create` + cache bookkeeping) stays serial on the
/// engine.
fn build_covered_prefix_cpu(
    layer_idx: usize,
    kv_cache: &KvCache,
    batch_size: usize,
    nkvh: usize,
    dh: usize,
    sigma: f32,
) -> Result<(Array3<f32>, Array3<f32>, DecodeCover)> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    const SALT: u64 = 0xC0FFEE_5EED;
    let mut crng = ChaCha20Rng::seed_from_u64(SALT ^ layer_idx as u64);
    let o_qk = sample_orthogonal(dh, &mut crng);
    let (c_v, c_v_inv) = sample_value_cover(dh, cover_kappa(), &mut crng);
    let kv_views: Vec<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> = (0..batch_size)
        .map(|b| kv_cache.view_b(layer_idx, b))
        .collect::<Result<Vec<_>>>()?;
    let prefix_len = kv_views[0].0.nrows();
    let (k_st, v_st) = stack_cache(&kv_views, batch_size, nkvh, dh);
    // perm (row gather) + σ on K — O4(a): row-level copies, per-head ChaCha
    // noise, parallel over the B·nkvh heads.
    let mut perm: Vec<usize> = (0..prefix_len).collect();
    perm.shuffle(&mut crng);
    let bh = batch_size * nkvh;
    let noise_seed = rand::RngCore::next_u64(&mut crng);
    let mut kp = Array3::<f32>::zeros((bh, prefix_len, dh));
    let mut vp = Array3::<f32>::zeros((bh, prefix_len, dh));
    {
        use ndarray::parallel::prelude::*;
        kp.outer_iter_mut()
            .into_par_iter()
            .zip(vp.outer_iter_mut().into_par_iter())
            .enumerate()
            .for_each(|(h, (mut kph, mut vph))| {
                let k_src = k_st.index_axis(Axis(0), h);
                let v_src = v_st.index_axis(Axis(0), h);
                for (i, &src) in perm.iter().enumerate() {
                    kph.row_mut(i).assign(&k_src.row(src));
                    vph.row_mut(i).assign(&v_src.row(src));
                }
                if sigma > 0.0 {
                    let mut hrng = ChaCha20Rng::seed_from_u64(noise_seed ^ h as u64);
                    for x in kph.iter_mut() {
                        let z: f32 = StandardNormal.sample(&mut hrng);
                        *x += sigma * z;
                    }
                }
            });
    }
    // Feature rotation: K·O_qk, V·O_v (shared O across heads →
    // GQA-broadcast-consistent).
    let k_cov = rotate_heads(kp.view(), o_qk.view());
    let v_cov = rotate_heads(vp.view(), c_v.view());
    Ok((k_cov, v_cov, DecodeCover { prefix_len, o_qk, c_v, c_v_inv }))
}

/// Build covered resident prefixes for **all GLOBAL layers** at the
/// prefill→decode handoff (O5). Gated by the caller on the capability-resolved
/// resident-cover decision; idempotent (skips layers already built).
/// Moves the one-time `build_covered_prefix` cost off the first decode step.
fn build_covered_prefix_all_global(
    cfg: &DecoderConfig,
    exec: &mut impl TrustedExecutor,
    kv_cache: &mut KvCache,
    batch_size: usize,
) -> Result<()> {
    let (nkvh, dh) = (cfg.num_key_value_heads, cfg.head_dim_value());
    let sigma = resident_cover_sigma();
    let pending: Vec<usize> = (0..cfg.num_hidden_layers)
        .filter(|&li| {
            matches!(cfg.effective_attention_class(li), AttentionClass::Global)
                && kv_cache.gpu_session(li).is_none()
        })
        .collect();
    // Layers are independent (per-layer seed, per-layer K/V): par-build
    // the CPU covers in bounded chunks (the transient covered K/V is
    // ~130 MB/layer at n=8192 — chunking caps peak memory), then upload
    // each chunk serially on the engine. The per-layer head parallelism
    // alone (B·nkvh = 8 at B=1) under-fills the cores; cross-layer
    // fan-out fixes that.
    profile::time("cover:build_covered_prefix+upload", || -> Result<()> {
        use rayon::prelude::*;
        for chunk in pending.chunks(4) {
            let built: Vec<(usize, (Array3<f32>, Array3<f32>, DecodeCover))> = chunk
                .par_iter()
                .map(|&li| {
                    Ok((
                        li,
                        build_covered_prefix_cpu(li, kv_cache, batch_size, nkvh, dh, sigma)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            for (li, (k_cov, v_cov, cover)) in built {
                let cap = kv_cache.capacity();
                let id = exec.resident_kv_create(k_cov.view(), v_cov.view(), cap)?;
                kv_cache.set_gpu_session(li, id);
                kv_cache.set_gpu_cover(li, cover);
            }
        }
        Ok(())
    })
}

fn decoder_block_cached_batched(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
    kv_cache: &mut KvCache,
    q_pos_offsets: &[usize],
) -> Result<Array2<f32>> {
    let batch_size = hidden.nrows();
    debug_assert_eq!(batch_size, q_pos_offsets.len());

    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V: hidden has shape (B, hidden_size). Under
    // SessionKind::PerSequence (default decode) or Single (shared-A
    // opt-in), offload_qkv produces (B, q_dim/kv_dim) outputs.
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv")
                        .view(),
                ),
            )
        })
    };

    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE: row `b` sits at absolute position `q_pos_offsets[b]`.
    // Apply rotation per-row by slicing the single row out, calling
    // apply_partial_at with start_pos = q_pos_offsets[b], and writing
    // back.
    //
    // `rotated_dim` follows the existing decoder_block_cached logic
    // for attention class. At decode the layer class is consistent
    // across all sequences (it's a property of the layer, not the
    // sequence), so resolve once outside the per-b loop.
    let layer_class = cfg.effective_attention_class(layer_idx as usize);
    let rotated_dim = match (layer_class, cfg.partial_rope) {
        (AttentionClass::Global, Some(_)) => cfg.rotated_dim(),
        _ => cfg.head_dim_value(),
    };
    profile::time("tee:rope", || {
        for b in 0..batch_size {
            let pos = q_pos_offsets[b];
            rope.apply_partial_at(
                q.slice_mut(ndarray::s![b..b + 1, ..]),
                cfg.num_attention_heads,
                pos,
                rotated_dim,
            );
            rope.apply_partial_at(
                k.slice_mut(ndarray::s![b..b + 1, ..]),
                cfg.num_key_value_heads,
                pos,
                rotated_dim,
            );
        }
    });

    // K=V sharing (Gemma 4 global): re-derive V from K post-RoPE so
    // that the V tensor stays identical to K after rotation.
    let kv_shared = cfg.kv_shared_in_global && matches!(layer_class, AttentionClass::Global);
    let v = if kv_shared { k.clone() } else { v };

    // Append one row per sequence to the KV cache.
    kv_cache.append_decode(layer_idx as usize, k.view(), v.view())?;

    // Per-sequence in-TEE causal attention over the full cached
    // prefix. n_q = 1 per sequence, n_kv = lens[b]. R1.4 stopgap as
    // in decoder_block_batched.
    //
    // D1.8: rayon-parallel over B. At n_q = 1 the inner attention
    // kernel (`causal_gqa_attention_cached` / `_swa_cached`) runs
    // serial over heads (its internal rayon threshold is n_q ≥ 64),
    // so an outer rayon-iter over the B sequences gives clean
    // B-parallel attention with no nested-rayon contention. We
    // pre-collect `kv_cache.view_b` views outside the parallel
    // section to keep Result plumbing out of the closure.
    let q_dim = cfg.num_attention_heads * cfg.head_dim_value();
    let mut ctx = Array2::<f32>::zeros((batch_size, q_dim));

    // Phase-4 perf wire-up (perm-attn-gpu-offload): route GLOBAL-layer
    // decode attention through the GPU-resident K/V session — create once
    // (first decode step, from the full cache) then append one row/step
    // and attend on-device, instead of the in-TEE per-sequence kernel.
    // Gated (default off → the in-TEE path below is unchanged); SWA layers
    // always stay in-TEE. NO cover/tail-in-TEE yet (those need the σ-vs-N
    // spike) — this measures the resident-attention decode-wall lever only.
    let use_gpu_cover = gpu_resident_cover_override()
        .unwrap_or_else(|| exec.supports_offloaded_attention())
        && matches!(layer_class, AttentionClass::Global);
    let use_gpu_resident = !use_gpu_cover
        && gpu_resident_attn_enabled()
        && matches!(layer_class, AttentionClass::Global);
    if use_gpu_cover {
        // Permuted-cover tail-in-TEE decode (perm-attn-gpu-offload, full
        // wire). Session-fixed cover (perm_kv + σ on K + O_qk on Q/K + O_v
        // on V, re-derived from a per-layer seed); frozen prefix on the GPU,
        // newest tokens in-TEE (partial-stats prefix attend + in-TEE tail +
        // online merge → closes the write-location channel; no per-step GPU
        // write, no per-block re-permute).
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};
        let (nqh, nkvh, dh) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        );
        let scale = 1.0_f32 / (dh as f32).sqrt();
        let sigma = resident_cover_sigma();
        profile::time("tee:attn_resident_cover", || -> Result<()> {
            const SALT: u64 = 0xC0FFEE_5EED;
            let q_st = stack_heads(q.view(), batch_size, nqh, dh); // (B·nqh,1,dh) plaintext

            // Create the covered session once; O is sampled here and CACHED
            // (re-deriving the Gram-Schmidt O every step is pure waste).
            if kv_cache.gpu_session(layer_idx as usize).is_none() {
                // One-time covered-prefix build + upload (the session-fixed
                // "re-permute" cost, paid once per layer; amortized over K).
                // Lazy fallback — normally built at the prefill→decode handoff
                // (O5, `build_covered_prefix_all_global`); this fires only if
                // that hoist was skipped.
                profile::time("cover:build_covered_prefix+upload", || {
                    build_covered_prefix_session(
                        exec,
                        layer_idx as usize,
                        kv_cache,
                        batch_size,
                        nkvh,
                        dh,
                        sigma,
                    )
                })?;
            }
            let id = kv_cache.gpu_session(layer_idx as usize).unwrap();
            // Cached cover (clone the O matrices — ~128 KB, cheap — to release
            // the kv_cache borrow across the exec/view_b calls below).
            let (o_qk, c_v_inv, prefix_len) = {
                let c = kv_cache.gpu_cover(layer_idx as usize).unwrap();
                (c.o_qk.clone(), c.c_v_inv.clone(), c.prefix_len)
            };

            // Prefix partial on GPU: q covered by O_qk (+σ), uncover acc by O_vᵀ.
            let q_cov = profile::time("cover:q_cover_tee", || {
                let mut qn = q_st.clone();
                if sigma > 0.0 {
                    let mut qrng = ChaCha20Rng::seed_from_u64(
                        SALT ^ (layer_idx as u64) ^ ((prefix_len as u64) << 20)
                            ^ q_pos_offsets[0] as u64,
                    );
                    for e in qn.iter_mut() {
                        let z: f32 = StandardNormal.sample(&mut qrng);
                        *e += sigma * z;
                    }
                }
                rotate_heads(qn.view(), o_qk.view())
            });
            let (acc_a_cov, m_a, l_a) = profile::time("cover:prefix_partial_gpu", || {
                exec.resident_kv_attend_partial(id, q_cov.view(), scale)
            })?;
            let acc_a = profile::time("cover:acc_uncover_tee", || {
                rotate_heads(acc_a_cov.view(), c_v_inv.view())
            });

            // In-TEE active tail [prefix_len..len): plaintext partial + merge.
            let kv_views: Vec<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> = (0..batch_size)
                .map(|b| kv_cache.view_b(layer_idx as usize, b))
                .collect::<Result<Vec<_>>>()?;
            let n_tail = kv_views[0].0.nrows().saturating_sub(prefix_len);
            let ctx_st = if n_tail == 0 {
                // Create step: prefix only → normalise acc_a by l_a.
                let mut out = acc_a;
                for h in 0..out.shape()[0] {
                    let l = l_a[(h, 0, 0)];
                    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
                    out.index_axis_mut(Axis(0), h).mapv_inplace(|x| x * inv);
                }
                out
            } else {
                let (tk, tv) = profile::time("cover:tail_build_tee", || {
                    stack_tail_expanded(&kv_views, prefix_len, batch_size, nqh, nkvh, dh)
                });
                let (acc_b, m_b, l_b) = profile::time("cover:tail_partial_tee", || {
                    attention_partial(q_st.view(), tk.view(), tv.view(), scale)
                });
                profile::time("cover:merge_tee", || {
                    merge_attention_partials(
                        acc_a.view(),
                        m_a.view(),
                        l_a.view(),
                        acc_b.view(),
                        m_b.view(),
                        l_b.view(),
                    )
                })
            };
            ctx = unstack_heads(ctx_st.view(), batch_size, nqh, dh);
            Ok(())
        })?;
    } else if use_gpu_resident {
        let (nqh, nkvh, dh) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        );
        let scale = 1.0_f32 / (dh as f32).sqrt();
        profile::time("tee:attn_resident_gpu", || -> Result<()> {
            let q_st = stack_heads(q.view(), batch_size, nqh, dh); // (B·nqh, 1, dh)
            let id = match kv_cache.gpu_session(layer_idx as usize) {
                None => {
                    // First decode step: create the session from the full
                    // cache (already includes this step's appended row).
                    let (k_st, v_st) = {
                        let kv_views: Vec<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> = (0
                            ..batch_size)
                            .map(|b| kv_cache.view_b(layer_idx as usize, b))
                            .collect::<Result<Vec<_>>>()?;
                        stack_cache(&kv_views, batch_size, nkvh, dh)
                    };
                    let cap = kv_cache.capacity();
                    let id = exec.resident_kv_create(k_st.view(), v_st.view(), cap)?;
                    kv_cache.set_gpu_session(layer_idx as usize, id);
                    id
                }
                Some(id) => {
                    // Later steps: append just this step's new K/V row.
                    let k_row = stack_heads(k.view(), batch_size, nkvh, dh);
                    let v_row = stack_heads(v.view(), batch_size, nkvh, dh);
                    exec.resident_kv_append(id, k_row.view(), v_row.view())?;
                    id
                }
            };
            let ctx_st = exec.resident_kv_attend(id, q_st.view(), scale)?; // (B·nqh, 1, dh)
            ctx = unstack_heads(ctx_st.view(), batch_size, nqh, dh);
            Ok(())
        })?;
    } else {
        let kv_views: Vec<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> = (0..batch_size)
            .map(|b| kv_cache.view_b(layer_idx as usize, b))
            .collect::<Result<Vec<_>>>()?;
        profile::time("tee:attn_cached_inplace_many", || {
            use ndarray::parallel::prelude::*;
            ctx.axis_chunks_iter_mut(ndarray::Axis(0), 1)
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut ctx_slice)| {
                    let q_b = q.slice(ndarray::s![b..b + 1, ..]);
                    let (k_cached, v_cached) = kv_views[b];
                    let ctx_b = match layer_class {
                        AttentionClass::Local { window } => causal_gqa_attention_swa_cached(
                            q_b,
                            k_cached,
                            v_cached,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            cfg.head_dim_value(),
                            q_pos_offsets[b],
                            window,
                        ),
                        AttentionClass::Global => causal_gqa_attention_cached(
                            q_b,
                            k_cached,
                            v_cached,
                            cfg.num_attention_heads,
                            cfg.num_key_value_heads,
                            cfg.head_dim_value(),
                            q_pos_offsets[b],
                        ),
                    };
                    ctx_slice.assign(&ctx_b);
                });
        });
    }

    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up")
                        .view(),
                ),
            )
        })
    };
    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));
    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

/// Cache-aware decoder block. Same compute path as the legacy
/// [`decoder_block`] but additionally appends the post-RoPE K, V to
/// `kv_cache` for layer `layer_idx`, and routes attention through the
/// asymmetric [`causal_gqa_attention_cached`] kernel so a single-row
/// Q (decode) can attend to the full cached prefix.
///
/// At prefill shape (n_q = n_kv, q_pos_offset = 0) this matches the
/// legacy block bit-for-bit (the asymmetric mask collapses to the
/// lower-triangular causal mask). At decode shape it's the harness's
/// single-token-per-step path.
///
/// OutAttnMult / permuted attention auto-switches are intentionally
/// not wired through this path yet — those are square-only kernels and
/// the fused permuted FlashAttention path lands in M1.10. Until then,
/// the cached block uses the in-TEE attention computation. This
/// matches the locked design decision: decode global attention stays
/// in-TEE always; long-context prefill global attention will use the
/// fused permuted kernel once M1.10 ships.
#[allow(clippy::too_many_arguments)]
fn decoder_block_cached(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
    kv_cache: &mut KvCache,
    q_pos_offset: usize,
) -> Result<Array2<f32>> {
    // Pre-attention RMSNorm.
    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V projections.
    //
    // For Gemma 4 global layers with `kv_shared_in_global` true, the
    // trained model ties `W_k = W_v` ("K equals V" trick). Two
    // mathematically-identical matmuls collapse to one — we compute K
    // once and reuse the result as V. The KV cache for these layers
    // is sized half as wide (one tensor instead of two) — see
    // `KvCache::new_with_sharing`.
    let layer_class = cfg.effective_attention_class(layer_idx as usize);
    let kv_shared = cfg.kv_shared_in_global && matches!(layer_class, AttentionClass::Global);

    let (mut q_new, mut k_new, v_new) = if offload {
        if kv_shared {
            // One masked matmul for Q, one for K=V. The K and V handles
            // map to the same backing weight when the model is loaded
            // with `wk` and `wv` Arc-shared; the executor doesn't need
            // to know that. We just skip the V offload and use K.
            let q =
                exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::Q), h_norm.view())?;
            let k =
                exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::K), h_norm.view())?;
            let v = k.clone();
            (q, k, v)
        } else {
            exec.offload_qkv(layer_idx, h_norm.view())?
        }
    } else {
        profile::time("tee:qkv_direct", || {
            let q = tee_matmul_bf16(
                h_norm.view(),
                layer
                    .wq
                    .as_ref()
                    .expect("offload=false requires layer.wq present (skip-layers mode)")
                    .view(),
            );
            let k = tee_matmul_bf16(
                h_norm.view(),
                layer
                    .wk
                    .as_ref()
                    .expect("offload=false requires layer.wk present (skip-layers mode)")
                    .view(),
            );
            let v = if kv_shared {
                k.clone()
            } else {
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present (skip-layers mode)")
                        .view(),
                )
            };
            (q, k, v)
        })
    };

    // Qwen3 QK-norm — per-head RMSNorm on Q and K **before** RoPE.
    // No-op for older models (norms = None). Applied per-head so each
    // attention head normalises its own `head_dim` slice independently;
    // gamma has length `head_dim`.
    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q_new.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k_new.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE — rotate Q and K at absolute positions
    // `q_pos_offset..q_pos_offset + n_q`. Per the Gemma 4 p-RoPE
    // recipe: global layers rotate only the first `rotated_dim` of
    // each head; local layers rotate the full head_dim.
    let rotated_dim = match (layer_class, cfg.partial_rope) {
        (AttentionClass::Global, Some(_)) => cfg.rotated_dim(),
        // Local layers always rotate the full head_dim (Gemma 4
        // spec). Models with `partial_rope = None` likewise use
        // full rotation everywhere.
        _ => cfg.head_dim_value(),
    };
    profile::time("tee:rope", || {
        rope.apply_partial_at(
            q_new.view_mut(),
            cfg.num_attention_heads,
            q_pos_offset,
            rotated_dim,
        );
        rope.apply_partial_at(
            k_new.view_mut(),
            cfg.num_key_value_heads,
            q_pos_offset,
            rotated_dim,
        );
    });
    // For K=V shared global layers, the V tensor must stay identical
    // to K after RoPE. The simplest correctness path: re-derive V
    // from K post-RoPE. (Earlier we cloned K into V before RoPE, so
    // the clone is now stale.) Cheap — one ndarray clone.
    let v_new = if kv_shared { k_new.clone() } else { v_new };

    // Append fresh K, V to the cache before attention so the kernel
    // sees the full prefix including the current step's contribution.
    kv_cache.append(layer_idx as usize, k_new.view(), v_new.view())?;
    let (k_cached, v_cached) = kv_cache.view(layer_idx as usize)?;

    // Per-layer hybrid attention dispatch. The class falls back to
    // `Global` for `attention_classes = None`, preserving the
    // Qwen3 / Llama behaviour byte-for-byte.
    let ctx = match layer_class {
        AttentionClass::Local { window } => profile::time("tee:attn_swa_cached", || {
            causal_gqa_attention_swa_cached(
                q_new.view(),
                k_cached,
                v_cached,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                q_pos_offset,
                window,
            )
        }),
        AttentionClass::Global => {
            // M1.10.1.2: route Global cached attention through the
            // permutation-shielded path when the per-batch auto-switch
            // engages. The threshold compares against `n_q` (number of
            // NEW Q rows this forward) — at decode shape (n_q=1) the
            // permuted overhead would dominate so we stay in-TEE; at
            // prefill (n_q = n_prompt ≥ threshold) the permuted path
            // engages. Falls back to in-TEE when offload=false or the
            // master switch is off — the M1.3 default behaviour.
            let n_q = q_new.shape()[0];
            if offload && cfg.perm_attention_enabled_for(n_q) {
                profile::time("tee:attn_permuted_cached", || {
                    causal_gqa_attention_permuted_cached(
                        exec,
                        q_new.view(),
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offset,
                    )
                })?
            } else {
                profile::time("tee:attn_cached", || {
                    causal_gqa_attention_cached(
                        q_new.view(),
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offset,
                    )
                })
            }
        }
    };

    // Output projection — fresh mask per the per-offload protocol.
    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present (skip-layers mode)")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    // Pre-FFN RMSNorm.
    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // SwiGLU FFN — same shape, same offload group as the legacy block.
    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present (skip-layers mode)")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

/// **M1.11 R1.3** — Batched forward pass over `B` sequences.
///
/// `input_ids[b]` is sequence b's token IDs. Sequences may differ in
/// length; right-padded internally to `n_max = max(len)`. Returns
/// `(hidden, seq_lens)` where `hidden` has shape
/// `(B, n_max, hidden_size)` (rows past `seq_lens[b]` are valid
/// numerically but represent positions the model never trained on —
/// callers gather the last *valid* row per sequence via `seq_lens`).
///
/// One `begin_prefill_pass(B, n_max)` bracket wraps the whole call;
/// the substrate samples B per-sequence masks (see `m1-11-batched-decode.md`
/// §3.4) and reuses them across every offload in the forward.
pub fn run_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[Vec<u32>],
) -> Result<(Array3<f32>, Vec<usize>)> {
    if input_ids.is_empty() {
        return Err(anyhow!("run_batched: input_ids must be non-empty"));
    }
    let batch_size = input_ids.len();
    let seq_lens: Vec<usize> = input_ids.iter().map(|s| s.len()).collect();
    let n_max = seq_lens.iter().copied().max().unwrap_or(0);
    if n_max == 0 {
        return Err(anyhow!(
            "run_batched: at least one sequence must have length > 0"
        ));
    }
    let d = cfg.hidden_size;

    // (B * n_max, d) flat embedding tensor. Pad rows stay zero.
    let mut h_flat = profile::time("tee:embed_lookup", || {
        let mut h = Array2::<f32>::zeros((batch_size * n_max, d));
        for (b, ids) in input_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                let row = weights.token_embedding.row(id as usize);
                for (j, v) in row.iter().enumerate() {
                    h[[b * n_max + i, j]] = v.to_f32();
                }
            }
        }
        h
    });

    let h_final = with_forward_session(
        exec,
        ForwardSessionShape::PrefillBatch { batch_size, n_max },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h_flat = decoder_block_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h_flat.view(),
                    batch_size,
                    n_max,
                    &seq_lens,
                    cfg.offload_layer(li),
                    None,
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h_flat.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )?;

    // Materialise (B, n_max, hidden) from the flat (B*n_max, hidden)
    // tensor.  Direct copy preserves contiguity guarantees that
    // downstream gather logic relies on.
    let mut out = Array3::<f32>::zeros((batch_size, n_max, d));
    for b in 0..batch_size {
        out.slice_mut(ndarray::s![b, .., ..])
            .assign(&h_final.slice(ndarray::s![b * n_max..(b + 1) * n_max, ..]));
    }
    Ok((out, seq_lens))
}

/// **M1.11 D1.4** — Batched prefill forward that populates the KV
/// cache for subsequent batched decode steps.
///
/// Same shape contract as [`run_batched`]: returns `(B, n_max, hidden)`
/// + per-sequence valid lengths. Additionally, each layer's
/// post-RoPE K/V valid prefix per sequence is appended to `kv_cache`
/// via `kv_cache.append_prefill(li, b, k_b, v_b)`. After this call,
/// `kv_cache.len_b(li, b) == input_ids[b].len()` for every (li, b).
///
/// The caller then runs `run_decode_step_batched` one step at a time
/// with `token_ids: &[u32]` of length B (typically the next sampled
/// token per sequence; finished sequences feed a pad token, which is
/// fine because their output rows are discarded).
pub fn run_prefill_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[Vec<u32>],
    kv_cache: &mut KvCache,
) -> Result<(Array3<f32>, Vec<usize>)> {
    if input_ids.is_empty() {
        return Err(anyhow!("run_prefill_batched: input_ids must be non-empty"));
    }
    let batch_size = input_ids.len();
    assert_eq!(
        batch_size,
        kv_cache.batch_size(),
        "input_ids B={} != kv_cache batch_size={}",
        batch_size,
        kv_cache.batch_size(),
    );
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let seq_lens: Vec<usize> = input_ids.iter().map(|s| s.len()).collect();
    let n_max = seq_lens.iter().copied().max().unwrap_or(0);
    if n_max == 0 {
        return Err(anyhow!(
            "run_prefill_batched: at least one sequence must have length > 0"
        ));
    }
    let d = cfg.hidden_size;

    let mut h_flat = profile::time("tee:embed_lookup", || {
        let mut h = Array2::<f32>::zeros((batch_size * n_max, d));
        for (b, ids) in input_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                let row = weights.token_embedding.row(id as usize);
                for (j, v) in row.iter().enumerate() {
                    h[[b * n_max + i, j]] = v.to_f32();
                }
            }
        }
        h
    });

    let h_final = with_forward_session(
        exec,
        ForwardSessionShape::PrefillBatch { batch_size, n_max },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h_flat = decoder_block_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h_flat.view(),
                    batch_size,
                    n_max,
                    &seq_lens,
                    cfg.offload_layer(li),
                    Some(kv_cache),
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h_flat.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )?;

    // O5 (perm-attn-gpu-offload): build the permuted-cover resident prefix for
    // all GLOBAL layers **now**, at the prefill→decode handoff, so the one-time
    // `build_covered_prefix` cost lands here instead of on the first decode
    // step. Gated on the decode-cover path; the decode block's lazy build
    // remains as an idempotent fallback.
    if gpu_resident_cover_override().unwrap_or_else(|| exec.supports_offloaded_attention()) {
        build_covered_prefix_all_global(cfg, exec, kv_cache, batch_size)?;
    }

    let mut out = Array3::<f32>::zeros((batch_size, n_max, d));
    for b in 0..batch_size {
        out.slice_mut(ndarray::s![b, .., ..])
            .assign(&h_final.slice(ndarray::s![b * n_max..(b + 1) * n_max, ..]));
    }
    Ok((out, seq_lens))
}

/// Batched-prefill decoder block over `(B * n_max, hidden)` rows with
/// per-sequence valid-lengths `seq_lens`. Mirrors [`decoder_block`]
/// except:
///
/// 1. RoPE is applied **per-sequence** (each sequence's row 0 is at
///    absolute position 0).
/// 2. Attention is computed **per-sequence** in-TEE over each
///    sequence's valid prefix `[0..seq_lens[b]]` (R1.3 stopgap;
///    R1.4 ships a batched kernel routed through `engine.fused_attention_batched`).
/// 3. Linear projections (Q/K/V, O, gate/up, down) go through the
///    substrate's `PerSequence` session — `exec.offload_*` calls
///    transparently apply the B per-sequence masks.
///
/// Pad rows (rows `seq_lens[b]..n_max` of sequence b) carry garbage
/// values through the entire forward; they're irrelevant for the
/// caller's last-token gather. We do NOT zero them out per layer — the
/// residual stream just propagates whatever the embedding lookup
/// placed there (zero, in `run_batched`'s case).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn decoder_block_batched(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    batch_size: usize,
    n_max: usize,
    seq_lens: &[usize],
    offload: bool,
    // M1.11 D1.4: when `Some`, append each sequence's post-RoPE K/V
    // valid prefix to the cache so subsequent `run_decode_step_batched`
    // calls can continue from this prefill state.
    kv_cache_prefill: Option<&mut KvCache>,
) -> Result<Array2<f32>> {
    debug_assert_eq!(hidden.nrows(), batch_size * n_max);

    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V — under PerSequence session, `offload_qkv` falls through
    // to 3 `offload_linear` calls each running the per-sequence
    // path (substrate handles slicing).
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present")
                        .view(),
                ),
            )
        })
    };

    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // Per-sequence RoPE — each sequence's row i sits at absolute
    // position i (start_pos = 0).
    profile::time("tee:rope", || {
        for b in 0..batch_size {
            let q_block = q.slice_mut(ndarray::s![b * n_max..(b + 1) * n_max, ..]);
            rope.apply(q_block, cfg.num_attention_heads);
            let k_block = k.slice_mut(ndarray::s![b * n_max..(b + 1) * n_max, ..]);
            rope.apply(k_block, cfg.num_key_value_heads);
        }
    });

    // M1.11 D1.4 — when running as part of a batched prefill that
    // initialises the decode KV cache, append each sequence's
    // post-RoPE K/V valid prefix.  Pad rows are *not* appended; the
    // KV cache stores only real-token K/V per sequence.
    if let Some(kv_cache) = kv_cache_prefill {
        for b in 0..batch_size {
            let valid_n = seq_lens[b];
            if valid_n == 0 {
                continue;
            }
            let k_b = k.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            let v_b = v.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            kv_cache.append_prefill(layer_idx as usize, b, k_b, v_b)?;
        }
    }

    // Per-sequence attention via B serial in-TEE `causal_gqa_attention`
    // calls, one per sequence's valid prefix. Pad rows get a zero
    // context (their residual stream is unchanged beyond the FFN
    // residual).
    //
    // TODO(R1.4): replace this loop with a single dispatch through
    // `engine.fused_attention_batched` — reshape Q/K/V to per-head-
    // batched `(B·num_heads, n_q, d_head)`, fold per-sequence right-
    // padding + causal into one additive `(B·num_heads, n_q, n_kv)`
    // mask, dispatch once. Only land this when the
    // `tee:attn_inplace_many` bucket grows past ~10% of batched wall
    // (Qwen3-Reranker-0.6B at B=8 is currently 3.4% — not worth the
    // engineering). The trigger is longer per-sequence n_max (cross-
    // encoder rerank with longer docs, batched extraction with
    // longer chunks, or D-phase generate_batched). See M1.11 plan
    // §3.2 and `docs/handoffs/2026-05-21-attn-offload-spike.md` for
    // the kernel-routing reasoning.
    let q_dim = cfg.num_attention_heads * cfg.head_dim_value();
    let mut ctx = Array2::<f32>::zeros((batch_size * n_max, q_dim));

    // Phase-6 prefill offload (perm-attn-gpu-offload): route GLOBAL-layer
    // prefill self-attention through the fused `cubek` kernel under the
    // feature-rotation cover, instead of the in-TEE B-loop. Gated default-off
    // (production unchanged); the rotation cover is session-fixed per layer
    // and exactly corrected in-TEE by `O_vᵀ`, so output matches in-TEE at the
    // fp16 floor. PERF wire only — the cover does not clear `WEIGHTS-PUB`.
    let layer_class = cfg.effective_attention_class(layer_idx as usize);
    let use_gpu_prefill = gpu_prefill_offload_override()
        .unwrap_or_else(|| exec.supports_offloaded_attention())
        && matches!(layer_class, AttentionClass::Global);
    if use_gpu_prefill {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        const PREFILL_SALT: u64 = 0xC0FFEE_BEEF;
        let (nqh, nkvh, dh) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        );
        let scale = 1.0_f32 / (dh as f32).sqrt();
        profile::time("tee:attn_prefill_offload", || -> Result<()> {
            // Session-fixed shared cover per layer (re-derived from a per-layer
            // seed; shared O across heads → GQA-broadcast-consistent).
            let mut crng = ChaCha20Rng::seed_from_u64(PREFILL_SALT ^ layer_idx as u64);
            let o_qk = sample_orthogonal(dh, &mut crng);
            let (c_v, c_v_inv) = sample_value_cover(dh, cover_kappa(), &mut crng);
            let group = nqh / nkvh;
            // Loop-batching: when every sequence is full length (the common
            // case — padded prompts share `n_max`), fold all B into ONE cubek
            // dispatch instead of B per-sequence calls (drops per-call launch +
            // convert/upload overhead ×B). Ragged batches fall back per-seq.
            let uniform = batch_size > 0 && seq_lens.iter().all(|&l| l == n_max);
            if uniform {
                // Fold + rotate all B sequences (in-TEE), K/V un-replicated at
                // Hkv (O2); the GPU sees only `Q·O_qk`, `K·O_qk`, `V·O_v` and
                // does the GQA broadcast on-device.
                let (q_cov, k_cov, v_cov) = profile::time("prefill_cover:rotate_tee", || {
                    let q_f = fold_heads_2d_batched(q.view(), batch_size, n_max, nqh, dh);
                    let k_f = fold_heads_2d_batched(k.view(), batch_size, n_max, nkvh, dh);
                    let v_f = fold_heads_2d_batched(v.view(), batch_size, n_max, nkvh, dh);
                    (
                        rotate_heads(q_f.view(), o_qk.view()),
                        rotate_heads(k_f.view(), o_qk.view()),
                        rotate_heads(v_f.view(), c_v.view()),
                    )
                });
                // cubek dispatched **per-sequence** (slice the batched covered
                // operands, gather into one (B·Hq,n,d) buffer for the batched
                // correction). Counter to the usual "batch bigger" rule, a
                // controlled warm A/B (`cubek_prefill_cover::cubek_dispatch_granularity`)
                // measures per-seq ~1.48× faster than one bh=B·Hq dispatch — an
                // artifact of cubek's *materialised* GQA expand (`repeat_dim`
                // builds a ~268 MB expanded K/V for the big dispatch vs small
                // reused per-seq ones). The principled fix (cubek kv-head
                // read-index, no materialisation) would let the single dispatch
                // win; until then, per-seq is the measured optimum.
                let ctx_raw = profile::time("prefill_cover:cubek_gpu", || -> Result<Array3<f32>> {
                    let mut ctx_raw_all = Array3::<f32>::zeros((batch_size * nqh, n_max, dh));
                    for b in 0..batch_size {
                        let qb = q_cov.slice(ndarray::s![b * nqh..(b + 1) * nqh, .., ..]);
                        let kb = k_cov.slice(ndarray::s![b * nkvh..(b + 1) * nkvh, .., ..]);
                        let vb = v_cov.slice(ndarray::s![b * nkvh..(b + 1) * nkvh, .., ..]);
                        let cb = exec.cubek_causal_attend(qb, kb, vb, group, scale)?;
                        ctx_raw_all
                            .slice_mut(ndarray::s![b * nqh..(b + 1) * nqh, .., ..])
                            .assign(&cb);
                    }
                    Ok(ctx_raw_all)
                })?;
                // Fused `O_vᵀ` correct + unfold straight into `ctx` (no
                // intermediate (B·Hq,n,d) array, no separate unfold copy).
                profile::time("prefill_cover:correct_tee", || {
                    correct_unfold_into(&mut ctx, ctx_raw.view(), c_v_inv.view(), n_max, nqh, dh);
                });
            } else {
                for b in 0..batch_size {
                    let valid_n = seq_lens[b];
                    if valid_n == 0 {
                        continue;
                    }
                    let q_b = q.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                    let k_b = k.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                    let v_b = v.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                    let (q_cov, k_cov, v_cov) = profile::time("prefill_cover:rotate_tee", || {
                        let q_f = fold_heads_2d(q_b, nqh, dh);
                        let k_f = fold_heads_2d(k_b, nkvh, dh);
                        let v_f = fold_heads_2d(v_b, nkvh, dh);
                        (
                            rotate_heads(q_f.view(), o_qk.view()),
                            rotate_heads(k_f.view(), o_qk.view()),
                            rotate_heads(v_f.view(), c_v.view()),
                        )
                    });
                    let ctx_raw = profile::time("prefill_cover:cubek_gpu", || {
                        exec.cubek_causal_attend(q_cov.view(), k_cov.view(), v_cov.view(), group, scale)
                    })?;
                    let ctx_b = profile::time("prefill_cover:correct_tee", || {
                        let ctx_cov = rotate_heads(ctx_raw.view(), c_v_inv.view());
                        unfold_heads_2d(ctx_cov.view(), nqh, dh)
                    });
                    ctx.slice_mut(ndarray::s![b * n_max..b * n_max + valid_n, ..])
                        .assign(&ctx_b);
                }
            }
            Ok(())
        })?;
        if std::env::var("GELO_DEBUG_OFFLOAD").is_ok() {
            let nfin = |a: &Array2<f32>| a.iter().filter(|x| !x.is_finite()).count();
            let m = ctx.iter().fold(0f32, |a, &x| a.max(x.abs()));
            let nf = ctx.iter().filter(|x| !x.is_finite()).count();
            eprintln!(
                "[DEBUG-off] L{layer_idx} b={batch_size} n_max={n_max} \
                 q_nf={} k_nf={} v_nf={} | ctx_max={m:.2e} ctx_nonfinite={nf}",
                nfin(&q), nfin(&k), nfin(&v)
            );
        }
    } else {
        profile::time("tee:attn_inplace_many", || {
            for b in 0..batch_size {
                let valid_n = seq_lens[b];
                if valid_n == 0 {
                    continue;
                }
                let q_b = q.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                let k_b = k.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                let v_b = v.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
                let ctx_b = causal_gqa_attention(
                    q_b,
                    k_b,
                    v_b,
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim_value(),
                );
                ctx.slice_mut(ndarray::s![b * n_max..b * n_max + valid_n, ..])
                    .assign(&ctx_b);
            }
        });
    }

    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present")
                        .view(),
                ),
            )
        })
    };
    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));
    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

#[cfg(test)]
mod prefill_offload_tests {
    use super::*;
    use crate::decoder::attention::causal_gqa_attention;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    /// f32 CPU folded causal attention over `(H, n, d)` — a cubek stand-in
    /// so the cover/fold/unfold path can be checked exactly in-TEE, with no
    /// GPU and no fp16. Query row i attends keys 0..=i.
    fn cpu_folded_causal(
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        scale: f32,
    ) -> Array3<f32> {
        let (h, n, d) = q.dim();
        let mut out = Array3::<f32>::zeros((h, n, d));
        for hi in 0..h {
            for i in 0..n {
                let mut scores = vec![f32::NEG_INFINITY; i + 1];
                let mut mx = f32::NEG_INFINITY;
                for (j, s) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for c in 0..d {
                        acc += q[(hi, i, c)] * k[(hi, j, c)];
                    }
                    *s = acc * scale;
                    mx = mx.max(*s);
                }
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for c in 0..d {
                    let mut acc = 0.0;
                    for (j, &w) in scores.iter().enumerate() {
                        acc += w * v[(hi, j, c)];
                    }
                    out[(hi, i, c)] = acc / sum;
                }
            }
        }
        out
    }

    /// The prefill-offload cover path (fold → GQA-expand → rotate →
    /// [folded causal attend] → `O_vᵀ` → unfold) must reproduce the
    /// production in-TEE `causal_gqa_attention` at the f32 floor. This pins
    /// the fold/expand/unfold indexing + the orthogonal cover cancellation
    /// (the cubek fp16 step itself is covered by `cubek_folded_causal_parity`).
    #[test]
    fn cover_prefill_matches_in_tee() {
        let (hq, hkv, d, n) = (8usize, 2usize, 16usize, 12usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut rng = ChaCha20Rng::seed_from_u64(0xA11CE);
        let mk = |cols: usize, rng: &mut ChaCha20Rng| {
            Array2::<f32>::from_shape_fn((n, cols), |_| {
                let z: f32 = StandardNormal.sample(rng);
                z * 0.1
            })
        };
        let q_b = mk(hq * d, &mut rng);
        let k_b = mk(hkv * d, &mut rng);
        let v_b = mk(hkv * d, &mut rng);

        let in_tee = causal_gqa_attention(q_b.view(), k_b.view(), v_b.view(), hq, hkv, d);

        let mut crng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
        let o_qk = sample_orthogonal(d, &mut crng);
        let o_v = sample_orthogonal(d, &mut crng);
        let group = hq / hkv;
        // Production folds + rotates K/V un-replicated (Hkv); the engine
        // broadcasts to Hq on-device. Mirror that here: fold/rotate Hkv, then
        // CPU-expand to Hq for the cubek stand-in.
        let q_f = fold_heads_2d(q_b.view(), hq, d);
        let k_f = fold_heads_2d(k_b.view(), hkv, d);
        let v_f = fold_heads_2d(v_b.view(), hkv, d);
        let q_cov = rotate_heads(q_f.view(), o_qk.view());
        let k_cov = rotate_heads(k_f.view(), o_qk.view());
        let v_cov = rotate_heads(v_f.view(), o_v.view());
        let expand3 = |t: &Array3<f32>| -> Array3<f32> {
            let (h, nn, dd) = t.dim();
            let mut e = Array3::<f32>::zeros((h * group, nn, dd));
            for qh in 0..h * group {
                let kvh = qh / group;
                for j in 0..nn {
                    for c in 0..dd {
                        e[(qh, j, c)] = t[(kvh, j, c)];
                    }
                }
            }
            e
        };
        let k_exp = expand3(&k_cov);
        let v_exp = expand3(&v_cov);
        let ctx_raw = cpu_folded_causal(q_cov.view(), k_exp.view(), v_exp.view(), scale);
        let ctx_cov = rotate_heads(ctx_raw.view(), o_v.t());
        let cover = unfold_heads_2d(ctx_cov.view(), hq, d);

        let max_abs = in_tee
            .iter()
            .zip(cover.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "cover prefill path diverged from in-TEE: max_abs={max_abs:.6}"
        );
    }

    /// The **batched** (loop-batched) prefill path — `fold_heads_2d_batched`
    /// (b-major) → leading-dim GQA expand (as cubek does) → `correct_unfold_into`
    /// — must reproduce per-sequence in-TEE `causal_gqa_attention` for every
    /// sequence of a uniform-length batch, at the f32 floor.
    #[test]
    fn cover_prefill_batched_matches_in_tee() {
        let (b, hq, hkv, d, n) = (3usize, 8usize, 2usize, 16usize, 12usize);
        let group = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let mut rng = ChaCha20Rng::seed_from_u64(0xB00B5);
        let mk = |cols: usize, rng: &mut ChaCha20Rng| {
            Array2::<f32>::from_shape_fn((b * n, cols), |_| {
                let z: f32 = StandardNormal.sample(rng);
                z * 0.1
            })
        };
        let q = mk(hq * d, &mut rng);
        let k = mk(hkv * d, &mut rng);
        let v = mk(hkv * d, &mut rng);

        // Reference: per-sequence in-TEE.
        let mut in_tee = Array2::<f32>::zeros((b * n, hq * d));
        for bi in 0..b {
            let qb = q.slice(ndarray::s![bi * n..(bi + 1) * n, ..]);
            let kb = k.slice(ndarray::s![bi * n..(bi + 1) * n, ..]);
            let vb = v.slice(ndarray::s![bi * n..(bi + 1) * n, ..]);
            in_tee
                .slice_mut(ndarray::s![bi * n..(bi + 1) * n, ..])
                .assign(&causal_gqa_attention(qb, kb, vb, hq, hkv, d));
        }

        // Batched cover path.
        let mut crng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
        let o_qk = sample_orthogonal(d, &mut crng);
        let o_v = sample_orthogonal(d, &mut crng);
        let q_cov = rotate_heads(fold_heads_2d_batched(q.view(), b, n, hq, d).view(), o_qk.view());
        let k_cov = rotate_heads(fold_heads_2d_batched(k.view(), b, n, hkv, d).view(), o_qk.view());
        let v_cov = rotate_heads(fold_heads_2d_batched(v.view(), b, n, hkv, d).view(), o_v.view());
        // GQA expand exactly as cubek's leading-dim group repeat: folded q-head
        // b*Hq+qh reads kv head b*Hkv + qh/group.
        let expand_b = |t: &Array3<f32>| -> Array3<f32> {
            let mut e = Array3::<f32>::zeros((b * hq, n, d));
            for fh in 0..b * hq {
                let bi = fh / hq;
                let qh = fh % hq;
                let src = bi * hkv + qh / group;
                for j in 0..n {
                    for c in 0..d {
                        e[(fh, j, c)] = t[(src, j, c)];
                    }
                }
            }
            e
        };
        let ctx_raw = cpu_folded_causal(q_cov.view(), expand_b(&k_cov).view(), expand_b(&v_cov).view(), scale);
        let mut cover = Array2::<f32>::zeros((b * n, hq * d));
        correct_unfold_into(&mut cover, ctx_raw.view(), o_v.t(), n, hq, d);

        let max_abs = in_tee
            .iter()
            .zip(cover.iter())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "batched cover prefill path diverged from in-TEE: max_abs={max_abs:.6}"
        );
    }
}

fn decoder_block(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
) -> Result<Array2<f32>> {
    // Pre-attention RMSNorm.
    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V projections — offloaded one mask shared across the three reads,
    // matching the BERT path.
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    // Qwen3 QK-norm — per-head RMSNorm on Q and K **before** RoPE.
    // No-op when the loaded checkpoint lacks `q_norm` / `k_norm`
    // (Qwen2 / LLaMA / Mistral).
    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE rotates Q and K only (V left alone) per-head.
    profile::time("tee:rope", || {
        rope.apply(q.view_mut(), cfg.num_attention_heads);
        rope.apply(k.view_mut(), cfg.num_key_value_heads);
    });

    // GQA + causal attention. When this layer is offloaded **and** the
    // auto-switch fires (sequence length ≥ threshold; see
    // `DecoderConfig::out_attn_mult_enabled_for`), route per-head Q·Kᵀ
    // through TwinShield OutAttnMult. Otherwise compute attention inside
    // the TEE — equally confidential (Q, K never cross PCIe), and faster
    // at short n where the 4-partition scheme's 4× FLOP widening loses
    // to a plain in-TEE matmul.
    let n = q.shape()[0];
    let ctx = if offload && cfg.out_attn_mult_enabled_for(n) {
        causal_gqa_attention_with_offload(
            exec,
            q.view(),
            k.view(),
            v.view(),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        )?
    } else if offload && cfg.perm_attention_enabled_for(n) {
        causal_gqa_attention_permuted(
            exec,
            q.view(),
            k.view(),
            v.view(),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        )?
    } else {
        profile::time("tee:attn_inplace", || {
            causal_gqa_attention(
                q.view(),
                k.view(),
                v.view(),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
            )
        })
    };

    // Output projection — fresh mask.
    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present (skip-layers mode)")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    // Pre-FFN RMSNorm.
    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // SwiGLU FFN: gate + up share the same input `h1_norm`, so one
    // `offload_linear_many` call shares the mask apply + batches the
    // matmul + batches the unapply across both projections.
    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present (skip-layers mode)")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}
