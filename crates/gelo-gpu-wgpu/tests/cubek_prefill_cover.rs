//! Prefill-attention offload via `cubek-attention` WITH the
//! feature-rotation cover, benchmarked against the in-TEE baseline.
//!
//! This file delivers three things (all `#[ignore]`, release-only — they
//! run against a real Vulkan/wgpu device and pay a one-time cubecl shader
//! compile):
//!
//! 1. `cubek_folded_causal_parity` — proves `cubek_attention_folded` with
//!    `causal=true` matches a CPU causal softmax-attention reference at the
//!    fp16 floor, AND that the full feature-rotation cover pipeline
//!    (rotate → GQA-expand → cubek → O_vᵀ-correct) round-trips back to the
//!    uncovered cubek result at the fp16 floor.
//!
//! 2. `prefill_attention_breakdown` — the headline measurement. At Qwen3-4B
//!    GQA layout (Hq=32, Hkv=8, d=128), B=8, prefill shapes n ∈ {2048,
//!    8192}: in-TEE `causal_gqa_attention` (B loop) vs the end-to-end
//!    cubek+cover secure path, with the cover overhead decomposed into
//!    {rotation+GQA-expand TEE, cubek GPU, O_vᵀ correction TEE}.
//!
//! Run:
//! ```text
//! cargo test --release -p gelo-gpu-wgpu --test cubek_prefill_cover \
//!   -- --ignored --nocapture
//! ```

use gelo_embedder::decoder::attention::causal_gqa_attention;
use gelo_gpu_wgpu::{cubek_attention_folded, cubek_attention_folded_gqa};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::time::Instant;

// Qwen3-4B GQA layout.
const HQ: usize = 32;
const HKV: usize = 8;
const D: usize = 128;
const GROUP: usize = HQ / HKV; // 4
const B: usize = 8;

// ─── helpers ──────────────────────────────────────────────────────────

/// Haar-ish orthogonal `d×d` matrix via Gram-Schmidt on a Gaussian.
/// (Replicates `gelo_protocol`'s `sample_haar_orthogonal`, which is
/// `pub(crate)`; modified Gram-Schmidt is sufficient for a cover that we
/// fully undo with the transpose.)
fn sample_orthogonal<R: Rng>(d: usize, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut a = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(rng));
    // Modified Gram-Schmidt over columns.
    for j in 0..d {
        // Orthogonalise column j against previous columns.
        for i in 0..j {
            let (col_i, col_j) = {
                let ci = a.column(i).to_owned();
                let cj = a.column(j).to_owned();
                (ci, cj)
            };
            let proj: f32 = col_i.dot(&col_j);
            let mut cj = a.column_mut(j);
            for k in 0..d {
                cj[k] = col_j[k] - proj * col_i[k];
            }
        }
        // Normalise.
        let norm = a.column(j).dot(&a.column(j)).sqrt();
        let mut cj = a.column_mut(j);
        for k in 0..d {
            cj[k] /= norm;
        }
    }
    a
}

/// Per-head causal CPU reference over `[B*Hq, n, d]` folded layout.
/// Query i attends keys j ≤ i. Scale = 1/sqrt(d).
fn ref_causal_folded(
    q: &Array3<f32>,
    k: &Array3<f32>,
    v: &Array3<f32>,
    scale: f32,
) -> Array3<f32> {
    let bh = q.shape()[0];
    let n_q = q.shape()[1];
    let n_kv = k.shape()[1];
    let d = q.shape()[2];
    let mut out = Array3::<f32>::zeros((bh, n_q, d));
    for h in 0..bh {
        for i in 0..n_q {
            // causal: query row i (position i in a square prefill) attends
            // keys 0..=i. With n_q == n_kv this is the standard mask.
            let last = i.min(n_kv - 1);
            let mut scores = vec![f32::NEG_INFINITY; n_kv];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=last {
                let mut acc = 0.0_f32;
                for c in 0..d {
                    acc += q[[h, i, c]] * k[[h, j, c]];
                }
                let s = acc * scale;
                scores[j] = s;
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0_f32;
            for j in 0..=last {
                scores[j] = (scores[j] - max).exp();
                sum += scores[j];
            }
            for c in 0..d {
                let mut acc = 0.0_f32;
                for j in 0..=last {
                    acc += scores[j] * v[[h, j, c]];
                }
                out[[h, i, c]] = acc / sum;
            }
        }
    }
    out
}

/// Synthetic per-sequence Q/K/V at Qwen3-4B layout, like the bench helper.
/// Q_b: (n, Hq*d); K_b/V_b: (n, Hkv*d).
fn make_qkv(n: usize, seed: u64) -> (Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut qs = Vec::with_capacity(B);
    let mut ks = Vec::with_capacity(B);
    let mut vs = Vec::with_capacity(B);
    for _ in 0..B {
        let q = Array2::from_shape_fn((n, HQ * D), |_| rng.random::<f32>() * 0.1 - 0.05);
        let k = Array2::from_shape_fn((n, HKV * D), |_| rng.random::<f32>() * 0.1 - 0.05);
        let v = Array2::from_shape_fn((n, HKV * D), |_| rng.random::<f32>() * 0.1 - 0.05);
        qs.push(q);
        ks.push(k);
        vs.push(v);
    }
    (qs, ks, vs)
}

/// Fold per-sequence (Q,K,V) into [B*Hq, n, d] with GQA-expansion of K/V.
fn fold_expand(
    qs: &[Array2<f32>],
    ks: &[Array2<f32>],
    vs: &[Array2<f32>],
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let n_q = qs[0].nrows();
    let n_kv = ks[0].nrows();
    let bh = B * HQ;
    let mut q = Array3::<f32>::zeros((bh, n_q, D));
    let mut k = Array3::<f32>::zeros((bh, n_kv, D));
    let mut v = Array3::<f32>::zeros((bh, n_kv, D));
    for b in 0..B {
        for qh in 0..HQ {
            let kvh = qh / GROUP;
            let q_off = qh * D;
            let kv_off = kvh * D;
            let idx = b * HQ + qh;
            q.index_axis_mut(Axis(0), idx)
                .assign(&qs[b].slice(ndarray::s![.., q_off..q_off + D]));
            k.index_axis_mut(Axis(0), idx)
                .assign(&ks[b].slice(ndarray::s![.., kv_off..kv_off + D]));
            v.index_axis_mut(Axis(0), idx)
                .assign(&vs[b].slice(ndarray::s![.., kv_off..kv_off + D]));
        }
    }
    (q, k, v)
}

/// κ-bounded invertible value cover `C_v = U·diag(s)·Vᵀ` (cond=κ, log-uniform
/// `s` ⇒ geometric mean 1) and its exact inverse. Mirrors forward.rs
/// `sample_value_cover`. κ=1 ⇒ orthogonal.
fn sample_cv<R: Rng>(d: usize, kappa: f32, rng: &mut R) -> (Array2<f32>, Array2<f32>) {
    if kappa <= 1.0 {
        let o = sample_orthogonal(d, rng);
        let oi = o.t().to_owned();
        return (o, oi);
    }
    let u = sample_orthogonal(d, rng);
    let vt = sample_orthogonal(d, rng);
    let (hi, lo) = (kappa.sqrt(), 1.0 / kappa.sqrt());
    let (lhi, llo) = (hi.ln(), lo.ln());
    let mut s = vec![1.0f32; d];
    s[0] = hi;
    s[d - 1] = lo;
    for sj in s.iter_mut().take(d - 1).skip(1) {
        *sj = (llo + rng.random::<f32>() * (lhi - llo)).exp(); // log-uniform
    }
    let mut us = u.clone();
    let mut vi = vt.t().to_owned();
    for j in 0..d {
        for i in 0..d {
            us[(i, j)] *= s[j];
            vi[(i, j)] /= s[j];
        }
    }
    (us.dot(&vt), vi.dot(&u.t()))
}

fn rel_max(a: &Array3<f32>, ref_: &Array3<f32>) -> f32 {
    let num = a
        .iter()
        .zip(ref_.iter())
        .map(|(x, r)| (x - r).abs())
        .fold(0.0_f32, f32::max);
    let den = ref_.iter().map(|r| r.abs()).fold(0.0_f32, f32::max);
    num / den.max(1e-12)
}

/// DETERMINISTIC same-process κ-sweep: does the `C_v` value cover degrade the
/// cubek fp16 round-trip as κ grows? Compares covered→cubek→corrected against
/// (i) uncovered cubek (isolates cover-induced fp16 error) and (ii) the CPU
/// f32 causal reference (absolute faithfulness), at several operand scales.
/// One process ⇒ autotune cached ⇒ clean κ comparison (no cross-process /
/// autoregressive confound).
#[test]
#[ignore = "real Vulkan device; the C_v fp16-drift diagnosis loop"]
fn cv_cover_fp16_drift_sweep() {
    let (b, hq, hkv, n, d) = (2usize, 4usize, 1usize, 256usize, D);
    let group = hq / hkv;
    let bh = b * hq;
    let scale = 1.0 / (d as f32).sqrt();

    // Sweep operand scale: 0.05 (the legacy synthetic), and realistic O(1)
    // (post-qk-norm Q/K rows are ~unit-RMS; V is unnormalised).
    for amp in [0.05_f32, 1.0_f32] {
        let mut rng = ChaCha20Rng::seed_from_u64(0x9A917E);
        let mk = |rng: &mut ChaCha20Rng| {
            Array3::from_shape_fn((bh, n, d), |_| (rng.random::<f32>() - 0.5) * 2.0 * amp)
        };
        let q = mk(&mut rng);
        let k = mk(&mut rng);
        let v = mk(&mut rng);

        let uncov = cubek_attention_folded(q.view(), k.view(), v.view(), scale, true);
        let cpu = ref_causal_folded(&q, &k, &v, scale);
        let cubek_vs_cpu = rel_max(&uncov, &cpu);
        println!(
            "\n=== amp={amp}  (cubek vs CPU baseline rel_max={cubek_vs_cpu:.2e}) ==="
        );
        println!(
            "{:>6} {:>16} {:>16} {:>10}",
            "kappa", "cover_vs_uncov", "cover_vs_cpu", "cond"
        );

        for kappa in [1.0_f32, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0] {
            let mut orng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
            let o_qk: Vec<Array2<f32>> =
                (0..hkv).map(|_| sample_orthogonal(d, &mut orng)).collect();
            let cv: Vec<(Array2<f32>, Array2<f32>)> =
                (0..hkv).map(|_| sample_cv(d, kappa, &mut orng)).collect();

            let mut q_rot = Array3::<f32>::zeros((bh, n, d));
            let mut k_rot = Array3::<f32>::zeros((bh, n, d));
            let mut v_rot = Array3::<f32>::zeros((bh, n, d));
            for fh in 0..bh {
                let kvh = (fh % hq) / group;
                q_rot
                    .index_axis_mut(Axis(0), fh)
                    .assign(&q.index_axis(Axis(0), fh).dot(&o_qk[kvh]));
                k_rot
                    .index_axis_mut(Axis(0), fh)
                    .assign(&k.index_axis(Axis(0), fh).dot(&o_qk[kvh]));
                v_rot
                    .index_axis_mut(Axis(0), fh)
                    .assign(&v.index_axis(Axis(0), fh).dot(&cv[kvh].0));
            }
            let ctx_raw =
                cubek_attention_folded(q_rot.view(), k_rot.view(), v_rot.view(), scale, true);
            let mut ctx = Array3::<f32>::zeros((bh, n, d));
            for fh in 0..bh {
                let kvh = (fh % hq) / group;
                ctx.index_axis_mut(Axis(0), fh)
                    .assign(&ctx_raw.index_axis(Axis(0), fh).dot(&cv[kvh].1));
            }
            let cover_drift = rel_max(&ctx, &uncov);
            println!(
                "{:>6} {:>16.2e} {:>16.2e} {:>10.1}",
                kappa,
                cover_drift,
                rel_max(&ctx, &cpu),
                kappa
            );
            // Regression lock: the C_v cover round-trip through fp16 must stay
            // near the fp16 floor and grow sub-linearly in κ (NOT ∝cond). This
            // is the exoneration of C_v as a source of accuracy drift
            // (diagnosed 2026-06-02). κ≤8 is the production-relevant band.
            if kappa <= 8.0 {
                assert!(
                    cover_drift < 5e-3,
                    "C_v cover fp16 drift regressed at κ={kappa}, amp={amp}: \
                     rel_max={cover_drift:.2e} (expected < 5e-3 ≈ fp16 floor)"
                );
            }
        }
    }
}

/// DIAGNOSIS LOOP (2026-06-02): does `cubek_attention_folded_gqa` (the on-device
/// GQA-broadcast variant the prefill offload calls per-sequence) produce NaN at
/// certain sequence lengths? The offload degenerated to `!!!!` on a b=1/n=23
/// prompt (NaN in L0 ctx); the kernel never sees batch size, so it must be
/// n-dependent. Sweep n at the Qwen3-4B per-seq shape (Hq=32, Hkv=8, d=128,
/// group=4), random unit-scale operands, causal — report NaN count.
#[test]
#[ignore = "real Vulkan device; cubek GQA NaN-vs-n diagnosis"]
fn cubek_gqa_nan_sweep() {
    let (hq, hkv, d) = (32usize, 8usize, D);
    let group = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let n = 24usize;
    println!("n={n}  {:>6} {:>10} {:>12}", "amp", "nan/inf", "max_abs");
    for amp in [0.5f32, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0] {
        let mut rng = ChaCha20Rng::seed_from_u64(0x5EED);
        let mk = |rng: &mut ChaCha20Rng, h: usize| {
            Array3::from_shape_fn((h, n, d), |_| (rng.random::<f32>() - 0.5) * 2.0 * amp)
        };
        let q = mk(&mut rng, hq);
        let k = mk(&mut rng, hkv);
        let v = mk(&mut rng, hkv);
        let out = cubek_attention_folded_gqa(q.view(), k.view(), v.view(), group, scale, true);
        let nf = out.iter().filter(|x| !x.is_finite()).count();
        let mx = out.iter().filter(|x| x.is_finite()).fold(0f32, |a, &x| a.max(x.abs()));
        println!("    {amp:>6.1} {nf:>10} {mx:>12.3e}{}", if nf > 0 { "   <-- NaN/Inf" } else { "" });
    }
}

// ─── Deliverable 2: parity ──────────────────────────────────────────────

#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `--release -- --ignored`"]
fn cubek_folded_causal_parity() {
    // Small synthetic problem: B=2, Hq=4, Hkv=1 → folded bh = 8.
    let b = 2usize;
    let hq = 4usize;
    let n = 24usize;
    let d = D;
    let scale = 1.0 / (d as f32).sqrt();
    let bh = b * hq;

    let mut rng = ChaCha20Rng::seed_from_u64(0x9A917Eu64);
    let q = Array3::from_shape_fn((bh, n, d), |_| rng.random::<f32>() * 0.1 - 0.05);
    let k = Array3::from_shape_fn((bh, n, d), |_| rng.random::<f32>() * 0.1 - 0.05);
    let v = Array3::from_shape_fn((bh, n, d), |_| rng.random::<f32>() * 0.1 - 0.05);

    // (a) cubek causal vs CPU causal reference.
    let got = cubek_attention_folded(q.view(), k.view(), v.view(), scale, true);
    let reference = ref_causal_folded(&q, &k, &v, scale);
    let max_abs = got
        .iter()
        .zip(reference.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0_f32, f32::max);
    println!("[parity] cubek causal vs CPU causal: max_abs={max_abs:.6}");
    assert!(
        max_abs < 5e-2,
        "cubek causal drift too large: {max_abs:.6} (expected < 5e-2)"
    );

    // (b) full cover round-trip: rotated+cubek+O_vᵀ-corrected ≈ uncovered cubek.
    // One O_qk per kv-head, one O_v per kv-head. Here hkv = 1.
    let hkv = 1usize;
    let group = hq / hkv;
    let mut orng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
    let o_qk: Vec<Array2<f32>> = (0..hkv).map(|_| sample_orthogonal(d, &mut orng)).collect();
    let o_v: Vec<Array2<f32>> = (0..hkv).map(|_| sample_orthogonal(d, &mut orng)).collect();

    // Rotate folded heads. Folded index = b*hq + qh; its kv-head = qh/group.
    let mut q_rot = Array3::<f32>::zeros((bh, n, d));
    let mut k_rot = Array3::<f32>::zeros((bh, n, d));
    let mut v_rot = Array3::<f32>::zeros((bh, n, d));
    for fh in 0..bh {
        let qh = fh % hq;
        let kvh = qh / group;
        q_rot
            .index_axis_mut(Axis(0), fh)
            .assign(&q.index_axis(Axis(0), fh).dot(&o_qk[kvh]));
        k_rot
            .index_axis_mut(Axis(0), fh)
            .assign(&k.index_axis(Axis(0), fh).dot(&o_qk[kvh]));
        v_rot
            .index_axis_mut(Axis(0), fh)
            .assign(&v.index_axis(Axis(0), fh).dot(&o_v[kvh]));
    }
    let ctx_raw = cubek_attention_folded(q_rot.view(), k_rot.view(), v_rot.view(), scale, true);
    // Correct: ctx = ctx_raw · O_vᵀ.
    let mut ctx = Array3::<f32>::zeros((bh, n, d));
    for fh in 0..bh {
        let qh = fh % hq;
        let kvh = qh / group;
        ctx.index_axis_mut(Axis(0), fh)
            .assign(&ctx_raw.index_axis(Axis(0), fh).dot(&o_v[kvh].t()));
    }
    let cover_max_abs = ctx
        .iter()
        .zip(got.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0_f32, f32::max);
    println!("[parity] cover round-trip vs uncovered cubek: max_abs={cover_max_abs:.6}");
    assert!(
        cover_max_abs < 5e-2,
        "cover round-trip drift too large: {cover_max_abs:.6} (expected < 5e-2)"
    );
    println!("[parity] PASS");
}

// ─── Deliverable 3: prefill bench ───────────────────────────────────────

fn run_prefill_cell(n: usize) {
    let scale = 1.0 / (D as f32).sqrt();
    println!("\n=== prefill n={n}  B={B}  Hq={HQ} Hkv={HKV} d={D} ===");

    let (qs, ks, vs) = make_qkv(n, 0xDEAD_BEEF ^ n as u64);

    // (a) IN-TEE baseline: B-loop of causal_gqa_attention (rayon over B,
    // each call itself parallelises over heads). Warm once, then time.
    let in_tee = || -> Vec<Array2<f32>> {
        (0..B)
            .into_par_iter()
            .map(|b| {
                causal_gqa_attention(qs[b].view(), ks[b].view(), vs[b].view(), HQ, HKV, D)
            })
            .collect()
    };
    let _warm = in_tee();
    let t = Instant::now();
    let tee_out = in_tee();
    let in_tee_ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&tee_out);

    // (b) CUBEK + FEATURE-ROTATION COVER, end-to-end.
    // Sample covers: one O_qk per kv-head (HKV), one O_v per kv-head (HKV).
    let mut orng = ChaCha20Rng::seed_from_u64(0xC0FFEE ^ n as u64);
    let o_qk: Vec<Array2<f32>> = (0..HKV).map(|_| sample_orthogonal(D, &mut orng)).collect();
    let o_v: Vec<Array2<f32>> = (0..HKV).map(|_| sample_orthogonal(D, &mut orng)).collect();

    // Warm cubek (compile shader) once on a representative folded shape.
    {
        let (qf, kf, vf) = fold_expand(&qs, &ks, &vs);
        let _ = cubek_attention_folded(qf.view(), kf.view(), vf.view(), scale, true);
    }

    let t_all = Instant::now();

    // Step 1–3: rotate (in-TEE) + fold + GQA-expand into [B*Hq, n, d].
    // Itemised: the destination alloc/zero (the GQA-expanded footprint —
    // 3 × bh·n·d f32) vs each operand's rotation. Q rotates Hq folded
    // heads; K/V each rotate Hkv source heads broadcast into Hq folded rows
    // (the 4× GQA-expand the un-replicated-upload lever, P2, removes).
    let t_rot = Instant::now();
    let bh = B * HQ;
    let t_alloc = Instant::now();
    let mut q_rot = Array3::<f32>::zeros((bh, n, D));
    let mut k_rot = Array3::<f32>::zeros((bh, n, D));
    let mut v_rot = Array3::<f32>::zeros((bh, n, D));
    let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;
    // Parallelise the rotation over folded heads.
    let rotate_into = |dst: &mut Array3<f32>, src_heads: usize, src: &[Array2<f32>], o: &[Array2<f32>]| {
        dst.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(fh, mut row)| {
                let b = fh / HQ;
                let qh = fh % HQ;
                let kvh = qh / GROUP;
                // Q is per-q-head; K/V are per-kv-head. Pick the source
                // column slab accordingly; the cover matrix is indexed by
                // kv-head in both cases (q-head qh shares kv-head qh/GROUP).
                let off = if src_heads == HQ { qh * D } else { kvh * D };
                let slab: ArrayView2<f32> = src[b].slice(ndarray::s![.., off..off + D]);
                row.assign(&slab.dot(&o[kvh]));
            });
    };
    let t_rq = Instant::now();
    rotate_into(&mut q_rot, HQ, &qs, &o_qk);
    let rot_q_ms = t_rq.elapsed().as_secs_f64() * 1e3;
    let t_rk = Instant::now();
    rotate_into(&mut k_rot, HKV, &ks, &o_qk);
    let rot_k_ms = t_rk.elapsed().as_secs_f64() * 1e3;
    let t_rv = Instant::now();
    rotate_into(&mut v_rot, HKV, &vs, &o_v);
    let rot_v_ms = t_rv.elapsed().as_secs_f64() * 1e3;
    let rot_ms = t_rot.elapsed().as_secs_f64() * 1e3;

    // Step 4: cubek GPU attend (causal).
    let t_gpu = Instant::now();
    let ctx_raw = cubek_attention_folded(q_rot.view(), k_rot.view(), v_rot.view(), scale, true);
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;

    // Step 5: O_vᵀ correction (in-TEE).
    let t_corr = Instant::now();
    let mut ctx = Array3::<f32>::zeros((bh, n, D));
    ctx.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(fh, mut row)| {
            let qh = fh % HQ;
            let kvh = qh / GROUP;
            row.assign(&ctx_raw.index_axis(Axis(0), fh).dot(&o_v[kvh].t()));
        });
    let corr_ms = t_corr.elapsed().as_secs_f64() * 1e3;

    let cubek_cover_ms = t_all.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&ctx);

    let ratio = in_tee_ms / cubek_cover_ms;
    println!("  in-TEE baseline (B-loop causal_gqa_attention): {in_tee_ms:9.3} ms");
    println!("  cubek + cover (end-to-end):                    {cubek_cover_ms:9.3} ms");
    println!("  ratio (in-TEE / cubek+cover):                  {ratio:9.3}x");
    println!("  cover breakdown:");
    println!("    rotation + GQA-expand (TEE): {rot_ms:9.3} ms");
    println!("      ├ alloc/zero dst (3×):     {alloc_ms:9.3} ms");
    println!("      ├ rotate Q (Hq heads):     {rot_q_ms:9.3} ms");
    println!("      ├ rotate K (Hkv→Hq):       {rot_k_ms:9.3} ms");
    println!("      └ rotate V (Hkv→Hq):       {rot_v_ms:9.3} ms");
    println!("    cubek GPU attend:            {gpu_ms:9.3} ms  (CUBEK_PROFILE=1 → per-stage on stderr)");
    println!("    O_vᵀ correction (TEE):       {corr_ms:9.3} ms");
    let overhead = rot_ms + corr_ms;
    println!(
        "    cover overhead (rot+corr):   {overhead:9.3} ms  ({:.1}% of cubek+cover)",
        100.0 * overhead / cubek_cover_ms
    );
}

#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `--release -- --ignored`"]
fn prefill_attention_breakdown() {
    run_prefill_cell(2048);
    // n=8192 is heavy (in-TEE is O(n²) per head); skip if it blows up.
    run_prefill_cell(8192);
}

/// Controlled A/B: **one batched `bh=B·Hq` cubek dispatch** vs **B per-sequence
/// `bh=Hq` dispatches**, at the production prefill shape, *warm* and averaged
/// over many iterations in one process — the low-variance instrument the noisy
/// full-model bench (`gelo_llm_prefill_decode_breakdown`, ±~1.6× cross-run on
/// `cubek_gpu`) cannot provide. Settles whether prefill should fold all B into
/// one `cubek_causal_attend` (GPU best practice) or loop per sequence. Both
/// take un-replicated K/V (`bh=·Hkv`) + `group` (the production O2 path).
#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `--release -- --ignored`"]
fn cubek_dispatch_granularity() {
    let n = 2048usize;
    let scale = 1.0 / (D as f32).sqrt();
    let mut rng = ChaCha20Rng::seed_from_u64(0xD15A7C6);
    let mk = |lead: usize, rng: &mut ChaCha20Rng| {
        Array3::<f32>::from_shape_fn((lead, n, D), |_| rng.random::<f32>() * 0.1 - 0.05)
    };
    // Q over all B·Hq folded heads; K/V un-replicated over B·Hkv (cubek expands
    // to Hq on-device via `group`), b-major to match the production fold.
    let q = mk(B * HQ, &mut rng);
    let k = mk(B * HKV, &mut rng);
    let v = mk(B * HKV, &mut rng);

    let batched = || cubek_attention_folded_gqa(q.view(), k.view(), v.view(), GROUP, scale, true);
    let per_seq = || {
        for b in 0..B {
            let qb = q.slice(ndarray::s![b * HQ..(b + 1) * HQ, .., ..]);
            let kb = k.slice(ndarray::s![b * HKV..(b + 1) * HKV, .., ..]);
            let vb = v.slice(ndarray::s![b * HKV..(b + 1) * HKV, .., ..]);
            std::hint::black_box(cubek_attention_folded_gqa(qb, kb, vb, GROUP, scale, true));
        }
    };

    // Warm (JIT compile + autotune) both shapes.
    std::hint::black_box(batched());
    per_seq();

    const ITERS: usize = 8;
    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(batched());
    }
    let batched_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

    let t = Instant::now();
    for _ in 0..ITERS {
        per_seq();
    }
    let per_seq_ms = t.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

    println!("\n=== cubek dispatch granularity  B={B} Hq={HQ} Hkv={HKV} d={D} n={n} ===");
    println!("  batched (1× bh={}):       {batched_ms:9.3} ms/iter", B * HQ);
    println!("  per-seq ({B}× bh={HQ}):       {per_seq_ms:9.3} ms/iter");
    println!(
        "  ratio (per-seq / batched):  {:9.3}x  (>1 ⇒ batched wins, as best practice predicts)",
        per_seq_ms / batched_ms
    );
}
