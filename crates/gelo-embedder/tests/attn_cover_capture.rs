//! Attention-cover adversary-view capture for the persistent-attention
//! security gates (`docs/plans/perm-attn-gpu-offload.md`).
//!
//! Runs a real Qwen3 prefill, reads the populated per-layer K/V out of the
//! KV cache (the real activations the cover protects), applies the cover —
//! `perm_kv` (row permutation over the n_kv axis), σ-noise on K, `O_qk`
//! feature-rotation on K, `O_v` feature-rotation on V — and writes the
//! **adversary view** (`k_sent`, `v_sent`) plus the **ground truth**
//! (`k_clean`, `v_clean`, `perm_kv`, `o_v.*`) to a safetensors file for the
//! Python HNM / JADE / anchor_ica gate to attack and score against.
//!
//! This is the prefill-only (cover-applied-once) snapshot: it captures one
//! fixed (`perm_kv`, `O_qk`, `O_v`) realization — the optimistic case the
//! gate attacks first. Per `gelo-protocol`'s cover, σ-noise is on K only;
//! V's privacy is `perm_kv` + `O_v` (un-noised) — exactly the gate-3 target.
//!
//! `#[ignore]` — loads real Qwen3 weights from the HF cache.
//!
//! Run (host or container):
//!   cargo test -p gelo-embedder --test attn_cover_capture -- --ignored --nocapture
//! Output: `$GELO_CAPTURE_DIR` (default `evals/aloepri-attacks/captures/`).

use std::borrow::Cow;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView2};
use rand::{RngCore, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use safetensors::{Dtype, View, serialize_to_file};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward::run_prefill;
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;

use gelo_embedder::common::HfTokenizer;
use gelo_protocol::{
    PlaintextExecutor, ReferenceCpuEngine, TrustedExecutor, WeightHandle, WeightKind,
};

use hf_hub::api::sync::ApiBuilder;

/// Owned tensor implementing safetensors' `View` for serialization.
struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}
impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn f32_tensor(a: ArrayView2<'_, f32>) -> OwnedTensor {
    let std = a.as_standard_layout();
    let data: Vec<u8> = std.iter().flat_map(|x| x.to_le_bytes()).collect();
    OwnedTensor {
        dtype: Dtype::F32,
        shape: vec![a.nrows(), a.ncols()],
        data,
    }
}

fn i64_vec_tensor(v: &[i64]) -> OwnedTensor {
    let data: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    OwnedTensor {
        dtype: Dtype::I64,
        shape: vec![v.len()],
        data,
    }
}

/// Random `d×d` orthogonal matrix via modified Gram-Schmidt (mirrors
/// `gelo_protocol::attention::sample_orthogonal`, which is crate-private).
fn sample_orthogonal(d: usize, rng: &mut ChaCha20Rng) -> Array2<f32> {
    let normal = StandardNormal;
    let mut m = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(rng));
    for i in 0..d {
        for j in 0..i {
            let dot: f32 = (0..d).map(|k| m[(i, k)] * m[(j, k)]).sum();
            for k in 0..d {
                m[(i, k)] -= dot * m[(j, k)];
            }
        }
        let norm: f32 = (0..d).map(|k| m[(i, k)] * m[(i, k)]).sum::<f32>().sqrt();
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for k in 0..d {
            m[(i, k)] *= inv;
        }
    }
    m
}

#[test]
#[ignore = "loads real Qwen3 weights from the HF cache"]
fn capture_attn_cover_adversary_view() -> Result<()> {
    let variant = Qwen3Variant::Q4B;
    let sigma: f32 = std::env::var("GELO_CAPTURE_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    // Rotation-only cover = the offloaded-PREFILL cover (feature rotation
    // O_qk/O_v, no permutation, σ=0). The Phase-5 WEIGHTS-PUB spike attacks
    // this view. Default off → the decode capture is unchanged.
    let rotation_only = std::env::var("GELO_CAPTURE_COVER")
        .map(|v| v.eq_ignore_ascii_case("rotation"))
        .unwrap_or(false);
    let sigma = if rotation_only { 0.0 } else { sigma };
    let prompt = std::env::var("GELO_CAPTURE_PROMPT").unwrap_or_else(|_| {
        "The mitochondria is the powerhouse of the cell. In distributed \
         systems, consensus protocols like Raft elect a leader to order writes."
            .to_string()
    });
    // Default to the repo's eval captures dir, anchored at the workspace
    // root (cargo runs tests with CWD = the crate dir, not the root).
    let out_dir = match std::env::var("GELO_CAPTURE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../evals/aloepri-attacks/captures"),
    };
    std::fs::create_dir_all(&out_dir).context("creating capture dir")?;

    // ── Load real model ────────────────────────────────────────────
    let cfg: DecoderConfig = variant.config();
    let api = ApiBuilder::new().with_progress(false).build()?;
    let repo = api.model(variant.hf_model_id().to_string());
    let tokenizer = HfTokenizer::from_file(&repo.get("tokenizer.json")?)?;
    let shard_paths = find_shards(&repo)?;
    let shard_refs: Vec<&std::path::Path> = shard_paths.iter().map(|p| p.as_path()).collect();
    let weights = DecoderWeights::from_safetensors(&shard_refs, &cfg)?;
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    // ── Real prefill → populated KV cache ──────────────────────────
    let max_tok: usize = std::env::var("GELO_CAPTURE_MAXTOK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let input_ids = tokenizer.encode(prompt.as_str(), max_tok)?;
    let n_kv = input_ids.len();
    assert!(n_kv >= 8, "need a non-trivial prompt; got {n_kv} tokens");
    let mut exec = PlaintextExecutor::new(ReferenceCpuEngine::new());
    // Provision per-layer weights into the executor (the offload path
    // needs them registered before the forward).
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        for (kind, w) in [
            (WeightKind::Q, &layer.wq),
            (WeightKind::K, &layer.wk),
            (WeightKind::V, &layer.wv),
            (WeightKind::O, &layer.wo),
            (WeightKind::FfnGate, &layer.w_gate),
            (WeightKind::FfnUp, &layer.w_up),
            (WeightKind::FfnDown, &layer.w_down),
        ] {
            exec.provision_weight_bf16(
                WeightHandle::new(li16, kind),
                w.as_ref().expect("offloadable weight").view(),
            )?;
        }
    }
    let mut kv = KvCache::new(cfg.num_hidden_layers, n_kv + 4, cfg.kv_dim());
    run_prefill(&cfg, &weights, &rope, &mut exec, &input_ids, &mut kv)?;

    let n_kv_heads = cfg.num_key_value_heads;
    let d_head = cfg.head_dim_value();
    let kv_dim = cfg.kv_dim();

    // Capture a representative spread of layers (env-overridable).
    let layers: Vec<usize> = std::env::var("GELO_CAPTURE_LAYERS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, cfg.num_hidden_layers / 2, cfg.num_hidden_layers - 1]);

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE_A77AC0);
    let mut tensors: Vec<(String, OwnedTensor)> = Vec::new();

    for &li in &layers {
        let (k_clean, v_clean) = kv.layer(li).view_b(0);
        let k_clean = k_clean.to_owned();
        let v_clean = v_clean.to_owned();
        assert_eq!(k_clean.dim(), (n_kv, kv_dim));

        // Cover: one fixed perm_kv over the n_kv axis (prefill-only).
        // In rotation-only mode the permutation is the identity (the
        // offloaded-prefill cover hides coordinates, not order).
        let mut perm: Vec<usize> = (0..n_kv).collect();
        if !rotation_only {
            perm.shuffle(&mut rng);
        }

        let mut k_sent = Array2::<f32>::zeros((n_kv, kv_dim));
        let mut v_sent = Array2::<f32>::zeros((n_kv, kv_dim));
        for h in 0..n_kv_heads {
            let c0 = h * d_head;
            let o_qk = sample_orthogonal(d_head, &mut rng);
            let o_v = sample_orthogonal(d_head, &mut rng);

            // Per-head slices, row-permuted; K noised then O_qk-rotated,
            // V O_v-rotated (un-noised).
            let mut k_perm = Array2::<f32>::zeros((n_kv, d_head));
            let mut v_perm = Array2::<f32>::zeros((n_kv, d_head));
            for (i, &src) in perm.iter().enumerate() {
                for c in 0..d_head {
                    let z: f32 = StandardNormal.sample(&mut rng);
                    k_perm[(i, c)] = k_clean[(src, c0 + c)] + sigma * z;
                    v_perm[(i, c)] = v_clean[(src, c0 + c)];
                }
            }
            let k_rot = k_perm.dot(&o_qk);
            let v_rot = v_perm.dot(&o_v);
            for i in 0..n_kv {
                for c in 0..d_head {
                    k_sent[(i, c0 + c)] = k_rot[(i, c)];
                    v_sent[(i, c0 + c)] = v_rot[(i, c)];
                }
            }
            tensors.push((format!("layer{li:03}.o_v.head{h:02}"), f32_tensor(o_v.view())));
        }

        let perm_i64: Vec<i64> = perm.iter().map(|&x| x as i64).collect();
        tensors.push((format!("layer{li:03}.k_clean"), f32_tensor(k_clean.view())));
        tensors.push((format!("layer{li:03}.v_clean"), f32_tensor(v_clean.view())));
        tensors.push((format!("layer{li:03}.k_sent"), f32_tensor(k_sent.view())));
        tensors.push((format!("layer{li:03}.v_sent"), f32_tensor(v_sent.view())));
        tensors.push((format!("layer{li:03}.perm_kv"), i64_vec_tensor(&perm_i64)));
    }

    // Token ids — the ground truth for the WEIGHTS-PUB dictionary attack.
    let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    tensors.push(("input_ids".to_string(), i64_vec_tensor(&input_ids_i64)));

    // Layer-0 value dictionary for the WEIGHTS-PUB Gram/norm attack. At
    // layer 0, V(t) = rms_norm(embed(t), norm_attn[0])·W_V[0] is
    // context-free (no attention, no positional dependence on the V path),
    // so it is a single value projection over the candidate embeddings —
    // not a 36-layer forward. Uses the model's own `rms_norm` + bf16
    // weights; the Python attack verifies it matches `v_clean` on the
    // prompt tokens (faithfulness check → no false negative).
    if std::env::var("GELO_CAPTURE_DICT").map(|v| v == "1").unwrap_or(false) {
        use gelo_embedder::decoder::rms_norm::rms_norm;
        let vocab = weights.token_embedding.nrows();
        let hidden = weights.token_embedding.ncols();
        let n_distract: usize = std::env::var("GELO_CAPTURE_DICT_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1800);
        let promptset: std::collections::HashSet<u32> = input_ids.iter().copied().collect();
        let mut cand: Vec<u32> = {
            let mut u: Vec<u32> = promptset.iter().copied().collect();
            u.sort_unstable();
            u
        };
        let n_true = cand.len();
        let mut dr = ChaCha20Rng::seed_from_u64(0xD1C7_5EED_u64);
        while cand.len() < n_true + n_distract {
            let t = (dr.next_u32() as usize % vocab) as u32;
            if !promptset.contains(&t) {
                cand.push(t);
            }
        }
        let mut emb = Array2::<f32>::zeros((cand.len(), hidden));
        for (i, &t) in cand.iter().enumerate() {
            let row = weights.token_embedding.row(t as usize);
            for c in 0..hidden {
                emb[(i, c)] = row[c].to_f32();
            }
        }
        let normed = rms_norm(
            emb.view(),
            weights.layers[0].norm_attn.as_slice().expect("norm_attn contiguous"),
            cfg.rms_norm_eps,
        );
        let wv0 = weights.layers[0].wv.as_ref().expect("wv[0]");
        let mut wv0_f = Array2::<f32>::zeros((hidden, kv_dim));
        for r in 0..hidden {
            for c in 0..kv_dim {
                wv0_f[(r, c)] = wv0[(r, c)].to_f32();
            }
        }
        let dict_v = normed.dot(&wv0_f); // (K, kv_dim), layer-0 V per candidate token
        tensors.push(("dict.v".to_string(), f32_tensor(dict_v.view())));
        let cand_i64: Vec<i64> = cand.iter().map(|&x| x as i64).collect();
        tensors.push(("dict.ids".to_string(), i64_vec_tensor(&cand_i64)));
        eprintln!(
            "[capture] dict: {} candidates ({} prompt-unique + {} distractors), layer-0 V (direct projection)",
            cand.len(),
            n_true,
            cand.len() - n_true
        );
    }

    let cover_str = if rotation_only {
        "O_qk(K) + O_v(V), rotation-only (no perm, sigma=0) — offloaded-prefill cover"
    } else {
        "perm_kv + sigma-noise(K) + O_qk(K) + O_v(V), prefill-only"
    };

    let st_path = out_dir.join("attn_cover.safetensors");
    let no_meta: Option<std::collections::HashMap<String, String>> = None;
    serialize_to_file(tensors.iter().map(|(k, t)| (k.clone(), t)), &no_meta, &st_path)
        .context("writing safetensors")?;

    // Sidecar meta the Python gate reads (manual JSON — no serde dep here).
    let layers_json = layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let meta = format!(
        "{{\n  \"schema_version\": \"attn-cover-1\",\n  \"model_id\": \"{}\",\n  \
         \"n_kv\": {n_kv},\n  \"kv_dim\": {kv_dim},\n  \"n_kv_heads\": {n_kv_heads},\n  \
         \"d_head\": {d_head},\n  \"sigma\": {sigma},\n  \"layers\": [{layers_json}],\n  \
         \"cover\": \"{cover_str}\",\n  \
         \"keys\": \"layer{{L:03}}.{{k_clean,v_clean,k_sent,v_sent,perm_kv,o_v.head{{H:02}}}}\"\n}}\n",
        variant.hf_model_id(),
    );
    std::fs::write(out_dir.join("attn_cover.meta.json"), meta)?;

    eprintln!(
        "[capture] wrote {} ({} layers × {} heads, n_kv={n_kv}, σ={sigma}) → {}",
        st_path.display(),
        layers.len(),
        n_kv_heads,
        out_dir.display(),
    );
    Ok(())
}

/// Find `*.safetensors` shards in the HF repo cache (single-file or sharded).
fn find_shards(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try the sharded index first; fall back to the single-file model.
    if let Ok(idx) = repo.get("model.safetensors.index.json") {
        let txt = std::fs::read_to_string(&idx)?;
        let mut names: Vec<String> = txt
            .split('"')
            .filter(|s| s.ends_with(".safetensors"))
            .map(|s| s.to_string())
            .collect();
        names.sort();
        names.dedup();
        let mut out = Vec::new();
        for n in names {
            out.push(repo.get(&n)?);
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    Ok(vec![repo.get("model.safetensors")?])
}
