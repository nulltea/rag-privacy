//! [`GpuOffloadEngine`] backed by **burn-cubecl** on the **wgpu/Vulkan** runtime.
//!
//! Replaces the prior cubecl-matmul direct path with `burn_tensor::Tensor::matmul`
//! over the `CubeBackend<Rt, …>` backend. burn-cubecl wires:
//! - Real autotune with disk-persistent cache (via `cubecl-runtime::TuneCache`,
//!   configured by workspace-root `cubecl.toml`).
//! - Lazy / deferred dispatch — sync only happens at `.into_data()`.
//! - Built-in buffer pooling and kernel fusion (`burn-cubecl-fusion`).
//!
//! The trait surface (`GpuOffloadEngine` from `gelo-protocol`) is unchanged.
//! The GELO mask round-trip math stays on the trusted/TEE side (CPU);
//! only the masked product `A·H` becomes a `Tensor` on the engine side.
//!
//! On Linux this dispatches via Vulkan; on macOS via Metal; on Windows via DX12.
//!
//! ## Precision modes
//!
//! - **`new()`** — default, f32 throughout. Highest fidelity, full
//!   U-Verify compatibility.
//! - **`new_fp16()`** — engine internal element type is f16 (`half::f16`).
//!   Inputs/outputs are converted at the f32 ↔ f16 boundary inside the
//!   engine; the trait surface remains f32 so trusted-side code is
//!   unchanged. Expected ~1.5–2× faster GEMM kernels on Vulkan
//!   `shader-f16`-capable adapters at the cost of ~3–4 decimal digits
//!   of precision. **U-Verify probes must be widened or disabled** under
//!   fp16 — the engine's matmul output is not bit-equal to the trusted
//!   side's f32 reference. Confirms via the `WgpuVulkanEngine::is_fp16()`
//!   accessor.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData, Transaction, activation};
// Backend runtime + device, selected at compile time: wgpu/Vulkan by
// default (OEM-agnostic), cubecl-cuda under the `cuda` feature. The rest
// of the file refers to the runtime/device only through `Rt` / `Dev`.
// `future` is backend-agnostic (`cubecl_common`); both the wgpu `gpu_ctx`
// init and the cubek `CUBEK_PROFILE` sync barrier use it, so it must be in
// scope under the cuda feature too (previously not-cuda-gated, which broke
// the cuda build at the cubek profile barrier).
use cubecl_common::future;
#[cfg(any(feature = "vulkan", not(feature = "cuda")))]
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};
#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
use cubecl_cuda::{CudaDevice, CudaRuntime};

#[cfg(any(feature = "vulkan", not(feature = "cuda")))]
type Rt = WgpuRuntime;
#[cfg(any(feature = "vulkan", not(feature = "cuda")))]
type Dev = WgpuDevice;
#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
type Rt = CudaRuntime;
#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
type Dev = CudaDevice;
use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use gelo_protocol::{GpuOffloadEngine, KvSessionId, MatmulToken, WeightHandle};

/// burn-cubecl backend specialised to f32 floats. The default engine
/// precision.
type CubeWgpu32 = CubeBackend<Rt, f32, i32, u8>;

/// burn-cubecl backend specialised to f16 floats. Used by the fp16
/// engine path. Requires the wgpu adapter to support the `shader-f16`
/// extension (true on AMD RDNA2/3, NVIDIA Maxwell+, most modern Intel
/// iGPUs).
type CubeWgpu16 = CubeBackend<Rt, f16, i32, u8>;

/// burn-cubecl backend specialised to bf16 floats. Used by the cubek
/// prefill-offload path to store the attention operands and intermediate
/// tiles with f32-range exponent — bf16's 8-bit exponent removes the f16
/// (max 65 504) overflow that NaNs the offloaded prefill on real Qwen3-4B
/// activations (`perm-attn-gpu-offload` BF16-kernel fix), at the cost of
/// 3 mantissa bits vs f16. On **CUDA** bf16 is fully `all_scalar`; on
/// **Vulkan** cubecl registers it as `Conversion | Buffer` only (gated on
/// `VK_KHR_shader_bfloat16`), which suffices because cubek keeps the
/// reductions in the f32 accumulator and only loads/stores bf16.
type CubeWgpuBf16 = CubeBackend<Rt, bf16, i32, u8>;

/// Backend device class, abstracted over the wgpu / CUDA split so the
/// engine reports device identity uniformly across both runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

/// Per-process GPU identity, captured once at first device init.
pub struct GpuContext {
    pub name: String,
    pub device_type: GpuDeviceType,
    pub backend: String,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

#[cfg(any(feature = "vulkan", not(feature = "cuda")))]
fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        let device = Dev::default();
        let setup = future::block_on(init_setup_async::<AutoGraphicsApi>(
            &device,
            RuntimeOptions::default(),
        ));
        let info = setup.adapter.get_info();
        // Match on the Debug string rather than `wgpu::DeviceType::*` so this
        // compiles even when the `vulkan` feature is enabled alongside the
        // default `cuda` (the two backends pull different `wgpu_types`
        // versions; the direct-`wgpu` enum would then mismatch the adapter's).
        let device_type = match format!("{:?}", info.device_type).as_str() {
            "DiscreteGpu" => GpuDeviceType::Discrete,
            "IntegratedGpu" => GpuDeviceType::Integrated,
            "VirtualGpu" => GpuDeviceType::Virtual,
            "Cpu" => GpuDeviceType::Cpu,
            _ => GpuDeviceType::Other,
        };
        GpuContext {
            backend: format!("{:?}", info.backend),
            name: info.name,
            device_type,
        }
    })
}

#[cfg(all(feature = "cuda", not(feature = "vulkan")))]
fn gpu_ctx() -> &'static GpuContext {
    // cubecl-cuda has no wgpu-style adapter enumeration; the CUDA context
    // is created lazily on the first `Backend::sync`. Report a fixed
    // identity for device 0 — nvidia-smi confirms which physical card.
    GPU_CTX.get_or_init(|| GpuContext {
        name: "CUDA device 0 (cubecl-cuda)".to_string(),
        device_type: GpuDeviceType::Discrete,
        backend: "Cuda".to_string(),
    })
}

/// Dispatch enum holding the precision-specific weight map.
enum WeightStore {
    F32(HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>),
    F16(HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>),
}

/// burn-cubecl/wgpu-backed offload engine.
///
/// Registered weights live device-resident as `Tensor<…, 2>` in either
/// f32 or f16. `clone_shared()` produces a second handle pointing at the
/// same weight cache. Precision is fixed at construction
/// ([`Self::new`] vs [`Self::new_fp16`]).
pub struct WgpuVulkanEngine {
    device: Dev,
    weights: Arc<Mutex<WeightStore>>,
    fp16: bool,
    /// Resident K/V sessions (perm-attn-gpu-offload Phase 2). Shared
    /// across `clone_shared` handles so a session created on one handle
    /// is visible on its clones (mirrors the weight-cache sharing).
    sessions: Arc<Mutex<HashMap<KvSessionId, ResidentKvSession>>>,
    next_session_id: Arc<AtomicU64>,
}

impl WgpuVulkanEngine {
    /// Initialise a Vulkan-preferred wgpu device via burn-cubecl, with
    /// f32 internal precision.
    pub fn new() -> Result<Self> {
        let _ = gpu_ctx();
        let device = Dev::default();
        <CubeWgpu32 as Backend>::sync(&device)
            .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;
        Ok(Self {
            device,
            weights: Arc::new(Mutex::new(WeightStore::F32(HashMap::new()))),
            fp16: false,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Initialise the engine with **f16 internal precision** for the
    /// GEMM kernel. Inputs/outputs cross the trait boundary as f32; the
    /// conversion to/from f16 happens inside `register_weight` and each
    /// `matmul*` call. See module docs for the U-Verify caveat.
    ///
    /// Fails if the adapter doesn't support `shader-f16`. (cubecl checks
    /// this lazily; the first matmul will surface the error.)
    pub fn new_fp16() -> Result<Self> {
        let _ = gpu_ctx();
        let device = Dev::default();
        <CubeWgpu16 as Backend>::sync(&device)
            .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;
        Ok(Self {
            device,
            weights: Arc::new(Mutex::new(WeightStore::F16(HashMap::new()))),
            fp16: true,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Second handle sharing the registered-weight cache with `self`.
    pub fn clone_shared(&self) -> Self {
        Self {
            device: self.device.clone(),
            weights: Arc::clone(&self.weights),
            fp16: self.fp16,
            sessions: Arc::clone(&self.sessions),
            next_session_id: Arc::clone(&self.next_session_id),
        }
    }
}

/// `Clone` delegates to `clone_shared` — both handles point at the same
/// `Arc`-backed weight cache, so the `Embedder::embed` rayon fan-out can
/// hand each worker its own engine handle without duplicating the
/// device-resident weight tensors.
impl Clone for WgpuVulkanEngine {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

/// Device-resident f16 K/V for the gate-1 persistent-attention
/// microbench (`docs/dev/logs/perm-attn-gpu-offload.md`). Holds the K/V
/// tensors on the GPU across decode steps so the per-step path uploads
/// only Q — isolating the resident-read cost from the per-call K/V
/// upload+convert that dominates `fused_attention_batched`. Seed of the
/// eventual session-resident K/V API (not the production surface yet).
pub struct ResidentKvF16 {
    k_t: Tensor<CubeWgpu16, 3>,
    v_t: Tensor<CubeWgpu16, 3>,
}

impl WgpuVulkanEngine {
    /// Upload + convert K/V to device-resident f16 tensors **once**.
    /// `k`, `v`: `(B·H, n_kv, d_head)`. fp16 engine only.
    pub fn upload_resident_kv(
        &self,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
    ) -> Result<ResidentKvF16> {
        if !self.fp16 {
            return Err(anyhow!("upload_resident_kv requires the fp16 engine"));
        }
        Ok(ResidentKvF16 {
            k_t: array3_to_tensor_f16(k, &self.device),
            v_t: array3_to_tensor_f16(v, &self.device),
        })
    }

    /// Attention against device-resident K/V: uploads only `q`
    /// `(B·H, n_q, d_head)`, computes `softmax(q·kᵀ·scale)·v`, reads
    /// back the context `(B·H, n_q, d_head)`. No mask (decode m=1 causal
    /// is a no-op). This is `fused_attention_batched` **minus** the
    /// per-call K/V upload+convert — the gate-1 resident-read
    /// measurement. fp16 engine only.
    pub fn attend_resident(
        &self,
        q: ArrayView3<'_, f32>,
        kv: &ResidentKvF16,
        scale: f32,
    ) -> Result<Array3<f32>> {
        if !self.fp16 {
            return Err(anyhow!("attend_resident requires the fp16 engine"));
        }
        let q_t = array3_to_tensor_f16(q, &self.device);
        let kt = kv.k_t.clone().permute([0, 2, 1]);
        let scores = q_t.matmul(kt).mul_scalar(scale);
        let probs = activation::softmax(scores, 2);
        let out = probs.matmul(kv.v_t.clone());
        tensor3_to_array_f16(out)
    }
}

/// Device-resident **growing** K/V session for the representative
/// decode microbench (gate 1, `docs/dev/logs/perm-attn-gpu-offload.md`).
/// Models the **optimistic prefill-only-permute** case: the cover is
/// applied once at `create_kv_session` (prefill); decode only *appends*
/// new rows and attends over the active slice — **no per-block
/// re-permute** (the N=∞ best case the security gate will later test).
/// K/V are pre-allocated to `capacity` on the n_kv axis so `append_kv`
/// is an O(1) in-place row write (`slice_assign` on a sole-owner tensor),
/// not an O(n) recopy.
pub struct ResidentKvSession {
    k_t: Tensor<CubeWgpu16, 3>,
    v_t: Tensor<CubeWgpu16, 3>,
    len: usize,
    capacity: usize,
    /// Optional additive score mask `(h_q, 1, len)` (0 / −∞), added to the
    /// `q·kᵀ` scores before the softmax max. Used by the ragged batched
    /// decode cover (Option B, chronicle §25): the frozen prefix is padded
    /// to the batch-max length and each row's padded key slots are masked
    /// to −∞ so a query never attends another row's padding. `None` on the
    /// uniform / single-sequence paths (no padding → no mask).
    mask: Option<Tensor<CubeWgpu16, 3>>,
}

impl WgpuVulkanEngine {
    /// Prefill: pre-allocate `(B·H, capacity, d_head)` and write the
    /// (already TEE-permuted/noised) prefix into `[0..n0]`. One-time;
    /// the per-step decode path never re-touches it. fp16 only.
    pub fn create_kv_session(
        &self,
        k_prefix: ArrayView3<'_, f32>,
        v_prefix: ArrayView3<'_, f32>,
        capacity: usize,
    ) -> Result<ResidentKvSession> {
        if !self.fp16 {
            return Err(anyhow!("create_kv_session requires the fp16 engine"));
        }
        let (bh, n0, d) = k_prefix.dim();
        if n0 > capacity {
            return Err(anyhow!("create_kv_session: prefix {n0} > capacity {capacity}"));
        }
        let k_full = Tensor::<CubeWgpu16, 3>::zeros([bh, capacity, d], &self.device);
        let v_full = Tensor::<CubeWgpu16, 3>::zeros([bh, capacity, d], &self.device);
        let k_t = k_full.slice_assign([0..bh, 0..n0, 0..d], array3_to_tensor_f16(k_prefix, &self.device));
        let v_t = v_full.slice_assign([0..bh, 0..n0, 0..d], array3_to_tensor_f16(v_prefix, &self.device));
        Ok(ResidentKvSession { k_t, v_t, len: n0, capacity, mask: None })
    }

    /// Decode append: write one token's `(B·H, 1, d_head)` K/V row at
    /// the current length (O(1) in-place). At capacity it overwrites the
    /// last slot (the bench holds context fixed); production grows `len`.
    pub fn append_kv(
        &self,
        session: &mut ResidentKvSession,
        k_row: ArrayView3<'_, f32>,
        v_row: ArrayView3<'_, f32>,
    ) -> Result<()> {
        let (bh, _one, d) = k_row.dim();
        let idx = session.len.min(session.capacity - 1);
        let kr = array3_to_tensor_f16(k_row, &self.device);
        let vr = array3_to_tensor_f16(v_row, &self.device);
        // mem::replace so each tensor is the sole owner during
        // slice_assign → cubecl mutates the resident buffer in place.
        let dummy = || Tensor::<CubeWgpu16, 3>::zeros([1, 1, 1], &self.device);
        let k_t = std::mem::replace(&mut session.k_t, dummy());
        let v_t = std::mem::replace(&mut session.v_t, dummy());
        session.k_t = k_t.slice_assign([0..bh, idx..idx + 1, 0..d], kr);
        session.v_t = v_t.slice_assign([0..bh, idx..idx + 1, 0..d], vr);
        if session.len < session.capacity {
            session.len += 1;
        }
        Ok(())
    }

    /// Attend over the active `[0..len]` slice: upload only `q`
    /// `(B·H, n_q, d_head)`, `softmax(q·kᵀ·scale)·v`, read back context.
    /// The per-step decode read against the growing resident cache.
    pub fn attend_session(
        &self,
        q: ArrayView3<'_, f32>,
        session: &ResidentKvSession,
        scale: f32,
    ) -> Result<Array3<f32>> {
        let (h_q, _nq, d) = q.dim();
        let q_t = array3_to_tensor_f16(q, &self.device);
        let (k_act, v_act) = self.resident_kv_expanded(session, h_q, d)?;
        let scores = q_t.matmul(k_act.permute([0, 2, 1])).mul_scalar(scale);
        let probs = activation::softmax(scores, 2);
        tensor3_to_array_f16(probs.matmul(v_act))
    }

    /// **Partial-stats attend** (perm-attn-gpu-offload Phase 3, decode
    /// tail-in-TEE). Returns the *unnormalised* online-softmax state over
    /// the resident prefix — `(acc, m, l)` with `acc = Σ exp(s−m)·V`,
    /// `m = max_j s_j`, `l = Σ exp(s−m)` — instead of the normalised
    /// context. The TEE merges this with the in-TEE tail's partial
    /// (`merge_attention_partials`) so the freshest ≤N tokens never reach
    /// the GPU (closes the per-step write-location side channel). Composed
    /// from burn primitives — no custom kernel; at decode (n_q=1, tiny
    /// scores) the non-fused form is fine. Shapes: `acc (h_q, n_q, d)`,
    /// `m`/`l` `(h_q, n_q, 1)`.
    pub fn attend_session_partial(
        &self,
        q: ArrayView3<'_, f32>,
        session: &ResidentKvSession,
        scale: f32,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let (h_q, _nq, d) = q.dim();
        let q_t = array3_to_tensor_f16(q, &self.device);
        let (k_act, v_act) = self.resident_kv_expanded(session, h_q, d)?;
        let scores = q_t.matmul(k_act.permute([0, 2, 1])).mul_scalar(scale);
        // Ragged batched decode (Option B): add the per-row padded-key mask
        // (0 / −∞) before the softmax so a query never attends another
        // sequence's padding slots. Shape `(h_q, 1, len)`, matches `scores`.
        let scores = match &session.mask {
            Some(mask) => scores.add(mask.clone()),
            None => scores,
        };
        let m = scores.clone().max_dim(2);
        let shifted = scores.sub(m.clone()).exp();
        let l = shifted.clone().sum_dim(2);
        let acc = shifted.matmul(v_act);
        // Batch the (acc, m, l) download into one device sync (one
        // Transaction) instead of three separate `.into_data()` calls.
        let (acc_d, m_d, l_d) = (acc.dims(), m.dims(), l.dims());
        let to_dim = |d: [usize; 3]| (d[0], d[1], d[2]);
        let mut data = Transaction::<CubeWgpu16>::default()
            .register(acc)
            .register(m)
            .register(l)
            .execute()
            .into_iter();
        Ok((
            tensor_data_to_array3_f16(data.next().unwrap(), to_dim(acc_d))?,
            tensor_data_to_array3_f16(data.next().unwrap(), to_dim(m_d))?,
            tensor_data_to_array3_f16(data.next().unwrap(), to_dim(l_d))?,
        ))
    }

    /// Slice the resident K/V to `[0..len]` and GQA-broadcast un-replicated
    /// kv heads up to `h_q` query heads (interleaved, on-device). Shared by
    /// `attend_session` / `attend_session_partial`.
    fn resident_kv_expanded(
        &self,
        session: &ResidentKvSession,
        h_q: usize,
        d: usize,
    ) -> Result<(Tensor<CubeWgpu16, 3>, Tensor<CubeWgpu16, 3>)> {
        let h_kv = session.k_t.dims()[0];
        let len = session.len;
        let mut k_act = session.k_t.clone().slice([0..h_kv, 0..len, 0..d]);
        let mut v_act = session.v_t.clone().slice([0..h_kv, 0..len, 0..d]);
        if h_q != h_kv {
            if h_q % h_kv != 0 {
                return Err(anyhow!(
                    "resident_kv_expanded: q heads {h_q} not a multiple of kv heads {h_kv}"
                ));
            }
            let group = h_q / h_kv;
            k_act = k_act
                .reshape([h_kv, 1, len, d])
                .repeat_dim(1, group)
                .reshape([h_q, len, d]);
            v_act = v_act
                .reshape([h_kv, 1, len, d])
                .repeat_dim(1, group)
                .reshape([h_q, len, d]);
        }
        Ok((k_act, v_act))
    }
}

impl WgpuVulkanEngine {
    /// `true` if this engine handle runs GEMM kernels in f16. Trusted-
    /// side code that needs bit-equal matmul output (e.g. U-Verify) must
    /// gate on this.
    pub fn is_fp16(&self) -> bool {
        self.fp16
    }

    /// Backend name (`"Vulkan"`/`"Metal"`/… on wgpu; `"Cuda"` under the
    /// `cuda` feature).
    pub fn backend(&self) -> String {
        gpu_ctx().backend.clone()
    }

    /// GPU identity captured at first init (`.name`, `.device_type`).
    pub fn adapter_info(&self) -> &'static GpuContext {
        gpu_ctx()
    }

    /// `true` if the selected device is a real GPU (discrete, integrated,
    /// or virtual) — not a software rasterizer like lavapipe.
    pub fn is_real_gpu(&self) -> bool {
        matches!(
            gpu_ctx().device_type,
            GpuDeviceType::Discrete | GpuDeviceType::Integrated | GpuDeviceType::Virtual
        )
    }
}

// ─── f32 conversion helpers ───────────────────────────────────────────

fn array2_to_tensor_f32(view: ArrayView2<'_, f32>, device: &Dev) -> Tensor<CubeWgpu32, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu32, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor_f32(view: ArrayView3<'_, f32>, device: &Dev) -> Tensor<CubeWgpu32, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu32, 3>::from_data(TensorData::new(v, [b, m, k]), device)
}

fn tensor2_to_array_f32(t: Tensor<CubeWgpu32, 2>) -> Result<Array2<f32>> {
    let shape = t.dims();
    let v: Vec<f32> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f32 tensor → Vec<f32>: {e:?}"))?;
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor3_to_array_f32(t: Tensor<CubeWgpu32, 3>) -> Result<Array3<f32>> {
    let shape = t.dims();
    let v: Vec<f32> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f32 tensor → Vec<f32>: {e:?}"))?;
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

// ─── f16 conversion helpers ───────────────────────────────────────────

fn array2_to_tensor_f16(view: ArrayView2<'_, f32>, device: &Dev) -> Tensor<CubeWgpu16, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    // f32→f16 via half's slice conversion, which dispatches to the F16C
    // `vcvtps2ph` hardware path at runtime (the scalar `f16::from_f32`
    // map does NOT auto-vectorise — it's the software bit-twiddle —
    // measured ~2.7 ns/elem vs ~0.2-0.3 ns SIMD; see the upload
    // decomposition in docs/dev/logs/perm-attn-gpu-offload.md).
    let std = view.as_standard_layout();
    let src = std.as_slice().expect("standard-layout slice is contiguous");
    // Parallel f32->f16 into an UNINIT buffer: kills the redundant
    // `vec![f16::ZERO]` zerofill and spreads the SIMD F16C convert across
    // cores. The single-threaded convert was ~9.6 s of the B=8 prefill
    // :upload bucket; chunked rayon cut it to ~2.4 s (chronicle: the
    // forced-sync attribution found burn's device transfer — not this
    // convert — is the residual ~42 s wall). f16 is a POD u16 wrapper, so
    // set_len before the chunked write (which covers every element) is
    // sound. 64K-elem chunks keep small (decode) inputs single-chunk.
    // Sub-bucket split of the fp16 offload host path (chronicle §24): the
    // f32→f16 convert vs the `from_data` HtoD upload (the latter stages
    // through pageable host memory and dominates — ~19.6 s of the B=8
    // prefill `engine:matmul*` wall vs ~2.3 s for the convert). These
    // sit *inside* the `engine:matmul`/`engine:matmul_many` parent wall.
    let dst = gelo_protocol::profile::time("engine:matmul:cvt", || {
        let mut dst: Vec<f16> = Vec::with_capacity(src.len());
        unsafe {
            dst.set_len(src.len());
        }
        use rayon::prelude::*;
        const CHUNK: usize = 1 << 16;
        dst.par_chunks_mut(CHUNK)
            .zip(src.par_chunks(CHUNK))
            .for_each(|(d, s)| d.convert_from_f32_slice(s));
        dst
    });
    gelo_protocol::profile::time("engine:matmul:upload", || {
        Tensor::<CubeWgpu16, 2>::from_data(TensorData::new(dst, [rows, cols]), device)
    })
}

/// **bf16-native** weight upload. Skips the bf16 → f32 host
/// intermediate that `view_to_f32` would otherwise force on the
/// loader. Each bf16 element is converted directly to f16 via the
/// `f16::from_f32(bf16::to_f32(x))` round-trip — same numeric path
/// as the f32 entry point but without ever materialising an f32
/// host copy of the full weight matrix.
fn array2_bf16_to_tensor_f16(
    view: ArrayView2<'_, bf16>,
    device: &Dev,
) -> Tensor<CubeWgpu16, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f16> = view
        .as_standard_layout()
        .iter()
        .map(|&x| f16::from_f32(x.to_f32()))
        .collect();
    Tensor::<CubeWgpu16, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

/// **bf16 → f32 GPU upload**. Used when the engine is in F32 mode but
/// the caller supplied bf16. Still avoids a host f32 array — the
/// per-element widening happens once during the upload Vec build.
fn array2_bf16_to_tensor_f32(
    view: ArrayView2<'_, bf16>,
    device: &Dev,
) -> Tensor<CubeWgpu32, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view
        .as_standard_layout()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    Tensor::<CubeWgpu32, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor_f16(view: ArrayView3<'_, f32>, device: &Dev) -> Tensor<CubeWgpu16, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    // Parallel SIMD f32→f16 (F16C vcvtps2ph) — mirrors array2_to_tensor_f16.
    // This is the K/V upload hot path; the convert was ~75% of the per-call
    // upload cost as a scalar loop (gate-1 decomposition). The 2D sibling
    // was parallelised but this 3D path was left single-threaded; fan the
    // SIMD convert across cores via par_chunks into an uninit buffer (f16 is
    // a POD u16 wrapper, every element is written, so set_len is sound).
    let std = view.as_standard_layout();
    let src = std.as_slice().expect("standard-layout slice is contiguous");
    let mut dst: Vec<f16> = Vec::with_capacity(src.len());
    unsafe {
        dst.set_len(src.len());
    }
    use rayon::prelude::*;
    const CHUNK: usize = 1 << 16;
    dst.par_chunks_mut(CHUNK)
        .zip(src.par_chunks(CHUNK))
        .for_each(|(d, s)| d.convert_from_f32_slice(s));
    Tensor::<CubeWgpu16, 3>::from_data(TensorData::new(dst, [b, m, k]), device)
}

/// bf16 sibling of [`array3_to_tensor_f16`] for the cubek bf16 storage
/// path. Same SIMD `convert_from_f32_slice` (half implements
/// `HalfFloatSliceExt` for `bf16` too); only the element type differs.
fn array3_to_tensor_bf16(view: ArrayView3<'_, f32>, device: &Dev) -> Tensor<CubeWgpuBf16, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    let std = view.as_standard_layout();
    let src = std.as_slice().expect("standard-layout slice is contiguous");
    let mut dst = vec![bf16::ZERO; src.len()];
    dst.convert_from_f32_slice(src);
    Tensor::<CubeWgpuBf16, 3>::from_data(TensorData::new(dst, [b, m, k]), device)
}

fn tensor2_to_array_f16(t: Tensor<CubeWgpu16, 2>) -> Result<Array2<f32>> {
    let shape = t.dims();
    let v_f16: Vec<f16> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f16 tensor → Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor3_to_array_f16(t: Tensor<CubeWgpu16, 3>) -> Result<Array3<f32>> {
    let shape = t.dims();
    let v_f16: Vec<f16> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f16 tensor → Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

fn registered_weight_f32(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>,
    handle: WeightHandle,
    input_cols: usize,
    op: &str,
) -> Result<Tensor<CubeWgpu32, 2>> {
    let weight = map
        .get(&handle)
        .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
        .clone();
    if input_cols != weight.dims()[0] {
        return Err(anyhow!(
            "{op} shape mismatch on {handle:?}: input cols {input_cols} != weight rows {}",
            weight.dims()[0]
        ));
    }
    Ok(weight)
}

fn registered_weight_f16(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>,
    handle: WeightHandle,
    input_cols: usize,
    op: &str,
) -> Result<Tensor<CubeWgpu16, 2>> {
    let weight = map
        .get(&handle)
        .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
        .clone();
    if input_cols != weight.dims()[0] {
        return Err(anyhow!(
            "{op} shape mismatch on {handle:?}: input cols {input_cols} != weight rows {}",
            weight.dims()[0]
        ));
    }
    Ok(weight)
}

fn registered_weights_f32(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>,
    handles: &[WeightHandle],
    input_cols: usize,
    op: &str,
) -> Result<Vec<Tensor<CubeWgpu32, 2>>> {
    handles
        .iter()
        .map(|h| registered_weight_f32(map, *h, input_cols, op))
        .collect()
}

fn registered_weights_f16(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>,
    handles: &[WeightHandle],
    input_cols: usize,
    op: &str,
) -> Result<Vec<Tensor<CubeWgpu16, 2>>> {
    handles
        .iter()
        .map(|h| registered_weight_f16(map, *h, input_cols, op))
        .collect()
}

fn tensor_data_to_array_f32(data: TensorData, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let v: Vec<f32> = data
        .into_vec()
        .map_err(|e| anyhow!("burn f32 TensorData -> Vec<f32>: {e:?}"))?;
    Array2::from_shape_vec((rows, cols), v).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor_data_to_array3_f16(data: TensorData, dims: (usize, usize, usize)) -> Result<Array3<f32>> {
    let v_f16: Vec<f16> = data
        .into_vec()
        .map_err(|e| anyhow!("burn f16 TensorData -> Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array3::from_shape_vec((dims.0, dims.1, dims.2), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

fn tensor_data_to_array_f16(data: TensorData, rows: usize, cols: usize) -> Result<Array2<f32>> {
    // `engine:matmul:readback` — host DtoH copy out of the device result
    // (same sub-bucket as the f16-out path; the f16→f32 widen below is the
    // f32-output path's extra, kept out of the bucket).
    let v_f16: Vec<f16> = gelo_protocol::profile::time("engine:matmul:readback", || {
        data.into_vec()
            .map_err(|e| anyhow!("burn f16 TensorData -> Vec<f16>: {e:?}"))
    })?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array2::from_shape_vec((rows, cols), v).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

/// **bf16-output** read-back from an f16 device tensor. The GPU result
/// is f16 on the wire either way; this narrows to a **bf16** host array
/// (half the bytes of the f32 read-back) so the TEE-side mask unapply
/// can run on bf16 storage (the fused `Dct4Mask::unapply_bf16_into_f32_rows`),
/// halving the DRAM traffic of the dominant `mask_unapply` bucket. The
/// f16 → bf16 hop loses no information the f16 GPU result didn't already
/// carry (both are 16-bit; bf16 trades mantissa for exponent range).
fn tensor_data_to_array_f16_raw(
    data: TensorData,
    rows: usize,
    cols: usize,
) -> Result<Array2<f16>> {
    // `engine:matmul:readback` = the mandatory host copy of the DtoH'd
    // result into the enclave (the mask-unapply consumes the secret covers,
    // so it must run in-TEE — `TEE-TRUST`). burn's `into_vec` can't take the
    // pinned read-back buffer by value (it's cubecl's pooled pinned alloc),
    // so it would `to_vec` into a FRESH per-call `Vec` — an `mmap` that
    // page-faults in and is `munmap`'d on drop, re-faulting every call
    // (~2.2 GB/s, ~22 s of the B=8 prefill — chronicle §24). Instead we view
    // the pinned bytes zero-copy and copy into a page-resident buffer from
    // `readback_pool` (~13 GB/s); the in-TEE unapply consumer returns the
    // buffer to the pool once drained.
    let v_f16: Vec<f16> = gelo_protocol::profile::time("engine:matmul:readback", || {
        let src = data
            .as_slice::<f16>()
            .map_err(|e| anyhow!("burn f16 TensorData -> &[f16]: {e:?}"))?;
        let mut buf = gelo_protocol::readback_pool::take(src.len());
        buf.copy_from_slice(src);
        Ok::<Vec<f16>, anyhow::Error>(buf)
    })?;
    Array2::from_shape_vec((rows, cols), v_f16).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

/// bf16-output read-back from an f32 device tensor (F32 weight store).
fn tensor_data_to_array_f16_from_f32(
    data: TensorData,
    rows: usize,
    cols: usize,
) -> Result<Array2<f16>> {
    let v_f32: Vec<f32> = data
        .into_vec()
        .map_err(|e| anyhow!("burn f32 TensorData -> Vec<f32>: {e:?}"))?;
    // Scalar map (see the f16 sibling above for why the SIMD slice
    // convert is not used here — alloc churn regresses production).
    let v: Vec<f16> = v_f32.into_iter().map(f16::from_f32).collect();
    Array2::from_shape_vec((rows, cols), v).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn execute_registered_many_f32(
    lhs: Tensor<CubeWgpu32, 2>,
    weights: Vec<Tensor<CubeWgpu32, 2>>,
) -> Result<Vec<Array2<f32>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu32>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    tx.execute()
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f32(data, rows, cols))
        .collect()
}

fn execute_registered_many_f16(
    lhs: Tensor<CubeWgpu16, 2>,
    weights: Vec<Tensor<CubeWgpu16, 2>>,
) -> Result<Vec<Array2<f32>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu16>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    // `engine:matmul:drain` — GPU sync (kernel enqueue is lazy; this is where
    // the matmuls run + are read back). Same sub-bucket as the f16-out path,
    // so decode (f32-output) and prefill (f16-out) report consistently.
    let datas = gelo_protocol::profile::time("engine:matmul:drain", || tx.execute());
    datas
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f16(data, rows, cols))
        .collect()
}

/// bf16-output sibling of [`execute_registered_many_f32`]: same fused
/// transaction, narrows each output to a bf16 host array.
fn execute_registered_many_f32_to_f16(
    lhs: Tensor<CubeWgpu32, 2>,
    weights: Vec<Tensor<CubeWgpu32, 2>>,
) -> Result<Vec<Array2<f16>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu32>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    tx.execute()
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f16_from_f32(data, rows, cols))
        .collect()
}

/// bf16-output sibling of [`execute_registered_many_f16`]: same fused
/// transaction, narrows each f16 output to a bf16 host array (half the
/// read-back bytes of the f32 path).
fn execute_registered_many_f16_raw(
    lhs: Tensor<CubeWgpu16, 2>,
    weights: Vec<Tensor<CubeWgpu16, 2>>,
) -> Result<Vec<Array2<f16>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu16>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    // `engine:matmul:drain` = the actual GPU sync (kernel enqueue is lazy;
    // `tx.execute` is where the queued matmuls run + are read back). On the
    // B=8 prefill this is only ~5.3 s — i.e. the GPU work itself is small;
    // the parent bucket is dominated by host marshalling (chronicle §24).
    let datas = gelo_protocol::profile::time("engine:matmul:drain", || tx.execute());
    datas
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f16_raw(data, rows, cols))
        .collect()
}

fn submit_registered_many_f32(
    lhs: Tensor<CubeWgpu32, 2>,
    weights: Vec<Tensor<CubeWgpu32, 2>>,
) -> Vec<MatmulToken> {
    weights
        .into_iter()
        .map(|w| {
            let pending = lhs.clone().matmul(w);
            MatmulToken::from_fn(move || tensor2_to_array_f32(pending))
        })
        .collect()
}

fn submit_registered_many_f16(
    lhs: Tensor<CubeWgpu16, 2>,
    weights: Vec<Tensor<CubeWgpu16, 2>>,
) -> Vec<MatmulToken> {
    weights
        .into_iter()
        .map(|w| {
            let pending = lhs.clone().matmul(w);
            MatmulToken::from_fn(move || tensor2_to_array_f16(pending))
        })
        .collect()
}

// ─── GpuOffloadEngine impl ─────────────────────────────────────────────

impl GpuOffloadEngine for WgpuVulkanEngine {
    // ── Resident K/V session (perm-attn-gpu-offload Phase 2) ─────────
    // Engine-owned; the session map is shared across clone_shared
    // handles so the forward path's per-layer engine clones see it.

    fn kv_create_session(
        &self,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        capacity: usize,
    ) -> Result<KvSessionId> {
        let session = self.create_kv_session(k, v, capacity)?;
        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        self.sessions.lock().expect("sessions mutex").insert(id, session);
        Ok(id)
    }

    fn kv_append(
        &self,
        id: KvSessionId,
        k_row: ArrayView3<'_, f32>,
        v_row: ArrayView3<'_, f32>,
    ) -> Result<()> {
        let mut map = self.sessions.lock().expect("sessions mutex");
        let s = map
            .get_mut(&id)
            .ok_or_else(|| anyhow!("kv_append: unknown session {id}"))?;
        self.append_kv(s, k_row, v_row)
    }

    fn kv_attend(
        &self,
        id: KvSessionId,
        q: ArrayView3<'_, f32>,
        scale: f32,
    ) -> Result<Array3<f32>> {
        let map = self.sessions.lock().expect("sessions mutex");
        let s = map
            .get(&id)
            .ok_or_else(|| anyhow!("kv_attend: unknown session {id}"))?;
        self.attend_session(q, s, scale)
    }

    fn kv_attend_partial(
        &self,
        id: KvSessionId,
        q: ArrayView3<'_, f32>,
        scale: f32,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let map = self.sessions.lock().expect("sessions mutex");
        let s = map
            .get(&id)
            .ok_or_else(|| anyhow!("kv_attend_partial: unknown session {id}"))?;
        self.attend_session_partial(q, s, scale)
    }

    fn kv_set_mask(&self, id: KvSessionId, mask: ArrayView3<'_, f32>) -> Result<()> {
        let mut map = self.sessions.lock().expect("sessions mutex");
        let s = map
            .get_mut(&id)
            .ok_or_else(|| anyhow!("kv_set_mask: unknown session {id}"))?;
        s.mask = Some(array3_to_tensor_f16(mask, &self.device));
        Ok(())
    }

    fn kv_refresh_block(
        &self,
        id: KvSessionId,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
    ) -> Result<()> {
        let cap = self
            .sessions
            .lock()
            .expect("sessions mutex")
            .get(&id)
            .map(|s| s.capacity)
            .ok_or_else(|| anyhow!("kv_refresh_block: unknown session {id}"))?;
        let fresh = self.create_kv_session(k, v, cap)?;
        self.sessions.lock().expect("sessions mutex").insert(id, fresh);
        Ok(())
    }

    fn kv_drop_session(&self, id: KvSessionId) -> Result<()> {
        self.sessions.lock().expect("sessions mutex").remove(&id);
        Ok(())
    }

    fn supports_offloaded_attention(&self) -> bool {
        // The fused cubek offload requires the fp16 engine; the f32 engine
        // falls back to in-TEE attention.
        self.fp16
    }

    fn prefers_f16_output(&self) -> bool {
        // In fp16 mode the registered-linear matmul output is already
        // 16-bit on the device, so the bf16 read-back loses nothing the
        // f16 result didn't already carry — opt the offload into the
        // bf16 unapply path (halves the `mask_unapply` DRAM traffic).
        // The f32 engine keeps the exact f32 read-back.
        self.fp16
    }

    fn cubek_causal_attend(
        &self,
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        group: usize,
        scale: f32,
    ) -> Result<Array3<f32>> {
        if !self.fp16 {
            return Err(anyhow!("cubek_causal_attend requires the fp16 engine"));
        }
        // cubek runs an independent client on the default device (same
        // adapter as this engine). The caller passes rotation-covered,
        // UN-REPLICATED K/V; the GQA broadcast to Hq happens on-device.
        Ok(cubek_attention_folded_gqa(q, k, v, group, scale, true))
    }

    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<'_, f32>) -> Result<()> {
        let mut guard = self.weights.lock().unwrap();
        match &mut *guard {
            WeightStore::F32(map) => {
                let t = array2_to_tensor_f32(weight, &self.device);
                map.insert(handle, t);
            }
            WeightStore::F16(map) => {
                let t = array2_to_tensor_f16(weight, &self.device);
                map.insert(handle, t);
            }
        }
        Ok(())
    }

    fn register_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<'_, bf16>,
    ) -> Result<()> {
        let mut guard = self.weights.lock().unwrap();
        match &mut *guard {
            WeightStore::F32(map) => {
                let t = array2_bf16_to_tensor_f32(weight, &self.device);
                map.insert(handle, t);
            }
            WeightStore::F16(map) => {
                let t = array2_bf16_to_tensor_f16(weight, &self.device);
                map.insert(handle, t);
            }
        }
        Ok(())
    }

    fn matmul(&self, handle: WeightHandle, input: ArrayView2<'_, f32>) -> Result<Array2<f32>> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                tensor2_to_array_f32(lhs.matmul(weight))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                tensor2_to_array_f16(lhs.matmul(weight))
            }
        }
    }

    fn matmul_many(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, f32>,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                execute_registered_many_f32(lhs, weights)
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                execute_registered_many_f16(lhs, weights)
            }
        }
    }

    /// **bf16-output** sibling of [`Self::matmul_many`]. Same f32 input
    /// upload + fused dispatch; the GPU result (f16 on the wire) is
    /// narrowed to a **bf16** host array instead of f32. Halves the
    /// read-back host buffer so the TEE-side mask unapply runs on bf16
    /// storage — the dominant `mask_unapply` bucket (perm-attn-gpu-offload,
    /// chronicle §12/§13 bf16-offload lever).
    fn matmul_many_f16_out(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, f32>,
    ) -> Result<Vec<Array2<f16>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many_f16_out")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                execute_registered_many_f32_to_f16(lhs, weights)
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many_f16_out")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                execute_registered_many_f16_raw(lhs, weights)
            }
        }
    }

    /// **R4 async** override. Splits the existing sync `matmul` into
    /// (upload + kernel issue) and (download), returning a
    /// [`MatmulToken`] whose closure captures the pending burn-tensor.
    ///
    /// The kernel issue (`lhs.matmul(weight)`) is non-blocking on
    /// burn-cubecl — the actual GPU sync happens inside the closure
    /// when the substrate drains the token via `into_array`. This
    /// frees the substrate's calling thread to run shield/cascade
    /// work for the next offload site while the GPU is busy.
    ///
    /// Plan: `docs/plans/m1-12-r4-async-overlap.md` §B.
    fn matmul_async(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> Result<MatmulToken> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                let pending = lhs.matmul(weight);
                Ok(MatmulToken::from_fn(move || tensor2_to_array_f32(pending)))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                let pending = lhs.matmul(weight);
                Ok(MatmulToken::from_fn(move || tensor2_to_array_f16(pending)))
            }
        }
    }

    /// **R4 async** override for `matmul_many`. Shares one upload of
    /// `input` across all N kernel launches (same algebra as the sync
    /// `matmul_many` but each output is captured into its own token
    /// rather than batched via [`Transaction`]). The first token
    /// drained triggers a device sync that completes *all* N kernels;
    /// subsequent token drains just read pre-completed buffers, so
    /// the bus savings of the sync path are preserved.
    fn matmul_many_async(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, f32>,
    ) -> Result<Vec<MatmulToken>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                Ok(submit_registered_many_f32(lhs, weights))
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                Ok(submit_registered_many_f16(lhs, weights))
            }
        }
    }

    /// Path β bf16-input override. The bf16 → device-precision
    /// conversion runs once during the upload Vec build via the
    /// existing `array2_bf16_to_tensor_*` helpers — no transient
    /// host f32 buffer is materialised. Compared to the default
    /// trait impl (`bf16 → f32 → f16 upload`), this saves one
    /// full-tensor DRAM pass at the substrate boundary.
    fn matmul_bf16_input(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, bf16>,
    ) -> Result<Array2<f32>> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f32(input, &self.device);
                tensor2_to_array_f32(lhs.matmul(weight))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f16(input, &self.device);
                tensor2_to_array_f16(lhs.matmul(weight))
            }
        }
    }

    /// Path β bf16-input variant of [`Self::matmul_many`]. Same
    /// fused-dispatch structure as the f32 path — one upload of the
    /// bf16 input (no f32 intermediate), shared across all N kernel
    /// launches via the Transaction-batched download.
    fn matmul_many_bf16_input(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, bf16>,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f32(input, &self.device);
                execute_registered_many_f32(lhs, weights)
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f16(input, &self.device);
                execute_registered_many_f16(lhs, weights)
            }
        }
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<'_, f32>,
        rhs: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        if lhs.ncols() != rhs.nrows() {
            return Err(anyhow!(
                "matmul_dynamic shape mismatch: lhs cols {} != rhs rows {}",
                lhs.ncols(),
                rhs.nrows()
            ));
        }
        if self.fp16 {
            let lhs_t = array2_to_tensor_f16(lhs, &self.device);
            let rhs_t = array2_to_tensor_f16(rhs, &self.device);
            tensor2_to_array_f16(lhs_t.matmul(rhs_t))
        } else {
            let lhs_t = array2_to_tensor_f32(lhs, &self.device);
            let rhs_t = array2_to_tensor_f32(rhs, &self.device);
            tensor2_to_array_f32(lhs_t.matmul(rhs_t))
        }
    }

    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<'_, f32>,
        rhs: ArrayView3<'_, f32>,
    ) -> Result<Array3<f32>> {
        let b = lhs.shape()[0];
        let lhs_k = lhs.shape()[2];
        let rhs_k = rhs.shape()[1];
        if rhs.shape()[0] != b || rhs_k != lhs_k {
            return Err(anyhow!(
                "matmul_dynamic_batched shape mismatch: lhs {:?} vs rhs {:?}",
                lhs.shape(),
                rhs.shape()
            ));
        }
        if self.fp16 {
            let lhs_t = array3_to_tensor_f16(lhs, &self.device);
            let rhs_t = array3_to_tensor_f16(rhs, &self.device);
            tensor3_to_array_f16(lhs_t.matmul(rhs_t))
        } else {
            let lhs_t = array3_to_tensor_f32(lhs, &self.device);
            let rhs_t = array3_to_tensor_f32(rhs, &self.device);
            tensor3_to_array_f32(lhs_t.matmul(rhs_t))
        }
    }

    fn softmax_batched(&self, input: ArrayView3<'_, f32>) -> Result<Array3<f32>> {
        // Last-axis softmax via burn_tensor::activation::softmax. Runs on
        // the wgpu device — used by permutation-shielded attention so the
        // softmax doesn't bounce back to the TEE between Q·Kᵀ and ·V.
        if self.fp16 {
            let t = array3_to_tensor_f16(input, &self.device);
            tensor3_to_array_f16(activation::softmax(t, 2))
        } else {
            let t = array3_to_tensor_f32(input, &self.device);
            tensor3_to_array_f32(activation::softmax(t, 2))
        }
    }

    /// **A1 (Phase 1b enabler)** — single-dispatch-chain fused
    /// attention.  Uploads Q, K, V, mask **once**; runs
    /// `Q·Kᵀ → scale → +mask → softmax → ·V` entirely on-device via
    /// chained `burn::Tensor` ops; downloads the output **once**.
    ///
    /// The five sub-ops still execute as separate burn kernels (no
    /// FlashAttention-style single-pass fusion — that would need a
    /// hand-rolled CubeCL kernel against the `O(B·n_q·n_kv)` scores
    /// intermediate).  But all intermediates live in GPU device memory
    /// — no GPU↔CPU round-trips between sub-ops, no K^T staging on
    /// the host.  This is the load-bearing change against the trait's
    /// default impl, which materialises `K^T` and `scores` host-side
    /// between dispatches.
    ///
    /// At decode m=1 the dominant residual cost is per-kernel launch
    /// latency on Vulkan (~0.2-0.5 ms per dispatch on Strix Halo).
    /// Removing those further requires either (a) a hand-rolled
    /// FlashAttention-style kernel that runs the chain in one
    /// dispatch, or (b) burn-cubecl's operator-fusion pass picking up
    /// the chain — under investigation.
    fn fused_attention_batched(
        &self,
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        scale: f32,
        mask: Option<ArrayView3<'_, f32>>,
    ) -> Result<Array3<f32>> {
        let (b, n_q, d_head) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let n_kv = k.shape()[1];
        debug_assert_eq!(q.shape()[0], b);
        debug_assert_eq!(k.shape(), &[b, n_kv, d_head]);
        debug_assert_eq!(v.shape(), &[b, n_kv, d_head]);
        if let Some(m) = mask {
            debug_assert_eq!(m.shape(), &[b, n_q, n_kv]);
        }

        if self.fp16 {
            let q_t = array3_to_tensor_f16(q, &self.device);
            let k_t = array3_to_tensor_f16(k, &self.device);
            let v_t = array3_to_tensor_f16(v, &self.device);
            // Device-side K^T via permute (no host transpose).
            let kt = k_t.permute([0, 2, 1]);
            let scores = q_t.matmul(kt);
            // A2: mask is None at decode (no-op) — skip the upload +
            // add-kernel-dispatch entirely.
            let scores = match mask {
                Some(m) => {
                    let mask_t = array3_to_tensor_f16(m, &self.device);
                    scores.mul_scalar(scale).add(mask_t)
                }
                None => scores.mul_scalar(scale),
            };
            let probs = activation::softmax(scores, 2);
            let out = probs.matmul(v_t);
            tensor3_to_array_f16(out)
        } else {
            let q_t = array3_to_tensor_f32(q, &self.device);
            let k_t = array3_to_tensor_f32(k, &self.device);
            let v_t = array3_to_tensor_f32(v, &self.device);
            let kt = k_t.permute([0, 2, 1]);
            let scores = q_t.matmul(kt);
            let scores = match mask {
                Some(m) => {
                    let mask_t = array3_to_tensor_f32(m, &self.device);
                    scores.mul_scalar(scale).add(mask_t)
                }
                None => scores.mul_scalar(scale),
            };
            let probs = activation::softmax(scores, 2);
            let out = probs.matmul(v_t);
            tensor3_to_array_f32(out)
        }
    }
}

/// Storage dtype for the cubek attention operands + intermediate tiles,
/// selected at runtime via `GELO_CUBEK_DTYPE` (`f16` default, `bf16`,
/// `f32`). cubek's public API only exposes the operand/out global dtype
/// (`AttentionGlobalTypes::from_single_dtype`) — the score/softmax tile
/// precisions are derived from it — so this is a whole-pipeline switch,
/// not a per-tile one.
///
/// - **f16** — the original 2-byte path; 10 mantissa bits but the 5-bit
///   exponent (max 65 504) overflows on real Qwen3-4B activations → NaN.
/// - **bf16** — the fix: 2-byte, f32-range 8-bit exponent (no overflow),
///   at 7 mantissa bits (a pass@1 risk gated at acceptance, not here).
/// - **f32** — 4-byte fallback: full range *and* precision, but ~2× the
///   upload (the dominant cost), so the prefill speedup roughly halves.
///   Kept as the in-pocket fallback if bf16 regresses pass@1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CubekDtype {
    F16,
    Bf16,
    F32,
}

impl CubekDtype {
    fn from_env() -> Self {
        match std::env::var("GELO_CUBEK_DTYPE").as_deref() {
            Ok("bf16") => CubekDtype::Bf16,
            Ok("f32") => CubekDtype::F32,
            _ => CubekDtype::F16,
        }
    }

    /// cubecl element `Type` for this dtype (cubek 0.2.0 `from_single_float_dtype`
    /// and `TensorHandle::new` take `impl Into<Type>`, not a raw `StorageType`).
    fn cube_type(self) -> cubecl::ir::Type {
        use cubecl::prelude::CubePrimitive;
        match self {
            CubekDtype::F16 => f16::as_type_native_unchecked(),
            CubekDtype::Bf16 => bf16::as_type_native_unchecked(),
            CubekDtype::F32 => f32::as_type_native_unchecked(),
        }
    }

    fn elem_size(self) -> usize {
        match self {
            CubekDtype::F16 | CubekDtype::Bf16 => 2,
            CubekDtype::F32 => 4,
        }
    }

    /// f32 host slice → LE bytes of this dtype, ready for cubek upload.
    fn f32_to_bytes(self, src: &[f32]) -> Vec<u8> {
        match self {
            CubekDtype::F16 => {
                let mut dst = vec![f16::ZERO; src.len()];
                dst.convert_from_f32_slice(src);
                bytemuck::cast_slice(&dst).to_vec()
            }
            CubekDtype::Bf16 => {
                let mut dst = vec![bf16::ZERO; src.len()];
                dst.convert_from_f32_slice(src);
                bytemuck::cast_slice(&dst).to_vec()
            }
            CubekDtype::F32 => bytemuck::cast_slice(src).to_vec(),
        }
    }

    /// cubek readback LE bytes of this dtype → f32 host vec (`n` elems).
    fn bytes_to_f32(self, bytes: &[u8], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n];
        match self {
            CubekDtype::F16 => {
                let h: &[f16] = bytemuck::cast_slice(&bytes[..n * 2]);
                h.convert_to_f32_slice(&mut out);
            }
            CubekDtype::Bf16 => {
                let h: &[bf16] = bytemuck::cast_slice(&bytes[..n * 2]);
                h.convert_to_f32_slice(&mut out);
            }
            CubekDtype::F32 => {
                out.copy_from_slice(bytemuck::cast_slice(&bytes[..n * 4]));
            }
        }
        out
    }
}

/// Max |element| of an f32 view — cheap operand-magnitude probe for the
/// which-tile-overflows diagnosis (logged under `GELO_DEBUG_OFFLOAD`).
fn max_abs3(view: ArrayView3<'_, f32>) -> f32 {
    view.iter().fold(0f32, |a, &x| a.max(x.abs()))
}

/// Minimum `n_kv` for the `blackbox` (tensor-core) kernel: below the seq_kv MMA
/// tile there is no valid instruction ("Matmul is not supported: no tile size"),
/// so tiny problems route to the portable `Unit` kernel. (The n-sweep gate
/// confirms blackbox clean for n_kv ≥ 16; n_kv=1 fails.)
const BLACKBOX_MIN_NKV: usize = 16;
/// `seq_q` stage-tile extent the blackbox kernel requires `seq_q` to be a
/// multiple of (`num_planes · partition_seq_q · tile_m` = 1·1·16 for the hint
/// below + the fp16 tensor-core instruction). Ragged `seq_q` is zero-padded up
/// to this and sliced back; `Unit` handles ragged `seq_q` natively.
const BLACKBOX_SEQ_Q_ALIGN: usize = 16;

/// Resolve the cubek attend strategy for a problem with `n_kv` keys, and whether
/// it is the tensor-core (`blackbox`) kernel.
///
/// **Default: blackbox** (cooperative-matmul / tensor cores) on **both** the CUDA
/// and Vulkan backends for `n_kv ≥ BLACKBOX_MIN_NKV`; **Unit** (portable,
/// ragged-safe) for tiny `n_kv` below the tile floor and for an explicit
/// `CUBEK_STRATEGY=unit`. `CUBEK_STRATEGY` (`blackbox`/`unit`) overrides the
/// default, but the tiny-`n_kv` guard still applies so blackbox is never launched
/// below the floor. cubek 0.2.0's `BlackboxAcceleratedStrategy` is no longer
/// `Default`, so the `Inferred` hint supplies minimal partition counts (1 each).
///
/// Blackbox was NaN-broken on the cubecl-wgpu / Vulkan backend on cubecl 0.9.0
/// (data-independent NaN for all n≥3 — a SPIR-V cooperative-matrix codegen bug,
/// not an f16 overflow); the cubek 0.2.0 / cubecl 0.10.0 bump fixed it. Verified
/// clean on this RTX 5090 via Vulkan (`cubek_gqa_nan_nsweep`, tile-aligned
/// n=16…2048, 0 NaN), so blackbox is now the default on Vulkan too. Note this is
/// cooperative-matrix-capable hardware (Nvidia via `VK_KHR_cooperative_matrix`);
/// Vulkan GPUs without it still need `CUBEK_STRATEGY=unit`.
fn cubek_strategy(n_kv: usize) -> (cubek_attention::launch::Strategy, bool) {
    use cubek_attention::launch::{BlueprintStrategy, Strategy};
    use cubek_attention::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;
    let want_blackbox = match std::env::var("CUBEK_STRATEGY").as_deref() {
        Ok("blackbox") => true,
        Ok("unit") => false,
        // Unset → production default: blackbox (tensor cores) on both backends.
        _ => true,
    };
    if want_blackbox && n_kv >= BLACKBOX_MIN_NKV {
        (
            Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
                BlackboxAcceleratedStrategy {
                    num_planes: 1,
                    seq_q: 1,
                    seq_kv: 1,
                },
            )),
            true,
        )
    } else {
        (Strategy::Unit(BlueprintStrategy::Inferred(())), false)
    }
}

/// Zero-pad `q`'s `seq_q` (axis 1) up to a multiple of `align`, returning the
/// padded copy only when padding is actually needed (`None` if already
/// aligned). The phantom query rows are zero vectors: under causal masking they
/// attend the real keys (`key ≤ query`, top-left aligned — cubek masks on
/// absolute positions, so the real rows' masking is unchanged), yielding a
/// finite softmax, and the caller slices them off the output. `seq_kv` is left
/// ragged — cubek's K/V staged reader masks it via `check_bounds.seq_kv`.
fn pad_seq_q_zeros(q: ArrayView3<'_, f32>, align: usize) -> Option<Array3<f32>> {
    let (bh, n_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let rem = n_q % align;
    if rem == 0 {
        return None;
    }
    let n_q_pad = n_q + (align - rem);
    let mut padded = Array3::zeros((bh, n_q_pad, d));
    padded.slice_mut(ndarray::s![.., ..n_q, ..]).assign(&q);
    Some(padded)
}

/// Folded-head attention via the `cubek-attention` portable kernel.
///
/// Treats the leading dim of `q`/`k`/`v` (`[B*Hq, n_q, d]` /
/// `[B*Hq, n_kv, d]`) as cubek's `batch` with `num_heads = 1` — each
/// folded head is an independent attention problem. K/V must already be
/// GQA-expanded to `Hq` by the caller, so the fold is uniform.
///
/// Inputs are f32 host arrays; converted to f16 LE bytes, uploaded as
/// device tensors, attended with `Strategy::Unit(BlueprintStrategy::Inferred(()))`
/// (the portable, non-tensor-core kernel), then read back and converted
/// f16 → f32. Returns `[B*Hq, n_q, d]`.
///
/// Acquires its own cubecl client on the default device (mirrors the
/// `cubek_attention_spike` test); independent of `WgpuVulkanEngine`
/// internals.
pub fn cubek_attention_folded(
    q: ArrayView3<'_, f32>,
    k: ArrayView3<'_, f32>,
    v: ArrayView3<'_, f32>,
    scale: f32,
    causal: bool,
) -> Array3<f32> {
    use cubecl::std::tensor::TensorHandle;
    use cubek_attention::definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions};
    use cubek_attention::launch::launch_ref;

    let bh = q.shape()[0];
    let n_q_real = q.shape()[1];
    let d = q.shape()[2];
    let n_kv = k.shape()[1];
    assert_eq!(k.shape()[0], bh, "K leading dim must match Q (GQA-expanded)");
    assert_eq!(v.shape()[0], bh);
    assert_eq!(k.shape()[2], d);
    assert_eq!(v.shape()[2], d);
    let _ = scale; // cubek derives scale = 1/sqrt(head_dim) internally.

    // Resolve the cubek strategy (blackbox tensor cores by default on both
    // backends for n_kv ≥ the tile floor; Unit otherwise). For blackbox, zero-pad seq_q to the
    // stage tile and slice the real rows back below (ragged-prompt support);
    // no-op for Unit / already-aligned seq_q.
    let (strategy, is_blackbox) = cubek_strategy(n_kv);
    let q_cow: ndarray::CowArray<'_, f32, ndarray::Ix3> =
        match is_blackbox
            .then(|| pad_seq_q_zeros(q, BLACKBOX_SEQ_Q_ALIGN))
            .flatten()
        {
            Some(p) => p.into(),
            None => q.into(),
        };
    let q = q_cow.view();
    let n_q = q.shape()[1];

    let device = Dev::default();
    let client = <Rt as cubecl::Runtime>::client(&device);

    // Storage dtype (f16 default / bf16 fix / f32 fallback) — whole-pipeline,
    // since the score/softmax tiles derive from the operand dtype.
    let dtype = CubekDtype::from_env();
    let dtype_ty = dtype.cube_type();
    let global_dtypes =
        AttentionGlobalTypes::from_single_float_dtype(dtype_ty, AttentionGlobalTypes::mask_dtype(&client));

    // Per-stage preparation breakdown (perm-attn-gpu-offload Phase 5a). The
    // cubek attend bucket lumps convert + upload + compute + readback into
    // one number; `CUBEK_PROFILE=1` decomposes it with device sync barriers
    // between stages (the barriers serialise otherwise-overlapped work, so
    // they perturb the wall slightly but cleanly attribute each stage). Off
    // by default → no extra syncs, identical to the un-instrumented path.
    let profile = std::env::var("CUBEK_PROFILE").as_deref() == Ok("1");
    let sync_barrier = |reason: &str| {
        if profile {
            let _ = client.flush();
            future::block_on(client.sync()).unwrap_or_else(|e| {
                panic!("cubek_attention_folded sync barrier ({reason}) failed: {e:?}")
            });
        }
    };

    // ── Stage 1: f32 → storage-dtype host convert via half's SIMD
    //    `convert_from_f32_slice` (F16C, runtime-detected) for f16/bf16, or a
    //    direct copy for f32 — the same path the engine's K/V upload uses
    //    (`array3_to_tensor_f16`). Timed per operand so the GQA-expanded K/V
    //    convert cost stays explicit. Returns LE bytes ready for upload.
    let to_bytes = |arr: ArrayView3<'_, f32>| -> Vec<u8> {
        let std = arr.as_standard_layout();
        let src = std.as_slice().expect("standard-layout slice is contiguous");
        dtype.f32_to_bytes(src)
    };

    let t = std::time::Instant::now();
    let q_bytes = to_bytes(q);
    let cvt_q = t.elapsed();
    let t = std::time::Instant::now();
    let k_bytes = to_bytes(k);
    let cvt_k = t.elapsed();
    let t = std::time::Instant::now();
    let v_bytes = to_bytes(v);
    let cvt_v = t.elapsed();

    // cubek shape: [batch, num_heads, seq, head_dim] with num_heads = 1.
    let q_shape = vec![bh, 1, n_q, d];
    let kv_shape = vec![bh, 1, n_kv, d];
    let out_shape = vec![bh, 1, n_q, d];
    let elem_size = dtype.elem_size();

    // ── Stage 2: host → device upload (DMA). Under the profile barrier the
    //    elapsed time is the real transfer; otherwise the alloc just enqueues.
    let t = std::time::Instant::now();
    let q_alloc = client.create_tensor_from_slice(&q_bytes, q_shape.clone().into(), elem_size);
    let k_alloc = client.create_tensor_from_slice(&k_bytes, kv_shape.clone().into(), elem_size);
    let v_alloc = client.create_tensor_from_slice(&v_bytes, kv_shape.clone().into(), elem_size);

    let q_tensor: TensorHandle<Rt> =
        TensorHandle::new(q_alloc.memory, q_shape, q_alloc.strides, dtype_ty);
    let k_tensor: TensorHandle<Rt> =
        TensorHandle::new(k_alloc.memory, kv_shape.clone(), k_alloc.strides, dtype_ty);
    let v_tensor: TensorHandle<Rt> =
        TensorHandle::new(v_alloc.memory, kv_shape, v_alloc.strides, dtype_ty);
    let out_tensor: TensorHandle<Rt> = TensorHandle::empty(&client, out_shape, dtype_ty);
    sync_barrier("upload");
    let upload_ms = t.elapsed();

    let options = AttentionOptions {
        causal,
        accumulator_precision: AccumulatorPrecision::default(),
    };

    // `strategy` (Unit vs blackbox) was resolved from `n_kv` at the head.

    // ── Stage 3: GPU attend (tiled online-softmax · V). The profile barrier
    //    isolates pure compute; without it this just enqueues and the cost
    //    surfaces in the readback sync below.
    let t = std::time::Instant::now();
    let out_handle = out_tensor.handle.clone();
    launch_ref::<Rt>(
        strategy,
        &client,
        q_tensor.binding(),
        k_tensor.binding(),
        v_tensor.binding(),
        None,
        out_tensor.binding(),
        &global_dtypes,
        options,
    )
    .expect("cubek_attention_folded launch failed");
    sync_barrier("compute");
    let compute_ms = t.elapsed();

    // ── Stage 4: device → host readback (includes the implicit sync if the
    //    profile barriers are off, in which case it absorbs upload+compute).
    let t = std::time::Instant::now();
    let out_bytes = client.read_one_unchecked(out_handle);
    let readback_ms = t.elapsed();

    // ── Stage 5: storage-dtype → f32 host convert of the output. f16/bf16
    //    SIMD-widen via `convert_to_f32_slice`; f32 is a copy (Phase-5a O1).
    let t = std::time::Instant::now();
    let n_out = bh * n_q * d;
    debug_assert_eq!(
        out_bytes.len(),
        n_out * elem_size,
        "cubek readback size mismatch"
    );
    let out_vec = dtype.bytes_to_f32(&out_bytes, n_out);
    let out = Array3::from_shape_vec((bh, n_q, d), out_vec)
        .expect("cubek out shape matches buffer");
    let cvt_out = t.elapsed();

    if profile {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let q_el = bh * n_q * d;
        let kv_el = bh * n_kv * d;
        eprintln!(
            "[cubek-prep] dtype={dtype:?} bh={bh} n_q={n_q} n_kv={n_kv} d={d}  \
             (Q {q_el} el, K/V {kv_el} el each)"
        );
        eprintln!(
            "  convert  Q {:7.2} ms  K {:7.2} ms  V {:7.2} ms  | total {:7.2} ms",
            ms(cvt_q),
            ms(cvt_k),
            ms(cvt_v),
            ms(cvt_q + cvt_k + cvt_v)
        );
        eprintln!("  upload (DMA, synced)        {:7.2} ms", ms(upload_ms));
        eprintln!("  compute (GPU attend)        {:7.2} ms", ms(compute_ms));
        eprintln!("  readback                    {:7.2} ms", ms(readback_ms));
        eprintln!("  convert-out (→f32)          {:7.2} ms", ms(cvt_out));
        let prep = cvt_q + cvt_k + cvt_v + upload_ms;
        eprintln!(
            "  ── prep (convert+upload) {:7.2} ms  vs compute {:7.2} ms  \
             (prep is {:.0}% of {:.2} ms)",
            ms(prep),
            ms(compute_ms),
            100.0 * ms(prep) / ms(prep + compute_ms + readback_ms + cvt_out),
            ms(prep + compute_ms + readback_ms + cvt_out)
        );
    }
    // Drop the phantom padded query rows (no-op when seq_q was already aligned).
    if n_q != n_q_real {
        out.slice(ndarray::s![.., ..n_q_real, ..]).to_owned()
    } else {
        out
    }
}

/// GQA-aware folded causal attention (perm-attn-gpu-offload Phase-5a O2).
/// Takes **un-replicated** K/V (`[Hkv, n_kv, d]`) plus `group = Hq/Hkv` and
/// `q` (`[Hq, n_q, d]`); converts + uploads K/V un-replicated (4× less host
/// convert and PCIe DMA than the GQA-expanded `cubek_attention_folded`), then
/// broadcasts them up to `Hq` **on-device** via `repeat_dim` — mirroring the
/// decode path's `resident_kv_expanded` — and bridges the burn tensors to
/// cubek's cubecl `TensorHandle`s (same client/device, so the buffers are
/// shared, no host round-trip). Returns `[Hq, n_q, d]`.
pub fn cubek_attention_folded_gqa(
    q: ArrayView3<'_, f32>,
    k: ArrayView3<'_, f32>,
    v: ArrayView3<'_, f32>,
    group: usize,
    scale: f32,
    causal: bool,
) -> Array3<f32> {
    use cubecl::std::tensor::TensorHandle;
    use cubek_attention::definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions};
    use cubek_attention::launch::launch_ref;

    let hq = q.shape()[0];
    let n_q_real = q.shape()[1];
    let d = q.shape()[2];
    let hkv = k.shape()[0];
    let n_kv = k.shape()[1];
    assert_eq!(v.shape()[0], hkv);
    assert_eq!(hq, hkv * group, "hq must equal hkv·group");
    let _ = scale; // cubek derives scale = 1/sqrt(head_dim) internally.

    // Resolve the cubek strategy (blackbox tensor cores by default on both
    // backends for n_kv ≥ the tile floor; Unit otherwise). For blackbox, zero-pad seq_q to the
    // stage tile and slice the real rows back below (ragged-prompt support);
    // no-op for Unit / already-aligned seq_q.
    let (strategy, is_blackbox) = cubek_strategy(n_kv);
    let q_cow: ndarray::CowArray<'_, f32, ndarray::Ix3> =
        match is_blackbox
            .then(|| pad_seq_q_zeros(q, BLACKBOX_SEQ_Q_ALIGN))
            .flatten()
        {
            Some(p) => p.into(),
            None => q.into(),
        };
    let q = q_cow.view();
    let n_q = q.shape()[1];

    let device = Dev::default();
    let client = <Rt as cubecl::Runtime>::client(&device);
    // Storage dtype (f16 default / bf16 fix / f32 fallback) — whole-pipeline.
    let dtype = CubekDtype::from_env();
    let dtype_ty = dtype.cube_type();
    let global_dtypes =
        AttentionGlobalTypes::from_single_float_dtype(dtype_ty, AttentionGlobalTypes::mask_dtype(&client));

    // Which-tile-overflows probe (GELO_DEBUG_OFFLOAD). The f16 storage NaN is
    // either the value tile (max|V| > 65 504) or the score tile (max|QKᵀᵀᵀ|/√d).
    // max|V| is exact; the score bound mq·mk·√d is a loose upper bound (no
    // matmul) that flags whether the score path *can* overflow f16. This tells
    // the post-mortem which fallback applies if the pass@1 gate fails: a
    // correctable V pre-scale (value-only) vs wider storage (score tile).
    if std::env::var("GELO_DEBUG_OFFLOAD").is_ok() {
        let (mq, mk, mv) = (max_abs3(q), max_abs3(k), max_abs3(v));
        let score_ub = mq * mk * (d as f32).sqrt();
        // Actual max |score| = max_{h,i,j} |Q[h,i,·]·K[h/group,j,·]| · (1/√d)
        // (cubek's internal scale). Cheap at probe shapes; only the max is
        // needed. exp(this) is what the f16 softmax tile must hold — f16
        // overflows at score ≈ 11.09 (exp = 65 520 > 65 504), so this, not the
        // operand range, is the real overflow seam (operands above are ≪ 65 504).
        let inv_sqrt_d = 1.0_f32 / (d as f32).sqrt();
        let mut score_max = 0.0_f32;
        for h in 0..hq {
            let kv = h / group;
            for i in 0..n_q {
                for j in 0..n_kv {
                    let mut dot = 0.0_f32;
                    for dd in 0..d {
                        dot += q[[h, i, dd]] * k[[kv, j, dd]];
                    }
                    score_max = score_max.max((dot * inv_sqrt_d).abs());
                }
            }
        }
        let exp_score = score_max.exp();
        eprintln!(
            "[cubek-tile] dtype={dtype:?} hq={hq} hkv={hkv} n_q={n_q} n_kv={n_kv} \
             max|Q|={mq:.2e} max|K|={mk:.2e} max|V|={mv:.2e} \
             score_ub≈{score_ub:.2e} |score|max={score_max:.2} exp={exp_score:.2e} \
             (f16 max 6.55e4)"
        );
    }

    // Un-replicated SIMD convert + upload, then expand K/V on-device
    // (reshape → repeat_dim → reshape), mirroring `resident_kv_expanded`; Q is
    // already per-q-head. The burn tensor element type differs per storage
    // dtype, so the build+expand+bridge is generated per dtype by `build`
    // (the on-device GQA broadcast and the `CubeTensor<Rt>` bridge are
    // identical across dtypes). `bf16` is the overflow fix; `f32` the fallback.
    macro_rules! build {
        ($conv:ident, $B:ty) => {{
            let q_b = $conv(q, &device).reshape([hq, 1, n_q, d]);
            let expand_kv = |t: Tensor<$B, 3>| -> Tensor<$B, 4> {
                if group > 1 {
                    t.reshape([hkv, 1, n_kv, d])
                        .repeat_dim(1, group)
                        .reshape([hq, 1, n_kv, d])
                } else {
                    t.reshape([hq, 1, n_kv, d])
                }
            };
            let k_b = expand_kv($conv(k, &device));
            let v_b = expand_kv($conv(v, &device));
            // Bridge burn `Tensor<$B, 4>` → cubek `TensorHandle<Rt>` via the
            // `From<CubeTensor<R>>` impl (burn-cubecl): the burn float primitive
            // *is* `CubeTensor<Rt>` for every `CubeBackend<Rt, _, _, _>`, so its
            // buffer handle is valid for cubek's launch on the same client
            // regardless of element type.
            let to_handle = |t: Tensor<$B, 4>| -> TensorHandle<Rt> {
                t.into_primitive().tensor().into()
            };
            (to_handle(q_b), to_handle(k_b), to_handle(v_b))
        }};
    }
    let (q_tensor, k_tensor, v_tensor) = match dtype {
        CubekDtype::F16 => build!(array3_to_tensor_f16, CubeWgpu16),
        CubekDtype::Bf16 => build!(array3_to_tensor_bf16, CubeWgpuBf16),
        CubekDtype::F32 => build!(array3_to_tensor_f32, CubeWgpu32),
    };
    let out_tensor: TensorHandle<Rt> = TensorHandle::empty(&client, vec![hq, 1, n_q, d], dtype_ty);

    let options = AttentionOptions {
        causal,
        accumulator_precision: AccumulatorPrecision::default(),
    };
    // `strategy` (Unit vs blackbox) was resolved from `n_kv` at the head.
    let out_handle = out_tensor.handle.clone();
    launch_ref::<Rt>(
        strategy,
        &client,
        q_tensor.binding(),
        k_tensor.binding(),
        v_tensor.binding(),
        None,
        out_tensor.binding(),
        &global_dtypes,
        options,
    )
    .expect("cubek_attention_folded_gqa launch failed");

    let out_bytes = client.read_one_unchecked(out_handle);
    let n_out = hq * n_q * d;
    debug_assert_eq!(
        out_bytes.len(),
        n_out * dtype.elem_size(),
        "cubek gqa readback size mismatch"
    );
    let out_vec = dtype.bytes_to_f32(&out_bytes, n_out);
    let out = Array3::from_shape_vec((hq, n_q, d), out_vec).expect("cubek gqa out shape matches buffer");
    // Drop the phantom padded query rows (no-op when seq_q was already aligned).
    if n_q != n_q_real {
        out.slice(ndarray::s![.., ..n_q_real, ..]).to_owned()
    } else {
        out
    }
}
