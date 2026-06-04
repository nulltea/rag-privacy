//! DCT-IV cascade mask — non-pow2 alternative to [`crate::hd3::Hd3Mask`].
//!
//! `A = D₃ · C · D₂ · C · D₁ · C` where each `C` is the orthonormal
//! DCT-IV transform of side length `n` and each `Dᵢ ∈ {-1,+1}^{n}` is
//! a fresh ±1 diagonal sampled per forward pass. The construction
//! mirrors [`crate::hd3::Hd3Mask`] with two structural differences:
//!
//! 1. **Works at arbitrary `n`** (no power-of-two requirement). DCT-IV
//!    has `O(n log n)` algorithms at any `n` (Bluestein / Lee-Wang).
//!    This eliminates the `s_pad = n.next_power_of_two()` pad penalty
//!    that costs ~50 % of TTFT for HD₃ at non-pow2 shapes (e.g.,
//!    n=2056 at Qwen3-4B: HD₃ pads to 4096 and pays 2× GPU GEMM cost;
//!    DCT-IV operates directly at n=2056).
//!
//! 2. **DCT-IV is self-inverse** (`C · C = I` exactly in the orthonormal
//!    form), so apply and unapply use the same transform with reversed
//!    cascade order. Same as Hadamard.
//!
//! ## Why DCT-IV (not DCT-II)
//!
//! DCT-II rows include a constant first row `C[0, :] = 1/√n` —
//! after `D₁ · C · x`, the first output coordinate is
//! `(D₁)_0 · (sum of x components) / √n`, which leaks information about
//! the input row-sum and one diagonal sign. DCT-IV has no constant
//! row; all rows are balanced cosine sequences with entries bounded
//! by `√(2/n)` (matching HD₃'s `1/√n` incoherence bound).
//!
//! See `docs/research/hd3-non-pow2-fix.md` §6 for the security
//! argument and threat-model survey that motivated DCT-IV over DCT-II,
//! block-diagonal HD₃, or other candidates.
//!
//! ## Cost (4B at n=2048, s = n + k_shield = 2056)
//!
//! | op | dense Haar | HD₃ at pow2 (s=2048) | HD₃ pad→4096 | DCT-IV (s=2056) |
//! |---|---|---|---|---|
//! | sample (per forward) | `O(s³)` | `O(s)` | `O(s)` | `O(s)` |
//! | apply / unapply (per call) | `O(s²·d)` | `O(s·d·log s)` | `O(s·d·log s_pad)` | `O(s·d·log s)` |
//! | GPU rows transmitted | `s` | `s` | `s_pad` (2× regression) | `s` (no pad) |
//!
//! DCT-IV per-call is ~3× slower than FWHT on CPU (one DCT-IV ≈ one
//! length-n real FFT plus pre/post twiddles via Bluestein at non-pow2
//! `n`), but eliminates the GPU pad regression entirely.
//!
//! ## References
//!
//! - Wang, *On Computing the Discrete Fourier and Cosine Transforms*, 1985 — O(N log N) DCT-IV recursion at any N.
//! - Tolimieri-An-Lu, *Algorithms for Discrete Fourier Transform and Convolution*, 1997 — FFT-via-DCT-IV equivalence.
//! - `rustdct` 0.7 — production DCT-IV implementation we delegate to.

use std::sync::Arc;

use half::f16;
use ndarray::Array2;
use rand::RngCore;
use rayon::prelude::*;
use rustdct::{Dct4, DctPlanner};

pub use crate::rng::MaskSeed;

/// Inner-DCT-IV work threshold above which `apply_in_place` parallelises
/// across columns via rayon. Below this the per-call rayon spawn
/// overhead (~100 µs) dominates the per-column DCT cost. Picked at
/// 2 048 columns × n rows ≈ 4 M FLOPs of inner work — slightly above
/// the threshold used in [`crate::hd3`].
const DCT4_RAYON_COL_THRESHOLD: usize = 64;

/// DCT-IV Hadamard-like cascade mask. Stores three ±1 diagonal vectors
/// of length `n` and an `Arc`-shared `Dct4<f32>` planner output.
///
/// `n` is unconstrained — works at any positive integer.
pub struct Dct4Mask {
    /// Side length the mask operates on.
    n: usize,
    /// First diagonal `D₁`, length `n`, ±1.0 values.
    d1: Vec<f32>,
    /// Second diagonal `D₂`, length `n`, ±1.0 values.
    d2: Vec<f32>,
    /// Third diagonal `D₃`, length `n`, ±1.0 values.
    d3: Vec<f32>,
    /// Per-pass normalisation collected across three DCT-IV invocations.
    /// `rustdct`'s DCT-IV is unnormalised: applying it twice scales
    /// each entry by `n/2` (empirically verified: a unit impulse at
    /// `n=8` recovers `4 · impulse` after two passes). So one pass is
    /// `√(n/2) · C_orthonormal · x`. After three passes the cumulative
    /// factor is `(n/2)^{3/2}`; we apply `(2/n)^{3/2}` once at the end
    /// of apply/unapply so the inner passes stay scale-free.
    inv_norm: f32,
    /// Cached DCT-IV planner output. Shared across calls; rustdct
    /// designs are `Send + Sync` so this is safe across rayon threads.
    dct4: Arc<dyn Dct4<f32> + Send + Sync>,
}

impl std::fmt::Debug for Dct4Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dct4Mask")
            .field("n", &self.n)
            .field("inv_norm", &self.inv_norm)
            // Omit dct4 (no Debug impl).
            .finish_non_exhaustive()
    }
}

impl Clone for Dct4Mask {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            d1: self.d1.clone(),
            d2: self.d2.clone(),
            d3: self.d3.clone(),
            inv_norm: self.inv_norm,
            dct4: Arc::clone(&self.dct4),
        }
    }
}

/// Round `n` up to the next **fast DCT-IV size**: the smallest `m ≥ n`
/// with `v₂(m) ≥ 4` (divisible by 16) and largest prime factor ≤ 11.
///
/// `stacked_n = n_data + shield_k` lands on whatever factorisation chance
/// gives, and rustdct/rustfft fall onto slow Rader/Bluestein paths when a
/// larger prime divides it. Measured on the cascade microbench (native,
/// ns/elem at d=9728, chronicle §15.2/§16): 2064 = 2⁴·3·**43** → 1.285;
/// 2112 = 2⁶·3·11 → **0.747** (1.72×); 8208 = 2⁴·3³·**19** → 1.168;
/// 8400 = 2⁴·3·5²·7 → **0.857** (1.36×). High power-of-two content beats
/// pure smoothness (2112 with an 11 outruns 7-smooth 2100), hence the
/// `v₂ ≥ 4` requirement; 13+ measured slow (2080 = 2⁵·5·13 → 1.05),
/// hence the ≤ 11 bound. Sizes of this form are dense (pad ≤ ~3% in the
/// 2k–16k range; 2064→2112 and 8208→8400 are both +2.3%).
///
/// The extra rows are zero/shield padding through the orthogonal mask —
/// the data-row round-trip is exact regardless of pad content, so this is
/// purely a transform-speed lever. `n < 32` passes through unchanged
/// (tiny shapes resolve to HD₃ in production; keeps small-shape tests
/// bit-identical).
pub fn next_fast_n(n: usize) -> usize {
    if n < 32 {
        return n;
    }
    fn odd_part_is_11_smooth(mut m: usize) -> bool {
        for p in [3usize, 5, 7, 11] {
            while m % p == 0 {
                m /= p;
            }
        }
        m == 1
    }
    // Smallest multiple of 16 that is ≥ n, then step by 16 until the odd
    // part is {3,5,7,11}-smooth. Terminates fast: pow2-times-small-odd
    // sizes are dense at every scale.
    let mut m = n.div_ceil(16) * 16;
    while !odd_part_is_11_smooth(m >> m.trailing_zeros()) {
        m += 16;
    }
    m
}

impl Dct4Mask {
    /// Sample a fresh DCT-IV mask at side length `n` (any positive int).
    /// Consumes `3·n` random bits — same orbit cardinality as HD₃ but
    /// without the pow2 constraint.
    pub fn fresh<R: RngCore>(n: usize, rng: &mut R) -> Self {
        assert!(n > 0, "Dct4Mask::fresh: n must be positive, got {n}");
        let sample_diag = |rng: &mut R| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                let mut bits = rng.next_u32();
                let take = (n - i).min(32);
                for _ in 0..take {
                    out.push(if bits & 1 == 0 { -1.0_f32 } else { 1.0_f32 });
                    bits >>= 1;
                }
                i += take;
            }
            out
        };
        let d1 = sample_diag(rng);
        let d2 = sample_diag(rng);
        let d3 = sample_diag(rng);
        // rustdct's DCT-IV is unnormalised. Three passes accumulate
        // `(n/2)^{3/2}`, so `inv_norm = (2/n)^{3/2}` makes the
        // composed operator exactly orthogonal.
        let inv_norm = (2.0_f32 / n as f32).powf(1.5);

        // Planner is cheap; rustdct internally caches plans by length.
        let dct4 = DctPlanner::new().plan_dct4(n);
        Self {
            n,
            d1,
            d2,
            d3,
            inv_norm,
            dct4,
        }
    }

    /// Deterministic constructor for tests.
    pub fn from_seed(n: usize, seed: MaskSeed) -> Self {
        let mut rng = seed.rng();
        Self::fresh(n, &mut rng)
    }

    /// Side length `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Apply the mask: `U = A · H`. `hidden` must have shape `(n, d)`.
    /// Returns a freshly-allocated `(n, d)` output buffer.
    ///
    /// Hot paths should prefer [`Self::apply_in_place`] to avoid the
    /// allocation + copy.
    pub fn apply(&self, hidden: ndarray::ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            hidden.nrows(),
            self.n,
            "Dct4Mask::apply: hidden has {} rows, expected {}",
            hidden.nrows(),
            self.n
        );
        let d = hidden.ncols();
        let mut buf = Array2::<f32>::zeros((self.n, d));
        if d > 0 {
            buf.assign(&hidden);
        }
        self.apply_in_place(&mut buf);
        buf
    }

    /// Apply the mask in place: `buf ← A · buf` (`A = D₃·C·D₂·C·D₁·C`).
    /// Buffer must have shape `(n, *)`.
    pub fn apply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Dct4Mask::apply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        let d = buf.ncols();
        let slice = buf
            .as_slice_mut()
            .expect("Dct4Mask::apply_in_place: buffer must be standard layout");
        dct4_cascade_apply_inplace_slice(
            slice,
            self.n,
            d,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`. `masked_output` must have
    /// shape `(n, p)`. Returns a freshly-allocated `(n, p)` buffer.
    ///
    /// Hot paths should prefer [`Self::unapply_in_place`].
    pub fn unapply(&self, masked_output: ndarray::ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            masked_output.nrows(),
            self.n,
            "Dct4Mask::unapply: masked_output has {} rows, expected {}",
            masked_output.nrows(),
            self.n
        );
        let p = masked_output.ncols();
        let mut buf = Array2::<f32>::zeros((self.n, p));
        if p > 0 {
            buf.assign(&masked_output);
        }
        self.unapply_in_place(&mut buf);
        buf
    }

    /// Remove the mask in place: `buf ← Aᵀ · buf`. Same shape contract
    /// as [`Self::unapply`].
    ///
    /// `Aᵀ = (D₃·C·D₂·C·D₁·C)ᵀ = Cᵀ·D₁ᵀ·Cᵀ·D₂ᵀ·Cᵀ·D₃ᵀ
    ///     = C·D₁·C·D₂·C·D₃` (since `C = Cᵀ` for DCT-IV and `Dᵢ = Dᵢᵀ`).
    /// Slice-flavored variant of [`Self::apply_in_place`] — see
    /// [`crate::hd3::Hd3Mask::apply_in_place_slice`] for the rationale.
    pub fn apply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::apply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_apply_inplace_slice(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    pub fn unapply_in_place(&self, buf: &mut Array2<f32>) {
        assert_eq!(
            buf.nrows(),
            self.n,
            "Dct4Mask::unapply_in_place: buf has {} rows, expected {}",
            buf.nrows(),
            self.n
        );
        let d = buf.ncols();
        let slice = buf
            .as_slice_mut()
            .expect("Dct4Mask::unapply_in_place: buffer must be standard layout");
        dct4_cascade_unapply_inplace_slice(
            slice,
            self.n,
            d,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// Slice-flavored variant of [`Self::unapply_in_place`].
    pub fn unapply_in_place_slice(&self, buf: &mut [f32], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::unapply_in_place_slice: buf has {} f32s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_unapply_inplace_slice(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// f16 in/out variant of [`Self::apply_in_place_slice`] — phase 3a
    /// of the f16 activation pipeline. Operand storage is f16; the
    /// tile-fused cascade widens on tile-load and narrows on tile-store
    /// so the in-tile arithmetic stays at f32 (the cascade's precision
    /// contract is unchanged from the f32 path). Halves DRAM traffic
    /// at the cascade boundary vs an f32 storage path.
    pub fn apply_in_place_slice_f16(&self, buf: &mut [f16], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::apply_in_place_slice_f16: buf has {} f16s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_apply_inplace_slice_f16(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// f16 in/out variant of [`Self::unapply_in_place_slice`].
    pub fn unapply_in_place_slice_f16(&self, buf: &mut [f16], cols: usize) {
        assert_eq!(
            buf.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::unapply_in_place_slice_f16: buf has {} f16s, expected n={} * cols={}",
            buf.len(), self.n, cols,
        );
        dct4_cascade_unapply_inplace_slice_f16(
            buf,
            self.n,
            cols,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }

    /// **Fused** f16 unapply (§16 follow-up): reads the `(n × cols)`
    /// f16 matmul output (read-only) and writes the unmasked **data
    /// rows** `[..out_rows]` directly into the `(out_rows × cols)` f32
    /// `out` from the cascade tiles — no in-place f16 narrow, no
    /// separate widen-copy pass, pad/shield rows never stored. One f16
    /// rounding *fewer* on every output element than
    /// [`Self::unapply_in_place_slice_f16`] + copy, so accuracy is
    /// equal-or-better.
    pub fn unapply_f16_into_f32_rows(
        &self,
        src: &[f16],
        cols: usize,
        out: &mut [f32],
        out_rows: usize,
    ) {
        assert_eq!(
            src.len(),
            self.n.saturating_mul(cols),
            "Dct4Mask::unapply_f16_into_f32_rows: src has {} f16s, expected n={} * cols={}",
            src.len(), self.n, cols,
        );
        assert_eq!(
            out.len(),
            out_rows.saturating_mul(cols),
            "Dct4Mask::unapply_f16_into_f32_rows: out has {} f32s, expected out_rows={} * cols={}",
            out.len(), out_rows, cols,
        );
        assert!(
            out_rows <= self.n,
            "Dct4Mask::unapply_f16_into_f32_rows: out_rows {} > n {}",
            out_rows, self.n,
        );
        dct4_cascade_unapply_f16_into_f32_rows(
            src,
            self.n,
            cols,
            out,
            out_rows,
            &self.d1,
            &self.d2,
            &self.d3,
            self.inv_norm,
            self.dct4.as_ref(),
        );
    }
}

/// In-place DCT-IV applied along axis 0 of a row-major `Array2`.
/// Equivalent to multiplying by the (unnormalised) DCT-IV matrix from
/// the left for each column independently.
///
/// Implementation: copy each column into a contiguous length-n scratch,
/// run `rustdct::Dct4::process_dct4` (with its internal scratch), copy
/// back. Rayon-parallel over columns when `d ≥ DCT4_RAYON_COL_THRESHOLD`.
///
/// **Superseded** by the tile-fused cascade
/// `dct4_cascade_apply_inplace_slice` — kept here as a parity-test
/// fallback. Will be removed after the cascade ships measurement
/// validation.
#[allow(dead_code)]
fn dct4_cols_inplace(buf: &mut Array2<f32>, dct4: &(dyn Dct4<f32> + Send + Sync)) {
    let n = buf.nrows();
    let d = buf.ncols();
    let slice = buf
        .as_slice_mut()
        .expect("dct4_cols_inplace: matrix must be standard layout");
    dct4_cols_inplace_slice(slice, n, d, dct4);
}

/// Slice-flavored body of [`dct4_cols_inplace`]. Operates on a row-major
/// `(n, d)` `&mut [f32]` directly so the batched per-block dispatch in
/// `crate::sim` can avoid materialising a per-block `Array2`.
#[allow(dead_code)]
fn dct4_cols_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(
        slice.len(),
        n.saturating_mul(d),
        "dct4_cols_inplace_slice: slice has {} f32s, expected n={} * d={}",
        slice.len(), n, d,
    );
    if n < 2 || d == 0 {
        return;
    }

    // Process columns. Each column is strided (stride `d` in row-major
    // (n, d) layout). We copy-out → DCT → copy-back per column.
    if d >= DCT4_RAYON_COL_THRESHOLD {
        // Rayon over column index. Each thread allocates its own
        // column buffer + DCT scratch (via thread-local cache below).
        (0..d).into_par_iter().for_each(|j| {
            COL_SCRATCH.with(|cell| {
                let mut state = cell.borrow_mut();
                state.col.resize(n, 0.0);
                let dct_scratch_len = dct4.get_scratch_len();
                if state.scratch.len() < dct_scratch_len {
                    state.scratch.resize(dct_scratch_len, 0.0);
                }
                // Split the RefMut into two non-overlapping slice borrows.
                let ColScratch { col, scratch } = &mut *state;
                let col = &mut col[..n];
                let scratch = &mut scratch[..dct_scratch_len];

                // Copy column j out: row i has slice[i·d + j].
                // SAFETY: bounds checked by buf shape; disjoint
                // columns across rayon workers (different j).
                let base = slice.as_ptr();
                for i in 0..n {
                    // SAFETY: `i·d + j < n·d = slice.len()`.
                    unsafe {
                        col[i] = *base.add(i * d + j);
                    }
                }

                dct4.process_dct4_with_scratch(col, scratch);

                // SAFETY: same bounds; writes to disjoint column.
                let base_mut = slice.as_ptr() as *mut f32;
                for i in 0..n {
                    unsafe {
                        *base_mut.add(i * d + j) = col[i];
                    }
                }
            });
        });
    } else {
        let mut col = vec![0.0_f32; n];
        let mut scratch = vec![0.0_f32; dct4.get_scratch_len()];
        for j in 0..d {
            for i in 0..n {
                col[i] = slice[i * d + j];
            }
            dct4.process_dct4_with_scratch(&mut col, &mut scratch);
            for i in 0..n {
                slice[i * d + j] = col[i];
            }
        }
    }
}

thread_local! {
    /// Per-thread reusable DCT scratch + column-copy buffer. Avoids
    /// allocating ~16 KB twice per column on the hot path; over
    /// 144 calls × `d` columns × 3 stages this would be ~140 MB of
    /// allocator churn per forward at Qwen3-4B.
    static COL_SCRATCH: std::cell::RefCell<ColScratch> = std::cell::RefCell::new(ColScratch::default());
}

#[derive(Default)]
struct ColScratch {
    col: Vec<f32>,
    scratch: Vec<f32>,
}

/// Tile width for the fused DCT-IV cascade. Picked so:
///  - 16 f32 = 64 B = one cache line: a tile row of the `(n, d)` input
///    is read/written as exactly one cache line per slice row (no
///    stride-`d` cache waste on copy-in / copy-out).
///  - The full tile working set `tile_d × n × 4 B` fits in L2 at the
///    production shape (16 × 4096 × 4 B = 256 KiB; Strix Halo L2 = 1 MiB
///    per core).
const DCT4_CASCADE_TILE: usize = 16;

thread_local! {
    /// Per-thread reusable scratch for the tile-fused cascade. Layout
    /// is `(tile_d, n)` row-major: each tile row is a contiguous
    /// length-`n` vector ready for DCT-IV. Reused across calls so the
    /// 256 KiB allocation amortises across the whole forward pass.
    static TILE_SCRATCH: std::cell::RefCell<TileScratch> =
        std::cell::RefCell::new(TileScratch::default());
}

#[derive(Default)]
struct TileScratch {
    /// `(tile_d, n)` row-major: `tile[j * n + i] = buf[i * d + tile_start + j]`.
    tile: Vec<f32>,
    /// Internal DCT scratch from `rustdct::Dct4::get_scratch_len`.
    dct_scratch: Vec<f32>,
}

/// Apply the DCT-IV cascade `A = D₃·C·D₂·C·D₁·C` in place, fused over
/// `DCT4_CASCADE_TILE`-column tiles. Equivalent to:
///   dct → ×D₁ → dct → ×D₂ → dct → ×D₃·inv_norm
/// run separately, but all six stages happen on each tile while it's
/// resident in L2 — slashing per-stage RAM round-trips from 6× the
/// buffer to one copy-in + one copy-out per tile (≈ 4× less RAM
/// traffic at the production shape).
///
/// Disjoint column tiles let us write through raw pointers under
/// rayon without data races.
fn dct4_cascade_apply_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            // SAFETY: tiles touch disjoint column ranges of `slice`.
            // Each iteration of `i` reads `tile_d` contiguous f32s
            // (one cache line at tile_d=16) from row `i` of slice.
            let slice_ptr = slice_addr as *mut f32;
            unsafe {
                copy_tile_in(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            // Cascade — 3 × (DCT + diag), all in-tile.
            cascade_apply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// Inverse cascade: `Aᵀ = C·D₁·C·D₂·C·D₃·inv_norm`. Same tile-fused
/// structure as `dct4_cascade_apply_inplace_slice`.
fn dct4_cascade_unapply_inplace_slice(
    slice: &mut [f32],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            let slice_ptr = slice_addr as *mut f32;
            unsafe {
                copy_tile_in(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            cascade_unapply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// SAFETY: caller must ensure
///   - `slice_ptr` points to at least `n * d` f32s,
///   - `tile_start + tile_d ≤ d`,
///   - `tile_buf.len() ≥ tile_d * n`,
///   - this thread is the only one writing to `slice[*, tile_start..tile_start+tile_d]`.
#[inline]
unsafe fn copy_tile_in(
    slice_ptr: *const f32,
    n: usize,
    d: usize,
    tile_start: usize,
    tile_d: usize,
    tile_buf: &mut [f32],
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let src = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, src slice valid for `tile_d` reads.
            tile_buf[j * n + i] = unsafe { *src.add(j) };
        }
    }
}

/// SAFETY: same preconditions as `copy_tile_in`; this thread is the only
/// one writing to `slice[*, tile_start..tile_start+tile_d]`.
#[inline]
unsafe fn copy_tile_out(
    tile_buf: &[f32],
    tile_d: usize,
    n: usize,
    slice_ptr: *mut f32,
    d: usize,
    tile_start: usize,
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let dst = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, dst valid for `tile_d` writes.
            unsafe { *dst.add(j) = tile_buf[j * n + i] };
        }
    }
}

/// f16 tile-load variant of `copy_tile_in`. Reads f16 from the
/// `slice` storage and widens to f32 in the L2-resident tile buffer
/// during the load itself — no transient f32 buffer is materialised in
/// DRAM. Halves the in-bound DRAM traffic at the tile boundary
/// (`tile_d * n * 2 B` of f16 reads instead of `tile_d * n * 4 B`).
///
/// SAFETY: same preconditions as `copy_tile_in` (disjoint tile column
/// ranges across rayon workers; valid for `n * d` f16 reads through
/// `slice_ptr`).
#[inline]
unsafe fn copy_tile_in_f16(
    slice_ptr: *const f16,
    n: usize,
    d: usize,
    tile_start: usize,
    tile_d: usize,
    tile_buf: &mut [f32],
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let src = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, src valid for `tile_d` reads.
            tile_buf[j * n + i] = unsafe { (*src.add(j)).to_f32() };
        }
    }
}

/// f16 tile-store variant of `copy_tile_out`. Narrows from the f32
/// tile buffer to f16 storage in the slice during the store itself —
/// halves the out-bound DRAM traffic at the tile boundary.
///
/// SAFETY: same preconditions as `copy_tile_out`.
#[inline]
unsafe fn copy_tile_out_f16(
    tile_buf: &[f32],
    tile_d: usize,
    n: usize,
    slice_ptr: *mut f16,
    d: usize,
    tile_start: usize,
) {
    for i in 0..n {
        // SAFETY: `i * d + tile_start + tile_d ≤ n * d`.
        let dst = unsafe { slice_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, dst valid for `tile_d` writes.
            unsafe { *dst.add(j) = f16::from_f32(tile_buf[j * n + i]) };
        }
    }
}

/// f16 in/out variant of `dct4_cascade_apply_inplace_slice`. Same
/// tile-fused cascade structure — only the tile boundary changes:
/// f16 → f32 on load (in-register widen), f32 → f16 on store
/// (in-register narrow). In-tile arithmetic stays at f32, so the
/// precision contract is unchanged from the f32 cascade.
///
/// Bandwidth comparison at a `(tile_d=16, n=4096)` tile:
///   - f32 cascade: 256 KiB load + 256 KiB store = 512 KiB DRAM/tile
///   - f16 cascade: 128 KiB load + 128 KiB store = 256 KiB DRAM/tile
fn dct4_cascade_apply_inplace_slice_f16(
    slice: &mut [f16],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            // SAFETY: tiles touch disjoint column ranges of `slice`.
            let slice_ptr = slice_addr as *mut f16;
            unsafe {
                copy_tile_in_f16(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            cascade_apply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out_f16(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// f32 tile-store for the **fused** unapply: writes only rows
/// `[..out_rows]` of the tile into the `(out_rows × d)` f32 output —
/// the data rows the caller keeps. Pad/shield rows above `out_rows`
/// are simply dropped (never stored), and the f32 tile value goes to
/// the output **without** the f16 narrow round-trip the in-place
/// variant pays.
///
/// SAFETY: caller must ensure `out_ptr` points to at least
/// `out_rows * d` f32s, `tile_start + tile_d ≤ d`, `out_rows ≤ n`,
/// and this thread is the only one writing
/// `out[*, tile_start..tile_start+tile_d]`.
#[inline]
unsafe fn copy_tile_rows_out_f32(
    tile_buf: &[f32],
    tile_d: usize,
    n: usize,
    out_ptr: *mut f32,
    d: usize,
    tile_start: usize,
    out_rows: usize,
) {
    debug_assert!(out_rows <= n);
    for i in 0..out_rows {
        // SAFETY: `i * d + tile_start + tile_d ≤ out_rows * d`.
        let dst = unsafe { out_ptr.add(i * d + tile_start) };
        for j in 0..tile_d {
            // SAFETY: `j < tile_d`, dst valid for `tile_d` writes.
            unsafe { *dst.add(j) = tile_buf[j * n + i] };
        }
    }
}

/// **Fused** f16 unapply: reads the `(n × d)` f16 matmul output
/// (read-only — `src` is not clobbered), runs the inverse cascade per
/// column tile, and stores the first `out_rows` rows of the result
/// **directly into the `(out_rows × d)` f32 `out`**. Replaces the
/// in-place-f16-then-widen-copy sequence in the offload unmask, which
/// paid (a) a full f16 narrow store of the whole stacked buffer,
/// (b) a separate single-threaded widen-copy pass of the data rows, and
/// (c) one extra f16 rounding on every output element. The fused store
/// happens cache-hot from the tile, in parallel across column tiles, and
/// skips the pad/shield rows entirely. (§16 follow-up — the
/// conversion/copy overhead measured ~2× the transform itself.)
fn dct4_cascade_unapply_f16_into_f32_rows(
    src: &[f16],
    n: usize,
    d: usize,
    out: &mut [f32],
    out_rows: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(src.len(), n.saturating_mul(d));
    debug_assert_eq!(out.len(), out_rows.saturating_mul(d));
    debug_assert!(out_rows <= n);
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 || out_rows == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let src_addr = src.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            // SAFETY: tiles read/write disjoint column ranges of
            // `src`/`out`; `src` is only read.
            let src_ptr = src_addr as *const f16;
            let out_ptr = out_addr as *mut f32;
            unsafe {
                copy_tile_in_f16(src_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            cascade_unapply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_rows_out_f32(tile_buf, tile_d, n, out_ptr, d, tile_start, out_rows);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// f16 in/out variant of `dct4_cascade_unapply_inplace_slice`.
/// Mirrors `dct4_cascade_apply_inplace_slice_f16`'s tile structure.
fn dct4_cascade_unapply_inplace_slice_f16(
    slice: &mut [f16],
    n: usize,
    d: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
) {
    debug_assert_eq!(slice.len(), n.saturating_mul(d));
    debug_assert_eq!(d1.len(), n);
    debug_assert_eq!(d2.len(), n);
    debug_assert_eq!(d3.len(), n);
    if n < 2 || d == 0 {
        return;
    }
    let tile = DCT4_CASCADE_TILE;
    let n_tiles = d.div_ceil(tile);
    let slice_addr = slice.as_mut_ptr() as usize;
    let process = |t_idx: usize| {
        let tile_start = t_idx * tile;
        let tile_d = (tile_start + tile).min(d) - tile_start;
        TILE_SCRATCH.with(|cell| {
            let mut state = cell.borrow_mut();
            let dct_scratch_len = dct4.get_scratch_len();
            if state.tile.len() < tile * n {
                state.tile.resize(tile * n, 0.0);
            }
            if state.dct_scratch.len() < dct_scratch_len {
                state.dct_scratch.resize(dct_scratch_len, 0.0);
            }
            let TileScratch { tile: tile_buf, dct_scratch } = &mut *state;
            let tile_buf = &mut tile_buf[..tile_d * n];
            let dct_scratch = &mut dct_scratch[..dct_scratch_len];

            let slice_ptr = slice_addr as *mut f16;
            unsafe {
                copy_tile_in_f16(slice_ptr, n, d, tile_start, tile_d, tile_buf);
            }

            cascade_unapply_in_tile(
                tile_buf, tile_d, n, d1, d2, d3, inv_norm, dct4, dct_scratch,
            );

            unsafe {
                copy_tile_out_f16(tile_buf, tile_d, n, slice_ptr, d, tile_start);
            }
        });
    };

    if d >= DCT4_RAYON_COL_THRESHOLD {
        (0..n_tiles).into_par_iter().for_each(process);
    } else {
        (0..n_tiles).for_each(process);
    }
}

/// Apply cascade: 3× DCT + 3× diag on the tile's `(tile_d, n)` row-major
/// buffer.
#[inline]
fn cascade_apply_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
    dct_scratch: &mut [f32],
) {
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₁ (±1 sign flip)
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d1);
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₂
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d2);
    // C
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    // ×D₃ · inv_norm
    apply_scaled_diag_in_tile(tile_buf, tile_d, n, d3, inv_norm);
}

/// Inverse cascade: `inv_norm·D₃ → C → D₂ → C → D₁ → C`.
#[inline]
fn cascade_unapply_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    d1: &[f32],
    d2: &[f32],
    d3: &[f32],
    inv_norm: f32,
    dct4: &(dyn Dct4<f32> + Send + Sync),
    dct_scratch: &mut [f32],
) {
    apply_scaled_diag_in_tile(tile_buf, tile_d, n, d3, inv_norm);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d2);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
    apply_sign_diag_in_tile(tile_buf, tile_d, n, d1);
    for j in 0..tile_d {
        dct4.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], dct_scratch);
    }
}

/// In-tile diag with ±1 entries: per `(tile_d, n)` row, flip the sign
/// of `tile[j, i]` when `diag[i] < 0`. No multiply when `diag[i] = +1`.
/// Mirrors `apply_diag_inplace_slice`'s sign-skip optimisation but in
/// the transposed layout (inner loop iterates `i`, contiguous in row j).
#[inline]
fn apply_sign_diag_in_tile(tile_buf: &mut [f32], tile_d: usize, n: usize, diag: &[f32]) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            if diag[i] < 0.0 {
                row[i] = -row[i];
            }
        }
    }
}

/// In-tile scaled diag: `tile[j, i] *= diag[i] * factor`. Always
/// multiplies (factor may be non-trivial — `inv_norm = (2/n)^{3/2}`).
#[inline]
fn apply_scaled_diag_in_tile(
    tile_buf: &mut [f32],
    tile_d: usize,
    n: usize,
    diag: &[f32],
    factor: f32,
) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            row[i] *= diag[i] * factor;
        }
    }
}

/// **SIMD-spike** branchless variant of [`apply_sign_diag_in_tile`]:
/// `row[i] *= diag[i]` with `diag` holding ±1.0. No data-dependent
/// branch, so the inner loop auto-vectorises (mulps/vmulps) instead of
/// the per-element predicted negate. Used by `dct4_cascade_vectorize_spike`
/// to A/B the diagonal glue against the branched baseline. Spike-only.
#[cfg(test)]
#[inline]
fn apply_sign_diag_in_tile_branchless(tile_buf: &mut [f32], tile_d: usize, n: usize, diag: &[f32]) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            row[i] *= diag[i];
        }
    }
}

/// **SIMD-spike** variant of [`apply_scaled_diag_in_tile`] taking a
/// pre-multiplied `scaled[i] = diag[i] * factor` (hoists the per-element
/// `* factor` out of the j-loop). Pure element-wise multiply → vectorises.
/// Spike-only.
#[cfg(test)]
#[inline]
fn apply_scaled_diag_in_tile_precomp(tile_buf: &mut [f32], tile_d: usize, n: usize, scaled: &[f32]) {
    for j in 0..tile_d {
        let row = &mut tile_buf[j * n..j * n + n];
        for i in 0..n {
            row[i] *= scaled[i];
        }
    }
}

/// In-place row-wise sign flip: `m[i, *] *= d[i]`. Identical contract
/// to [`crate::hd3::apply_diag_inplace`]; copied here to avoid a
/// cross-module re-export. Superseded by `apply_sign_diag_in_tile`.
#[allow(dead_code)]
fn apply_diag_inplace(m: &mut Array2<f32>, d: &[f32]) {
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_inplace: matrix must be standard layout");
    apply_diag_inplace_slice(slice, d, cols);
}

#[allow(dead_code)]
fn apply_diag_inplace_slice(slice: &mut [f32], d: &[f32], cols: usize) {
    let n_rows = d.len();
    debug_assert_eq!(
        slice.len(),
        n_rows.saturating_mul(cols),
        "apply_diag_inplace_slice: slice has {} f32s, expected d.len()={} * cols={}",
        slice.len(), n_rows, cols,
    );
    if cols == 0 {
        return;
    }
    let total_work = n_rows.saturating_mul(cols);
    if total_work >= crate::hd3::FWHT_RAYON_WORK_THRESHOLD {
        slice
            .par_chunks_mut(cols)
            .zip(d.par_iter())
            .for_each(|(row, &sign)| {
                if sign < 0.0 {
                    for v in row.iter_mut() {
                        *v = -*v;
                    }
                }
            });
    } else {
        for (row_idx, &sign) in d.iter().enumerate() {
            if sign < 0.0 {
                let row_offset = row_idx * cols;
                for v in &mut slice[row_offset..row_offset + cols] {
                    *v = -*v;
                }
            }
        }
    }
}

/// Fused diagonal + scalar pass — mirror of `crate::hd3::apply_diag_scaled_inplace`.
/// Multiplies row `i` by `d[i] * factor` in one full-tensor pass,
/// replacing the `apply_diag_inplace + scale_inplace` pair at the
/// D₃ boundary of `apply` / `unapply`. Superseded by
/// `apply_scaled_diag_in_tile`.
#[allow(dead_code)]
fn apply_diag_scaled_inplace(m: &mut Array2<f32>, d: &[f32], factor: f32) {
    let cols = m.ncols();
    let slice = m
        .as_slice_mut()
        .expect("apply_diag_scaled_inplace: matrix must be standard layout");
    apply_diag_scaled_inplace_slice(slice, d, cols, factor);
}

#[allow(dead_code)]
fn apply_diag_scaled_inplace_slice(slice: &mut [f32], d: &[f32], cols: usize, factor: f32) {
    let n_rows = d.len();
    debug_assert_eq!(
        slice.len(),
        n_rows.saturating_mul(cols),
        "apply_diag_scaled_inplace_slice: slice has {} f32s, expected d.len()={} * cols={}",
        slice.len(), n_rows, cols,
    );
    if cols == 0 {
        return;
    }
    let total_work = n_rows.saturating_mul(cols);
    if total_work >= crate::hd3::FWHT_RAYON_WORK_THRESHOLD {
        slice
            .par_chunks_mut(cols)
            .zip(d.par_iter())
            .for_each(|(row, &sign)| {
                let mult = sign * factor;
                for v in row.iter_mut() {
                    *v *= mult;
                }
            });
    } else {
        for (row_idx, &sign) in d.iter().enumerate() {
            let row_offset = row_idx * cols;
            let mult = sign * factor;
            for v in &mut slice[row_offset..row_offset + cols] {
                *v *= mult;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
        let normal = StandardNormal;
        Array2::from_shape_fn((n, d), |_| normal.sample(rng))
    }

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// `unapply(apply(H) · W) ≈ H · W` to f32 noise at non-pow2 sizes
    /// (the whole point of DCT-IV) and pow2 sizes (sanity).
    #[test]
    fn dct4_round_trip_preserves_matmul() {
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        for &(n, d, p) in &[
            (8usize, 8, 8),
            (12, 8, 8),       // non-pow2
            (17, 13, 19),     // non-pow2 prime-ish
            (64, 64, 32),
            (257, 128, 64),   // non-pow2 prime — Bluestein-DCT path
            (256, 128, 128),
            (2056, 2048, 1024), // Qwen3-4B QKV-like shape, non-pow2
        ] {
            let mask = Dct4Mask::fresh(n, &mut rng);
            let h = sample_normal(&mut rng, n, d);
            let w = sample_normal(&mut rng, d, p);
            let target = h.dot(&w);

            let u = mask.apply(h.view());
            let v = u.dot(&w);
            let recovered = mask.unapply(v.view());

            assert_eq!(recovered.dim(), target.dim());
            // Tolerance: matmul depth d plus DCT-IV accumulation depth
            // (rustdct's algorithm depth is O(log n) for pow2, O(log²n)
            // for Bluestein non-pow2) times three cascade stages. Use a
            // conservative bound similar to HD₃.
            let depth = d as f32 + 3.0 * (n as f32).log2().max(1.0) * 4.0;
            let target_max = target
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max)
                .max(1.0);
            let tol = 8.0 * depth * f32::EPSILON * target_max;
            let err = max_abs(&recovered, &target);
            assert!(
                err <= tol,
                "round-trip max abs error at (n={n}, d={d}, p={p}): {err:.3e} > tol {tol:.3e}"
            );
        }
    }

    /// `Aᵀ · A == I` to f32 noise. DCT-IV cascade is orthogonal by
    /// construction.
    #[test]
    fn dct4_orthogonality() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        for &n in &[8usize, 12, 17, 32, 64, 257] {
            let mask = Dct4Mask::fresh(n, &mut rng);
            let id = Array2::<f32>::eye(n);
            let a = mask.apply(id.view());
            let ata = a.t().dot(&a);
            let id_target = Array2::<f32>::eye(n);
            let depth = n as f32 + 3.0 * (n as f32).log2().max(1.0) * 4.0;
            let tol = 16.0 * depth * f32::EPSILON;
            let err = max_abs(&ata, &id_target);
            assert!(
                err <= tol,
                "AᵀA - I max abs error at n={n}: {err:.3e} > tol {tol:.3e}"
            );
        }
    }

    /// Different seeds produce different masks; same seed reproduces.
    #[test]
    fn dct4_deterministic_from_seed() {
        let seed_a = MaskSeed::from_bytes([42u8; 32]);
        let seed_b = MaskSeed::from_bytes([43u8; 32]);
        let m_a1 = Dct4Mask::from_seed(64, seed_a);
        let m_a2 = Dct4Mask::from_seed(64, seed_a);
        let m_b = Dct4Mask::from_seed(64, seed_b);
        assert_eq!(m_a1.d1, m_a2.d1);
        assert_eq!(m_a1.d2, m_a2.d2);
        assert_eq!(m_a1.d3, m_a2.d3);
        assert_ne!(m_a1.d1, m_b.d1);
    }

    /// `Dct4Mask::fresh(0)` rejects zero.
    #[test]
    #[should_panic(expected = "n must be positive")]
    fn dct4_rejects_zero() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let _ = Dct4Mask::fresh(0, &mut rng);
    }

    /// Round-trip relative RMS at the realistic Qwen3-4B
    /// non-pow2 long-context shape (n=2056, d=2560).
    #[test]
    fn dct4_round_trip_relative_error_at_long_n() {
        let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
        let n = 2056;
        let d = 2560;
        let p = 1024;
        let mask = Dct4Mask::fresh(n, &mut rng);
        let h = sample_normal(&mut rng, n, d);
        let w = sample_normal(&mut rng, d, p);
        let target = h.dot(&w);
        let u = mask.apply(h.view());
        let v = u.dot(&w);
        let recovered = mask.unapply(v.view());
        let target_rms = (target.iter().map(|v| v * v).sum::<f32>() / target.len() as f32).sqrt();
        let err_rms = ((recovered
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>())
            / target.len() as f32)
            .sqrt();
        assert!(
            err_rms / target_rms < 1e-3,
            "round-trip relative rms at long-n (n={n}): {:.3e}",
            err_rms / target_rms
        );
    }

    /// Phase 3a parity: `apply_in_place_slice_f16` matches
    /// `apply_in_place_slice` on f16-quantised inputs within
    /// f16-floor tolerance. Both paths run the same f32 cascade
    /// internally; the only delta is the f16↔f32 round-trip at the
    /// tile boundaries.
    #[test]
    fn dct4_apply_f16_parity_at_production_shape() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        // Production prefill shape: n=2056 (post-shield), d=2560.
        let n = 2056;
        let d = 2560;
        let mask = Dct4Mask::fresh(n, &mut rng);
        let h = sample_normal(&mut rng, n, d);
        // Quantise to f16 once so both paths consume the same
        // already-rounded input — measures only the cascade-internal
        // precision delta, not input quantisation noise.
        let h_f16: Vec<f16> = h.iter().map(|&v| f16::from_f32(v)).collect();
        let mut h_f32_q: Vec<f32> = h_f16.iter().map(|v| v.to_f32()).collect();
        let mut h_f16_buf = h_f16.clone();

        mask.apply_in_place_slice(&mut h_f32_q, d);
        mask.apply_in_place_slice_f16(&mut h_f16_buf, d);

        let max_abs = h_f32_q
            .iter()
            .zip(h_f16_buf.iter())
            .map(|(f, b)| (f - b.to_f32()).abs())
            .fold(0.0_f32, f32::max);
        // f16 mantissa = 8 bits effective; per-element abs error
        // bounded by ~max(|out|) · 2⁻⁸. Output of a DCT-IV cascade on
        // unit-normal input is ~O(1); 5e-2 is a comfortable bound.
        assert!(
            max_abs < 5e-2,
            "dct4 apply_in_place_slice_f16 max abs delta {max_abs} exceeds f16-floor bound 5e-2"
        );
    }

    /// Phase 3a parity for unapply.
    #[test]
    fn dct4_unapply_f16_parity_at_production_shape() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let n = 2056;
        let d = 2560;
        let mask = Dct4Mask::fresh(n, &mut rng);
        let h = sample_normal(&mut rng, n, d);
        let h_f16: Vec<f16> = h.iter().map(|&v| f16::from_f32(v)).collect();
        let mut h_f32_q: Vec<f32> = h_f16.iter().map(|v| v.to_f32()).collect();
        let mut h_f16_buf = h_f16.clone();

        mask.unapply_in_place_slice(&mut h_f32_q, d);
        mask.unapply_in_place_slice_f16(&mut h_f16_buf, d);

        let max_abs = h_f32_q
            .iter()
            .zip(h_f16_buf.iter())
            .map(|(f, b)| (f - b.to_f32()).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 5e-2,
            "dct4 unapply_in_place_slice_f16 max abs delta {max_abs} exceeds f16-floor bound 5e-2"
        );
    }

    /// Phase 3a end-to-end correctness: f16 round-trip
    /// `unapply_f16(apply_f16(H) · W) ≈ H · W` at the production
    /// shape, within f16-quantisation-floor relative RMS.
    #[test]
    fn dct4_f16_round_trip_preserves_matmul() {
        let mut rng = ChaCha20Rng::from_seed([19u8; 32]);
        let n = 2056;
        let d = 2560;
        let p = 256;
        let mask = Dct4Mask::fresh(n, &mut rng);
        let h = sample_normal(&mut rng, n, d);
        let w = sample_normal(&mut rng, d, p);
        let target = h.dot(&w);

        // f16 storage path: apply via f16, dot in f32, unapply via f16.
        let mut h_f16: Vec<f16> = h.iter().map(|&v| f16::from_f32(v)).collect();
        mask.apply_in_place_slice_f16(&mut h_f16, d);
        let h_masked_f32: Vec<f32> = h_f16.iter().map(|v| v.to_f32()).collect();
        let h_masked = Array2::from_shape_vec((n, d), h_masked_f32).unwrap();
        let v = h_masked.dot(&w);
        // unapply takes the (n, p) masked output back — narrow it to
        // f16 and run f16 unapply.
        let mut v_f16: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
        mask.unapply_in_place_slice_f16(&mut v_f16, p);
        let recovered = Array2::from_shape_vec((n, p), v_f16.iter().map(|v| v.to_f32()).collect()).unwrap();

        let target_rms = (target.iter().map(|v| v * v).sum::<f32>() / target.len() as f32).sqrt();
        let err_rms = ((recovered
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>())
            / target.len() as f32)
            .sqrt();
        // f16 mantissa noise at long-n accumulates to ~1% relative;
        // 3e-2 is the documented f16-floor for matmul round-trip per
        // the activation-pipeline plan §1.3.
        assert!(
            err_rms / target_rms < 3e-2,
            "dct4 f16 round-trip relative rms at long-n (n={n}): {:.3e}",
            err_rms / target_rms
        );
    }

    /// `next_fast_n` picks the measured-fast sizes, is idempotent on
    /// already-fast sizes, passes tiny shapes through, and never shrinks.
    #[test]
    fn next_fast_n_picks_fast_sizes() {
        // The two production shapes from the §15.2/§16 sweep.
        assert_eq!(next_fast_n(2064), 2112); // 2⁴·3·43 → 2⁶·3·11
        assert_eq!(next_fast_n(8208), 8400); // 2⁴·3³·19 → 2⁴·3·5²·7
        // Already-fast sizes are fixed points.
        for &n in &[2048usize, 2112, 2160, 2304, 8400] {
            assert_eq!(next_fast_n(n), n);
        }
        // Tiny shapes pass through unchanged (HD₃ territory).
        for n in 1..32 {
            assert_eq!(next_fast_n(n), n);
        }
        // Never shrinks; result divisible by 16 with ≤11-smooth odd part.
        for n in (32..4096).step_by(7) {
            let m = next_fast_n(n);
            assert!(m >= n && m % 16 == 0, "n={n} m={m}");
            let mut odd = m >> m.trailing_zeros();
            for p in [3usize, 5, 7, 11] {
                while odd % p == 0 {
                    odd /= p;
                }
            }
            assert_eq!(odd, 1, "n={n} m={m} not 11-smooth");
            // 16 absolute slack (multiple-of-16 rounding) + 10% relative.
            // Production DCT-IV sizes are large (the Auto pad-ratio
            // threshold sends small shapes to HD₃), where the relative
            // term dominates and stays ≤ ~3% in practice.
            assert!(m - n <= 16 + n / 10, "pad too large at n={n} m={m}");
        }
    }

    /// Fused `unapply_f16_into_f32_rows` matches the f32 reference
    /// within the f16 floor, and is at least as accurate as the
    /// in-place-f16-then-widen-copy sequence it replaces (it skips the
    /// final f16 narrow on the output).
    #[test]
    fn dct4_fused_f16_unapply_parity() {
        let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
        for &(n, data_n, d) in &[(64usize, 48usize, 80usize), (2112, 2048, 256)] {
            let mask = Dct4Mask::fresh(n, &mut rng);
            let src_f32 = sample_normal(&mut rng, n, d);
            let src_f16: Vec<f16> = src_f32.iter().map(|&v| f16::from_f32(v)).collect();

            // f32 reference on the f16-quantised input.
            let widened =
                Array2::from_shape_vec((n, d), src_f16.iter().map(|v| v.to_f32()).collect())
                    .unwrap();
            let reference = mask.unapply(widened.view());

            // Fused path.
            let mut out = vec![0.0f32; data_n * d];
            mask.unapply_f16_into_f32_rows(&src_f16, d, &mut out, data_n);

            // Old path: in-place f16 + widen copy.
            let mut buf = src_f16.clone();
            mask.unapply_in_place_slice_f16(&mut buf, d);

            let mut max_fused = 0.0f32;
            let mut max_old = 0.0f32;
            for i in 0..data_n {
                for j in 0..d {
                    let r = reference[(i, j)];
                    max_fused = max_fused.max((out[i * d + j] - r).abs());
                    max_old = max_old.max((buf[i * d + j].to_f32() - r).abs());
                }
            }
            // Fused is exact vs the f32 reference up to in-tile noise
            // (same cascade, no output narrowing); the old path adds a
            // f16 rounding on top.
            assert!(
                max_fused <= 5e-2,
                "(n={n}) fused vs f32 reference: {max_fused:.3e}"
            );
            assert!(
                max_fused <= max_old + 1e-6,
                "(n={n}) fused ({max_fused:.3e}) should be ≤ old path ({max_old:.3e})"
            );
        }
    }

    /// **Spike microbench** (§16 follow-up): split the production f16
    /// unapply bucket into transform vs conversion/copy overhead. Times,
    /// at the production stacked size, (a) the f32 in-place unapply
    /// (transform floor), (b) the f16 in-place unapply (adds the tile
    /// widen/narrow), and (c) the full production composite — f16
    /// unapply + fresh f32 output alloc + data-row widen copy, exactly
    /// what `unmask_per_sequence_f16` does per call.
    /// Run: `cargo test --release -p gelo-protocol --lib dct4_f16_unapply_overhead_spike -- --ignored --nocapture`
    #[test]
    #[ignore = "perf microbench: f16 unapply overhead split (transform vs convert/copy)"]
    fn dct4_f16_unapply_overhead_spike() {
        use std::time::Instant;
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let n: usize = 2112; // next_fast_n(2048 + 16)
        let data_n: usize = 2048;
        let mask = Dct4Mask::fresh(n, &mut rng);
        eprintln!("=== f16 unapply overhead split (n={n}, data_n={data_n}) ===");
        for &d in &[4096usize, 9728] {
            let src_f32: Vec<f32> = (0..n * d).map(|_| StandardNormal.sample(&mut rng)).collect();
            let src_f16: Vec<half::f16> =
                src_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
            let reps = (40usize * 2048 / d).max(4);

            // (a) f32 in-place transform floor.
            let mut buf = src_f32.clone();
            mask.unapply_in_place_slice(&mut buf, d);
            let t = Instant::now();
            for _ in 0..reps {
                mask.unapply_in_place_slice(&mut buf, d);
            }
            let f32_t = t.elapsed().as_secs_f64() / reps as f64;

            // (b) f16 in-place (adds per-tile widen/narrow).
            let mut buf_bf = src_f16.clone();
            mask.unapply_in_place_slice_f16(&mut buf_bf, d);
            let t = Instant::now();
            for _ in 0..reps {
                mask.unapply_in_place_slice_f16(&mut buf_bf, d);
            }
            let f16_t = t.elapsed().as_secs_f64() / reps as f64;

            // (c) production composite: f16 unapply + fresh f32 output
            // alloc + data-row widen copy (the unmask_per_sequence_f16
            // tail).
            let t = Instant::now();
            for _ in 0..reps {
                mask.unapply_in_place_slice_f16(&mut buf_bf, d);
                let mut out = vec![0.0f32; data_n * d];
                for (o, i) in out.iter_mut().zip(buf_bf[..data_n * d].iter()) {
                    *o = i.to_f32();
                }
                std::hint::black_box(&out);
            }
            let comp_t = t.elapsed().as_secs_f64() / reps as f64;

            // (d) alloc-cycle only: fresh zeroed output + drop per rep
            // (the mmap/page-fault/zero/munmap churn glibc pays for
            // >32 MB blocks).
            let t = Instant::now();
            for _ in 0..reps {
                let out = vec![0.0f32; data_n * d];
                std::hint::black_box(&out);
            }
            let alloc_t = t.elapsed().as_secs_f64() / reps as f64;

            // (e) widen copy only, into a reused (warm) buffer.
            let mut out_warm = vec![0.0f32; data_n * d];
            let t = Instant::now();
            for _ in 0..reps {
                for (o, i) in out_warm.iter_mut().zip(buf_bf[..data_n * d].iter()) {
                    *o = i.to_f32();
                }
                std::hint::black_box(&out_warm);
            }
            let copy_t = t.elapsed().as_secs_f64() / reps as f64;

            let ms = |s: f64| s * 1e3;
            eprintln!(
                "  d={d:5}: f32 transform {:7.2} ms | f16 in-place {:7.2} ms (tile-convert +{:.2}) | composite {:7.2} ms (alloc+copy +{:.2})",
                ms(f32_t), ms(f16_t), ms(f16_t - f32_t), ms(comp_t), ms(comp_t - f16_t)
            );
            eprintln!(
                "           alloc-cycle alone {:7.2} ms | warm widen-copy alone {:7.2} ms",
                ms(alloc_t), ms(copy_t)
            );
        }
    }

    /// **Spike microbench** (chronicle §13.3 follow-up): is the DCT-IV
    /// cascade unapply worth vectorising? Reports (1) the full
    /// `unapply_in_place_slice` rate at the three real offload output
    /// widths, and (2) a per-tile attribution across the three cost
    /// centres — the rustdct FFT (already SIMD), our diagonal multiplies
    /// (`apply_{sign,scaled}_diag_in_tile`), and the tile transpose
    /// (`copy_tile_in/out`, scalar gather/scatter). The split tells us
    /// how much headroom vectorising *our* glue can recover vs the
    /// FFT-bound floor.
    ///
    /// Run: `cargo test --release -p gelo-protocol --lib dct4_cascade_vectorize_spike -- --ignored --nocapture`
    #[test]
    #[ignore = "perf microbench: DCT-IV cascade unapply rate + FFT/diag/transpose attribution"]
    fn dct4_cascade_vectorize_spike() {
        use std::time::Instant;
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let n: usize = std::env::var("DCT4_BENCH_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2064);
        let mask = Dct4Mask::fresh(n, &mut rng);
        eprintln!("=== DCT-IV cascade unapply spike (n={n}) ===");

        // --- (1) full unapply rate at the real offload output widths ---
        // 1024 = kv_dim (K, V), 4096 = q_dim (Q), 9728 = intermediate
        // (gate, up). These are the per-output unapply widths at Qwen3-4B.
        for &d in &[1024usize, 4096, 9728] {
            let mut buf: Vec<f32> = (0..n * d)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            mask.unapply_in_place_slice(&mut buf, d); // warmup
            let t0 = Instant::now();
            mask.unapply_in_place_slice(&mut buf, d);
            let one = t0.elapsed().as_secs_f64();
            let reps = ((1.0 / one).round() as usize).clamp(3, 2000);
            let t = Instant::now();
            for _ in 0..reps {
                mask.unapply_in_place_slice(&mut buf, d);
            }
            let el = t.elapsed().as_secs_f64();
            let per_ms = el * 1000.0 / reps as f64;
            let ns_elem = el * 1e9 / (reps as f64 * (n * d) as f64);
            eprintln!("  full unapply d={d:5}: {per_ms:8.3} ms/call  {ns_elem:6.3} ns/elem  ({reps} reps)");
        }

        // --- (2) per-tile attribution: FFT vs diag vs transpose ---
        let tile = DCT4_CASCADE_TILE;
        let dct = mask.dct4.as_ref();
        let mut dct_scratch = vec![0f32; dct.get_scratch_len()];
        let mut tile_buf: Vec<f32> = (0..tile * n)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        // Backing (n × tile) slice for the transpose round-trip.
        let mut slice: Vec<f32> = (0..n * tile)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let reps = 2000usize;

        // FFT: the 3 cascade DCT-IV passes (3 × tile column-DCTs).
        let t = Instant::now();
        for _ in 0..reps {
            for _ in 0..3 {
                for j in 0..tile {
                    dct.process_dct4_with_scratch(&mut tile_buf[j * n..j * n + n], &mut dct_scratch);
                }
            }
        }
        let fft = t.elapsed().as_secs_f64() / reps as f64;

        // Diag: the 3 cascade diagonal multiplies (D₁, D₂ sign + D₃·norm).
        let t = Instant::now();
        for _ in 0..reps {
            apply_sign_diag_in_tile(&mut tile_buf, tile, n, &mask.d1);
            apply_sign_diag_in_tile(&mut tile_buf, tile, n, &mask.d2);
            apply_scaled_diag_in_tile(&mut tile_buf, tile, n, &mask.d3, mask.inv_norm);
        }
        let diag = t.elapsed().as_secs_f64() / reps as f64;

        // Transpose: copy_tile_in + copy_tile_out (one round-trip).
        let sptr_const = slice.as_ptr();
        let sptr_mut = slice.as_mut_ptr();
        let t = Instant::now();
        for _ in 0..reps {
            unsafe {
                copy_tile_in(sptr_const, n, tile, 0, tile, &mut tile_buf);
                copy_tile_out(&tile_buf, tile, n, sptr_mut, tile, 0);
            }
        }
        let transpose = t.elapsed().as_secs_f64() / reps as f64;

        let total = fft + diag + transpose;
        let us = |s: f64| s * 1e6;
        eprintln!("  per-tile (tile={tile}, n={n}) attribution:");
        eprintln!("    FFT (3× DCT-IV, rustdct/SIMD): {:8.2} µs  {:5.1}%", us(fft), 100.0 * fft / total);
        eprintln!("    diag (D1/D2/D3, our glue):     {:8.2} µs  {:5.1}%", us(diag), 100.0 * diag / total);
        eprintln!("    transpose (copy_tile_in/out):  {:8.2} µs  {:5.1}%", us(transpose), 100.0 * transpose / total);
        eprintln!("    --> our-glue (diag+transpose): {:5.1}% of cascade compute", 100.0 * (diag + transpose) / total);

        // --- (3) SIMD A/B: branchless diag vs the branched baseline ---
        let scaled3: Vec<f32> = mask.d3.iter().map(|&v| v * mask.inv_norm).collect();
        let t = Instant::now();
        for _ in 0..reps {
            apply_sign_diag_in_tile_branchless(&mut tile_buf, tile, n, &mask.d1);
            apply_sign_diag_in_tile_branchless(&mut tile_buf, tile, n, &mask.d2);
            apply_scaled_diag_in_tile_precomp(&mut tile_buf, tile, n, &scaled3);
        }
        let diag_simd = t.elapsed().as_secs_f64() / reps as f64;
        // Projected full-cascade impact if the SIMD diag replaced the baseline.
        let total_simd = fft + diag_simd + transpose;
        let cascade_speedup = total / total_simd;
        eprintln!("  SIMD diag A/B:");
        eprintln!("    diag baseline (branched):  {:8.2} µs", us(diag));
        eprintln!("    diag branchless (SIMD):    {:8.2} µs  ({:.2}× on diag)", us(diag_simd), diag / diag_simd);
        eprintln!("    --> projected cascade:     {:.2}× ({:.1}% faster); diag share {:.1}%→{:.1}%",
            cascade_speedup, 100.0 * (1.0 - 1.0 / cascade_speedup),
            100.0 * diag / total, 100.0 * diag_simd / total_simd);
        eprintln!("    (cascade is ~25% of prefill wall → ~{:.1}% prefill at most)",
            100.0 * (1.0 - 1.0 / cascade_speedup) * 0.25);
    }
}
