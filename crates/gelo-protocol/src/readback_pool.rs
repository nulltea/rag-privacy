//! Thread-local recycling pool for f16 GPU read-back buffers.
//!
//! ## Why this exists (chronicle §24)
//!
//! The fp16 offload reads each matmul result back into the enclave so the
//! mask-unapply (which consumes the *secret* covers) can run in-TEE — a
//! mandatory host round-trip under the `TEE-TRUST` boundary, not something
//! that can move to the GPU. The read-back itself is fast (the DtoH copy
//! into a pinned buffer runs at ~46 GB/s), but burn's `TensorData::into_vec`
//! cannot take ownership of cubecl's pooled pinned buffer, so it falls back
//! to a **fresh per-call `Vec` allocation + memcpy**. At B=8 each output is
//! up to ~319 MB; a fresh allocation is a `mmap` whose pages fault in (the
//! kernel zero-fills each before the copy overwrites it) and is `munmap`'d
//! on drop — so the next read-back re-faults from scratch. Measured: a fresh
//! `to_vec` runs at ~2.2 GB/s vs ~13 GB/s into a resident buffer (5.9×), and
//! this churn was ~22 s of the B=8 prefill `engine:matmul:readback` bucket.
//!
//! This pool recycles the destination allocation so its pages stay resident
//! across calls. The engine [`take`]s a buffer (resident → no fault), copies
//! the pinned read-back into it, and hands back an `Array2` over it; the
//! in-TEE consumer [`give`]s the buffer back once the unapply has drained it.
//! The offload→unapply cycle is synchronous on one thread, so `take`/`give`
//! land on the same thread-local. Missing a `give` (e.g. an off-by-default
//! snapshot path that clones) is harmless — the next `take` just allocates.

use std::cell::RefCell;

use half::f16;

/// Retain at most this many buffers per thread. A fused `matmul_many`
/// returns ≤3 outputs live at once (QKV) and sizes repeat every layer, so a
/// small cap keeps every hot size resident without unbounded growth.
const MAX_POOLED: usize = 8;

thread_local! {
    static POOL: RefCell<Vec<Vec<f16>>> = const { RefCell::new(Vec::new()) };
}

/// Take a buffer of exactly `len` elements, reusing a resident allocation
/// (capacity ≥ `len`) when one is available. The returned buffer's contents
/// are **uninitialised** — the caller must overwrite all `len` elements
/// (the read-back copy does). `f16` is a POD `u16` wrapper with no `Drop`,
/// so `set_len` before a full overwrite is sound.
pub fn take(len: usize) -> Vec<f16> {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if let Some(i) = p.iter().position(|b| b.capacity() >= len) {
            let mut b = p.swap_remove(i);
            // SAFETY: capacity ≥ len; caller overwrites all len elements
            // before any read, and f16 needs no initialisation/drop.
            unsafe { b.set_len(len) };
            return b;
        }
        let mut b = Vec::with_capacity(len);
        // SAFETY: as above — capacity == len, caller overwrites all of it.
        unsafe { b.set_len(len) };
        b
    })
}

/// Return a buffer for reuse once its data has been consumed. Dropped
/// silently if the pool is already full (bounds retained memory).
pub fn give(buf: Vec<f16>) {
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < MAX_POOLED {
            p.push(buf);
        }
    });
}
