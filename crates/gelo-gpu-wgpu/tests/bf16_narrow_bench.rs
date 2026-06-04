//! Microbench (§17/§6 follow-up): the offload read-back narrows the GPU
//! matmul output to bf16 host storage. Today it's a scalar
//! `.map(|x| bf16::from_f32(x.to_f32()))`; this measures whether half's
//! SIMD slice converts (`HalfFloatSliceExt`, F16C/AVX under
//! target-cpu=native) beat it at the production output shape. Decides
//! whether to convert `tensor_data_to_array_bf16_from_{f16,f32}`.
//!
//! Run: `cargo test --release -p gelo-gpu-wgpu --test bf16_narrow_bench -- --ignored --nocapture`

use half::slice::HalfFloatSliceExt;
use half::{bf16, f16};
use std::time::Instant;

const N: usize = 2112 * 9728; // gate∥up read-back at n=2048 prefill

fn bench<F: FnMut()>(label: &str, mut f: F, reps: usize) {
    for _ in 0..3 {
        f();
    } // warm
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let per = t.elapsed().as_secs_f64() / reps as f64;
    eprintln!(
        "  {label:42} {:8.3} ms  {:6.3} ns/elem",
        per * 1e3,
        per * 1e9 / N as f64
    );
}

#[test]
#[ignore = "perf microbench: bf16 read-back narrow — scalar map vs half SIMD slice convert"]
fn bf16_narrow_scalar_vs_simd() {
    let reps = 50;
    eprintln!("=== bf16 read-back narrow (N={N} = 2112x9728) ===");

    // ---- f16 GPU output → bf16 host (the fp16-engine read-back) ----
    let src_f16: Vec<f16> = (0..N).map(|i| f16::from_f32((i % 1000) as f32 * 0.01 - 5.0)).collect();
    let mut out = vec![bf16::ZERO; N];

    bench(
        "f16→bf16 scalar map (current)",
        || {
            let v: Vec<bf16> = src_f16.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            std::hint::black_box(&v);
        },
        reps,
    );
    let mut f32_tmp = vec![0.0f32; N];
    bench(
        "f16→bf16 SIMD (via f32 tmp, reused bufs)",
        || {
            src_f16.convert_to_f32_slice(&mut f32_tmp);
            out.convert_from_f32_slice(&f32_tmp);
            std::hint::black_box(&out);
        },
        reps,
    );

    // ---- f32 GPU output → bf16 host (the f32-engine read-back) ----
    let src_f32: Vec<f32> = (0..N).map(|i| (i % 1000) as f32 * 0.01 - 5.0).collect();
    bench(
        "f32→bf16 scalar map (current)",
        || {
            let v: Vec<bf16> = src_f32.iter().map(|&x| bf16::from_f32(x)).collect();
            std::hint::black_box(&v);
        },
        reps,
    );
    bench(
        "f32→bf16 SIMD (convert_from_f32_slice)",
        || {
            out.convert_from_f32_slice(&src_f32);
            std::hint::black_box(&out);
        },
        reps,
    );
}
