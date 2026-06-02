#!/usr/bin/env python3
"""Root-cause isolation for the C_v fp16 accuracy degradation (tier-3 gate).

Isolates the per-application fp16-conditioning error of the value cover C_v as a
function of κ, on REAL captured V, with NO GPU / autotune / autoregression — so
it is deterministic and attributes the error purely to (cond=κ) × fp16.

The offload value path is, per covered layer:
    out_cov = P · fp16(V · C_v)          [GPU, fp16]
    out     = out_cov · C_v⁻¹            [TEE, f32]      ≈  P · V
We measure  ‖out − P·V‖ / ‖P·V‖  vs κ, for:
  - roundtrip:   P = I              (raw value round-trip, per row)
  - attn_peaked: P one-hot-ish      (peaked softmax — output ≈ a single V row)
  - attn_diffuse:P ~ uniform        (diffuse softmax — averaging)
and compares singular-value laws: UNIFORM(1/√κ,√κ) [current] vs LOG-UNIFORM
[geometric-mean-1 fix]. Also reports cond and the det drift (geom-mean^d).

Run in the gelo-attack container:
  GELO_CV_SRC=captures_cv_covalign_k1.0 \
      evals/aloepri-attacks/run-in-container.sh \
      python3 evals/aloepri-attacks/analysis_cv_fp16_error.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
SRC = HERE / os.environ.get("GELO_CV_SRC", "captures_cv_covalign_k1.0")
RNG = np.random.default_rng(0xC0FFEE)


def orthogonal(d: int) -> np.ndarray:
    q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    return q


def make_cv(d: int, kappa: float, law: str):
    """C_v = U·diag(s)·Vᵀ with cond = kappa; return (C, C_inv, s)."""
    if kappa <= 1.0:
        o = orthogonal(d)
        return o, o.T, np.ones(d)
    u, vt = orthogonal(d), orthogonal(d)
    hi, lo = np.sqrt(kappa), 1.0 / np.sqrt(kappa)
    s = np.empty(d)
    s[0], s[-1] = hi, lo
    if law == "uniform":
        s[1:-1] = RNG.uniform(lo, hi, d - 2)          # current construction
    else:  # log-uniform → geometric mean 1
        s[1:-1] = np.exp(RNG.uniform(np.log(lo), np.log(hi), d - 2))
    c = (u * s) @ vt
    c_inv = (vt.T / s) @ u.T
    return c, c_inv, s


def attn_error(V: np.ndarray, kappa: float, law: str, P: np.ndarray) -> float:
    """Relative error of the fp16 value-cover round-trip through contraction P."""
    d = V.shape[1]
    C, C_inv, _ = make_cv(d, kappa, law)
    Vc = (V @ C).astype(np.float16).astype(np.float32)   # fp16 storage on GPU
    out_cov = (P @ Vc).astype(np.float16).astype(np.float32)  # fp16 contraction
    out = out_cov @ C_inv                                # f32 TEE correction
    ref = P @ V
    return float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-12))


def main() -> int:
    meta = json.loads((SRC / "attn_cover.meta.json").read_text())
    d, H = meta["d_head"], meta["n_kv_heads"]
    L = meta["layers"][0]
    f = safe_open(str(SRC / "attn_cover.safetensors"), framework="numpy")
    v_clean = f.get_tensor(f"layer{L:03}.v_clean").astype(np.float32)  # (n, H·d)
    n = v_clean.shape[0]
    print(f"# C_v fp16-error isolation — src={SRC.name} layer={L} n={n} d_head={d} H={H}")
    print(f"# per-head error of  out·C_v⁻¹  vs  P·V,  contraction P; fp16 storage+contract\n")

    # Contractions: peaked (≈ one row) and diffuse (uniform average) softmax.
    P_peaked = np.zeros((n, n), np.float32)
    P_peaked[np.arange(n), RNG.integers(0, n, n)] = 1.0
    P_diffuse = np.full((n, n), 1.0 / n, np.float32)

    kappas = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    for law in ("uniform", "log-uniform"):
        print(f"## singular-value law: {law}")
        # report det drift (geom mean^d) at kappa=6
        _, _, s6 = make_cv(d, 6.0, law)
        print(f"#   kappa=6: cond={s6.max()/s6.min():.2f}  geom-mean(s)={np.exp(np.mean(np.log(s6))):.3f}  "
              f"det-drift={np.exp(d*np.mean(np.log(s6))):.2e}")
        hdr = f"{'kappa':>6} {'roundtrip(P=I)':>15} {'attn_peaked':>13} {'attn_diffuse':>13}"
        print(hdr)
        for k in kappas:
            # average over heads
            rt, ap, ad = [], [], []
            for h in range(H):
                Vh = v_clean[:, h * d:(h + 1) * d]
                rt.append(attn_error(Vh, k, law, np.eye(n, dtype=np.float32)))
                ap.append(attn_error(Vh, k, law, P_peaked))
                ad.append(attn_error(Vh, k, law, P_diffuse))
            print(f"{k:>6} {np.mean(rt):>15.2e} {np.mean(ap):>13.2e} {np.mean(ad):>13.2e}")
        print()

    print("# Reading: error should scale ~linearly in kappa (cond × fp16 ε≈5e-4).")
    print("# Compounding across ~36 covered GLOBAL layers (×√L..×L) + autoregressive")
    print("# amplification turns a per-layer ~1e-3 into off-distribution degeneracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
