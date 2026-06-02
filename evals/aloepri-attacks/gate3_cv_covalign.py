#!/usr/bin/env python3
"""Stage-2 — covariance/Procrustes recoverability attack on the non-orthogonal
value cover C_v (the re-scoped lever; see docs/dev/logs/perm-attn-gpu-offload.md
*Adapting covariant obfuscation to the offload*).

This generalizes gate3_weights_anchor.py (which aligned the *orthogonal* O_v via
eigenvectors only) to a *general invertible* C_v. With v_sent = V·C and a
population-covariance anchor ≈ Cov(V),

    Cov(v_sent) = Cᵀ · Cov(V) · C,

a 2nd-moment recovery must also match the *eigenvalues* (C non-orthogonal scales
them), so the de-cover map whitens v_sent and re-colours it to the anchor's
spectrum, with per-eigenvector sign fixed by the 3rd moment:

    D = Vs · diag(sqrt(La / Ls)) · Vaᵀ ,   de-covered = v_sent · D ≈ V .

Recovery is fundamentally limited to a 2nd-moment fit: it is exact only when the
anchor covariance equals the instance covariance (self-anchor), and is left
unresolved by the instance-vs-population mismatch *plus* the non-orthogonal
spectrum — strictly harder than the O_v case. We score recovery of the clean V
columns with the same Hungarian-matched |corr| metric as gate3_weights_anchor.

NO-PLAINTEXT faithful: the population anchor is a DIFFERENT prompt's clean V
(captures2 / prompt B), built by the attacker from public data. The `self*`
column (target's own clean covariance) is a METHOD-VALIDATION upper bound only.

Run in the gelo-attack container:
  GELO_CV_TARGET=captures_cv_covalign_k6.0 \
      evals/aloepri-attacks/run-in-container.sh \
      python3 evals/aloepri-attacks/gate3_cv_covalign.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
TARGET = HERE / os.environ.get("GELO_CV_TARGET", "captures_cv_covalign_k6.0")
ANCHOR = HERE / os.environ.get("GELO_CV_ANCHOR", "captures2")  # prompt B proxy


def best_match_corr(rec: np.ndarray, true: np.ndarray) -> float:
    R = rec - rec.mean(0, keepdims=True)
    T = true - true.mean(0, keepdims=True)
    R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-9
    T /= np.linalg.norm(T, axis=0, keepdims=True) + 1e-9
    C = np.abs(R.T @ T)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def cov(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(0, keepdims=True)
    return (xc.T @ xc) / max(len(x) - 1, 1)


def canon_signs(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    proj = X @ V
    sk = ((proj - proj.mean(0)) ** 3).mean(0)
    s = np.sign(sk)
    s[s == 0] = 1.0
    return V * s


def recover_Cv_map(v_sent: np.ndarray, v_anchor: np.ndarray) -> np.ndarray:
    """Eigenvalue-aware whitening de-cover map: D s.t. v_sent·D ≈ V.
    Whiten v_sent (cov→I), re-colour to the anchor spectrum, sign-fix by 3rd
    moment. Reduces to the orthogonal recover_Ov when the spectra match."""
    Ls, Vs = np.linalg.eigh(cov(v_sent))
    La, Va = np.linalg.eigh(cov(v_anchor))
    Vs = canon_signs(v_sent, Vs[:, ::-1])
    Va = canon_signs(v_anchor, Va[:, ::-1])
    Ls = np.clip(Ls[::-1], 1e-9, None)
    La = np.clip(La[::-1], 1e-9, None)
    return Vs @ np.diag(np.sqrt(La / Ls)) @ Va.T


def main() -> int:
    tmeta = json.loads((TARGET / "attn_cover.meta.json").read_text())
    ameta = json.loads((ANCHOR / "attn_cover.meta.json").read_text())
    d, H = tmeta["d_head"], tmeta["n_kv_heads"]
    shared = sorted(set(tmeta["layers"]) & set(ameta["layers"]))

    print("# Stage-2 — covariance/Procrustes recoverability attack on C_v")
    print(f"# target {TARGET.name} (n_kv={tmeta['n_kv']}, cv_kappa={tmeta.get('cv_kappa')})  "
          f"anchor {ANCHOR.name} (n_kv={ameta['n_kv']}, DIFFERENT prompt)  d_head={d} heads={H}")
    print("# corr ≈ no_attack ⇒ C_v HOLDS;  corr → 1 ⇒ C_v BROKEN\n")
    print(f"{'layer':>6} {'no_attack':>10} {'covalign(pub)':>14} {'covalign(self*)':>16} {'verdict@pub':>12}")

    tf = safe_open(str(TARGET / "attn_cover.safetensors"), framework="numpy")
    af = safe_open(str(ANCHOR / "attn_cover.safetensors"), framework="numpy")

    agg = []
    for L in shared:
        tag = f"layer{L:03}"
        perm = tf.get_tensor(f"{tag}.perm_kv")
        v_sent = tf.get_tensor(f"{tag}.v_sent")
        v_clean = tf.get_tensor(f"{tag}.v_clean")
        v_anchor = af.get_tensor(f"{tag}.v_clean")   # prompt B clean V = population proxy

        base, pub, slf = [], [], []
        for h in range(H):
            sl = slice(h * d, (h + 1) * d)
            vs, va = v_sent[:, sl], v_anchor[:, sl]
            vc = v_clean[:, sl][perm]
            base.append(best_match_corr(vs, vc))
            pub.append(best_match_corr(vs @ recover_Cv_map(vs, va), vc))
            slf.append(best_match_corr(vs @ recover_Cv_map(vs, v_clean[:, sl]), vc))
        b, p, s = np.mean(base), np.mean(pub), np.mean(slf)
        agg.append((b, p, s))
        verdict = "BREAKS" if p > b + 0.25 else "holds"
        print(f"{L:>6} {b:>10.3f} {p:>14.3f} {s:>16.3f} {verdict:>12}")

    b, p, s = (np.mean([a[i] for a in agg]) for i in range(3))
    print(f"\n# mean: no_attack={b:.3f}  covalign(pub)={p:.3f}  covalign(self*)={s:.3f}")
    print("# self* (target's own clean cov as anchor) is a METHOD-VALIDATION upper bound, not a security claim.")
    print(f"# verdict @ WEIGHTS-PUB: {'C_v BREAKS' if p > b + 0.25 else 'C_v HOLDS'} "
          "(population-covariance anchor from different data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
