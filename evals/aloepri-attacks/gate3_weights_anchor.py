#!/usr/bin/env python3
"""Gate 3 under the WEIGHTS-PUB assumption — covariance-alignment attack on O_v.

Threat model (see docs/dev/logs/perm-attn-gpu-offload.md): a WEIGHTS-PUB
adversary knows the public model, so it can estimate the *population* value
covariance Cov(V) by running the model on public data. Because the value
cover is V·O_v and the row permutation perm_kv does not affect a covariance
taken over the rows,

    Cov(v_sent) = Cov(V·O_v) = O_vᵀ · Cov(V) · O_v ,

eigen-aligning the observed covariance to a population-covariance ANCHOR
recovers O_v up to per-eigenvector sign (resolved here via the 3rd moment)
and up to rotations inside degenerate eigenspaces. We then un-rotate v_sent
and score recovery of the clean V columns — the same matched-|corr| metric
gate3_ica / gate3_jade use, so the numbers are directly comparable.

This is the attack the blind gate-3 (FastICA / JADE) never ran: a
WEIGHTS-PUB attacker does not try to pin O_v blindly — it aligns to a known
covariance. Crucially the attack never targets O_v's *coordinates*; it only
needs the O_v-invariant second-moment structure.

NO-PLAINTEXT faithful: the anchor covariance is taken from a DIFFERENT
prompt (captures2 / prompt B) than the target (captures / prompt A) — a
population proxy the attacker builds from its own public data, never the
victim's clean activations. A `self` column (target's own clean covariance
as anchor) is reported ONLY as a method-validation upper bound; it is not a
security claim.

Run in the gelo-attack container:
  evals/aloepri-attacks/run-in-container.sh \
      python3 evals/aloepri-attacks/gate3_weights_anchor.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
TARGET = HERE / "captures"   # prompt A — the victim
ANCHOR = HERE / "captures2"  # prompt B — attacker's public-data population proxy


def best_match_corr(rec: np.ndarray, true: np.ndarray) -> float:
    """Mean Hungarian-matched |correlation| between columns of `rec` and
    `true` — sign- and permutation-invariant (the standard source-recovery
    metric, matching gate3_ica/jade)."""
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
    """Canonicalise each eigenvector's sign so its projection has positive
    3rd moment — makes the sign convention consistent across the two
    (independently-decomposed) covariances."""
    proj = X @ V
    sk = ((proj - proj.mean(0)) ** 3).mean(0)
    s = np.sign(sk)
    s[s == 0] = 1.0
    return V * s


def recover_Ov(v_sent: np.ndarray, v_anchor: np.ndarray) -> np.ndarray:
    """O_v estimate from eigen-aligning Cov(v_sent) to Cov(v_anchor)."""
    _, Vo = np.linalg.eigh(cov(v_sent))
    _, Va = np.linalg.eigh(cov(v_anchor))
    Vo = canon_signs(v_sent, Vo[:, ::-1])   # descending eigenvalue order
    Va = canon_signs(v_anchor, Va[:, ::-1])
    return Va @ Vo.T                         # ≈ O_v   (since Vo ≈ O_vᵀ·Va)


def main() -> int:
    tmeta = json.loads((TARGET / "attn_cover.meta.json").read_text())
    ameta = json.loads((ANCHOR / "attn_cover.meta.json").read_text())
    d, H = tmeta["d_head"], tmeta["n_kv_heads"]
    shared = sorted(set(tmeta["layers"]) & set(ameta["layers"]))

    print(f"# gate 3 @ WEIGHTS-PUB — covariance-alignment attack on O_v")
    print(f"# target {TARGET.name} (n_kv={tmeta['n_kv']})  anchor {ANCHOR.name} "
          f"(n_kv={ameta['n_kv']}, DIFFERENT prompt)  d_head={d} heads={H}")
    print(f"# corr ≈ no_attack ⇒ O_v HOLDS;  corr → 1 ⇒ O_v BROKEN\n")
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
            vc = v_clean[:, sl][perm]                # clean V, perm-aligned (scoring only)
            base.append(best_match_corr(vs, vc))
            pub.append(best_match_corr(vs @ recover_Ov(vs, va).T, vc))
            slf.append(best_match_corr(vs @ recover_Ov(vs, v_clean[:, sl]).T, vc))
        b, p, s = np.mean(base), np.mean(pub), np.mean(slf)
        agg.append((b, p, s))
        verdict = "BREAKS" if p > b + 0.25 else "holds"
        print(f"{L:>6} {b:>10.3f} {p:>14.3f} {s:>16.3f} {verdict:>12}")

    b, p, s = np.mean([a[0] for a in agg]), np.mean([a[1] for a in agg]), np.mean([a[2] for a in agg])
    print(f"\n# mean: no_attack={b:.3f}  covalign(pub)={p:.3f}  covalign(self*)={s:.3f}")
    print(f"# self* (target's own clean cov as anchor) is a METHOD-VALIDATION upper bound, not a security claim.")
    print(f"# verdict @ WEIGHTS-PUB: {'O_v BREAKS' if p > b + 0.25 else 'O_v HOLDS'} "
          f"(population-covariance anchor from different data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
