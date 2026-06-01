#!/usr/bin/env python3
"""Phase-5 spike — WEIGHTS-PUB token recovery via the O_v-invariant value
norm/Gram dictionary. Works for BOTH offload covers (selected by capture dir):

  * rotation-only PREFILL cover  (no perm)  → recovers token AND position.
  * permutation DECODE cover     (perm_kv)  → recovers token IDENTITY per slot
        (membership); position stays hidden by π.

Mechanism (see docs/dev/logs/perm-attn-gpu-offload.md). O_v is applied per head as a
d_head-orthogonal, so the per-head value NORM is O_v-INVARIANT and (for the decode
cover) permutation only relabels rows:

    || v_sent[i, head h] ||  =  || V(token at slot i)[head h] ||.

Under WEIGHTS-PUB the attacker computes the layer-0 dictionary
V(t) = rms_norm(embed(t))*W_V from public weights (context-free at layer 0), forms
each candidate's H-head norm fingerprint, and matches every slot's observed
fingerprint to the nearest dictionary token. The true token at physical slot i is
input_ids[perm_kv[i]] (perm is identity for the rotation cover).

NO-PLAINTEXT / WEIGHTS-PUB faithful: the dictionary is from public weights;
input_ids / v_clean / perm_kv are used only to SCORE and verify.

Capture dir via GELO_GATE_CAPDIR (default captures_prefill). Run in container:
  GELO_GATE_CAPDIR=captures_decode evals/aloepri-attacks/run-in-container.sh \
      python3 evals/aloepri-attacks/gate3_prefill_dict.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
CAP = HERE / os.environ.get("GELO_GATE_CAPDIR", "captures_prefill")


def main() -> int:
    meta = json.loads((CAP / "attn_cover.meta.json").read_text())
    d, H = meta["d_head"], meta["n_kv_heads"]
    L = meta["layers"][0]
    f = safe_open(str(CAP / "attn_cover.safetensors"), framework="numpy")
    tag = f"layer{L:03}"
    v_sent = f.get_tensor(f"{tag}.v_sent")            # (n, kv_dim) = (perm·V)·O_v
    v_clean = f.get_tensor(f"{tag}.v_clean")          # (n, kv_dim) = V  (clean order; verify/score)
    perm = f.get_tensor(f"{tag}.perm_kv").astype(np.int64)   # slot i holds clean row perm[i]
    ids = f.get_tensor("input_ids").astype(np.int64)  # token at clean position q
    dv = f.get_tensor("dict.v")                        # (K, kv_dim) layer-0 V per candidate
    dids = f.get_tensor("dict.ids").astype(np.int64)   # (K,)
    n, K = v_sent.shape[0], dv.shape[0]
    permuted = not np.array_equal(perm, np.arange(n))

    print(f"# WEIGHTS-PUB token-dictionary attack — capture: {CAP.name} (layer {L})")
    print(f"# cover: {meta['cover']}")
    print(f"# n={n} slots, K={K} dictionary tokens, "
          f"{'PERMUTED (decode)' if permuted else 'identity perm (prefill)'}\n")

    def head_norms(M):  # (rows, kv_dim) -> (rows, H)
        return np.stack(
            [np.linalg.norm(M[:, h * d:(h + 1) * d], axis=1) for h in range(H)], axis=1
        )

    # --- dictionary faithfulness: dict[ids[q]] must match v_clean[q] (clean order) ---
    id2idx = {int(t): i for i, t in enumerate(dids)}
    rel = [
        np.linalg.norm(v_clean[q] - dv[id2idx[int(ids[q])]]) / (np.linalg.norm(v_clean[q]) + 1e-9)
        for q in range(n) if int(ids[q]) in id2idx
    ]
    relerr = float(np.mean(rel)) if rel else float("nan")
    print(f"# dict faithfulness: mean rel-err = {relerr:.2e} over {len(rel)} prompt tokens "
          f"({'OK' if relerr < 5e-2 else 'SUSPECT'})")

    # --- attack: per-slot H-head norm fingerprint → nearest dictionary token ---
    fp_obs, fp_dict = head_norms(v_sent), head_norms(dv)
    mu, sd = fp_dict.mean(0), fp_dict.std(0) + 1e-9
    dist = np.linalg.norm(((fp_obs - mu) / sd)[:, None, :] - ((fp_dict - mu) / sd)[None, :, :], axis=2)
    order = np.argsort(dist, axis=1)

    ranks = []
    for i in range(n):
        t = int(ids[perm[i]])           # true token at physical slot i
        if t in id2idx:
            ranks.append(int(np.where(order[i] == id2idx[t])[0][0]))
    ranks = np.array(ranks)

    print(f"\n{'metric (per-slot token identity)':>34} {'attack':>9} {'chance':>9}")
    for k in (1, 5, 10, 50):
        print(f"{'top-'+str(k):>34} {(ranks < k).mean():>9.3f} {k / K:>9.4f}")
    print(f"{'median true-token rank':>34} {int(np.median(ranks)):>9d} {'(/'+str(K)+')':>9}")

    top1 = (ranks < 1).mean()
    leak = top1 > 10 * (1 / K) and top1 > 0.02
    membership = "token MEMBERSHIP LEAKS" if leak else "membership HOLDS"
    print(f"\n# verdict @ WEIGHTS-PUB: {membership} (top-1 {top1:.3f})")
    if permuted:
        print("# decode cover: this is the per-slot identity (the multiset). ORDER is a"
              " SEPARATE channel — hidden by perm_kv, measured weak by the gate-2 seriation"
              " attack (|tau|~0.1). So: membership via weights, order via permutation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
