# HumanEval accuracy gate — GPU-offloaded GELO

The final, heavy accuracy gate for the offloaded-attention path (acceptance
tier-5 in `docs/dev/logs/perm-attn-gpu-offload.md`). **Run sparingly, after full
wire-in — not as a debug loop.** If accuracy regresses, localise with
microbenches + theory (`cv_cover_fp16_drift_sweep`, `analysis_cv_fp16_error`),
then re-run this once to confirm.

**Reference:** Plain Qwen3-4B via llama.cpp = **6/20** (HumanEval pass@1, n=20,
greedy; `docs/prototype/aloepri-llm.html`). We compare the offloaded **secure**
path against this, not against an in-TEE re-run.

## Always gate on coherence first

Before the heavy run, a 1–3 prompt coherence check that the model returns
relevant, coherent output:

```
GELO_GPU_PREFILL_OFFLOAD=1 GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 \
GELO_COVER_KAPPA=6 GELO_BENCH_VARIANT=4b GELO_BENCH_N=24 \
  cargo test --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench \
    cover_greedy_parity -- --ignored --nocapture
```

Eyeball the `PARITY_TOKENS` line; if it's incoherent, stop — debug before spending the benchmark.

## Three-step flow

1. **Vendor the subset** (once; needs `datasets` — the AloePri eval env, **not**
   the gelo-attack container). Pins the exact 20 problems via `task_ids.json`
   (same as the AloePri Qwen3-4B runs → directly comparable to Plain = 6/20):
   ```
   python3 evals/humaneval-gate/prepare_subset.py            # → subset.jsonl
   ```
2. **Generate** (host, GPU) — the secure cell `B` (offload + `C_v`, κ=6):
   ```
   GELO_GPU_PREFILL_OFFLOAD=1 GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 \
   GELO_COVER_KAPPA=6 GELO_BENCH_VARIANT=4b \
     cargo test --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench \
       humaneval_gate_generate -- --ignored --nocapture     # → completions.jsonl
   ```
3. **Score** (container or host with python3):
   ```
   python3 evals/humaneval-gate/score.py                    # pass@1 X/20 vs 6/20
   ```

## Ablation (acceptance tier-5)

- **B — offload + `C_v` (κ=6, secure):** the gate. Accept iff `B ≥ 6/20 − ε`.
- **A — in-TEE** (drop the offload flags) — only if you want a fresh local baseline; otherwise the 6/20 llama.cpp reference is the baseline.
- **C — offload − cover (κ=1):** run **only if B regresses**, to attribute the loss — `A`-vs-`C` = the offload itself; `C`-vs-`B` = the cover. Move `completions.jsonl` between runs (or pass a path to `score.py`).

Determinism note: the offload (cubek autotune + fp16) is not bit-reproducible
across processes; pass@1 over 20 problems is the in-budget signal, not a
publishable number.
