---
type: handoff
status: current
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, dgpu, attention, gpu-offload, cubek, blackbox, ragged, humaneval, accuracy, perf, cv-cover]
companion: [perm-attn-gpu-offload, gpu-offloaded-attention-with-value-cover]
supersedes: [2026-06-03-candle-fa-spike-and-cubek-ragged-patch]
---

# Handoff — blackbox ragged-prompt offload (caller-side padding) + HumanEval acceptance gate PASS

## TL;DR

- **Ragged-prompt support for the blackbox (tensor-core) prefill offload shipped**
  via **caller-side `n_q` padding** (not a cubek patch — see *Decision* below).
- **HumanEval accuracy acceptance gate PASSES: pass@1 = 7/20 = in-TEE parity**
  (offload + blackbox + `C_v` κ=6, Qwen3-4B; llama.cpp plain = 6/20). No regression
  from offloading or from securing it.
- All changes are **uncommitted** on `dgpu-nvidia-bringup` (commit only when asked).
- Prefill-offload default-on now blocks on **one** item: the `WEIGHTS-PUB`/`C_v`
  **security acceptance** (policy sign-off). Accuracy + correctness are cleared.

Design write-up: [`gpu-offloaded-attention-with-value-cover.md`](../dev/prototype/gpu-offloaded-attention-with-value-cover.md).
Living log: [`perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md).
cubek-gap diagnosis + upstream/fork research: [`2026-06-02-offload-attention-collapse-sm120-rootcause.md`](2026-06-02-offload-attention-collapse-sm120-rootcause.md).

## Decision (this session) — padding, not a cubek patch

The cubek ragged-`seq_q` limit is a **genuine, unaddressed upstream gap** (confirmed
by an authenticated GitHub API sweep — no release/issue/PR/branch/fork fix; recorded
in the rootcause handoff). Two fixes were on the table: **(a) caller-side `n_q`
padding** (our code, no fork) vs **(b) an in-kernel cubek predication patch**.

Investigating (b) showed it is **real kernel work, not a 2-line change**: the
blackbox query reader (`cubek-attention/.../reader/query.rs::get_tile`) reads the
query tile straight from gmem (`to_linear_slice`, unchecked), distinct from the
K/V `FullStageGlobalReader` which staged-reads with bounds — so a correct fix needs
smem-staged/predicated query loads (or a new checked-read primitive), with
GPU-loop iteration. Given that, and that padding is <1% overhead at n=2048, the
user chose **(a)**. (b) is parked (see *Parked alternatives*).

## What landed (uncommitted, `dgpu-nvidia-bringup`)

1. **Caller-side `n_q` padding** — `crates/gelo-gpu-wgpu/src/lib.rs`:
   `cubek_blackbox_seq_q_align()` (→ `Some(16)` under `CUBEK_STRATEGY=blackbox`,
   else `None`; 16 = stage extent for the `num_planes=1`/`partition=1` strategy ×
   fp16 MMA m=16) and `pad_seq_q_zeros()`, wired into `cubek_attention_folded` and
   `cubek_attention_folded_gqa` via `CowArray` (borrowed when aligned, owned when
   padded), slicing phantom rows off the output. K/V left ragged (cubek's staged
   reader masks `seq_kv`). Correct because cubek masks causal on **absolute**
   positions, so real rows are unaffected and zero phantom rows attend real keys
   then get discarded.
   - **Validated** (`crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs`,
     `CUBEK_STRATEGY=blackbox`): `cubek_gqa_nan_nsweep` clean (0 NaN) across ragged
     n = 8…2048; `cubek_folded_causal_parity` (ragged n=24) max_abs **2.7e-5** vs
     CPU. Unit path unaffected. **Small-KV floor:** n_kv below the tensor-core tile
     (a handful of tokens) has no valid MMA tile — degenerate only.
2. **HumanEval gate instrumentation** — `crates/gelo-gpu-wgpu/tests/qwen3_m1_12_r1_q1_microbench.rs`
   `humaneval_gate_generate`: per-prompt `profile` capture accumulated into a
   cumulative per-op table (prefill `tee:attn_prefill_offload`/`prefill_cover:*` +
   decode `tee:attn_resident_cover`/`cover:*` + shared matmul/mask), with
   prompt-length stats. New env knobs: `GELO_HE_LIMIT` (cap prompts; smoke=1),
   `GELO_HE_PROFILE_EACH` (per-prompt dump). ⚠ `GELO_HE_OUT` must be **absolute**
   (cargo-test CWD is the crate dir, not the repo root).
3. **Dev-log update** (`perm-attn-gpu-offload.md`): compressed the falsified-fp16
   debugging narrative to conclusions + handoff refs (paper-devlog rule), fixed the
   stale "blackbox refuses ragged n" claims → "landed via padding", recorded the
   7/20 pass, added `---` section separators, renamed the duplicate `## Threat
   model` header.
4. **Prototype-note created**: `docs/dev/prototype/gpu-offloaded-attention-with-value-cover.md`
   (6-section design+results description).

## Acceptance gate result (2026-06-03)

`GELO_GPU_PREFILL_OFFLOAD=1 GELO_GPU_RESIDENT_COVER=1 GELO_RESIDENT_SIGMA=0.01 GELO_COVER_KAPPA=6 GELO_BENCH_VARIANT=4b CUBEK_STRATEGY=blackbox`,
20 prompts, `GELO_HE_MAXTOK=384` → **`completions.jsonl`** → `score.py`:

- **pass@1 = 7/20 = in-TEE GELO baseline (7/20)**; plain Qwen3-4B/llama.cpp = 6/20.
  `B ≥ A−ε` (ε=0) → cell C (κ=1) not needed.
- All 20 prompts (38–391 tok, ragged) generated cleanly on blackbox via padding,
  no NaN/launch failure. The ~6 misses are model-quality artifacts (markdown/prose
  contamination, logic errors) of the same class as in-TEE — not offload defects.
- Production-shape perf (from the dev-log, B=8 n=2048, κ=6): prefill **1.79× (Unit) /
  3.16× (blackbox)**, decode **4.85× / 5.25×** vs in-TEE; securing with `C_v` is
  perf-free. (HumanEval prompts are short/decode-dominated; the prefill-heavy
  production shape is where blackbox's 3.16× lands.)

## Remaining work (priority order)

1. **`WEIGHTS-PUB`/`C_v` security acceptance** — the **only** gate left for
   prefill-offload default-on. `C_v` κ=6 passed the membership + covariance gates
   (dev-log *Stage-2*), but acceptance is a policy sign-off on the residual
   (full-vocab top-5 ≈ 0.047) and on extending validation to **deeper contextual
   layers + production context lengths (2k–16k)** — covariance-alignment may
   strengthen as instance→population covariance at long context.
2. **Commit the session's work** (when asked) — `/code-review` first; decide what
   ships (the padding + instrumentation + docs).
3. **Small robustness follow-ups** (flagged, not done): (a) route prompts below the
   blackbox KV-tile floor (n_kv ≲ a handful) to the Unit kernel at the offload call
   site; (b) make `cubek_gqa_nan_nsweep` a dual-strategy regression lock that starts
   the blackbox sweep at the KV floor (it currently fails at n=1 under blackbox —
   the tiny-KV limit, not our bug).
4. **Open perf levers (no security/accuracy impact)**: cubek kv-head read-index
   (kills the materialised GQA expand), upload-bandwidth probe.

## Parked alternatives (not pursued; revisit only if needed)

- **In-kernel cubek patch (b).** Fork cloned at `/home/timo/cubek`, branch
  `fix/blackbox-ragged-seqq` off `release/0.2` (= our pinned cubek 0.2.0, cubecl
  0.10.0 — compatible; `main`/0.3.0-pre pins a git-rev cubecl and is **not**
  compatible). No commits. The two edits would be: predicate the query load in
  `reader/query.rs::get_tile` (route through `to_source_pos_checked`/`is_in_bounds`)
  + relax `routines/blackbox_accelerated.rs::validate()`; prove on
  `cubek_gqa_nan_nsweep`. Worth upstreaming if pursued.
- **candle-flash-attn / cuDNN / Rust-CUDA.** Evaluated, **not** spiked: burn's
  attention *is* cubek (no escape); candle-flash-attn (Dao) is the strongest kernel
  but a different runtime (cudarc, not cubecl) → buffer-bridging out of cubecl +
  CUDA-only + sm_120 support unverified (FA2 tops at sm90); only a fallback if we
  ever leave cubecl. Details in the rootcause handoff's alternatives note.

## Gotchas

- `--release` mandatory; CUDA is a default feature on this box; `GELO_HE_OUT` must
  be absolute. Perf is single-sample (±~7%).
- The padding align (16) tracks `cubek_strategy_from_env`'s blackbox partition
  counts — if those change, update `cubek_blackbox_seq_q_align`'s constant.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## Suggested skills

- **`code-review`** — before committing the working tree (padding + instrumentation
  + dev-log/prototype docs).
- **`verify`** — re-confirm the 7/20 + blackbox-ragged behaviour if any of the
  offload path changes before commit.
