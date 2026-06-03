---
type: handoff
status: stale
created: 2026-06-03
updated: 2026-06-03
tags: [gelo, dgpu, attention, gpu-offload, cubek, candle-flash-attn, cuda, sm120, ragged, blackbox, tensor-core, patch]
companion: [2026-06-02-offload-attention-collapse-sm120-rootcause, perm-attn-gpu-offload]
superseded_by: 2026-06-03-blackbox-ragged-padding-and-humaneval-gate
archive_reason: >
  The planned path (candle-flash-attn spike, then a vendored cubek ragged-blackbox
  patch) was overtaken same-day: the user chose caller-side n_q padding instead,
  which shipped and passed the HumanEval gate (7/20 = in-TEE parity). Retained for
  the candle-spike go/no-go criteria and the parked in-kernel cubek-patch mechanics.
  See the superseding handoff.
---

# Handoff — ragged blackbox attention: candle-flash-attn spike, then vendored cubek patch

## What this session is for

Two tasks, in order:

1. **candle-flash-attn integration spike** — decide whether Dao's CUDA FlashAttention
   (via `candle-flash-attn`) is a viable alternative to patching cubek for the
   ragged-prompt prefill offload. Timeboxed; the outcome is a go/no-go note, not an
   integration.
2. **Prototype the vendored cubek patch (the committed direction)** — make the
   tensor-core (`BlackboxAccelerated`) kernel accept ragged `n_q` by completing the
   query-read predication, using the user's fork **cloned at `/home/timo/cubek`**
   (`github.com/nulltea/cubek`).

**Do not re-derive the diagnosis or the upstream/fork research** — it is already
written up. Read these first:

- Prior handoff (full diagnosis + the authenticated GitHub API upstream/fork sweep):
  [`2026-06-02-offload-attention-collapse-sm120-rootcause.md`](2026-06-02-offload-attention-collapse-sm120-rootcause.md)
  — see its new section *"Ragged blackbox kernel — confirmed an unaddressed upstream gap"*.
- Dev-log (design + conclusions, paper source):
  [`../dev/logs/perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md).

## Settled context (don't relitigate)

- The blackbox ragged-`n_q` limit is a **genuine, unaddressed upstream gap**, not a
  missing config: no release fix (0.2.0 newest; `main`/0.3.0-pre.1 still has it), no
  issue, no PR (only #150 which *added* the reject), no branch, and of 34 forks only
  3 are trivially ahead (none touch attention). Confirmed by source read of `main` +
  authenticated API sweep.
- **Not algorithmic.** Tensor-core FA handles ragged `n` everywhere via `ceil_div` +
  boundary-tile predication. cubek's own `Unit` routine already does ragged `n`.
- **Root cause = one unchecked reader.** The accelerated query reader
  (`cubek-attention/src/components/global/simple/reader/query.rs`, `QueryReader::get_tile`)
  slices the query tile straight from gmem (unchecked `to_source_pos` /
  `to_linear_slice`). PR #150 chose to *reject* non-divisible `seq_q` in
  `routines/blackbox_accelerated.rs::validate()` (`"Stage seq_q must divide problem
  seq_q"`) instead of predicating. The predication primitive
  (`AttentionGlobalLayout::is_in_bounds` / `to_source_pos_checked` in
  `components/global/layout.rs`, gated by the comptime `check_bounds.seq_q` flag) is
  already wired through `components/global/simple/setup.rs` as `check_row_bounds` and
  is used by every other reader + the `Unit` query path — the accelerated query read
  is the **sole consumer that ignores it**.
- **Alternative-library verdict (this session):** none beats patching cubek.
  - **burn-cubecl / `burn_attention`** = cubek under the hood (Burn 0.20 introduced
    CubeK *for* its flash attention) → same gap, no help.
  - **candle-flash-attn / cuDNN-via-cudarc / mistral.rs** = different runtime
    (candle/cudarc, not cubecl), CUDA-only, require bridging cubecl `CudaRuntime`
    device buffers out (unsupported cross-context ptr sharing, or copies that erode
    the offload), heavy nvcc/CUTLASS build. Fallbacks only.
  - **Rust-CUDA / cuda-oxide / KAIO** = kernel-authoring infra, no production FA;
    that's "write our own", already rejected.
  - cubek patch stays inside cubecl: shares buffers, keeps the `Unit`/Vulkan dev
    path, no new runtime/build, fix is tiny + upstreamable.

## Task 1 — candle-flash-attn spike: what to actually determine

The spike exists to *confirm or kill* candle-flash-attn as a fallback. Answer these,
cheapest-to-falsify first; stop and write no-go the moment one fails hard:

1. **sm_120 support (highest kill-risk).** RTX 5090 is Blackwell **consumer sm_120**.
   Dao FlashAttention-2 ships kernels for sm80/86/89/90; **FA3 is sm90 (Hopper) only**.
   Verify candle-flash-attn even *builds and runs correct* on sm_120 — this is the
   exact class of failure that bit cubek originally (see prior handoff). If it falls
   back / fails to compile for sm_120, candle-FA is dead for this box → no-go.
2. **Runtime/buffer bridge.** Our operands are cubecl `TensorHandle`s on `CudaRuntime`
   (`cubecl-cuda` driver API). candle uses `cudarc`. Determine the cheapest legal way
   to hand operands across (device-ptr/context sharing feasibility vs forced
   host/D2D copy) and **measure the copy overhead** — if it's a host round-trip per
   layer it likely eats the offload win.
3. **Feature fit.** Needs: fp16, causal mask, GQA (Hq=32/Hkv=8), arbitrary `n`. The
   prefill rotation cover uses the *standard public causal mask* (no permuted mask, no
   partial-stats output needed) — so this is in candle-FA's wheelhouse *if* (1)+(2)
   pass.
4. **Build cost** in our workspace (nvcc + CUTLASS, CI implications).

Likely outcome (hypothesis): no-go on (1) and/or (2) → proceed to Task 2. Record the
result in the prior handoff's "alternatives" note either way.

## Task 2 — vendored cubek patch (the deliverable)

**Goal:** `BlackboxAccelerated` accepts ragged `n_q` (predicate the partial query
tile) and is proven correct on the existing gate at ragged `n`.

### ⚠ #1 gotcha — version compatibility (resolve before editing)

The fork at `/home/timo/cubek` is on **`main`** (cubek `0.3.0-pre.1`, reorganized
`src/forward/` + `src/backward/` layout) and pins **cubecl via a git rev**
(`f683f2d…`, ~0.11-pre). Our workspace pins **cubecl `=0.10.0`, burn `=0.21.0`,
cubek-attention `=0.2.0`** (flat `src/routines/` layout) — see `Cargo.toml:90-116`.
Patching against the fork's `main` will drag in an incompatible cubecl and **won't
build** against our stack.

→ **Branch the fork from the `0.2.0` release** (check `git -C /home/timo/cubek tag` /
`git branch -a` for the `v0.2.0` tag or `release/0.2`), apply the patch there, and
point `[patch.crates-io]` at that branch/worktree. Only then are paths
`src/routines/blackbox_accelerated.rs` (flat) and a cubecl-0.10.0 dep correct.
(Upstreaming later can target `main`/`forward/` separately.)

### The two edits (on the 0.2.0 line)

1. `crates/cubek-attention/src/components/global/simple/reader/query.rs` —
   `QueryReader::get_tile`: route the query-tile load through the **checked** accessor
   (`to_source_pos_checked` / `is_in_bounds`) so out-of-range rows are predicated
   (zero-filled), instead of the raw `.slice(...).to_linear_slice()`. Mirror how the
   `Unit` query path / the K/V/mask readers already do it.
2. `crates/cubek-attention/src/routines/blackbox_accelerated.rs` — `validate()`:
   remove/relax the `"Stage seq_q must divide problem seq_q"` rejection so
   `check_bounds.seq_q = true` is allowed. The output **writer** already predicates on
   `check_bounds.seq_q`, so the masked epilogue store needs no change.

Worth a look while there: `check_bounds()` in `src/definition/blueprint.rs` computes
`seq_q` with what looks like reversed `is_multiple_of` args (`elements_in_stage_seq_q
.is_multiple_of(problem.seq_q)`); it errs toward *enabling* the check (safe), but
confirm the partial-tile case sets the flag correctly once the reject is lifted.

### Wiring + proving it

- `[patch.crates-io]` in the workspace root `Cargo.toml`:
  `cubek-attention = { path = "/home/timo/cubek/crates/cubek-attention" }` (or `git`
  + branch). Keep the `version = "=0.2.0"` requirement satisfied.
- Our call sites (no change expected): `crates/gelo-gpu-wgpu/src/lib.rs` —
  `cubek_strategy_from_env()` (~:1293), `cubek_attention_folded` (~:1323),
  `cubek_attention_folded_gqa` (~:1505); strategy via `CUBEK_STRATEGY` (`unit`
  default / `blackbox`).
- **Gate (already built):** `crates/gelo-gpu-wgpu/tests/cubek_prefill_cover.rs::cubek_gqa_nan_nsweep`.
  Prove clean (0/N NaN) under `CUBEK_STRATEGY=blackbox` at **ragged** `n` (the n-sweep
  hits non-tile-aligned lengths). Today blackbox passes only tile-aligned `n`; success
  = ragged `n` now also clean.
- Then re-run the end-to-end check and the canonical bench (CLAUDE.md) to confirm the
  ~2.45× attend win survives on ragged prompts.

### If the patch stalls — fallback already analysed

Option (1) **caller-side `n_q` padding** to a multiple of `elements_in_stage_seq_q()`
+ slice-back (sound under causal masking; overhead ≈ `(E−1)/n` ≤ ~0.7% at n=2048),
implemented entirely in `gelo-gpu-wgpu`, no cubek change. Ship this if the patch needs
more time than available.

## Build / env notes

- `--release` mandatory (cubecl shader compile). CUDA is a **default feature** on this
  box (`crates/gelo-gpu-wgpu/Cargo.toml`); CUDA 13.0 / driver 595.71.05 / sm_120
  present. Non-CUDA hosts: `--no-default-features --features blas`.
- With CUDA default, any `GELO_GPU_PREFILL_OFFLOAD=1` gate that routes blackbox on the
  *unpatched* kernel panics on ragged `n` — that's the bug under repair.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit
  only when asked.

## Security note (action required)

A short-lived, repo-scoped fine-grained GitHub PAT was used **inline** during this
session's API research, so its value is in the transcript/tool logs on this machine.
**Revoke it now** (it was read/write on a single repo; revocation fully closes it).
The token value is intentionally **not** reproduced in this handoff.

## Suggested skills

- **`diagnose`** — Task 2 is a reproduce→fix→prove loop with the gate already built
  (`cubek_gqa_nan_nsweep` + `CUBEK_STRATEGY=blackbox` at ragged `n`); use it to drive
  the patch and avoid "it ran ≠ it's correct" (perf benches time NaN kernels silently).
- **`verify`** — after the patch, confirm the ~2.45× blackbox attend win holds on
  ragged prompts via the canonical bench (CLAUDE.md), not just the unit gate.
- **`code-review`** — before committing the `[patch.crates-io]` wiring + any
  `gelo-gpu-wgpu` changes; decide what to keep (padding fallback vs patch).
