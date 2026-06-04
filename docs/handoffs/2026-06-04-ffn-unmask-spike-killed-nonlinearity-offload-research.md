---
type: handoff
status: current
created: 2026-06-04
updated: 2026-06-04
tags: [gelo, security-spike, ffn-permutation, swiglu, weights-pub, aloepri, fission, threat-model]
companion: [2026-06-03-dgpu-perf-session-next-levers, aloepri-vs-gelo, gpu-offloaded-attention-with-value-cover]
# Resolves focus #1 of 2026-06-03-dgpu-perf-session-next-levers; that doc's other foci remain live (companion, not superseded).
---

# Handoff — FFN-intermediate-unmask spike: KILLED under `WEIGHTS-PUB`; next: deep literature research on off-trusted-path nonlinearity

## What this session settled

The prior handoff's **focus #1** ("Eliminate the FFN-intermediate unmask",
[`2026-06-03-dgpu-perf-session-next-levers.md`](2026-06-03-dgpu-perf-session-next-levers.md))
was a security spike. **Verdict: not viable under the standing
`WEIGHTS-PUB` default.** The lever is shelved (revisit only under an
explicit `WEIGHTS-BLIND` deployment + AloePri-gate, or a second-trust-domain
architecture — see Next).

Two concrete artifacts produced:

1. **Threat model written into `CLAUDE.md`** (new top section "Threat model
   (split TEE ⟷ untrusted GPU)"). Codifies `TEE-TRUST` / `GPU-ADV` /
   `WEIGHTS-PUB` / `NO-PLAINTEXT` (canonical source:
   [`../dev/logs/perm-attn-gpu-offload.md`](../dev/logs/perm-attn-gpu-offload.md)
   § "Threat model — standardized assumptions") plus the **cover invariant
   design rule**: *no orthogonal-or-permutation cover, however refreshed,
   hides token membership under `WEIGHTS-PUB`; only a non-orthogonal,
   per-session, activation-space cover does* (measured top-1 = 1.000 break,
   [`../dev/prototype/gpu-offloaded-attention-with-value-cover.md`](../dev/prototype/gpu-offloaded-attention-with-value-cover.md) §2).

2. **AloePri + Fission analysis** (this conversation; not yet written to a
   doc — see "Owed").

## Why the lever dies (the argument, condensed)

- The cost targeted is the orthogonal mask round-trip around SwiGLU
  (`Aᵀ` unapply of gate∥up ≈19 456 cols + `A` apply of down-input ≈9 728
  cols; ≈ −2.5 s ≈ 20%+ of B=1 prefill).
- To eliminate it the row-mask `A` must be **replaced** by a feature-axis
  cover that survives SwiGLU. **The only linear transform that commutes
  through `silu(g) ⊙ u` is a per-feature permutation** (+ a diagonal
  scaling on the *u*-path only). Scaling on the *g*/silu path and any
  additive blind on either path are non-correctable
  (`silu(g+r) ≠ silu(g)+h(r)`, `silu(sg) ≠ s·silu(g)`).
- A per-feature permutation is **bakeable into resident weights** but is
  **fixed per session** → preserves per-column moments → recoverable by
  column-moment matching against public `W_gate/W_up` → full intermediate
  plaintext → invert to `h1_norm` → prompt. This is *strictly weaker* than
  the orthogonal `O_v` cover that already failed the membership gate
  (top-1 = 1.000). It exposes the FFN intermediate (incl. `h1_norm` input)
  as feature-permuted **plaintext**, not just a leaked invariant.

## How the two requested papers bear on it

Both were examined as the requested follow-up; **both confirm the negative
verdict rather than rescue the lever** — each relies on a protection
GELO's threat model denies.

- **AloePri** (`AloePri.pdf`, ByteDance 2603.01499; FFN obfuscation §5.2.4):
  intermediate cover = permutation `Ẑ_ffn` (exact) + diagonal scaling
  `Ĥ_ffn` (**approximate** — applied to both gate & up, doesn't commute
  through silu; this is the source of AloePri's 0–3.5% accuracy loss),
  baked into **shipped-obfuscated** weights with two-sided key matrices
  `P̂/Q̂`. Its secrecy needs the server to **lack plaintext `W`** — voided
  by `WEIGHTS-PUB` (resident-clear weights ⇒ `W⁺·W̃` inverts any bare
  transform). The `C_v` doc already rejected exactly this ("baked-weight
  obfuscation (full AloePri)") for the value cover; same verdict transfers
  to FFN. `aloepri-vs-gelo.md` rows #3/#8 already scored AloePri's
  weight-space permutations as adding nothing under public weights.

- **Fission** (eprint 2025/653, Nillion+Meta; *could not fetch PDF — IACR
  403s the fetcher*; mechanism from Nillion's technical writeup): linear
  ops via Beaver-triple MPC; **non-linearities computed in the clear by a
  separate evaluator network** — MPC nodes shuffle the matmul output,
  reconstruct the *shuffled plaintext* at the evaluator, evaluate, re-share,
  unshuffle. **The additive sharing is removed *before* the nonlinearity,
  not carried through it** (so "does the additive blind commute over
  SwiGLU?" → no, and Fission never needs it to). Its security at the
  reveal point rests on (a) a **fresh full-tensor shuffle**, (b) a
  **second non-colluding party** that (c) **lacks the weights**. GELO has
  *one* untrusted party (the GPU), co-located with the host, holding the
  weights — it can offer none of the three. The only non-adversarial place
  GELO can reconstruct the intermediate is the **TEE**, which = today's
  unmask-in-TEE flow. **GELO already *is* the Fission pattern** (TEE =
  trusted evaluator); the unmask round-trip is the price of doing it with
  one party / no additive blind / public weights.

## Next session — deeper literature research (the user's stated focus)

Goal: survey techniques/schemes/papers for **running an elementwise
nonlinearity (SiLU/SwiGLU/GELU) off the trusted path when the linear-compute
party is untrusted, holds public weights, and is a single non-colluding
domain** — i.e. close (or definitively bound) the FFN-intermediate gap that
permutation/orthogonal/additive covers cannot. Concrete angles to chase:

- **Functional/structural covers that commute through silu** beyond
  permutation — is there any (non-permutation) group action under which
  SwiGLU is equivariant and that is *not* weight-bakeable-hence-recoverable?
  (Strong prior: no, but prove/cite it.)
- **Polynomial / spline approximations of silu** that turn the nonlinearity
  into a low-degree map an *orthogonal*-masked operand could survive
  (cf. CryptoNets/CKKS-style activation approximation, BumbleBee, Iron,
  BOLT, NEXUS). Trade: approximation error vs. greedy-parity (the project
  holds 7/20 HumanEval parity — any approx must defend that).
- **Selective / lookup-table nonlinearity** (Fission-class, Sigma, SIRNN,
  CrypTFlow2 LUT) and **FSS for activations** — what minimal extra party /
  preprocessing makes them sound, and is a *second attested TEE in an
  independent trust domain* (the only Fission-adaptation that works for us)
  ever justifiable on latency? (Repo survey already files Fission as
  "order-of-magnitude slower than TEE+GPU",
  [`../research/private-reranking-research.md`](../research/private-reranking-research.md):477.)
- **Covariant-obfuscation follow-ups / citations of AloePri** and the
  **SoK 2026/935** and **PP-LLM 2026/105** surveys (noted in
  [`../plans/m1-10-security-review.md`](../plans/m1-10-security-review.md) §1.8).
- Re-confirm the negative: is there *any* published scheme that reveals a
  fixed-permuted nonlinearity input to a **weight-holding** adversary and
  survives a moment-matching attack? If not, state it as a lower bound.

Deliverable: a `research`-type note under `docs/research/` with the
three-way map (GELO ⟷ AloePri ⟷ Fission ⟷ new findings), the
SwiGLU-commutation algebra, and a ranked list of any candidate that clears
`WEIGHTS-PUB`. If none clears it, record the bound and close the lever
formally.

## Owed (small, from this session)

1. **Write the spike note** — the AloePri/Fission/SwiGLU analysis above is
   only in this conversation. Offered but not yet written:
   `docs/research/` note + a `reference`-type pointer for the Fission paper.
   Fold into the Next-session research note rather than a standalone.
2. **Refresh `aloepri-vs-gelo.md` header** — its scope box still says
   "model-weight privacy is NOT a goal" and dates to 2026-05-18, *predating*
   the `WEIGHTS-PUB` decision (2026-05-29) and `C_v` (2026-06-02). Its
   technique-by-technique verdicts still hold; the framing needs a
   "superseded threat-model" note.
3. The prior handoff's **focus #1 is now resolved (killed)** — its #2 (B=8
   matmul scaling), #3 (decode tail-fold), and the "Owed/open" list
   (HumanEval gate re-run, decode bimodality, **push the unpushed 13-commit
   branch after `code-review`**, U-Verify-vs-16-bit-GPU precision fix) are
   **unchanged and still open** — see
   [`2026-06-03-dgpu-perf-session-next-levers.md`](2026-06-03-dgpu-perf-session-next-levers.md).
   Do not re-scope focus #1.

## Suggested skills

- **`deep-research`** — the stated next step. Pass a refined question, e.g.
  "schemes for computing SiLU/SwiGLU/GELU off the trusted path when the
  linear-compute party is a single, non-colluding, public-weight-holding
  adversary; emphasis on covers that commute through the nonlinearity or
  low-error polynomial approximations; for each, does it survive a
  moment-matching attack by a weight-holding adversary?"
- **`grill-with-docs`** — if a candidate survives and a design forms
  (esp. the second-trust-domain / approximate-silu directions), grill the
  leakage model against `CLAUDE.md`'s cover-invariant rule and the `C_v`
  precedent before building.
- **`code-review`** — still owed on the unpushed 13-commit
  `dgpu-nvidia-bringup` branch (carried from prior handoff).

## Gotchas

- `CLAUDE.md` now carries the threat model — treat it as the one-paragraph
  canonical statement; the long form is `perm-attn-gpu-offload.md`.
- Fission PDF (`https://eprint.iacr.org/2025/653.pdf`) and abstract page
  both **403 the WebFetch tool**; mechanism details came from Nillion's
  blog writeup — verify against the PDF via a browser / `gh`-style auth
  fetch before quoting numbers.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
