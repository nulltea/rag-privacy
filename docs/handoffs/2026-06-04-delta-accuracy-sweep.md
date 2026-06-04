---
type: handoff
status: current
created: 2026-06-04
updated: 2026-06-04
tags: [sok, fidelity, accuracy-loss, methodology, li-et-al, section-8]
companion: [scheme-categorization, survey-corpus]
---

# Handoff — Δacc (accuracy-loss vs plaintext) sweep for non-cryptographic families

## Why

Adopting a reporting convention from Li et al. 2025 (PTI survey, arXiv 2505.10315): they
report fidelity as **plaintext acc / encrypted acc / LOSS** per benchmark task, a cleaner,
more rigorous form than our qualitative "≈plain / approx". Decision (grill 2026-06-04):
adopt **Δacc** (single accuracy-loss-vs-plaintext figure) — but **cryptographic schemes
only for now** (done this pass in `tab:crypto-inf`). This handoff covers extending Δacc to
the **other families** where the numbers exist.

## Done this pass (reference for the convention)

`tab:crypto-inf` Fidelity column → **Δacc** form:
- PermLLM: exact (0) · SHAFT: ≈0 (≈plain.) · CryptoMoE: −0.8% (99.2% retained) ·
  Fission: ≈0 (≈plain.) · Euston: approx† (FHE; NEXUS anchor loss 0.16–0.38%).
- Caption credits Li et al.'s plain/enc/loss convention; a hardware-basis note flags that
  latencies are paper-reported on differing hardware (SIGMA/NEXUS GPU) and not co-measured.

## The task

For the **non-cryptographic families**, find and report a Δacc (encrypted/protected vs
plaintext accuracy-loss, with the task/benchmark) **where the primary papers report it**;
mark `n/r` where they don't. Then surface it in the §8 fidelity view (`tab:performance`
and/or `tab:deployment`) and reconcile with the per-family "Fidelity" ratings already there.

Families + first-pass leads (verify against primary sources; do not fabricate):
- **TEE (Tier 3):** exact by construction (the enclave runs the real model) → Δacc ≈ 0 for
  PipeLLM / H100-CC / Opal; confirm none introduce approximation.
- **Static obfuscation:** these DO lose utility — AloePri ~0; SGT −0.4% util; OSNIP; Eguard
  (F1 94→5 is a *privacy* metric, not utility — do not confuse); TextObfuscator −1pp.
  Report task + Δacc per scheme.
- **Differential privacy:** the privacy–utility curve is the point — DP-Forward ~94% SST-2;
  SPARSE 65% util @ε=5; report Δacc *at a stated ε* (ε must accompany the number).
- **Hybrid split:** GELO / TwinShield / ObfuscaTune — typically exact or near-exact (TEE does
  the non-linears); confirm and give Δacc.
- **DP-RAG / output-DP:** Δacc is an output-quality metric, not model accuracy — flag the
  different notion rather than forcing it into the column.

## Method
- Primary-source numbers only; `n/r` where unreported (per the Completeness discipline — do
  not leave a silent blank). EdgeQuake `document_get_md` per scheme (NOT bundled queries;
  one aspect per query — see memory `feedback-edgequake-one-aspect-per-query`). Note: Δacc
  per-task often lives in paper tables EdgeQuake drops to placeholders (Part E #20) → may
  need the PDF.
- **ε must travel with every DP Δacc** (a DP accuracy number is meaningless without its ε).
- Record per-scheme in `scheme-categorization.md` (Part B / a new Δacc column); reconcile the
  §8 qualitative fidelity ratings (H/M/L) against the gathered Δacc.

## Watch-outs (don't conflate)
- **Utility-loss vs privacy-metric:** Eguard's "F1 94→5" and CAPRISE's "Vec2Text BLEU 83→12"
  are *privacy* (attack-defeated) numbers, NOT fidelity loss. Δacc = task-utility loss only.
- **Retrieval ≠ model accuracy:** crypto-retrieval fidelity is exact (PIR/ORAM/SSE) or
  approx-β (DCPE) — a recall/exactness notion, not Δacc. Leave retrieval rows as-is.

## Files
- `manuscript/sections/03_cryptographic.tex` (`tab:crypto-inf` — the done exemplar)
- `manuscript/sections/08_synthesis.tex` (`tab:performance`, `tab:deployment` — targets)
- `docs/research/scheme-categorization.md` (Part B — add Δacc)
- Li et al. methodology: its Table 1 is the convention reference (plain/enc/loss/task).

## Verify
`cd manuscript && make` → 0 undefined; watch table widths if a Δacc column is added to §8.
