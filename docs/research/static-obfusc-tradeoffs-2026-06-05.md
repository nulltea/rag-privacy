---
type: research
status: current
created: 2026-06-05
updated: 2026-06-05
tags: [static-obfuscation, tradeoffs, accuracy, performance, security, sok-corpus, section5]
companion: [aloepri-h-beta-interaction-2026-05-27, aloepri-attacks, obfuscation-round2-2026-06-05, scheme-categorization]
---

# Static-Obfuscation trade-off landscape — accuracy × performance × security

Built for PDF round-5 comment #6 (§5.3 "must be much more nuanced … create a markdown doc
to track accuracy, performance, and security trade-offs for each scheme"). The point: for
every §5 scheme there is a **control knob** that simultaneously moves all three axes, so
"≈1× overhead, near-lossless" is never the whole story. Once this is agreed, decide §5
reporting structure (separate Accuracy subsection vs combined "Performance & Accuracy" vs
accuracy-folded-into-attacks).

Grounding: ★ = extracted from the EdgeQuake-indexed primary source this session; † = carried
from corpus / round-2 notes, verify before manuscript. Companion-novel recovery numbers
(AloePri attack suite) live in `aloepri-attacks.md` and stay out of the SoK tables.

## The shared shape of the trade-off

Each scheme has a **hiding-strength knob**. Turning it up improves security (lower attack
success) and degrades at least one of {accuracy, performance}. The schemes differ in
*which* of the other two axes pays, and whether the cost is online (latency) or offline
(training/setup).

| Scheme | Knob | ↑knob ⇒ security | ↑knob ⇒ accuracy | ↑knob ⇒ performance | Dominant cost locus |
|---|---|---|---|---|---|
| **AloePri**★ | keymat half-expansion `h` (d_obs=d+2h); RoPE-window `β`; noise `α_e,α_h` | ↑ (larger hiding space) | ↓↓ (κ(K_d) compounds through L layers; β≥4 collapses) | ↓ (wider residual stream = more compute) | online width + accuracy cliff |
| **SGT**★ | MI-loss strength of trained transform | ↑ | ↓ (−0.3–0.5 pp util) | ≈0 online | **offline training** (~6 GPU-h → 2 days) |
| **OSNIP**★ | null-space dimension / orthogonality | ↑ (KNN-ASR→0) | ↓ (near-lossless) | ≈0 online (0.96 ms) | offline encryptor training |
| **Eguard**★ | dual-level MI objective; projection rank | ↑ (inversion F1→~4%) | ↓ (>98% task acc) | ↓ (~1.7× inference) | online ~1.7× + offline 1.6–3.4× train |
| **KV-Cloak**★ | block-permutation size `b` (key space `O((b²d+bd²)·b!)`) | ↑ (b! brute-force wall) | — (lossless, exact attention) | ↓ (<0.45% prefill) | online (tiny); offline matrix fusion |
| **ObfusLM**★ | cluster size `k`, DP budget `ε` ((k,ε)-anonymity) | ↑ (k↑ → Top-1 ↓ ~7× from k=5→20) | ↓↓ (−4% clf / −6% gen; longer texts worse) | ≈0 online (1.29 s/inf) | **offline fine-tune** (>10 h server) |
| **EE**† | key-space size (combinatorial) | ? (no formal proof) | — (exact) | ≈0 (near-plaintext) | none — but security unverified |
| **CodeCipher**★ | PPL threshold (obfuscation degree) | ↑ (edit distance↑) | ↓ (Pass@1 50.6→47.6 at fixed degree) | ≈0 online | offline discrete-gradient opt. |
| **IOI**★ | group size `n`, classes `|C|` | ↑ (1/C(r,2mn) pairing) | ↓ (decision resolution noise) | ↓↓ (2mn× inference calls) | **online call multiplier** |

## Per-scheme notes

### AloePri — the cleanest three-way coupling
The expansion `h` is the lever the reviewer flagged. `d_obs = d + 2h` adds a heuristic
hiding subspace, but (a) widens every matmul (perf), and (b) raises κ(K_d), whose per-layer
perturbation **multiplies** across L=36 layers — a 37% κ jump (h=128→256) becomes ≈10⁵×
logit-map amplification, so accuracy falls off a cliff (HumanEval 8/20 at h=128/β=2 →
readable-but-incoherent at h=256). `β` (RoPE-pair window) is a *discrete* security/accuracy
bifurcation: β=2 cancels exactly in Q·Kᵀ (accuracy preserved), β≥4 is generically non-zero
(collapse). So AloePri's "0–3.5% acc loss, ≈1×" is the *sweet-spot* (h=128, β=2) reading,
not a free lunch. Detail: `aloepri-h-beta-interaction-2026-05-27.md`.

### SGT / OSNIP / Eguard — the offline-training family
All three move the cost offline by training a transform/encryptor/projector. Security comes
from the trained objective (MI loss); the residual risk is that the *training* under-fits the
attacker (SGT's info-theoretic claim is contested — AloePri reports it broken by IMA with
>90% TTRSR, >50% acc loss, see `scheme-categorization.md` E#9). Eguard alone pays a visible
online tax (~1.7×) by running the projection at inference.

### KV-Cloak — the near-free one (different asset)
Lossless and <0.45% because S,M are fused into weights offline and P̂ is a cheap per-block
permutation. Security scales with block size b (b! term). The only "cost" is the TEE key
custody for the MB-scale secret matrices. This is the scheme with the *weakest* three-way
tension — its knob barely touches accuracy/perf — which is exactly why it is the strongest
deployability story in the family.

### ObfusLM — anonymity-set vs utility
k and ε trade EIA-resistance against utility monotonically (k=5→20 drops Top-1 recovery ~7×).
Unique in protecting generated output, but the offline fine-tune on obfuscated embeddings
(>10 h) is a real adoption cost and ties the obfuscation to a specific fine-tuned checkpoint.

### EE / CodeCipher / IOI — edge cases
EE: exact + near-zero, but **no formal security** (combinatorial hand-wave) → its "security"
axis is unverified; report with caveat. CodeCipher: code-LLM-specific; obfuscation degree
(PPL threshold) trades against Pass@k; LLMs still partially de-obfuscate (34%). IOI:
classification-only; the security knob (group size) is also the performance knob (2mn× calls)
— the tightest accuracy/perf/security coupling but a narrow use case.

## Reporting-structure options for §5 (decision pending)

1. **Combined "Performance & Accuracy" subsection** — one subsection, per-scheme knob → cost
   on both axes; security stays in §5.3 attacks. Cleanest; surfaces the coupling.
2. **Separate Accuracy subsection** — most explicit, +1 subsection.
3. **Accuracy folded into attacks** — frames accuracy-vs-privacy as one curve (e.g. ObfusLM k,
   AloePri h); tightest three-way coupling but buries fidelity from a perf-only reader.

Recommendation pending user review: option 1, with a compact "knob → (sec, acc, perf)" table
lifted from this doc.

## To verify before manuscript (★→confirm, †→ground)
- EE security basis (†): confirm there is genuinely no formal argument (commercial whitepaper).
- ObfusLM k=5→20 "Top-1 ↓ 7×" figure (★ from Fig 3 prose) — confirm exact number.
- Eguard 1.6–3.4× train and ~1.7× inference (★) — confirm against paper table.
- SGT "broken by IMA" (cross-scheme claim from AloePri eval) — keep as *reported-by-AloePri*.
