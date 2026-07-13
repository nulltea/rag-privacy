# §4 (TEE) four-lens audit synthesis — 2026-06-09

Target: `manuscript/sections/04_tee.tex` (freshly written + round-8-revised). Lenses:
/humanize (humanize-auditor), /proofread (proofreader), /term-audit (inline), /verify-claims
(claim-verifier, fresh context, 11 claims). Build at audit time: clean, 30pp, 0 undefined.

Headline: **voice is clean (0 HIGH AI-tells), one HIGH factual error (Ascend 16% misattribution),
the rest are mechanical grammar/consistency + a few register choices.**

## A. FACTUAL (verify-claims) — 9/11 SUPPORTED

| ID | Finding | Sev | Resolution |
|---|---|---|---|
| F1 | "Ascend-CC's 16% at a 50-token input" is **GPT-Neo-125M**, not Llama. Llama-2/3 stays <0.1% at all lengths incl. 50 tokens. As written it implies Llama hits 16% at short sequences. | HIGH | Attribute 16% to the tiny GPT-Neo-125M model; Llama stays <0.1% even at short sequences |
| F2 | "oblivious retrieval roughly 29× higher-throughput" — 29.5× is QUERY throughput (ingest is 9.05×). "retrieval"=query so correct, but ambiguous. | tighten | → "29× higher in query throughput" |
| F3 | H100 "on the order of 1000 TFLOPS FP16" = dense figure (1,979 w/ sparsity). | tighten | → "dense FP16"; keeps apples-to-apples with Ascend 256 (also dense) |
| — | TEE.Fail (vendor out-of-scope verbatim: Intel "no current mitigation", AMD "does not plan to publish updates"; <$1000 DDR5; SGX/TDX/SEV-SNP+conf-GPU; deterministic AES-XTS) | ✓ | confirmed |
| — | deterministic AES-XTS + dropped Merkle-integrity tree vs client SGX | ✓ | confirmed |
| — | WeSee (S&P'24, CVE-2024-25742, AMD firmware fix); LeftoverLocals (CVE-2023-4969, vendor-patched) | ✓ | confirmed |
| — | Opal 1.57× query latency, $1.40M vs $21.13M (15×); TensorTEE 4.0×/2.1%/sim-training; PipeLLM <19.6%; H100 4-8%/bounce-buffer/TEE-IO | ✓ | confirmed |

## B. PROOFREAD

| ID | Line | Finding | Fix |
|---|---|---|---|
| P1 | 23 | `\cite{kaviani_opal}` (sole outlier; all others `\citep`) | → `\citep` |
| P2 | 85-86 | unit spacing: `$1000$ TFLOPS`/`$256$ TFLOPS` vs `\,TB/s`, `7\,nm` | → `\,TFLOPS` |
| P3 | 82 | "and Ascend-CC under 0.1%" elided verb doesn't carry | → "and Ascend-CC adds under 0.1%" |
| P4 | 41 | "(Opal and Ascend-CC the exceptions)" missing verb | → "(Opal and Ascend-CC are the exceptions)" |
| P5 | 116 | "which most deployed systems do not yet" trailing verb | → "do not yet provide" |
| P6 | 58-65 | Opal sentence ~75 words (semicolon + 3 "and") | split after "rolled-back execution" |
| P7 | 179-183 | structural-residue sentence ~55 words + comma-ambiguous list | split after "memory bus"; parenthesize the optimization list |
| P8 | 177-179 | "structural residue, not an oversight... but a class..." comma-splice apposition | colon after "residue" |
| P9 | 74 | three-party comma apposition reads as run-on | parenthesize "(the data owner, the model owner, and the cloud)" |
| P10 | 22/58 | near-verbatim repeat: "the corpus's strongest adversary, a malicious host that may tamper with computation outside the enclave" | vary the second occurrence |
| P11 | tab | TEE.Fail row overfull-hbox risk | verify in PDF (clean build reported no overfull) |

## C. HUMANIZE — 0 HIGH, 2 MED, 5 LOW (cosmetic only)
- Zero em-dash violations, zero boilerplate/cliché/sycophancy/hedge-stacking. Bold-name scheme
  paragraphs = structure, not formulaic tell.
- 2 MED = the two run-ons already in P6/P7. LOW hyphenation/tricolon = domain-mandatory, no action.

## D. TERM-AUDIT (register / word choice)

| ID | Line | Current | Proposed | Note |
|---|---|---|---|---|
| T1 | 117 | "Opal sits apart from the rest:" | "Opal is the exception:" | matches §4's own "exceptions" (line 41); less colloquial |
| T2 | 71 | "copy the plaintext out" | "exfiltrate the plaintext" | canonical security term |
| T3 | 184 | "cannot outrun this without the vendor" | "cannot close this without the vendor" | metaphor → standard |
| T4 | 82 | "each platform's own TEE tax" | keep / "overhead" | borderline idiom — DECISION |
| T5 | 177 | "the structural residue" | keep / "the structural remainder" | borderline metaphor — DECISION |
| T6 | 95-104 | "trust machinery" ×2 / "trust cost" / "TEE tax" | standardize? | three phrasings for overhead — DECISION |

## Resolution plan
- **Apply unambiguously (no grill):** F2, F3, P1–P5, P8, P9, P11-check.
- **Grill:** F1 reframe; T1–T6 register batch; P6/P7 splits + P10 redundancy.

## RESOLVED (2026-06-09, build clean 30pp/0 undef/0 overfull)
All 21 items applied. Grilled decisions: F1 → **dropped the 16% example** (Llama <0.1% across
lengths; charge visible only on very small models/short inputs, no number). Register: applied
**all** (T1 sits-apart→is-the-exception, T2 copy-out→exfiltrate, T3 outrun→close, T4 TEE-tax→
overhead, T5 residue→remainder, T6 trust-machinery→trust-mechanisms + removed competing phrasings).
Run-ons: split both (Opal, structural-remainder w/ parenthesized Oxford list) + trimmed the repeated
"strongest adversary" phrase (line 58 → "confronts that malicious host directly"). Mechanical: F2
(29× query throughput), F3 (dense FP16), P1 (\citep), P2 (\,TFLOPS), P3/P4/P5 (elided verbs),
P8 (colon), P9 (parenthesized 3-party list), P11 (no overfull). Verify-claims: 9/11 clean; the two
issues (F1 GPT-Neo misattribution, F2 query-vs-ingest) fixed. Voice was 0-HIGH (cosmetic only).
