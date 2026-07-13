# PDF review round 8 — comment audit (2026-06-09)

Source: `pdfannots` extraction of annotated `main.pdf` (saved 22:43). All 36 comments on §4 (TEE),
pp.7–9. **36 of 36 resolved** (35 ✅ · 1 ℹ️). Reconciled: JSON total 36 (35 Highlight + 1 Text);
markdown showed 35, recovered the free-floating Text note (#13). Build after fixes: clean, 30pp,
0 undefined, no overfull. Committed in `<pending>`.

Legend: ✅ done · ℹ️ answered, no standalone change · ⏸ deferred · ❌ won't-fix · ⬜ pending.

Two substantive reframes (grilled + researched): **(A)** dropped the self-referential "baseline"
framing from §4, deferring cross-family comparison to §9. **(B)** rebuilt the side-channel
limitation: LeftoverLocals/WeSee are vendor-patched; the one structural case is TEE.Fail (physical
DDR5 interposition on deterministic AES-XTS, which AMD+Intel declare out of scope, no fix), reframed
from "blind spot" to "deliberate vendor-threat-model exclusion" + the why-silicon-trust-is-fragile
meta-argument; propagated the same reframe to §9 ssec:sidechannel and resolved its Spectre/Foreshadow
`\needswork` (cites spectre, foreshadow, tee_fail).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | p7 intro | "now vendor-plural" = correction-commentary | ✅ | Removed; now "We cover two such accelerators: a confidential NVIDIA GPU and a confidential NPU" |
| 2 | p7 intro | baseline point duplicated | ✅ | Dropped "performance/trust baseline" clause; intro no longer self-describes as baseline |
| 3 | p7 intro | "enclave construction" arbitrary | ✅ | → "enclave provisioning" |
| 4 | p7 intro | bad sentence structure | ✅ | Rewritten: "The hardware requirement, rather than any performance gap, is what motivates the…hybrid split" |
| 5 | p7 intro | Ascend-CC is also malicious-host | ✅ | Intro now names both Opal (adds ORAM) and Ascend-CC as the malicious-host schemes |
| 6 | p7 Appr | "Four steps recur" | ✅ | → "The workflow has four parts" |
| 7 | p7 Appr | "measurement" arbitrary | ✅ | Glossed: "the enclave's measurement (the hash of the loaded code, model, and configuration)" (standard attestation term) |
| 8 | p7 Appr | why "output handling"? | ✅ | Renamed → "operational discipline…forbidden by policy rather than enforced by the hardware" (policy, not a TEE mechanism) |
| 9 | p8 Appr | bandwidth only? capacity matters | ✅ | → bounded by "memory capacity, which caps the model size that fits, and…absence of a GPU, which throttles throughput" |
| 10 | p8 Appr | redundant + literal glitch | ✅ | Removed redundant band lead-in; paragraph now opens "The Tier-3 schemes run on a trusted accelerator. The confidential GPU places…" |
| 11 | p8 Appr | drop baseline notion entirely | ✅ | Removed "baseline the rest of the paper measures against"; comparison deferred to §9 (reframe A) |
| 12 | p8 Appr | how does Opal resist host tampering? | ✅ | Added: verifies integrity+freshness of every result against attested Merkle-authenticated state, rejecting tampered/rolled-back execution |
| 13 | p8 Text | (Q) doesn't E2E enclave-key encryption already stop tampering? | ℹ️ | Answered in the Opal edit: encryption hides content but not host control of scheduling/mapping/replay, so integrity needs explicit verification (confidentiality ≠ integrity) |
| 14 | p8 Appr | drop "need not be an NVIDIA GPU" | ✅ | Dropped; → "Ascend-CC places the enclave on a discrete NPU instead" |
| 15 | p8 Appr | confusing attestation sentence | ✅ | → "It attests each operator binary before execution, so a malicious host cannot substitute a rogue operator that would copy the plaintext out" |
| 16 | p8 Perf | relative overhead misses absolute perf | ✅ | Added absolute-baseline contrast (H100 ≈1000 TFLOPS FP16 / 3.35 TB/s HBM3 vs Ascend 910A ≈256 TFLOPS FP16, lower bandwidth) + non-comparability caveat |
| 17 | p8 Perf | why amortized? | ✅ | Explained: trust cost is a fixed per-request charge (channel encryption + one-time setup) that does not scale with token count |
| 18 | p8 Perf | 16% referenced as if known | ✅ | Now introduced+explained: the fixed charge "spread over almost no work" at 50 tokens, <0.1% at production lengths |
| 19 | p8 Perf | same (Ascend 16%) | ✅ | Same edit as #18 |
| 20 | p8 Perf | what is "Hopper installed base"/"current"? | ✅ | → "On today's deployed Hopper-generation GPUs" |
| 21 | p8 Perf | "finds dominates" grammar | ✅ | → "finds this staging, not the GPU computation, to be the dominant source" |
| 22 | p8 Perf | "near 1.2×" of what? | ✅ | → "near 1.2× the unprotected latency" |
| 23 | p8 Perf | "training in simulation"? | ✅ | → "evaluated on training workloads in architectural simulation rather than on deployed silicon" |
| 24 | p8 Perf | "at the protocol level" sw/hw? | ✅ | → "encrypts the link in the PCIe controller itself" (hardware) |
| 25 | p8 Perf | what installed base? | ✅ | → "which most deployed systems do not yet" |
| 26 | p8 Perf | "above the band" arbitrary | ✅ | → "Opal sits apart from the rest" |
| 27 | p8 Perf | "29× lower infrastructure"? validate | ✅ | Corrected (verified vs Opal paper): ORAM adds 1.57× query latency over a non-oblivious enclave; O(log N) access is ≈29× higher-throughput and ~order-of-magnitude cheaper at scale than in-enclave full-scan for the same privacy |
| 28 | p8 Lim | duplicate "structural not closed" | ✅ | Limitations rewritten; duplication removed |
| 29 | p8 Lim | "scheme" used for hardware | ✅ | Limitations now uses "hardware and platforms"/"designs"/"families"; caption states items are hardware not only schemes |
| 30 | p8 Lim | "schemes"; most attacks patched; find structural ones | ✅ | Reclassified: WeSee (AMD firmware fix) + LeftoverLocals (vendor-patched) → patched band; TEE.Fail → structural/vendor-declined band (researched) |
| 31 | p8 Lim | not a "blind spot" | ✅ | Reframed to "a class the vendor threat model deliberately excludes" in §4 and §9 (reframe B) |
| 32 | p8 Lim | why is silicon-trust bad long-term? | ✅ | Added meta-argument: guarantee holds only within vendor threat model; performance optimizations (deterministic enc, speculation, shared caches) keep reopening channels; a hardware-anchored design can't outrun this without the vendor |
| 33 | p9 Lim | "manufactured" → "forged" | ✅ | → "forge one outright once the bus encryption is broken" |
| 34 | p9 Lim | "fresh quotes" standard? | ✅ | → "a fresh, nonce-bound quote on each session" (standard attestation freshness) |
| 35 | p9 keyinsight | drop "vendor-plural" | ✅ | → "which more than one vendor now supplies but which a commodity device still lacks" |
| 36 | p9 keyinsight | "one risk" then two | ✅ | → single risk: "a physical or microarchitectural attack on the silicon the guarantee rests on" |
