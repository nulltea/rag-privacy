# PDF review round 5 — comment audit (2026-06-05)

Source: `pdfannots` extraction of the annotated `main.pdf` (8 annotations; markdown bucket
showed 6, 2 free-floating Text notes recovered from JSON). All comments target §5 Static
Obfuscation (the round just written). **8 of 8 resolved** (7 ✅ · 1 ℹ️; 1 answered without a manuscript edit). Committed in `9cfdedf`.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred (reason + where tracked) · ❌ won't-fix (reason) · ⬜ pending (transient).

| # | Page / loc | Comment | Status | What changed |
|---|---|---|---|---|
| 1 | p8 §5.1 | "Focusing on only data or covariant is too broad." Mention data-only/covariant briefly in the §5 intro; Approaches spine should instead be **Mechanisms per scheme**, each scheme name in bold — same structure as §7 Hybrid Split Approaches. | ✅ | §5 intro now mentions the data-only/covariant split briefly (and why we don't organize by it). §5.1 rewritten §7-style: 6 bold-name mechanism paras (AloePri, SGT, OSNIP, Eguard, KV-Cloak, ObfusLM). |
| 2 | p8 §5.1 | "token substitution (SANTEXT/CUSTEXT" — **Drop this.** | ✅ | Removed from Approaches; SANTEXT/RANTEXT now only in the intro as the out-of-scope text-sanitization example. |
| 3 | p8 §5.1 | "learned embedding transforms (SGT [49]," — current embedding transforms are by definition **not data-only**. | ✅ | Dropped the "data-only" label on embedding transforms; intro states every representation-level scheme also transforms weights. |
| 4 | p8 (Text) | "there is actually no such split as data-only or covariant aside from token substitution, but we are not going to report on it." | ✅ | Intro states the only genuinely data-only variant is token substitution (out of scope); section organized by mechanism, not by that axis. |
| 5 | p8 §5.2 | §5.2 Representative Schemes should be **subsumed into Approaches** — same structure as §7. | ✅ | Representative Schemes subsection removed; its content folded into §5.1 Approaches. Section is now Approaches / Performance / Limitations. |
| 6 | p8 §5.3 | §5.3 Performance must be **much more nuanced**; need **accuracy** treatment; **first create a markdown doc tracking accuracy×performance×security per scheme**. | ✅ | (1) Doc built: `docs/research/static-obfusc-tradeoffs-2026-06-05.md`. (2) User chose **combined "Performance & Accuracy"**: §5.2 rewritten with separate Performance / Accuracy passages framing the headline ≈1×/near-lossless as a sweet-spot of a single hiding knob; added **`tab:obfusc-tradeoffs`** (knob → accuracy cost → perf cost, all 9 schemes); AloePri h/β cliff + ObfusLM k/ε + Eguard projection surfaced. \needswork removed. |
| 7 | p10 Table 5 | `tab:obfusc-attacks` "misses a lot of attacks." | ✅ | Table expanded 4→6 rows (added nearest-neighbour EIA, covariant weight-fingerprint); published-attack taxonomy. Companion recovery numbers kept out (\needswork) per scope rule. |
| 8 | p10 (Text) | Pointer: `docs/research/aloepri-attacks.md` (the attack list for #7). | ℹ️ | Pointer consumed: aloepri-attacks.md informed the expanded taxonomy + the trade-off doc; companion-novel recovery numbers deliberately excluded from the SoK table. No standalone edit. |
