# PDF review round 4 — comment audit (2026-06-05)

Source: `pdfannots` extraction of the annotated `main.pdf` (saved 2026-06-05 18:31; comments
preserved here since rebuilds overwrite the annotated PDF). All 20 comments on §7 (Hybrid
Split), pages 9–10. Resolved via `/address-comments with /grill-me`.

**20 of 20 resolved** (19 ✅ · 1 ℹ️). Reconciliation gate: `pdfannots -f json` → **M = 20**
(19 anchored Highlights + 1 free-floating Text, #4) = audit rows = Σ terminal statuses; **0
rows ⬜**. PASS. Build: clean — 21 pp, 0 undefined refs/cites, 1 pre-existing 1.25 pt overfull.

Legend: ✅ done · ℹ️ answered, no change · ⏸ deferred · ❌ won't-fix. Decisions (grill): #2 drop
all CNN roots · #3 CMIF→§7 TEE+DP member · #4 "Approach"→"Approaches" paper-wide · #11 Portcullis
dropped · #19 build attack table.

| # | Loc | Comment (abbrev.) | Status | What changed |
|---|---|---|---|---|
| 1 | §7 intro | "near-baseline performance" not factually true | ✅ | Rewrote scope ¶: hybrid sits *below* confidential-GPU & static-obfusc on raw speed (pays TEE↔GPU comm cost), but more secure than static-obfusc + no confidential-HW dependence; deployment ranking deferred to §9. |
| 2 | §7 intro | Drop CNN-era roots + cites; keep ≤1 | ✅ | Deleted the Slalom/DarKnight/ShadowNet/SOTER/TEESlice lineage sentence + all 5 `\cite`s (now uncited; bib keys orphaned). Kept **none** (user chose drop-all). |
| 3 | §7.1 | Add CMIF (2509.09091) | ✅ | Added CMIF as a TEE+DP hybrid member in Approaches (embedding in CPU-TEE + RNM input-token DP, no per-layer return) + `\cite{cmif}` (refs.bib) + a row in tab:inf-perf. |
| 4 | §7 (free) | Rename all "Approach" subsections | ✅ | `\subsection{Approach}`→`Approaches` across §3, §4, §5, §6, §7 (paper-wide). |
| 5 | §7.1 | (SGX/TDX)→TDX/SEV-SNP; remove SGX | ✅ | §7 core now "CPU-TEE (Intel TDX or AMD SEV-SNP)"; §2 taxonomy "Intel SGX/TDX"→"Intel TDX"; §4 comment fixed. **0 "SGX" in rendered PDF.** |
| 6 | §7.1 | "or a confidential GPU" — none? | ✅ | Scoped to GELO: "a CPU-TEE, or in GELO's case a confidential GPU." Answered: GELO *is* the trusted-GPU + untrusted-GPU scheme (#18). |
| 7 | §7.1 | "disclosive" ambiguous | ✅ | "disclosive"→"revealing". |
| 8 | §7.1 | "mixes each batch's hidden states" — batch? | ✅ | Clarified: "fresh secret invertible matrix $A$ for each offloaded batch (a block of token hidden states $H$)"; offloads Q/K/V/O only. |
| 9 | §7.1 | ObfuscaTune ~5% — which params? | ✅ | Named: "input embedding, output head, LayerNorm/dropout (~5%); offloads attention+MLP weights (95%)." |
| 10 | §7.1 | "one-time obfuscation" — per inference/token? | ✅ | Corrected: matrices are **static, set up once and reused across inferences (refreshed periodically)** — neither per-token nor per-inference. |
| 11 | §7.1 | Portcullis — explain; really hybrid? | ✅ | **Dropped entirely** (user decision). Removed from §7 Approaches + tab:inf-perf. Investigation: TDX PII-anon gateway to a black-box LLM, no untrusted-GPU model offload → not a hybrid-split scheme. |
| 12 | §7.1 | "SCX encodes" — proper term? | ✅ | Reworded: SCX "protects the KV-cache not by encryption but by **permutation** (token + feature) + redundant embeddings + Laplace noise ((ε,0)-DP)." |
| 13 | §7.2 | Representative Schemes redundant | ✅ | Merged into a single "Approaches" subsection; the redundant §7.2 removed. |
| 14 | §7.3 | GELO 76%/20–30% misleading | ✅ | Reframed: synthetic **single-offload microbenchmark** (~80% IPC-dominated), **U-shaped** in batch (rises at large n, O(n³) matrix-gen), Q/K/V/O only — "best-case lower bound, not full-inference." tab:inf-perf cell annotated "microbench." (EdgeQuake-confirmed.) |
| 15 | §7.3 | TwinShield 87% — for what? | ✅ | Stated: "~87% of **total** transformer-inference computation offloaded." |
| 16 | §7.3 | 5.4× include U-Verify? | ✅ | **Fixed an error in the draft** (was written as a "slowdown"): 5.4× is a **speedup** vs TEE-only in the **verifiable** config (incl. U-Verify); ~6.7× privacy-only; 4.0–6.1× vs prior verifiable. |
| 17 | §7.3 | "practical frontier" — not really; move to Synthesis | ✅ | Removed the over-claim from §7; reframed advantage as *economic* (commodity GPUs), named the TEE↔GPU **serialization** as the binding constraint, and deferred the cross-family deployment ranking to §9 (`\cref{sec:synthesis}`). *Note: synthesis-side elaboration of the serialization point itself is a follow-up §9 pass (material in `docs/research/tee-gpu-serialization-bottleneck-2026-06-05.md`).* |
| 18 | §7.3 | GELO H200+L40S weird — explain | ✅ | Added: cluster-economics rationale — scarce confidential GPUs (H200) as trusted core, offload heavy GEMMs to cheaper untrusted commodity GPUs (L40S) to maximize throughput. |
| 19 | §7.4 | Limitations too thin; need attack survey + table | ✅ | Built **tab:hybrid-attacks** — **6 attacks** after a thorough double-check + Gram-row split: Game of Arrows (USENIX'25, ObfuscaTune weight recovery >98%), **Hidden No More** (ICML'25 2505.18332, prompt reconstruction from hidden states — *headline input-recon attack, initially missed*), static-basis break (2602.11088), GELO Gram-matrix leak, GELO ICA/BSS+anchor, KV-cache reconstruction (2508.09442); **LeftoverLocals** (CVE-2023-4969) in prose as the enabling VRAM-read threat. **Bib consolidation:** Game-of-Arrows→existing `arrowmatch` (deleted dup `game_of_arrows`); removed a duplicate `hnm`; +`luo_shadowcache`, `cmif`, `leftoverlocals`. **Correction of record:** "Shuffling-Defense break / arXiv 2605.04901" in `obfuscation-round2` is **unverified** (web doesn't confirm the id; matches HNM 2505.18332) — do NOT cite 2605.04901. |
| 20 | §7.4 | Gram mitigation — GELO or TwinShield? | ℹ️ | Answered: **GELO's own** (§3.2.1 — non-orthogonal A + shield vectors). Existing `\citep{belikov_gelo}` is correct; no change. |

## Cross-file / paper-wide changes
- `02_evaluation_framework.tex`: tab:inf-perf — dropped Portcullis row, added CMIF row, annotated GELO cell "microbench."; §2 TCB axis "Intel SGX/TDX"→"Intel TDX".
- §3/§4/§5/§6: "Approach"→"Approaches".
- `refs.bib`: +3 (`cmif`, `game_of_arrows`, `luo_shadowcache`); 5 CNN-root keys + `kvshield` + `portcullis` now uncited (orphaned — safe to prune in a later cleanup; do not appear in the rendered bibliography).

## Open follow-up (not a PDF comment)
- Weave the serialization-bottleneck discussion into §9 synthesis (#17 synthesis-side), from the round-4 serialization research doc.
