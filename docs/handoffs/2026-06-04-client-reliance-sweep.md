---
type: handoff
status: current
created: 2026-06-04
updated: 2026-06-04
tags: [sok, deployment-readiness, client-reliance, section-8, threat-model]
companion: [scheme-categorization, survey-corpus]
---

# Handoff — cross-scheme client-reliance sweep (§8)

## Why this exists

The SoK's trust-anchor axis (§2.1) collapses to a 3-tier **server-side** hardware-trust
ladder (Tier 1 = "no trusted hardware"). That ladder is silent on **client-side** work, so
"trust anchor: none" reads as zero cost when it is not: client-side-PIR/ORAM retrieval
(Compass, Pacmann) and 2PC inference where the user is a compute party (PermLLM) make the
**client load-bearing** — it stores access state and runs a multi-round retrieval/compute
loop. This client burden is a first-order deployment / adoption / UX factor (a phone or
browser tab is a poor ORAM controller). Grilled + scoped 2026-06-04
(`quality_reports/plans/2026-06-04_client-reliance.md`).

## Decisions of record (do not relitigate)

1. **Framing = rigorous "split".** Keep trust anchor = *none* (the confidentiality claim is
   real — no trusted third party/hardware), but mark client-heavy Tier-1 schemes
   **"none; client-relied"** and separate *trusted-not-to-leak* from *relied-on-to-function*.
   Already applied to Pacmann/Compass in `tab:threat-taxonomy` + §2.1 Tier-1 prose.
2. **Rubric = 3-level ordinal** (judge on client state + compute + online-rounds):
   - **thin** — client only encrypts/decrypts a query (non-interactive FHE: Euston; send an
     obfuscated embedding / noisy vector: DP-Forward).
   - **stateful** — client holds keys + does light per-query work (DCPE/CAPRISE; SSE/EMM
     search-key holders).
   - **thick** — client stores access state AND runs a multi-round retrieval/compute loop
     (Compass ORAM controller, Pacmann client-side traversal; 2PC inference as a party: PermLLM).
3. **§8 home = `tab:deployment` Client column + the "client-reliance asymmetry" finding**
   (already written in §8.3). Crypto family filled (`thin--thick`); other families flagged `\tq`.
4. **Blast radius of this sweep = annotate/fill, not restructure.** Fill the `\tq` cells;
   add a per-scheme appendix table only if the family-level column proves too lossy.

## Done already (this pass)

- §2.1: `tab:threat-taxonomy` Pacmann/Compass trust anchor → "none; client-relied"; Tier-1
  prose now states "none" is about trust, not where the work runs, and names the load-bearing
  Tier-1 schemes.
- §3.3: access-hiding paragraph gained a client-load-bearing sentence (Compass ORAM +
  position map; Pacmann client-side traversal; server holds bulk) → cross-ref §8.
- §8.3: `tab:deployment` gained a **Client** column (crypto = `thin--thick`; TEE / static
  obfuscation / DP / hybrid split = `\tq`) + the **"Finding: the client-reliance asymmetry"**
  paragraph. Build clean (27pp, 0 undefined).

## The task

Classify **every in-comparison scheme** (the ~50 in `survey-corpus.md` §1–§7 / the rows of
the §3–§7 family tables) as **thin / stateful / thick**, with one line of evidence each, then:
1. Fill the four `\tq` cells in `tab:deployment` with the family-level value or range.
2. Decide whether a **per-scheme appendix table** (Scheme | Client | what the client stores |
   what it computes | rounds) is warranted — likely yes, since reliance varies within families.
3. Record per-scheme results in `docs/research/scheme-categorization.md` (new column in the
   Part A tables: "Client role"), and reconcile any that contradict the family cell.

## How to do it (method)

- **Derive first from existing data** — `scheme-categorization.md` Part A already has
  *Behavior / Trust anchor / Knows / Protects / Residual leak / Basis* per scheme; the client
  role is largely inferable from the primitive + who does what (e.g. "client-side ORAM",
  "2-server SS" → client thin; "user-side ranking" → stateful).
- **EdgeQuake `document_get_md`** (NOT `query` — it doesn't index the setup/protocol sections
  well; see Part E #18/#20) for any scheme whose client/server split is unclear. Doc IDs are in
  the `document_list` output (e.g. Compass `7b372edb`, Pacmann `f358376b`, PermLLM `0d0659e4`).
- **Do NOT ingest new docs** (user directive 2026-06-04). Web-fetch only if EdgeQuake lacks it.

## First-pass hypotheses to verify (seed, not truth)

- **Crypto inference:** PermLLM **thick** (client is a 2PC party, runs nonlinear eval on
  permuted data); Fission **thin** (client secret-shares input; MPC+evaluators do the work);
  SHAFT/Euston **thin** (send shares/ciphertext, receive result). CryptoMoE **thin** (client a
  2PC party but server-driven — verify).
- **Crypto retrieval:** Compass/Pacmann **thick**; TRSE **stateful** (paper: user-side
  ranking); CAPRISE/SAP **stateful** (client holds DCPE key, encrypts vectors); p²RAG **thin**
  (2-server, client sends shares); Panther **thin** (single-server, client sends query);
  PIR-RAG **stateful** (client cluster map?); RAGtime-PIANO **stateful** (client PIR state);
  XorMM/FLASH/PeGraph **stateful** (client holds SSE search keys); GORAM **thin–stateful** (3PC).
- **TEE (Tier 3):** **thin** (client ships data to the enclave; the TEE does everything).
- **Static obfuscation:** **stateful** — client runs the obfuscation transform locally and
  holds the (often trained) obfuscator: SGT/OSNIP/Eguard need a client-side projector/model;
  AloePri a client-side covariant transform. This is a real, under-reported client cost.
- **DP:** **thin–stateful** — DP-Forward runs the perturbed forward pass client-side
  (stateful); query-only noise (RemoteRAG) is lighter.
- **Hybrid split (TEE + obfusc):** **thin–stateful** — the TEE+GPU do the masking server-side;
  client typically ships data. Verify GELO/TwinShield/ObfuscaTune.

Net hypothesis (worth stating as a finding if it holds): **obfuscation and crypto-retrieval —
the two families the paper currently scores best on the deployment frontier — carry the
heaviest hidden client burden.** That would sharpen, or complicate, the §8 headline.

## Files
- `manuscript/sections/02_evaluation_framework.tex` (trust anchor, `tab:threat-taxonomy`)
- `manuscript/sections/08_synthesis.tex` (`tab:deployment`, §8.3 finding)
- `docs/research/scheme-categorization.md` (Part A — add "Client role" column)
- `manuscript/survey-corpus.md` (scheme inventory)
- Plan: `quality_reports/plans/2026-06-04_client-reliance.md`

## Verify when done
`cd manuscript && make` → 0 undefined; no new overfull >20pt (watch `tab:deployment` width if
cells lengthen — currently held with `\scriptsize\setlength{\tabcolsep}{4pt}`).
