# Abstract — three-lens audit (humanize + term-audit + proofread), 2026-06-11

File: `manuscript/sections/00_abstract.tex` (~190 words, one paragraph).

## Headline
Voice is clean: **0 HIGH AI-tells**, no clichés, no boilerplate, em-dash-free, no sycophancy.
The real issues are a **consistency/logic bug** (the third criterion is named two different things)
and the **utility/fidelity** abstract↔body mismatch. A few minor clarity nits.

## A. The colliding-triples bug (HIGH — flagged by both proofread F2/F5 and humanize)
The abstract introduces the lens and the finding with two near-identical triples whose **third item
silently differs**:
- L10 (lens): "scored on performance, utility, and **threat-model fit**"
- L11–12 (finding): "each attains at most two of performance, utility, and **a strong security basis**"

Both are individually correct against the body (the deployment-readiness lens in §1/§2 is
{performance, fidelity, threat-model fit}; the §9 design-space geometry is {performance, fidelity,
**security-basis strength**}). But juxtaposed in the abstract, with 2 of 3 items identical, the
swapped third item reads as an error and undercuts the "two of three" claim. The lens criterion
(threat-model fit) and the geometry axis (security basis) are genuinely different things;
the abstract conflates them. **Needs a framing fix** (see grill).

## B. utility vs fidelity (HIGH — abstract↔body mismatch)
Abstract says "utility" (×2); body + both table headers say "fidelity" (~40 uses/11 files). A reader
cannot map "utility" to "fidelity." Resolve: propagate `utility` globally, or revert the abstract to
`fidelity`. (This is the long-deferred axis-term decision.)

## C. Proofread — minor
| Line | Sev | Issue | Fix |
|---|---|---|---|
| 13–14 | LOW | "excludes the side channels" could misread as "blocks" (opposite of intended) | "leaves out of its threat model the side channels…" / "places out of scope" |
| 9 | MED-LOW | "We organize… readiness, the likelihood…, scored on…, across…, both dense and graph" stacks 4 appositives | split with a colon + period |
| 5 | MED | RAG is expanded but MPC/FHE/TEE are not (self-containedness inconsistency) | expand on first use, or accept as well-known field acronyms |
| 3 | LOW | comma before "or" in a 2-arm either/or | optional removal |

## D. term-audit (mine)
| Line | Issue | Suggested |
|---|---|---|
| 15 | "custom software" is vague | the paper's term is a "custom serving path" (§9); use that or "non-stock serving software" |
| 15 | "client reliance" | KEEP — matches the paper's own term (§9 "client-reliance") |

## Recommendation
Voice needs nothing. Fix A (the triple collision) and B (utility/fidelity) before submission;
C/D are quick polish. No rewrite — targeted edits.
