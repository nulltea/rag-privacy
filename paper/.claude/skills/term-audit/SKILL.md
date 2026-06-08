---
name: term-audit
description: Audit and (after approval) fix prose word-choice and register in manuscript sections — arbitrary/invented synonyms where a canonical term exists, vague terms, non-standard terminology, and colloquial or metaphor phrasings that break academic register. Use when the user asks to audit/fix "arbitrary, vague, or non-standard terms," "colloquial/metaphor phrasing," "word choice," or "terminology" in a `.tex`/`.qmd`/`.md` section, or says `/term-audit` or `/dejargon`. Distinct from /humanize (AI-voice tells, read-only) and /proofread (grammar/typos/overflow).
---

# /term-audit — Word-Choice & Register Audit (propose → grill → fix)

The *diction* lens: is each word the standard, precise, register-appropriate choice? Detects and, **only after the user approves via a grill**, fixes. It proposes; the user decides; then it applies. Never auto-fixes.

## Not this skill

- **/humanize** — AI-voice tells (boilerplate transitions, em-dash overuse, tricolons, symmetric paragraph shapes). Read-only. Overlaps only on metaphor lexicon.
- **/proofread** — grammar, typos, overflow, citation format.
- **/verify-claims** — factual/citation correctness.

Run alongside; none substitutes.

## Detection categories

1. **Invented synonym for a canonical term.** A coined or rephrased word used where the field (or the paper itself) has an established term. *e.g.* "crossings" → "round-trips"; "two knobs" → "two choices".
2. **Vague term.** Imprecise filler. *e.g.* "buys two things" → "offers two advantages"; "two of them need care" → "warrant caution".
3. **Non-standard terminology.** A coined compound or label not used in the literature. *e.g.* "cluster-economics choice" → "deployment choice".
4. **Colloquial / metaphor register.** Informal or figurative phrasing that breaks academic register. *e.g.* "reason to exist" → "rationale"; "trace a single curve" → "follow one trend"; "sweet-spot batch" → "optimal batch"; "the lever is" / "fill the stalls" (AI-voice metaphors); sentence-initial "Worse,".

## Calibration (do NOT flag)

Standard domain jargon is correct and stays: e.g. GEMM, round-trip, KV-cache, binding constraint, U-shaped, GELU, enclave, FLOP. Established terms in *this* paper stay (check `survey-corpus.md` / the section's own usage). Flag **patterns and clear cases**, not every word. When unsure whether a term is canonical, verify against a primary source rather than guessing.

## Workflow

1. **Read** the target file fully (the section, not a snippet — register drifts mid-section).
2. **Flag** candidates grouped by the 4 categories, each with a concrete **before → after** and the line number. Skip standard jargon (calibration above).
3. **Grill** with `AskUserQuestion` (multiSelect), one question per category group (≤4 options each), each option = one fix with its before→after in the description. Note borderline items in the preamble, not as options.
4. **Apply** only approved fixes (Edit). If two fixes share a sentence, merge into one Edit to avoid conflicts. The user may be editing concurrently — re-Read before each edit if a write fails.
5. **Rebuild & verify** — `cd manuscript && make`; confirm `0 undefined` refs/cites and **0 prose em-dashes** (`grep -n -- '---' <file> | grep -v '%'`). No new overfull boxes.
6. **Report** — list every applied fix as a before→after table; name what was left as borderline and why.

## Hard rules (non-negotiable)

- **No correction-commentary in the artifact.** The fix reads as if always correct. Never write "not X", "formerly Y", "instead of", or otherwise narrate the change in prose, captions, or notes. Rationale stays in chat. (memory: `feedback-no-correction-commentary`)
- **Canonical / primary-source terms only.** Replace with the paper's own term or the established field term, verified — never an invented synonym (that is the very thing this skill removes). (memory: `feedback-standard-terminology`)
- **No em dashes in prose.** Replacements must not introduce `---`; split sentences or use commas/parentheses/colons. (memory: `feedback-no-em-dashes`)
- **Propose, then apply.** No edits before the grill is answered.
- **Avoid tautology when substituting.** If `canonical-term` already appears adjacent, reword (drop the noun, use a pronoun) rather than repeat it.

## Output

- Edits applied to the file (after approval); clean build verified.
- A before→after summary in the conversation. No standalone report file unless asked.
