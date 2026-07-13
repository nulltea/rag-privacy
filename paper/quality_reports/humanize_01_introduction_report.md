# §1 Introduction — Three-Lens Audit (humanize + term-audit + proofread), 2026-06-11

File: `manuscript/sections/01_introduction.tex` (~830 prose words). Focus: the just-rewritten §1.1 Motivation.

## Headline

**§1.1 (the rewrite) is clean on AI voice** — 0 HIGH tells across all 10 categories (no boilerplate
transitions, no cliché lexicon, no hedge stacking, no sycophancy, em-dash-free). The work is:
(1) a pre-existing em-dash sweep in §1.2–§1.4 (hard-rule, NOT §1.1), (2) one citation-command
bug, (3) a few small §1.1 wording/precision fixes.

## A. Em-dash rule violations (HARD RULE) — 13 prose `---`, all in §1.2–§1.4, NONE in §1.1

| Section | Lines | Note |
|---|---|---|
| §1.2 Scope | 35, 95, 102, 108 | criteria---perf; attestation---a; graph-RAG---a; protection---schemes |
| §1.3 Contributions | 120, 125, 126, 128, 129, 133 | every itemize bullet uses an em-dash appositive (densest cluster) |
| §1.4 Organization | 141, 142, 144 | layer---dense; mechanisms---is; together---the |

Table band `---` (L65, 75–82) and `\cref{}--\cref{}` en-dashes (L124, 141) excluded. These predate
the no-em-dash rule (§1 was missed, like §2 before it). Mechanical sweep: → colon / comma / split.

## B. Proofread

| Line | Sev | Issue | Fix |
|---|---|---|---|
| 129 | HIGH | lone bare `\cite{bodea_sok}` breaks file-wide `\citep`/`\citet` convention | → `\citep{bodea_sok}` (fix outright) |
| 48–52 | MED | inline `Li et al.~\citep{}` / `Chen et al.~\citep{}` / `Andreoletti et al.~\citep{}` (author is subject) | → `\citet{}` (renders identically; semantic cleanliness) — judgment call |
| 11 | MED | "In a 2024 industry survey, half of enterprises already **adopt**" — past-dated survey, present verb | → "already **adopted**" |
| 12–13 | LOW | "the strongest are reachable" leans on elided noun | → "the strongest **models** are reachable" |
| 9–10 | LOW | "the EU AI Act" can break across line | → "the EU~AI Act" (non-breaking tie) |
| 53 | LOW | "trust-minimising" (UK) vs US spelling elsewhere | → "trust-minimizing" |

## C. term-audit (§1.1, mine)

| Line | Sev | Issue | Suggested |
|---|---|---|---|
| 10 | MED | "make this incompatible with personal, health, or privileged **data**" — regulations restrict *processing/transfer*, not the data itself; my trim dropped "processing" | → "incompatible with **processing** personal, health, or privileged data" |
| 20 | LOW | "The hard part is" — plain but slightly informal register | → "The difficulty is" (original), or keep if straightforwardness preferred |

## D. Cosmetic (optional, not acted unless asked)

- §1.1 L29 "This systematization supplies that comparison." + §1.3 L114 "This paper is a SoK." — canonical
  "This [artifact]…" cadence. Both read as deliberate punch; humanize rates acceptable. Leave.
- §1.1 L25–27: two stacked lists in one sentence (families + stages). Mild machine cadence; could drop the
  "(embedding, reranking, generation)" parenthetical since §1.4 repeats it. Optional.

## Recommendation

Genuine AI-voice: **cosmetic only** (~0 HIGH/1000 words). The substantive task is the §1.2–§1.4 em-dash
sweep (hard-rule conformance, mirrors the §2 sweep) + the L129 `\cite` fix + a handful of §1.1 polish edits.
