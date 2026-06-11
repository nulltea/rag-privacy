# CLAUDE.MD -- Academic Project Development with Claude Code

<!-- HOW TO USE: Replace [BRACKETED PLACEHOLDERS] with your project info.
     Keep this file under ~150 lines — Claude loads it every session.
     See the guide at docs/workflow-guide.html for full documentation. -->

**Project:** [YOUR PROJECT NAME]
**Institution:** [YOUR INSTITUTION]
**Branch:** main

---

## Core Principles

- **Plan first** -- enter plan mode before non-trivial tasks; save plans to `quality_reports/plans/`
- **Verify after** -- compile and confirm output at the end of every task
- **Quality gates** -- nothing ships below 80/100
- **[LEARN] tags** -- when corrected, save `[LEARN:category] wrong → right` to [MEMORY.md](MEMORY.md)

Cross-session context lives in [MEMORY.md](MEMORY.md); past plans, specs, and session logs are in [quality_reports/](quality_reports/).

---

## Writing Style (HARD RULES for `manuscript/**/*.tex` prose)

**Standard register only.** Use plain, standard, precise academic vocabulary. Do NOT use colloquial, conversational, or figurative words/phrases where a literal standard term exists. This has been a recurring failure; treat it as non-negotiable, on par with the no-em-dash rule.

- **Avoid** conversational verbs and idioms: e.g. "answer it", "boils/comes down to", "at once", "at hand", "the whole point", "story", "spare", "sore point", "buys" (meaning gains), "a lot", "kind/sort of", "sweet spot", "knob" (in prose), "reads differently". Avoid figurative framing when a literal term exists (e.g. "geometry" for a trade-off; "concede/concession" for giving something up).
- **Prefer the plainest standard term.** "answer it" → "are the appropriate choice"; "the whole point" → "is essential"; "at once" → "simultaneously / concurrently"; "reads differently" → "applies differently"; "concede a constraint" → "relax a constraint"; a scheme "gives up a criterion" (not "concedes"); name what is tunable (the hiding) rather than calling fidelity "tunable".
- **Sweep, don't patch.** When a term is flagged once, `grep` the WHOLE file and fix every instance; do not reintroduce it in rewrites. Fixing only the cited line is a failure — this is the exact mistake that produced this rule.
- **When unsure** whether a word is standard register, choose the plainer, more literal alternative.
- **No em-dashes** in prose (`---`); use canonical field terms or the paper's own defined terms, not invented synonyms; never narrate a correction inside the artifact.

---

## Folder Structure

```
[YOUR-PROJECT]/
├── CLAUDE.MD                    # This file
├── .claude/                     # Rules, skills, agents, hooks
├── Bibliography_base.bib        # Centralized bibliography
├── Figures/                     # Figures and images
├── manuscript/                  # arXiv LaTeX paper sources + figures
├── scripts/                     # Utility scripts + R code
├── quality_reports/             # Plans, session logs, merge reports, decision records
├── explorations/                # Research sandbox (see rules)
├── templates/                   # Session log, quality report templates
└── master_supporting_docs/      # Papers and supporting material
```

---

## Commands

```bash
# LaTeX (3-pass, XeLaTeX) — run from the manuscript directory
xelatex -interaction=nonstopmode paper.tex
bibtex paper
xelatex -interaction=nonstopmode paper.tex
xelatex -interaction=nonstopmode paper.tex

# Quality score
python scripts/quality_score.py manuscript/paper.tex

# Surface-count sync (README ↔ CLAUDE.md ↔ guide ↔ landing page)
./scripts/check-surface-sync.sh
```

---

## Quality Thresholds (advisory)

| Score | Checkpoint | Meaning |
|-------|------|---------|
| 80 | Commit | Good enough to save |
| 90 | PR | Ready for deployment |
| 95 | Excellence | Aspirational |

Enforced by `/commit` (halts + asks for override); not enforced by a git pre-commit hook.

---

## Skills Quick Reference

| Command | What It Does |
|---------|-------------|
| `/compile-latex [file]` | 3-pass XeLaTeX + bibtex |
| `/new-diagram [snippet] [output.tex]` | Scaffold a TikZ diagram from the gallery with prevention + review |
| `/proofread [file]` | Grammar/typo/overflow review |
| `/validate-bib` | Cross-reference citations |
| `/commit [msg]` | Stage, commit, PR, merge |
| `/lit-review [topic]` | Literature search + synthesis |
| `/research-ideation [topic]` | Research questions + strategies |
| `/interview-me [topic]` | Interactive research interview |
| `/review-paper [file]` | Manuscript review (single-pass / `--adversarial` / `--peer <journal>` simulated pipeline) |
| `/respond-to-referees [report] [manuscript]` | R&R cross-reference + response draft |
| `/data-analysis [dataset]` | End-to-end R analysis |
| `/audit-reproducibility [paper]` | Enforce replication tolerance thresholds on paper ↔ code |
| `/learn [skill-name]` | Extract discovery into persistent skill |
| `/context-status` | Show session health + context usage |
| `/deep-audit` | Repository-wide consistency audit |
| `/permission-check` | Diagnose permission layers when prompts fire unexpectedly |
| `/seven-pass-review` | Seven-pass adversarial manuscript review (parallel forked subagents) |
| `/verify-claims [file]` | Chain-of-Verification fact-check (forked verifier, fresh context) |
| `/checkpoint [topic]` | Save a structured state snapshot (active plan, decisions, file pointers, next actions) before stopping or handing off |
| `/preregister [--style osf|aspredicted|aea-rct]` | Draft a preregistration document (OSF / AsPredicted / AEA RCT Registry) from a research spec |
| `/humanize [file]` | Detect AI-voice tells in academic prose (read-only audit; no rewrite) |
| `/prompt [text] [depth:light|standard|deep]` | Reformat informal input into a structured six-section prompt, then execute |
| `/prompt-only [text] [depth] [--save path]` | Same formatting as `/prompt`, but emits the prompt as a reusable artifact (no execution) |
| `/compress-session [slug]` | Distil current session into structured notes before auto-compaction (vs `/checkpoint` for natural stops) |
| `/promote-memory [filter]` | Five-critic council that votes on which `[LEARN]` entries graduate from personal-memory.md to MEMORY.md |
| `/stata-replication [paper-or-data]` | End-to-end Stata pipeline scaffold + execution via `stata-mcp` (mirrors `/data-analysis` for R) |

---

<!-- CUSTOMIZE: Replace placeholder rows ([your-env]) with your own.
     Delete the rows marked "(example — delete)" once you've added yours. -->

## LaTeX Custom Environments

| Environment | Effect | Use Case |
| --- | --- | --- |
| `[your-env]` | [Description] | [When to use] |
| `definitionbox[Title]` | Blue-bordered titled box | Formal definitions *(example — delete)* |

---

## Current Project State

| Section | Source | Key Content |
| --- | --- | --- |
| 1: [Topic] | `manuscript/paper.tex` | [Brief description] |
