#!/usr/bin/env python3
"""Score the GELO offload HumanEval completions (pass@1) — final accuracy gate.

Reads `subset.jsonl` (problems) + `completions.jsonl` (Rust-generated, from
`humaneval_gate_generate`), truncates each completion at the canonical stop
sequences (the reference llama.cpp stopped there), runs the canonical
`check(entry_point)` harness in a subprocess, and reports pass@1 X/N against
the **Plain Qwen3-4B reference = 6/20** (llama.cpp, docs/prototype/aloepri-llm.html).

Ablation reading (acceptance tier-5): run the Rust generator for
  B = offload + C_v (κ=6, secure)         → this is the gate
  A = in-TEE (no offload flags)           → baseline (or reuse the 6/20 ref)
  C = offload − cover (κ=1)               → ONLY if B regresses, to attribute
and compare the three X/N. Heavy gate — run sparingly; debug regressions with
microbenches + theory, not by re-running this.

Run (no GPU; just executes the candidate code):
  python3 evals/humaneval-gate/score.py [completions.jsonl]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n\n\n", "\nif __name__"]
EXEC_TIMEOUT_S = 10
REFERENCE = "Plain Qwen3-4B (llama.cpp) = 6/20"


def truncate(completion: str) -> str:
    """Cut at the earliest stop sequence — equivalent to the reference's
    server-side `stop`, applied post-hoc to a fixed-length generation."""
    cut = len(completion)
    for s in STOP_SEQUENCES:
        j = completion.find(s)
        if j != -1:
            cut = min(cut, j)
    return completion[:cut]


def run_check(prompt: str, completion: str, test: str, entry_point: str) -> tuple[bool, str]:
    program = prompt + completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = Path(f.name)
    try:
        r = subprocess.run([sys.executable, str(path)], capture_output=True,
                           timeout=EXEC_TIMEOUT_S, text=True)
        return (r.returncode == 0), ("" if r.returncode == 0 else (r.stderr or r.stdout or "")[:160])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"exec error: {e}"
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    comp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "completions.jsonl"
    problems = {json.loads(l)["idx"]: json.loads(l)
                for l in (HERE / "subset.jsonl").read_text().splitlines() if l.strip()}
    comps = {json.loads(l)["idx"]: json.loads(l)
             for l in comp_path.read_text().splitlines() if l.strip()}

    passed = 0
    print(f"# HumanEval pass@1 — {comp_path.name}  (reference: {REFERENCE})\n")
    print(f"{'idx':>3} {'task_id':>14} {'pass':>5}  note")
    for idx in sorted(problems):
        p = problems[idx]
        c = comps.get(idx, {}).get("completion", "")
        ok, err = run_check(p["prompt"], truncate(c), p["test"], p["entry_point"])
        passed += ok
        print(f"{idx:>3} {p['task_id']:>14} {'OK' if ok else 'x':>5}  {'' if ok else err}")

    n = len(problems)
    print(f"\n# pass@1 = {passed}/{n} ({100*passed/n:.0f}%)   reference {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
