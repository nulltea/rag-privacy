#!/usr/bin/env python3
"""Vendor the HumanEval n=20 subset for the GELO offload accuracy gate.

Writes `subset.jsonl` (one problem per line: idx, task_id, prompt, test,
entry_point) — consumed by the Rust generator (`humaneval_gate_generate`) and
the scorer (`score.py`), so all three share the exact same problems.

Problem selection, in priority order:
  1. `task_ids.json` (pinned) — the **exact 20 problems** the AloePri Qwen3-4B
     runs used, so our pass@1 is directly comparable to the Plain = 6/20
     reference. This is the default (the file is committed).
  2. else a seed-42 sample of `GELO_HE_N` problems.

Dataset source (auto): uses `datasets` if installed; otherwise falls back to
fetching `HumanEval.jsonl.gz` via stdlib `urllib`+`gzip` (no `datasets` needed —
works on the host). Override the URL with `GELO_HE_URL`.
  python3 evals/humaneval-gate/prepare_subset.py
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

N = int(os.environ.get("GELO_HE_N", "20"))
SEED = int(os.environ.get("GELO_HE_SEED", "42"))
HERE = Path(__file__).resolve().parent
HE_URL = os.environ.get(
    "GELO_HE_URL",
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
)


def load_problems() -> list[dict]:
    """All 164 HumanEval problems as dicts (task_id/prompt/test/entry_point)."""
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [dict(ds[i]) for i in range(len(ds))]
    except Exception:  # noqa: BLE001 — fall back to the canonical jsonl.gz
        import gzip
        import urllib.request
        raw = urllib.request.urlopen(HE_URL, timeout=60).read()
        text = gzip.decompress(raw).decode()
        return [json.loads(l) for l in text.splitlines() if l.strip()]


def main() -> int:
    problems = load_problems()
    by_id = {p["task_id"]: p for p in problems}

    pin = HERE / "task_ids.json"
    if pin.exists():
        task_ids = json.loads(pin.read_text())
        src = f"pinned task_ids.json ({len(task_ids)})"
    else:
        task_ids = [problems[i]["task_id"] for i in random.Random(SEED).sample(range(len(problems)), N)]
        src = f"seed-{SEED} sample (n={N})"

    out = HERE / "subset.jsonl"
    with out.open("w") as f:
        for i, t in enumerate(task_ids):
            p = by_id[t]
            f.write(json.dumps({
                "idx": i, "task_id": t, "prompt": p["prompt"],
                "test": p["test"], "entry_point": p["entry_point"],
            }) + "\n")
    print(f"wrote {out} — {len(task_ids)} problems ({src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
