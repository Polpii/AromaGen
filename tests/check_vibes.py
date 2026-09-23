"""
Check that the local brain can tell this installation's vibes apart.

    python tests/check_vibes.py
    python tests/check_vibes.py --file samples/vibe_eval_vibes.jsonl --show

The vibes are named by the piece, not by sentiment, so the accuracy measured for
the earlier positive/neutral/negative palette says nothing about them. Run this
after renaming a vibe, rewriting a `theme` in recipes.py, or touching the prompt
in local_brain.py: a 1.5B model is sensitive to phrasing in a way Claude is not.

It prints accuracy, where the confusions go, and every mistake with the model's
own confidence -- a confident mistake is what moves the room wrongly; a hesitant
one is mostly absorbed by the vote tracker.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Behave like the venue: cache only, no network.
import os  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from aromagen.local_brain import LocalClassifier  # noqa: E402
from aromagen.recipes import AVAILABLE_RECIPES  # noqa: E402


class Line:
    def __init__(self, text):
        self.text, self.started_at = text, time.time()

    def __str__(self):
        return self.text


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path,
                        default=ROOT / "samples" / "vibe_eval_vibes.jsonl")
    parser.add_argument("--show", action="store_true", help="print every answer")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    items = [json.loads(line) for line in
             args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {r.key for r in AVAILABLE_RECIPES}
    unknown = {item["label"] for item in items} - known
    if unknown:
        print(f"the file expects vibes that are not mounted: {sorted(unknown)}")
        print(f"available: {sorted(known)}")
        return 1

    brain = LocalClassifier()
    print(f"{len(items)} snippets, {brain.description}\n")
    brain.classify_sync([Line("warm up")])

    correct, latencies, confusion = 0, [], Counter()
    errors = []
    for item in items:
        t0 = time.perf_counter()
        decision = brain.classify_sync([Line(t) for t in item["lines"]])
        latencies.append(time.perf_counter() - t0)
        got, want = decision.recipe, item["label"]
        confusion[(want, got)] += 1
        if got == want:
            correct += 1
        else:
            errors.append((item, decision))
        if args.show:
            mark = "ok " if got == want else "BAD"
            print(f"  {mark} {item['id']} [{item['lang']}] want {want:11} "
                  f"got {got:11} {decision.confidence:.2f}  {decision.reason}")

    print(f"\naccuracy {correct}/{len(items)} = {correct / len(items):.0%}"
          f"   p50 {statistics.median(latencies) * 1000:.0f} ms")

    for vibe in sorted(known):
        rows = [i for i in items if i["label"] == vibe]
        if rows:
            print(f"  {vibe:11} {confusion[(vibe, vibe)]}/{len(rows)}")

    hard = [i for i in items if i.get("hard")]
    if hard:
        hard_ok = sum(1 for i in hard if not any(e[0]["id"] == i["id"] for e in errors))
        print(f"  {'traps':11} {hard_ok}/{len(hard)}")

    if errors:
        print("\nmistakes:")
        for item, decision in errors:
            tag = f" [{item['hard']}]" if item.get("hard") else ""
            print(f"  {item['id']} want {item['label']:11} got {decision.recipe:11} "
                  f"confidence {decision.confidence:.2f}{tag}")
            for line in item["lines"]:
                print(f"        {line}")
    return 0 if correct == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
