"""
Regression check: the engine must keep classifying while a burst is diffusing.

    python tests/check_engine_bursts.py

Found when the local brain landed. Bursts were awaited inside the decision loop,
so the room went deaf for the whole 15 s of every burst. With a 15 s
classification window that is worse than a delay: speech heard during a burst
could fall out of the window before it was ever judged. A replayed argument
passed without a single negative vote.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aromagen.classify import Decision          # noqa: E402
from aromagen.device import MockAromaDevice     # noqa: E402
from aromagen.engine import AromaEngine, EngineConfig  # noqa: E402
from aromagen.recipes import AVAILABLE_RECIPES, DEFAULT_RECIPE  # noqa: E402

# Any vibe other than the resting one, so the engine has a reason to switch.
TARGET = next(r.key for r in AVAILABLE_RECIPES if r.key != DEFAULT_RECIPE)


class CountingBrain:
    """Always votes for TARGET, and records when it was asked."""

    name = "counting"

    def __init__(self):
        self.calls = []

    async def classify(self, window):
        self.calls.append(time.monotonic())
        return Decision(TARGET, 0.9, 0.9, "test", self.name).clamp()


class Line:
    def __init__(self, text):
        self.text, self.started_at = text, time.time()

    def __str__(self):
        return self.text


async def main() -> int:
    brain = CountingBrain()
    cfg = EngineConfig(burst_seconds=4.0, min_classify_interval=0.5, min_new_words=1,
                       tick=0.1, cooldown=0.1, min_dwell=0.0, refresh_interval=0.5,
                       min_evidence=0.1, silence_timeout=60.0)
    engine = AromaEngine(MockAromaDevice(verbose=False), brain, cfg)
    task = asyncio.create_task(engine.run())

    for i in range(50):
        engine.feed(Line(f"line {i}, this is getting ridiculous"))
        await asyncio.sleep(0.2)
        if engine.diffusing:
            break
    if not engine.diffusing:
        print("FAIL: no burst ever started")
        return 1

    burst_started = time.monotonic()
    for i in range(12):                       # keep talking through the burst
        engine.feed(Line(f"still talking {i}, stop it now"))
        await asyncio.sleep(0.2)
    still_diffusing = engine.diffusing
    during = [t for t in brain.calls if t >= burst_started]

    engine.stop()
    await asyncio.wait_for(task, 15)

    print(f"burst still diffusing after 2.4 s: {still_diffusing}")
    print(f"classifications while the burst was diffusing: {len(during)}")
    if len(during) < 2:
        print("FAIL: the engine went deaf during a burst")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
