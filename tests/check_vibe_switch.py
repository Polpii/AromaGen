"""
Regression check: a vibe change must reach the device at once.

    python tests/check_vibe_switch.py

The piece is meant to follow the room without hesitating: as soon as the votes
favour another vibe it switches, and sprays that pump immediately -- even when a
burst of the previous vibe is still going out, which is cut short.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aromagen.classify import Decision                  # noqa: E402
from aromagen.device import MockAromaDevice             # noqa: E402
from aromagen.engine import AromaEngine, EngineConfig   # noqa: E402
from aromagen.recipes import AVAILABLE_RECIPES, BY_KEY, DEFAULT_RECIPE  # noqa: E402

OTHERS = [r.key for r in AVAILABLE_RECIPES if r.key != DEFAULT_RECIPE]
FIRST, SECOND = OTHERS[0], OTHERS[1]
BURST = 8.0


class ScriptedBrain:
    """Votes for whatever `answer` currently says, with high confidence."""

    name = "scripted"

    def __init__(self):
        self.answer = FIRST

    async def classify(self, window):
        return Decision(self.answer, 0.95, 0.95, "test", self.name).clamp()


class Line:
    def __init__(self, text):
        self.text, self.started_at = text, time.time()

    def __str__(self):
        return self.text


def valve_bit(recipe_key: str) -> int:
    (valve,) = BY_KEY[recipe_key].valves()      # one pump per vibe
    return 1 << valve


async def main() -> int:
    brain = ScriptedBrain()
    device = MockAromaDevice(verbose=False)
    bursts = []                                 # (time, recipe key)

    def on_event(kind, payload):
        if kind == "burst":
            bursts.append((time.monotonic(), payload[0].key))

    cfg = EngineConfig(burst_seconds=BURST, min_classify_interval=0.3,
                       min_new_words=1, tick=0.1, silence_timeout=60.0)
    engine = AromaEngine(device, brain, cfg, on_event=on_event)
    task = asyncio.create_task(engine.run())

    for i in range(60):
        engine.feed(Line(f"line {i} with a few words in it"))
        await asyncio.sleep(0.2)
        if engine.state.current == FIRST and engine.diffusing:
            break
    if not (engine.state.current == FIRST and engine.diffusing):
        print(f"FAIL: '{FIRST}' never started diffusing")
        return 1
    first_started = max(t for t, k in bursts if k == FIRST)

    brain.answer = SECOND                       # the room changes its mind
    for i in range(60):
        engine.feed(Line(f"another line {i} with a few words"))
        await asyncio.sleep(0.1)
        if engine.state.current == SECOND and engine.diffusing:
            break

    engine.stop()
    await asyncio.wait_for(task, 15)

    second_opened = [t for t, byte in device.history
                     if byte & valve_bit(SECOND) and t > first_started]
    if not second_opened:
        print(f"FAIL: the '{SECOND}' pump never opened")
        return 1
    gap = second_opened[0] - first_started
    print(f"'{FIRST}' burst started, then the room turned to '{SECOND}'")
    print(f"'{SECOND}' pump opened {gap:.1f}s after the '{FIRST}' burst began "
          f"(that burst was due to run {BURST:.0f}s)")
    if gap >= BURST:
        print("FAIL: the change waited for the previous burst to finish")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
