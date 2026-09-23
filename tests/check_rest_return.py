"""
Regression check: the resting vibe takes the room back slowly, the others fast.

    python tests/check_rest_return.py

Almost every uneventful stretch of talk votes for the resting vibe, so with
symmetric rules it won back confession and resonance within seconds. Replays
votes every 5 s through the tracker and the engine's rest hold, with the
engine's default settings, on a simulated clock.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aromagen.engine import EngineConfig          # noqa: E402
from aromagen.recipes import DEFAULT_RECIPE       # noqa: E402
from aromagen.vibe import VibeTracker             # noqa: E402

CFG = EngineConfig()
STEP = 5.0
CONFIDENCE = 0.9


def simulate(first: str, hold: float, then) -> float:
    """Room says `first` for `hold` s, then `then(i)` per vote; seconds until it changes."""
    v = VibeTracker(half_life=CFG.vibe_half_life, min_evidence=CFG.min_evidence,
                    switch_share=CFG.switch_share, switch_lead=CFG.switch_lead,
                    resting=DEFAULT_RECIPE, resting_weight=CFG.resting_vote_weight,
                    return_lead=CFG.return_lead, _last_decay=0.0)
    current, since = DEFAULT_RECIPE, 0.0
    t = 0.0

    def step(vibe):
        nonlocal current, since
        v.vote(vibe, CONFIDENCE, now=t)
        leader, _why = v.should_switch(current, now=t)
        if leader is None:
            return False
        if leader == DEFAULT_RECIPE and t - since < CFG.rest_hold:
            return False
        current, since = leader, t
        return True

    while t < hold:
        step(first)
        t += STEP
    start, i = t, 0
    while t - start < 900:
        before = current
        step(then(i))
        if current != before:
            return t - start
        i += 1
        t += STEP
    return float("inf")


def main() -> int:
    ok = True

    def check(label, got, cond, want):
        nonlocal ok
        passed = cond(got)
        ok &= passed
        shown = "never" if got == float("inf") else f"{got:.0f}s"
        print(f"  {'ok ' if passed else 'BAD'} {label}: {shown} ({want})")

    print("back to rest, after confession held for...")
    for hold, floor in ((5, 40), (60, 50), (300, 60)):
        got = simulate("confession", hold, lambda i: DEFAULT_RECIPE)
        check(f"{hold:3.0f}s", got, lambda g, f=floor: f <= g < 900, f"at least {floor}s")

    got = simulate("confession", 60, lambda i: "confession" if i % 3 == 0 else DEFAULT_RECIPE)
    check("confession resurfacing 1 vote in 3", got, lambda g: g == float("inf"), "holds")

    got = simulate("confession", 60, lambda i: "resonance")
    check("confession -> resonance", got, lambda g: g <= 30, "30s at most")

    got = simulate(DEFAULT_RECIPE, 60, lambda i: "confession")
    check("rest -> confession", got, lambda g: g <= 20, "20s at most")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
