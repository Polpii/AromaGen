"""
The room's overall mood, estimated from decaying votes.

The difference between this and simply acting on the latest classification is
the whole point of the installation. A single classification says what the last
forty seconds sounded like; visitors experience the room over minutes. One loud
laugh in an otherwise tense room should not flip the scent, and neither should
one sharp remark in a cheerful one.

So every classification is a VOTE, not an order. Votes decay exponentially, so
the estimate reflects the recent past with the most recent moments weighted
highest, and old evidence fades instead of being cut off by a window edge.

Three guards decide when the expressed mood may actually change:

  evidence  a mood needs several votes behind it, never one;
  share     the leader must hold a real share of the total, not merely be first
            past the post in a three-way split;
  lead      it must be clearly ahead of what is currently being diffused, so a
            near-tie leaves the room as it is.

Everything here is pure bookkeeping: no I/O, no device, no clock beyond the one
passed in. That makes the policy testable in isolation, which matters for
something meant to run unattended for hours.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class VibeTracker:
    """
    Decaying vote accumulator over recipe keys.

    `half_life` is how long it takes a vote to count half as much. At 75 s, a
    mood that stopped being expressed two minutes ago still carries about a
    third of its weight -- enough inertia that the room does not flicker, little
    enough that it follows a genuine change within a minute or two.
    """

    # Tuned to follow the room rather than smooth it: whoever leads the vote is
    # diffused, as soon as there is a vote at all. The guards stay in the code so
    # the piece can be slowed down again from the command line -- raise
    # min_evidence and switch_lead to bring back the old inertia.
    half_life: float = 35.0
    min_evidence: float = 0.8       # roughly one confident vote
    switch_share: float = 0.34      # simply being the leader of three
    switch_lead: float = 1.0        # no head start required over the current vibe

    # The resting vibe is the answer whenever nothing much is happening, so it
    # collects votes far more easily than the others and would win back every
    # moment of confession or resonance within seconds. Returning to it is made
    # deliberately harder: its votes count less, and it must lead clearly.
    resting: str = None             # the vibe the room falls back to
    resting_weight: float = 1.0     # a vote for the resting vibe counts this much
    return_lead: float = 1.0        # it must be this many times ahead to take back

    _scores: dict = field(default_factory=dict)
    _last_decay: float = field(default_factory=time.monotonic)

    # --- bookkeeping -----------------------------------------------------
    def _decay(self, now: float = None) -> None:
        now = now if now is not None else time.monotonic()
        elapsed = now - self._last_decay
        self._last_decay = now
        if elapsed <= 0 or not self._scores:
            return
        factor = math.pow(0.5, elapsed / self.half_life)
        self._scores = {k: v * factor for k, v in self._scores.items() if v * factor > 1e-3}

    def vote(self, recipe: str, weight: float, now: float = None) -> None:
        """Record one classification. `weight` should be the model's confidence."""
        self._decay(now)
        if recipe == self.resting:
            weight *= self.resting_weight
        self._scores[recipe] = self._scores.get(recipe, 0.0) + max(0.0, weight)

    def reset(self, recipe: str = None, weight: float = 1.0, now: float = None) -> None:
        """Forget everything, optionally seeding a single mood (used on silence)."""
        self._scores = {recipe: weight} if recipe else {}
        self._last_decay = now if now is not None else time.monotonic()

    # --- reading ---------------------------------------------------------
    # Every reader takes an optional `now`, so the whole policy can be replayed
    # against a simulated clock. Half an injectable clock is worse than none:
    # writing with a fake time and reading with the real one silently decays
    # everything to nothing.
    def evidence(self, now: float = None) -> float:
        self._decay(now)
        return sum(self._scores.values())

    def shares(self, now: float = None) -> dict:
        """Each mood's share of the total weight, 0..1."""
        self._decay(now)
        total = sum(self._scores.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in self._scores.items()}

    def leader(self, now: float = None) -> tuple:
        """(recipe, share) of the strongest mood, or (None, 0.0) with no votes."""
        shares = self.shares(now)
        if not shares:
            return None, 0.0
        best = max(shares, key=lambda k: shares[k])
        return best, shares[best]

    # --- policy ----------------------------------------------------------
    def should_switch(self, current: str, now: float = None) -> tuple:
        """
        Decide whether the diffused mood should change.

        Returns (new_recipe_or_None, explanation). The explanation is always
        filled in, so an operator watching the console can see why the room is
        holding rather than being left to guess.
        """
        shares = self.shares(now)
        if not shares:
            return None, "no evidence yet"

        total = sum(self._scores.values())
        if total < self.min_evidence:
            return None, f"evidence {total:.1f} < {self.min_evidence:.1f}"

        leader, share = self.leader(now)
        if leader == current:
            return None, f"'{leader}' already diffusing ({share:.0%})"
        if share < self.switch_share:
            return None, (f"'{leader}' only {share:.0%} of the room "
                          f"(needs {self.switch_share:.0%})")

        current_share = shares.get(current, 0.0)
        lead = (max(self.switch_lead, self.return_lead) if leader == self.resting
                else self.switch_lead)
        if current_share > 0 and share < current_share * lead:
            return None, (f"'{leader}' {share:.0%} vs '{current}' "
                          f"{current_share:.0%}, needs {lead:.1f}x to take over")
        return leader, f"'{leader}' holds {share:.0%} of the room"

    def summary(self, now: float = None) -> str:
        """Compact one-line view of the vote split, for the console."""
        shares = self.shares(now)
        if not shares:
            return "no evidence"
        parts = " ".join(f"{k} {v:.0%}" for k, v in
                         sorted(shares.items(), key=lambda kv: -kv[1]))
        return f"{parts}  (evidence {sum(self._scores.values()):.1f})"
