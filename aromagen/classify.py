"""
Classifying a snippet of conversation into an olfactory recipe.

Three implementations:

  LocalClassifier   - (local_brain.py) a small language model on this machine.
                      The default, because the venue has no internet.
  ClaudeClassifier  - one Claude call per decision, output constrained by a JSON
                      schema. Needs the network.
  LexiconClassifier - keyword counting, no dependency, no latency. The fallback
                      behind either of the others.

`FallbackClassifier` puts the lexicon behind the chosen primary with a circuit
breaker: after several consecutive failures it switches to the lexicon for a
while, instead of retrying in a loop and stalling the pipeline.

Both languages are supported, English first, French second.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import unicodedata
from dataclasses import dataclass

from .recipes import (AVAILABLE_RECIPES, BY_KEY, DEFAULT_RECIPE, RECIPES,
                      catalogue_for_prompt)

log = logging.getLogger("aromagen.classify")

# Haiku by default: this runs a classification every few seconds for hours on
# end, so the per-call price is what decides whether a day-long exhibition is
# affordable. Pass --model claude-opus-5 for a sharper read of subtext.
MODEL = "claude-haiku-4-5"

# `output_config.effort` is rejected by Haiku 4.5 and the other 4.5-generation
# models; it only exists from Opus 4.5 upwards. Sending it anyway is a 400.
MODELS_WITHOUT_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5")

SYSTEM_PROMPT = f"""You are the nose of a scent diffuser in a public space. You read a rough transcript of what has just been said in the room and judge the OVERALL MOOD of that stretch of conversation.

The room speaks English or French, sometimes both. Handle either without comment.

Moods:
{catalogue_for_prompt()}

How your answer is used: it is one VOTE among many, averaged over several minutes. You are not switching the scent yourself, so do not try to compensate or hedge -- report what this stretch actually sounds like and let the average do its work.

Rules:
- Judge the TONE of the exchange, not its subject. An argument about a spreadsheet is negative; a warm chat about a funeral is not.
- Judge the WHOLE stretch, not its sharpest line. One irritated remark inside an otherwise easy conversation does not make the room negative.
- The transcript is noisy: missing words, homophones, several speakers blurred together. Do not over-read a single word.
- "{DEFAULT_RECIPE}" is the honest answer whenever the room is just talking -- logistics, work, small talk, nothing charged. It is the most common answer, not a failure.
- `confidence` is the weight your vote carries. Be strict: below 0.4 for a reading you would not defend.
- `intensity` is unused for now; report how strongly the mood comes across.
- `reason`: at most 12 words, in English, naming what decided it."""

_SCHEMA = {
    "type": "object",
    "properties": {
        # Only mounted recipes are offered: the model cannot pick a blend
        # the hardware could not produce.
        "recipe": {"type": "string", "enum": [r.key for r in AVAILABLE_RECIPES]},
        "confidence": {"type": "number"},
        "intensity": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["recipe", "confidence", "intensity", "reason"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    recipe: str
    confidence: float
    intensity: float
    reason: str
    source: str          # "claude" | "lexicon" | "silence"
    latency: float = 0.0

    def clamp(self) -> "Decision":
        if self.recipe not in BY_KEY:
            log.warning("Unknown recipe returned (%r), falling back to %s",
                        self.recipe, DEFAULT_RECIPE)
            self.recipe = DEFAULT_RECIPE
            self.confidence = min(self.confidence, 0.3)
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        # Floor at 0.35: below that the puff is too short to smell at all, so
        # there would be no point diffusing.
        self.intensity = max(0.35, min(float(self.intensity), 1.0))
        return self


def _normalize(text: str) -> str:
    """Lowercase, strip accents, keep alphanumerics -- so French matches too."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else " " for c in text)


class LexiconClassifier:
    """Keyword counting, weighted by how recent each utterance is."""

    name = "lexicon"

    async def classify(self, window: list) -> Decision:
        t0 = time.monotonic()
        scores = {r.key: 0.0 for r in AVAILABLE_RECIPES}
        now = time.time()

        for utt in window:
            haystack = " " + _normalize(str(utt)) + " "
            age = max(0.0, now - getattr(utt, "started_at", now))
            weight = 0.5 ** (age / 45.0)          # 45 s half-life
            for recipe in AVAILABLE_RECIPES:
                for kw in recipe.keywords:
                    if f" {_normalize(kw)} " in haystack:
                        scores[recipe.key] += weight

        best = max(scores, key=lambda k: scores[k])
        top = scores[best]
        if top <= 0:
            return Decision(DEFAULT_RECIPE, 0.25, 0.4, "no keyword matched",
                            self.name, time.monotonic() - t0).clamp()

        rest = sum(scores.values()) - top
        margin = top / (top + rest) if (top + rest) else 1.0
        # The lexicon is crude: cap its confidence so it never drives changes as
        # assertively as Claude does.
        confidence = min(0.35 + 0.4 * margin, 0.75)
        intensity = min(0.4 + 0.15 * top, 1.0)
        hits = int(round(top))
        return Decision(best, confidence, intensity, f"{hits} lexical hit(s)",
                        self.name, time.monotonic() - t0).clamp()


class ClaudeClassifier:
    """One Claude call per decision, response constrained by a JSON schema."""

    name = "claude"

    def __init__(self, model: str = MODEL, timeout: float = 9.0,
                 effort: str = "low", api_key=None):
        from anthropic import AsyncAnthropic

        self.model = model
        self.effort = effort
        # No retry on purpose. A retry doubles worst-case latency, and a scent
        # that lands 20 s late is worse than one chosen by the lexicon on time.
        # The FallbackClassifier below picks up the failure immediately.
        self._client = AsyncAnthropic(timeout=timeout, max_retries=0,
                                      **({"api_key": api_key} if api_key else {}))

    @staticmethod
    def _render(window: list) -> str:
        now = time.time()
        lines = []
        for utt in window:
            age = now - getattr(utt, "started_at", now)
            lines.append(f"[{age:4.0f}s ago] {utt}")
        return ("Transcript of the room:\n" + "\n".join(lines)
                + "\n\nWhich olfactory mood?")

    async def classify(self, window: list) -> Decision:
        t0 = time.monotonic()
        output_config = {"format": {"type": "json_schema", "schema": _SCHEMA}}
        if self.model not in MODELS_WITHOUT_EFFORT:
            output_config["effort"] = self.effort
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            output_config=output_config,
            messages=[{"role": "user", "content": self._render(window)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("request declined by the model")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return Decision(data["recipe"], data["confidence"], data["intensity"],
                        data["reason"], self.name, time.monotonic() - t0).clamp()


class FallbackClassifier:
    """
    A primary classifier -- the local model or Claude -- with the lexicon behind it.

    Circuit breaker: `failures_before_trip` consecutive failures disable the
    primary for `trip_seconds`, so a broken model or a dead network costs one
    lexicon vote each time instead of stalling every decision.
    """

    name = "auto"

    def __init__(self, primary, secondary, failures_before_trip: int = 3,
                 trip_seconds: float = 90.0):
        self.primary = primary
        self.secondary = secondary
        self.failures_before_trip = failures_before_trip
        self.trip_seconds = trip_seconds
        self._failures = 0
        self._blocked_until = 0.0

    @property
    def degraded(self) -> bool:
        return self.primary is None or time.monotonic() < self._blocked_until

    @property
    def description(self) -> str:
        """What is actually judging the room, for the dashboard."""
        if self.primary is None:
            return "keyword lexicon"
        return (getattr(self.primary, "description", None)
                or getattr(self.primary, "model", None) or self.primary.name)

    async def classify(self, window: list) -> Decision:
        if not self.degraded:
            try:
                decision = await self.primary.classify(window)
                self._failures = 0
                return decision
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failures += 1
                log.warning("%s classifier failed (%s) [%d/%d]", self.primary.name,
                            exc, self._failures, self.failures_before_trip)
                if self._failures >= self.failures_before_trip:
                    self._blocked_until = time.monotonic() + self.trip_seconds
                    self._failures = 0
                    log.error("Falling back to the lexicon for %.0fs", self.trip_seconds)
        return await self.secondary.classify(window)


def build_classifier(brain: str = "local", model: str = MODEL, effort: str = "low"):
    """
    Assemble the mood classifier, always backed by the keyword lexicon.

    `brain` is "local" (the default: a language model on this machine, no
    network needed), "claude" (needs the internet) or "lexicon" (keywords only).
    A primary that cannot be built degrades to the lexicon with a loud error
    rather than stopping the installation.
    """
    lexicon = LexiconClassifier()
    if brain == "lexicon":
        log.info("Classifying with the keyword lexicon only")
        return FallbackClassifier(None, lexicon)
    if brain not in ("local", "claude"):
        raise ValueError(f"unknown brain {brain!r}")
    try:
        if brain == "local":
            from .local_brain import LocalClassifier

            primary = LocalClassifier()
            log.info("Classifying with %s on this machine (backup: lexicon)",
                     primary.description)
        else:
            primary = ClaudeClassifier(model=model, effort=effort)
            log.info("Classifying with %s over the network (backup: lexicon)", model)
        return FallbackClassifier(primary, lexicon)
    except Exception as exc:
        log.error("%s classifier unavailable (%s) - KEYWORD LEXICON ONLY", brain, exc)
        return FallbackClassifier(None, lexicon)
