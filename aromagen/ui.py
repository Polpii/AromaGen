"""
Live dashboard for the running installation.

Left: what the room is saying, with the detected language. Right: how the vote
currently stands, and underneath it the scent being expressed with a light that
is green only while air is actually moving through the cartridges.

The point is to make the object's reasoning visible. An installation that
changes scent for reasons nobody can see is impossible to trust or to tune, and
the interesting moments are usually the ones where it decides NOT to change.

It reads the engine directly on every frame rather than keeping its own copy of
the state, so the screen cannot drift out of step with what is really happening.
Events only supply the things the engine does not keep: the conversation
backlog, the last reason, alerts.
"""

from __future__ import annotations

import collections
import logging
import time

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .recipes import AVAILABLE_RECIPES, BY_KEY, DEFAULT_RECIPE

# One colour per mood, used everywhere: bars, title, light. Consistency is what
# lets someone read the screen at a glance from across a room.
MOOD_COLOUR = {
    "confession": "orchid",
    "solitude": "steel_blue1",
    "resonance": "bright_green",
}
# A vibe not named above still gets a stable colour of its own.
PALETTE = ("steel_blue1", "orchid", "bright_green", "gold1", "cyan1", "salmon1")
FALLBACK_COLOUR = {r.key: PALETTE[i % len(PALETTE)]
                   for i, r in enumerate(AVAILABLE_RECIPES)}
DEFAULT_COLOUR = "grey70"

BAR_WIDTH = 18
# Wide enough for the longest vibe name, so nothing is truncated when renamed.
LABEL_WIDTH = max((len(r.key) for r in AVAILABLE_RECIPES), default=8) + 1


def _glyphs() -> dict:
    """
    Block characters where the console can render them, ASCII where it cannot.

    A legacy Windows console is cp1252 and raises UnicodeEncodeError on the
    block and bullet characters -- which would take the whole installation down
    on its first frame. Degrading the look is obviously preferable.
    """
    import sys

    encoding = getattr(sys.stdout, "encoding", "") or "ascii"
    try:
        "█●○─".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return {"full": "#", "empty": "-", "on": "*", "off": "o"}
    return {"full": "█", "empty": "─", "on": "●", "off": "○"}


GLYPH = _glyphs()
FULL, EMPTY = GLYPH["full"], GLYPH["empty"]


def mood_colour(key: str) -> str:
    return MOOD_COLOUR.get(key) or FALLBACK_COLOUR.get(key, DEFAULT_COLOUR)


class AlertHandler(logging.Handler):
    """Keeps the last warning or error so the dashboard can show it."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.last = ""
        self.at = 0.0

    def emit(self, record):
        self.last = f"{record.name.split('.')[-1]}: {record.getMessage()}"
        self.at = time.time()


class Dashboard:
    """
    Console dashboard. Exposes `on_event` so it can replace the plain reporter,
    and `run()` to drive the refresh loop alongside the engine.
    """

    def __init__(self, engine_ref: dict, model_name: str = "", history: int = 200):
        self.engine_ref = engine_ref
        self.model_name = model_name
        self.lines: collections.deque = collections.deque(maxlen=history)
        self.last_vote = None
        self.last_reason = "waiting for the room to say something"
        self.notice = ""
        self.notice_at = 0.0
        self.started = time.time()
        self.alerts = AlertHandler()
        logging.getLogger().addHandler(self.alerts)

    # --- engine events ---------------------------------------------------
    def on_event(self, kind, payload) -> None:
        if kind == "utterance":
            lang = getattr(payload, "language", "") or "??"
            self.lines.append((time.time(), lang, str(payload)))
        elif kind == "decision":
            decision, _summary = payload
            self.last_vote = decision
            self.last_reason = decision.reason
        elif kind == "held":
            self.last_reason = str(payload)
        elif kind == "burst":
            recipe, intensity, why = payload
            self.last_reason = why
            self._notify(f"diffusing {recipe.label} at {intensity:.0%}")
        elif kind == "silence":
            self._notify(f"nothing heard for {payload:.0f}s - resting")
        elif kind == "saturated":
            limit, window = payload
            self._notify(f"saturation guard: {limit} bursts in "
                         f"{window / 60:.0f} min, skipping")
        elif kind == "burst_failed":
            self._notify("diffuser unreachable - burst lost")
        elif kind == "device_ready":
            self._notify("diffuser connected")
        elif kind == "device_absent":
            self._notify("diffuser not found - check the side switch")

    def _notify(self, message: str) -> None:
        self.notice = message
        self.notice_at = time.time()

    # --- panels ----------------------------------------------------------
    def _conversation(self, height: int) -> Panel:
        rows = Table.grid(padding=(0, 1), expand=True)
        rows.add_column(style="grey42", no_wrap=True, width=8)
        rows.add_column(style="grey58", no_wrap=True, width=4)
        rows.add_column(overflow="fold", ratio=1)

        # Roughly two screen lines per utterance once wrapped; showing the tail
        # is what matters, the backlog is only there so nothing is lost.
        visible = max(1, (height - 2) // 2)
        for when, lang, text in list(self.lines)[-visible:]:
            rows.add_row(time.strftime("%H:%M:%S", time.localtime(when)),
                         f"({lang})", Text(text, style="white"))
        if not self.lines:
            rows.add_row("", "", Text("listening...", style="grey42 italic"))
        return Panel(rows, title="[b]Conversation[/b]", border_style="grey35",
                     padding=(0, 1))

    def _votes(self) -> Panel:
        engine = self.engine_ref.get("engine")
        shares = engine.vibe.shares() if engine else {}
        evidence = sum(engine.vibe._scores.values()) if engine else 0.0
        current = engine.state.current if engine else ""

        bars = Table.grid(padding=(0, 1))
        bars.add_column(no_wrap=True, width=LABEL_WIDTH)
        bars.add_column(no_wrap=True, width=BAR_WIDTH)
        bars.add_column(no_wrap=True, justify="right", width=4)

        for key in (r.key for r in AVAILABLE_RECIPES):
            share = shares.get(key, 0.0)
            filled = int(round(share * BAR_WIDTH))
            colour = mood_colour(key)
            bar = Text(FULL * filled, style=colour)
            bar.append(EMPTY * (BAR_WIDTH - filled), style="grey27")
            label = Text(key, style=f"bold {colour}" if key == current else colour)
            bars.add_row(label, bar, Text(f"{share:.0%}", style="grey62"))

        vote = self.last_vote
        footer = Text()
        footer.append(f"evidence {evidence:.1f}", style="grey50")
        if vote is not None:
            footer.append("   last vote ", style="grey50")
            footer.append(vote.recipe, style=mood_colour(vote.recipe))
            footer.append(f" {vote.confidence:.0%} ", style="grey50")
            footer.append(f"[{vote.source}]", style="grey35")

        return Panel(Group(bars, Text(), footer),
                     title="[b]How the room is voting[/b]",
                     border_style="grey35", padding=(1, 2))

    def _current(self) -> Panel:
        engine = self.engine_ref.get("engine")
        key = engine.state.current if engine else DEFAULT_RECIPE
        live = engine.diffusing if engine else False
        recipe = BY_KEY.get(key)
        colour = mood_colour(key)

        light = Text()
        light.append(GLYPH["on"] if live else GLYPH["off"],
                     style="bold bright_green" if live else "bold grey35")
        light.append("  DIFFUSING NOW" if live else "  idle",
                     style="bold bright_green" if live else "grey42")

        title = Text(recipe.label.upper() if recipe else key.upper(),
                     style=f"bold {colour}")
        blend = Text(recipe.describe() if recipe else "", style="grey58")
        why = Text(self.last_reason, style="grey46 italic")

        return Panel(Group(light, Text(), title, blend, Text(), why),
                     title="[b]Scent[/b]", border_style=colour, padding=(1, 2))

    def _status(self) -> Text:
        engine = self.engine_ref.get("engine")
        st = engine.state if engine else None
        bar = Text()

        connected = getattr(engine.device, "connected", False) if engine else False
        bar.append(f"{GLYPH['on'] if connected else GLYPH['off']} diffuser  ",
                   style="green" if connected else "red")
        if self.model_name:
            bar.append(f"{self.model_name}  ", style="grey42")
        if st:
            bar.append(f"{st.bursts} bursts  {st.switches} changes  ", style="grey42")
            if st.suppressed:
                bar.append(f"{st.suppressed} capped  ", style="yellow")
        elapsed = time.time() - self.started
        bar.append(f"up {elapsed / 60:.0f}m  ", style="grey42")

        # A recent alert outranks a routine notice: problems must not scroll by.
        if self.alerts.last and time.time() - self.alerts.at < 30:
            bar.append(f"! {self.alerts.last}", style="bold yellow")
        elif self.notice and time.time() - self.notice_at < 12:
            bar.append(self.notice, style="grey62")
        return bar

    # --- assembly --------------------------------------------------------
    def render(self, height: int = 30):
        layout = Layout()
        layout.split_column(Layout(name="body"), Layout(name="status", size=1))
        layout["body"].split_row(Layout(name="left", ratio=3),
                                 Layout(name="right", ratio=2))
        layout["right"].split_column(Layout(name="votes", size=9),
                                     Layout(name="scent"))
        layout["left"].update(self._conversation(height - 1))
        layout["votes"].update(self._votes())
        layout["scent"].update(self._current())
        layout["status"].update(Align.left(self._status()))
        return layout

    async def run(self, stop_event) -> None:
        """Refresh until `stop_event` is set. Runs alongside the engine."""
        import asyncio

        from rich.console import Console

        console = Console()
        with Live(self.render(console.size.height), console=console,
                  screen=True, refresh_per_second=8, transient=False) as live:
            while not stop_event.is_set():
                live.update(self.render(console.size.height))
                await asyncio.sleep(0.12)
