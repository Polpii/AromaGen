"""
Orchestrator: transcription -> classification -> mood estimate -> diffusion.

Designed for an unattended public installation, which drives every choice here.

The object does not react to sentences; it expresses the room's mood. A
classification is a vote into `VibeTracker`, whose decaying average is what
actually gets diffused. Three moods only -- positive, neutral, negative -- so
that visitors can perceive the difference at all.

The rules below exist because a smell is not a screen. It takes tens of seconds
to fill a room and minutes to clear, it mixes with whatever came before, and a
nose stops noticing a constant one within minutes:

  sliding window     a classification judges the last 45 seconds of talk;
  cadence            at most one classification every few seconds, and only if
                     something new has been said -- this also bounds API spend
                     over a day-long run;
  vote, not order    the mood is the decaying average of many classifications,
                     so no single sentence can flip the room;
  dwell time         an expressed mood holds for at least a minute, because
                     changing sooner only mixes two scents into mud;
  refresh            the standing mood is re-emitted, quietly, as it fades;
  slow return        the resting vibe takes the room back slowly, because it is
                     what every uneventful stretch of talk votes for;
  silence            nothing heard for 90 s returns the room to rest;
  saturation guard   a hard ceiling on how much is diffused per ten minutes, so
                     a busy afternoon cannot flood the space.

Classification does not pause during a burst -- pausing would cost the whole
burst duration in responsiveness -- but two bursts never overlap: the power
supply could not take it, and two scents at once mean nothing.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field

from .classify import Decision
from .recipes import BY_KEY, DEFAULT_RECIPE
from .vibe import VibeTracker

log = logging.getLogger("aromagen.engine")


@dataclass
class EngineConfig:
    # --- listening -------------------------------------------------------
    window_seconds: float = 45.0        # how much talk one classification judges
    min_classify_interval: float = 5.0  # max classifier cadence
    min_new_words: int = 5              # new material required before reclassifying

    # --- mood estimate ---------------------------------------------------
    vibe_half_life: float = 35.0        # how fast old votes fade
    min_evidence: float = 0.8           # roughly one confident vote
    switch_share: float = 0.34          # simply being the leader of three
    switch_lead: float = 1.0            # no head start over the current vibe
    # Returning to the resting vibe is slow on purpose. Almost any stretch of
    # talk that is not a confession or a meeting of minds votes for it, so with
    # symmetric rules it won back confession and resonance within seconds.
    # Measured (votes every 5 s): after a minute of confession it now takes about
    # 65 s of pure small talk to fall back, against 25 s before; and a confession
    # that keeps resurfacing even one vote in three holds indefinitely.
    resting_vote_weight: float = 0.5    # a resting vote counts half
    return_lead: float = 2.0            # resting must be twice as strong to take back
    rest_hold: float = 45.0             # confession/resonance last at least this long

    # --- diffusion -------------------------------------------------------
    min_dwell: float = 0.0              # a vibe change is never held back
    # Tuned for a space with a fan clearing the air between bursts. Without one,
    # halve the burst and double the refresh interval: a scent that never clears
    # both saturates the room and stops being noticed.
    refresh_interval: float = 45.0      # re-emit the standing mood after this
    refresh_damping: float = 0.8        # a refresh is slightly quieter
    burst_seconds: float = 15.0         # burst duration at intensity 1
    intensity_gain: float = 1.0         # global strength multiplier
    cooldown: float = 3.0               # minimum rest between two bursts

    # --- installation guards ---------------------------------------------
    silence_timeout: float = 90.0       # nothing heard -> back to the resting vibe
    silence_intensity: float = 0.6      # the resting scent, still below full
    saturation_window: float = 600.0    # ten minutes
    # 20 x 15 s is a 50% duty cycle at the ceiling; steady state sits near 33%
    # (15 s every 45 s). The pump tolerates that, and the hard 20 s cap in
    # device.py still applies to any single burst.
    max_bursts_per_window: int = 20

    tick: float = 0.4
    connect_timeout: float = 30.0       # wait for the BLE link before listening


@dataclass
class EngineState:
    current: str = DEFAULT_RECIPE
    last_switch: float = 0.0
    last_burst: float = 0.0
    last_classify: float = 0.0
    last_speech: float = field(default_factory=time.time)
    pending_words: int = 0
    bursts: int = 0
    switches: int = 0
    suppressed: int = 0                 # bursts skipped by the saturation guard
    resting: bool = True                # true while the room is silent


class AromaEngine:
    """
    Consumes `Utterance` objects and drives the diffuser.

    `on_event(kind, payload)` lets a UI follow what happens without the engine
    needing to know how any of it is displayed.
    """

    def __init__(self, device, classifier, config: EngineConfig = None, on_event=None):
        self.device = device
        self.classifier = classifier
        self.cfg = config or EngineConfig()
        self.on_event = on_event or (lambda kind, payload: None)
        self.state = EngineState()
        self.vibe = VibeTracker(half_life=self.cfg.vibe_half_life,
                                min_evidence=self.cfg.min_evidence,
                                switch_share=self.cfg.switch_share,
                                switch_lead=self.cfg.switch_lead,
                                resting=DEFAULT_RECIPE,
                                resting_weight=self.cfg.resting_vote_weight,
                                return_lead=self.cfg.return_lead)
        self.window: list = []
        self.inbox: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._burst_times: collections.deque = collections.deque()
        self._diffusing = False
        self._burst_task = None    # the burst currently going out, if any
        self._pending = None       # mood change decided during a burst
        self._stop = asyncio.Event()

    @property
    def diffusing(self) -> bool:
        """True while a burst is actually going out -- drives the UI's light."""
        return self._diffusing

    # --- input -----------------------------------------------------------
    def feed(self, utterance) -> None:
        """Add a turn of speech (callable from any coroutine)."""
        try:
            self.inbox.put_nowait(utterance)
        except asyncio.QueueFull:
            log.warning("Engine inbox full, utterance dropped")

    # --- main loops ------------------------------------------------------
    async def run(self) -> None:
        await self.device.start()
        # start() only launches the connection supervisor. Without this wait the
        # first burst goes out before the BLE link exists and is lost. Not a
        # failure either way: the supervisor keeps trying in the background and
        # bursts resume as soon as the link is up.
        if hasattr(self.device, "wait_connected"):
            ready = await self.device.wait_connected(self.cfg.connect_timeout)
            self.on_event("device_ready" if ready else "device_absent", None)
        try:
            await asyncio.gather(self._ingest_loop(), self._decide_loop())
        finally:
            await self._finish_burst()
            await self.device.close()

    def stop(self) -> None:
        self._stop.set()

    async def _ingest_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utt = await asyncio.wait_for(self.inbox.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            self.window.append(utt)
            self.state.pending_words += len(str(utt).split())
            self.state.last_speech = time.time()
            self.state.resting = False
            self.on_event("utterance", utt)

    def _prune(self) -> None:
        cutoff = time.time() - self.cfg.window_seconds
        self.window = [u for u in self.window
                       if getattr(u, "started_at", cutoff) >= cutoff]

    async def _decide_loop(self) -> None:
        cfg, st = self.cfg, self.state
        while not self._stop.is_set():
            await asyncio.sleep(cfg.tick)
            self._prune()
            now = time.time()

            # A change decided during a burst is applied as soon as it ends.
            if not self._diffusing and self._pending is not None:
                pending, self._pending = self._pending, None
                await self._express(pending, "queued while diffusing")
                continue

            if now - st.last_speech > cfg.silence_timeout:
                await self._go_quiet()
                continue

            if not self.window:
                continue
            if st.pending_words < cfg.min_new_words:
                continue
            if time.monotonic() - st.last_classify < cfg.min_classify_interval:
                continue

            st.last_classify = time.monotonic()
            st.pending_words = 0
            try:
                decision = await self.classifier.classify(list(self.window))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("Classification failed (%s)", exc)
                continue

            self.vibe.vote(decision.recipe, decision.confidence)
            self.on_event("decision", (decision, self.vibe.summary()))
            await self._consider_change()

    # --- mood policy -----------------------------------------------------
    async def _consider_change(self) -> None:
        """Let the vote accumulator decide whether the room should change."""
        cfg, st = self.cfg, self.state
        leader, why = self.vibe.should_switch(st.current)

        if leader is None:
            # Already expressing the leading mood: keep it alive as it fades.
            since_burst = time.monotonic() - st.last_burst if st.last_burst else 1e9
            if since_burst >= cfg.refresh_interval:
                await self._express(st.current, "refresh", refresh=True)
            else:
                self.on_event("held", why)
            return

        since_switch = time.monotonic() - st.last_switch if st.last_switch else 1e9
        if since_switch < cfg.min_dwell:
            self.on_event("held", f"{why}, but '{st.current}' has only held "
                                  f"{since_switch:.0f}s of {cfg.min_dwell:.0f}s")
            return
        if leader == DEFAULT_RECIPE and since_switch < cfg.rest_hold:
            self.on_event("held", f"{why}, but '{st.current}' holds at least "
                                  f"{cfg.rest_hold:.0f}s ({since_switch:.0f}s so far)")
            return
        await self._express(leader, why)

    async def _go_quiet(self) -> None:
        """
        Nothing heard for a while: return to the resting scent and stay there.

        The accumulator is cleared rather than left to decay, so that whatever
        was said before the room emptied cannot colour the next conversation.
        """
        cfg, st = self.cfg, self.state
        if not st.resting:
            st.resting = True
            self.vibe.reset(DEFAULT_RECIPE)
            self.on_event("silence", cfg.silence_timeout)

        if self._diffusing:
            return
        since_burst = time.monotonic() - st.last_burst if st.last_burst else 1e9
        if st.current != DEFAULT_RECIPE:
            await self._express(DEFAULT_RECIPE, "nothing heard", resting=True)
        elif since_burst >= cfg.refresh_interval:
            await self._express(DEFAULT_RECIPE, "resting", refresh=True, resting=True)

    # --- diffusion -------------------------------------------------------
    def _saturated(self) -> bool:
        """True once the ceiling on diffusion for the last window is reached."""
        cutoff = time.monotonic() - self.cfg.saturation_window
        while self._burst_times and self._burst_times[0] < cutoff:
            self._burst_times.popleft()
        return len(self._burst_times) >= self.cfg.max_bursts_per_window

    async def _express(self, recipe_key: str, why: str, refresh: bool = False,
                       resting: bool = False) -> None:
        cfg, st = self.cfg, self.state

        changing = recipe_key != st.current

        if self._diffusing:
            if not changing:
                # A refresh of the standing vibe can always wait its turn; two
                # bursts must never overlap.
                self._pending = recipe_key
                return
            # A vibe change is the one thing worth cutting a burst short for:
            # the piece should answer the room now, not in fifteen seconds. The
            # valves are shut on the way out, so nothing is left open.
            await self._finish_burst()

        since_burst = time.monotonic() - st.last_burst if st.last_burst else 1e9
        if since_burst < cfg.cooldown and not changing:
            self._pending = recipe_key
            self.on_event("held", f"resting {since_burst:.0f}s of "
                                  f"{cfg.cooldown:.0f}s before the next burst")
            return

        if self._saturated():
            # A ceiling matters more than any single burst: visitors stop
            # noticing a scent that is always present, and the cartridges are
            # finite. Skipping is the correct behaviour, not a failure.
            st.suppressed += 1
            self.on_event("saturated", (cfg.max_bursts_per_window,
                                        cfg.saturation_window))
            return

        if changing:
            previous = st.current
            st.current = recipe_key
            st.last_switch = time.monotonic()
            st.switches += 1
            why = f"{why} (was '{previous}')"

        intensity = cfg.silence_intensity if resting else 1.0
        if refresh:
            intensity *= cfg.refresh_damping
        intensity = max(0.0, min(intensity * cfg.intensity_gain, 1.0))

        recipe = BY_KEY[recipe_key]
        self._diffusing = True
        st.bursts += 1
        self._burst_times.append(time.monotonic())
        self.on_event("burst", (recipe, intensity, why))
        # Diffuse in the background. Awaiting it here blocked the decision loop
        # for the whole burst: the room went deaf for 15 s at a time, and with a
        # short classification window, speech heard during a burst could fall out
        # of the window before it was ever judged. A replayed argument passed
        # without one negative vote. tests/check_engine_bursts.py guards this.
        self._burst_task = asyncio.create_task(self._diffuse(recipe, intensity),
                                               name="burst")

    async def _diffuse(self, recipe, intensity: float) -> None:
        st = self.state
        try:
            ok = await self.device.diffuse(recipe, self.cfg.burst_seconds, intensity)
            if not ok:
                self.on_event("burst_failed", recipe)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("Diffusion failed (%s)", exc)
        finally:
            # Timestamped at the END, so `cooldown` and `refresh_interval` count
            # real rest rather than the burst's own duration.
            st.last_burst = time.monotonic()
            self._diffusing = False

    async def _finish_burst(self) -> None:
        """On shutdown, let a burst in flight shut its valves before closing."""
        task = self._burst_task
        if task is not None and not task.done():
            task.cancel()           # device.diffuse() shuts the valves on cancel
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
