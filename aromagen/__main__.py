"""
AromaGen command line.

    python -m aromagen --recipes              recipe catalogue
    python -m aromagen --list-audio           available microphone inputs
    python -m aromagen --test-valves          hardware test, valve by valve
    python -m aromagen --text "..."           classify one text, no microphone
    python -m aromagen --replay conv.txt      replay a written conversation
    python -m aromagen                        live: microphone + diffuser

Add --mock to any command to run everything without hardware: valve states are
printed to the console instead.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

from .classify import MODEL as CLASSIFIER_MODEL
from .classify import build_classifier
from .device import burst_schedule, make_device
from .env import describe_credentials, load_env_file
from .engine import AromaEngine, EngineConfig
from .ingredients import EMPTY_VALVES, INGREDIENTS, valve_name
from .local_brain import LOCAL_WINDOW_SECONDS
from .recipes import BY_KEY, RECIPES, availability_report

log = logging.getLogger("aromagen")


# --- display --------------------------------------------------------------
def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def make_reporter(engine_ref: dict):
    """
    Console view for whoever is watching the installation.

    It has to answer two questions at a glance: what is the room doing, and why
    is the object doing what it does. Silent behaviour is the hardest to trust,
    so every decision to HOLD is printed with its reason too.
    """
    def report(kind, payload):
        if kind == "utterance":
            tag = f" [{payload.language}]" if getattr(payload, "language", "") else ""
            print(f'{_stamp()}  "{payload}"{tag}')
        elif kind == "decision":
            d, vibe = payload
            print(f"{_stamp()}  vote {d.recipe:8} conf={d.confidence:.2f} "
                  f"[{d.source} {d.latency:.1f}s] {d.reason}")
            print(f"{_stamp()}       room: {vibe}")
        elif kind == "held":
            print(f"{_stamp()}       holding "
                  f"'{engine_ref['engine'].state.current}' -- {payload}")
        elif kind == "silence":
            print(f"{_stamp()}  -- nothing heard for {payload:.0f}s, "
                  f"returning to the resting scent --")
        elif kind == "saturated":
            limit, window = payload
            print(f"{_stamp()}  -- saturation guard: {limit} bursts already in "
                  f"the last {window / 60:.0f} min, skipping --")
        elif kind == "burst":
            recipe, intensity, why = payload
            print(f"{_stamp()}  ### {recipe.label.upper()} at {intensity:.0%} "
                  f"-- {why}\n            {recipe.describe()}")
        elif kind == "burst_failed":
            print(f"{_stamp()}  !!! diffuser unreachable, burst lost")
        elif kind == "device_ready":
            print(f"{_stamp()}  Diffuser connected and ready.")
        elif kind == "device_absent":
            print(f"{_stamp()}  Diffuser not found: check the side switch (slide "
                  f"it up). Reconnection continues in the background, bursts "
                  f"will resume on their own.")
    return report


def show_recipes() -> None:
    print("\nIngredients mounted on the device:")
    for ing in INGREDIENTS:
        print(f"  valve {ing.valve}  {ing.name:22} {ing.family:18} {ing.notes}")
    for valve in EMPTY_VALVES:
        print(f"  valve {valve}  {'(empty)':22} -")
    print("\nRecipes:")
    for r in RECIPES:
        flag = "" if r.available else f"   [UNAVAILABLE: needs {'+'.join(r.missing())}]"
        print(f"\n  {r.key.upper()} - {r.label}{flag}")
        print(f"    {r.describe()}")
        if r.available:
            print(f"    valves {sorted(r.valves())}")
        print(f"    {r.theme}")
    print(f"\n{availability_report()}\n")


def show_audio() -> None:
    try:
        from .transcribe import list_input_devices
        devices = list_input_devices()
    except Exception as exc:
        print(f"Could not list audio inputs: {exc}")
        return
    print("\nAvailable audio inputs (--mic N):")
    for idx, name, rate in devices:
        print(f"  [{idx:2}] {name}  ({rate:.0f} Hz)")
    print()


# --- modes ----------------------------------------------------------------
async def run_valve_test(args) -> None:
    """Open each valve in turn: checks the wiring and the cartridges."""
    device = make_device(mock=args.mock, address=args.address)
    await device.start()
    if hasattr(device, "wait_connected"):
        print("Connecting to the diffuser...")
        if not await device.wait_connected(args.connect_timeout):
            print("Diffuser not found. Switch it on, or rerun with --mock.")
            await device.close()
            return
    try:
        for valve in range(1, 7):
            print(f"\n  valve {valve} -> {valve_name(valve)} ({args.hold:.0f}s)")
            await device.apply(True, frozenset({valve}))
            await asyncio.sleep(args.hold)
            await device.all_off()
            await asyncio.sleep(1.0)
        print("\nTest complete, all outputs are shut.")
    finally:
        await device.close()


class Line:
    """A written utterance, for --text and --replay."""

    def __init__(self, text: str):
        self.text = text
        self.started_at = time.time()
        self.language = ""

    def __str__(self) -> str:
        return self.text


async def run_one_text(args, classifier) -> None:
    """Classify a single text and show the valve sequence it would produce."""
    decision = await classifier.classify([Line(args.text)])
    recipe = BY_KEY[decision.recipe]
    print(f"\n  text      : {args.text}")
    print(f"  recipe    : {recipe.label} ({decision.recipe})")
    print(f"  confidence: {decision.confidence:.2f}   intensity: {decision.intensity:.2f}")
    print(f"  source    : {decision.source} in {decision.latency:.2f}s")
    print(f"  reason    : {decision.reason}")
    print(f"  blend     : {recipe.describe()}")
    print("  sequence  :")
    for step in burst_schedule(recipe, args.burst, decision.intensity):
        names = ", ".join(sorted(str(v) for v in step.valves))
        print(f"      valve {names:10} for {step.hold:.1f}s")
    print()


async def run_engine(args, feeder, classifier) -> None:
    """Shared core of the live and replay modes."""
    cfg = EngineConfig(burst_seconds=args.burst,
                       min_dwell=args.dwell,
                       min_classify_interval=args.interval,
                       window_seconds=args.window,
                       intensity_gain=args.gain,
                       cooldown=args.cooldown,
                       silence_timeout=args.silence,
                       rest_hold=args.rest_hold,
                       vibe_half_life=args.half_life,
                       max_bursts_per_window=args.max_bursts)
    if args.fast:
        # Demo pacing only. Never run an installation like this: it exists to
        # make a two-minute test show behaviour that normally takes an hour.
        cfg.min_dwell /= 5
        cfg.rest_hold /= 5
        cfg.refresh_interval /= 5
        cfg.cooldown /= 5
        cfg.vibe_half_life /= 5
        cfg.min_evidence = 1.0
        cfg.min_classify_interval = min(cfg.min_classify_interval, 2.0)
        cfg.silence_timeout /= 5
        cfg.max_bursts_per_window = 999

    device = make_device(mock=args.mock, address=args.address)
    ref: dict = {}

    dashboard = None
    # A redirected stdout (a log file, a pipe) cannot host a live screen, and
    # would fill the file with escape codes. Plain lines are right there.
    interactive = sys.stdout.isatty()
    if not args.plain and interactive:
        try:
            from .ui import Dashboard

            dashboard = Dashboard(ref, model_name=getattr(
                classifier, "description", classifier.name))
        except Exception as exc:
            # A missing or unhappy terminal must never stop the installation:
            # fall back to plain lines and carry on.
            log.warning("Dashboard unavailable (%s), using plain output", exc)

    on_event = dashboard.on_event if dashboard else make_reporter(ref)
    engine = AromaEngine(device, classifier, cfg, on_event=on_event)
    ref["engine"] = engine

    tasks = [asyncio.create_task(engine.run(), name="engine")]
    ui_stop = asyncio.Event()
    if dashboard:
        # Quiet the console while the dashboard owns the screen; warnings still
        # reach it through its own log handler and show in the status bar.
        logging.getLogger("aromagen").setLevel(logging.WARNING)
        tasks.append(asyncio.create_task(dashboard.run(ui_stop), name="ui"))

    engine_task = tasks[0]
    feeder_task = asyncio.create_task(feeder(engine), name="feeder")
    try:
        await feeder_task
        if args.replay:
            # Let the engine digest the end of the conversation.
            await asyncio.sleep(cfg.min_classify_interval + 4)
    except asyncio.CancelledError:
        pass
    finally:
        engine.stop()
        ui_stop.set()
        feeder_task.cancel()
        try:
            await asyncio.wait_for(engine_task, timeout=10)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            engine_task.cancel()
        for task in tasks[1:]:
            task.cancel()
        st = engine.state
        print(f"\nSummary: {st.bursts} burst(s), {st.switches} mood change(s), "
              f"{st.suppressed} suppressed by the saturation guard, "
              f"final mood '{st.current}'.")


def make_live_feeder(args, model=None):
    async def feeder(engine):
        from .transcribe import LiveTranscriber

        queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        transcriber = LiveTranscriber(
            asyncio.get_running_loop(), queue, model=model,
            model_size=args.whisper, language=args.lang, device_index=args.mic)
        transcriber.start()
        detector = ("loudness + Silero, quiet voices included" if transcriber.uses_silero
                    else "loudness only, quiet voices may be missed")
        print(f"{_stamp()}  Listening ({detector}). Ctrl+C to stop.")
        try:
            while True:
                engine.feed(await queue.get())
        finally:
            transcriber.stop()
    return feeder


def make_replay_feeder(args):
    async def feeder(engine):
        with open(args.replay, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        print(f"{_stamp()}  Replaying {len(lines)} lines "
              f"({args.replay_delay:.1f}s apart).")
        for line in lines:
            engine.feed(Line(line))
            await asyncio.sleep(args.replay_delay)
    return feeder


def make_brain(args):
    """
    Build the mood classifier. Call it on the main thread, and after Whisper in
    live mode: the local model checks how much VRAM is left before choosing the
    GPU, so Whisper's share has to be taken already.
    """
    return build_classifier(brain=args.brain, model=args.model)


# --- entry point ----------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m aromagen",
        description="Diffuse a scent chosen from the surrounding conversation.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    mode = p.add_argument_group("modes")
    mode.add_argument("--recipes", action="store_true", help="print the catalogue")
    mode.add_argument("--list-audio", action="store_true", help="list microphones")
    mode.add_argument("--test-valves", action="store_true", help="hardware test")
    mode.add_argument("--text", help="classify a single text and exit")
    mode.add_argument("--replay", help="replay a file of lines (one per line)")

    hw = p.add_argument_group("hardware")
    hw.add_argument("--mock", action="store_true", help="simulated device, console only")
    hw.add_argument("--address", help="BLE address, skips the scan")
    hw.add_argument("--connect-timeout", type=float, default=25.0)
    hw.add_argument("--hold", type=float, default=3.0,
                    help="seconds per valve for --test-valves")

    au = p.add_argument_group("audio")
    au.add_argument("--mic", type=int, default=None, help="input index (--list-audio)")
    au.add_argument("--whisper", default=None,
                    choices=("tiny", "base", "small", "medium",
                             "large-v3-turbo", "large-v3"),
                    help="model size; defaults to large-v3-turbo on a GPU, "
                         "small on CPU")
    au.add_argument("--device", default=None, choices=("cuda", "cpu"),
                    help="force the backend; autodetected otherwise")
    au.add_argument("--lang", default="auto", choices=("en", "fr", "auto"),
                    help="'auto' (default) picks English or French per "
                         "utterance; 'en'/'fr' force one and skip detection")

    br = p.add_argument_group("brain")
    br.add_argument("--brain", default="local", choices=("local", "claude", "lexicon"),
                    help="who judges the mood: a model on this machine (default, "
                         "no internet needed), Claude (needs internet), or keywords")
    br.add_argument("--no-llm", action="store_true", help=argparse.SUPPRESS)
    br.add_argument("--model", default=CLASSIFIER_MODEL,
                    help="Claude model, with --brain claude")
    br.add_argument("--dwell", type=float, default=0.0,
                    help="minimum seconds a vibe stays before it may change")
    br.add_argument("--interval", type=float, default=5.0,
                    help="minimum seconds between classifications")
    br.add_argument("--silence", type=float, default=90.0,
                    help="seconds without speech before returning to solitude")
    br.add_argument("--rest-hold", type=float, default=45.0,
                    help="minimum seconds confession/resonance hold before "
                         "solitude may take back")
    br.add_argument("--half-life", type=float, default=35.0,
                    help="seconds for a vote to count half as much")
    br.add_argument("--max-bursts", type=int, default=20,
                    help="ceiling on bursts per 10 min (saturation guard)")
    br.add_argument("--burst", type=float, default=15.0,
                    help="burst duration in seconds (max 20)")
    br.add_argument("--window", type=float, default=None,
                    help="seconds of talk one classification judges "
                         "(default 15 with the local brain, 45 with Claude)")
    br.add_argument("--gain", type=float, default=1.0,
                    help="burst strength multiplier")
    br.add_argument("--cooldown", type=float, default=3.0,
                    help="minimum rest between two bursts")
    br.add_argument("--fast", action="store_true",
                    help="compress all timings (demos and tests)")
    br.add_argument("--replay-delay", type=float, default=2.5)

    p.add_argument("--plain", action="store_true",
                   help="scrolling text instead of the live dashboard")
    p.add_argument("--online", action="store_true",
                   help="let libraries reach the Hugging Face hub; by default it is "
                        "never contacted, exactly as at the venue")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.online:
        # The venue has no internet. Behave that way everywhere, so a missing
        # model fails at once with a clear message on the day it is prepared,
        # never as a network timeout in front of visitors. Must be set before
        # any Hugging Face library is imported; all of them are imported lazily.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.no_llm:
        args.brain = "lexicon"
    if args.window is None:
        # The local model is measurably better on short stretches; Claude on
        # longer ones. The vote tracker handles the long-term view either way.
        args.window = LOCAL_WINDOW_SECONDS if args.brain == "local" else 45.0

    load_env_file()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("bleak").setLevel(logging.WARNING)
    if args.brain == "claude":
        log.info("%s", describe_credentials())
    log.info("%s", availability_report())

    if args.recipes:
        show_recipes()
        return 0
    if args.list_audio:
        show_audio()
        return 0

    try:
        if args.test_valves:
            asyncio.run(run_valve_test(args))
        elif args.text:
            asyncio.run(run_one_text(args, make_brain(args)))
        elif args.replay:
            asyncio.run(run_engine(args, make_replay_feeder(args), make_brain(args)))
        else:
            # Whisper MUST be loaded here: main thread, before the microphone
            # (PortAudio) and the Windows BLE stack are initialised. Loading it
            # afterwards from a thread causes an access violation that kills the
            # process without leaving a single line of Python traceback.
            from .transcribe import load_model
            model = load_model(args.whisper, device=args.device)
            brain = make_brain(args)
            asyncio.run(run_engine(args, make_live_feeder(args, model), brain))
    except KeyboardInterrupt:
        print("\nStopped. All outputs are shut.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
