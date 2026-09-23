"""
Diffuse one recipe on a loop, forever. No microphone, no classifier.

    python run_neutral.py                    # neutral, 10 s burst, 8 s rest
    python run_neutral.py --burst 15 --rest 5
    python run_neutral.py --recipe warm --intensity 0.6
    python run_neutral.py --mock             # console only, no hardware

Ctrl+C shuts every output before returning.

The rest between bursts is not optional politeness: it keeps the pump duty cycle
sane and gives the scent time to actually reach the room instead of being pushed
into air that is already saturated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from aromagen.device import MAX_BURST_SECONDS, burst_schedule, make_device
from aromagen.env import load_env_file
from aromagen.recipes import AVAILABLE_RECIPES, BY_KEY, DEFAULT_RECIPE

log = logging.getLogger("run_neutral")


async def loop_forever(args) -> None:
    recipe = BY_KEY[args.recipe]
    device = make_device(mock=args.mock, address=args.address)

    print(f"Recipe : {recipe.label} -- {recipe.describe()}")
    print("Plan   : " + " -> ".join(
        f"valve {list(s.valves)[0]} {s.hold:.1f}s"
        for s in burst_schedule(recipe, args.burst, args.intensity)))
    print(f"Cycle  : {args.burst:.0f}s burst at {args.intensity:.0%}, "
          f"then {args.rest:.0f}s rest. Ctrl+C to stop.\n")

    await device.start()
    if hasattr(device, "wait_connected"):
        print("Connecting to the diffuser...")
        if not await device.wait_connected(args.connect_timeout):
            print("Diffuser not found. Check the side switch is UP, or use --mock.")
            await device.close()
            return
        print("Connected.\n")

    count = 0
    try:
        while True:
            count += 1
            ok = await device.diffuse(recipe, args.burst, args.intensity)
            if ok:
                print(f"  burst {count} done")
            else:
                # Never abort on a dropped link: the device layer reconnects on
                # its own, so just wait and let the next cycle go out.
                print(f"  burst {count} FAILED (diffuser unreachable) - "
                      f"waiting for the link to come back")
            await asyncio.sleep(args.rest)
    finally:
        print("\nShutting all outputs...")
        await device.close()


def main() -> int:
    load_env_file()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("bleak").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", default=DEFAULT_RECIPE,
                   choices=[r.key for r in AVAILABLE_RECIPES])
    p.add_argument("--burst", type=float, default=10.0,
                   help=f"seconds of diffusion per cycle (max {MAX_BURST_SECONDS:.0f})")
    p.add_argument("--rest", type=float, default=8.0,
                   help="seconds of silence between bursts")
    p.add_argument("--intensity", type=float, default=1.0, help="0.0 to 1.0")
    p.add_argument("--mock", action="store_true", help="console only, no hardware")
    p.add_argument("--address", help="BLE address, skips the scan")
    p.add_argument("--connect-timeout", type=float, default=30.0)
    args = p.parse_args()

    try:
        asyncio.run(loop_forever(args))
    except KeyboardInterrupt:
        print("Stopped. All outputs are shut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
