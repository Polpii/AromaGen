"""
Wait for the diffuser to advertise, then connect and exercise all six valves.

The device only advertises intermittently (side switch, reboots). Rather than
scanning and then running a separate command -- and missing the window -- this
chains discovery, connection and test in one go.
"""

import asyncio
import time

from bleak import BleakClient, BleakScanner

from aromagen.device import (CHARACTERISTIC_UUID, DEVICE_NAME, burst_schedule,
                             describe_byte, pack_byte)
from aromagen.ingredients import INGREDIENTS
from aromagen.recipes import BY_KEY

TOTAL_WAIT = 600.0   # 10 minutes of watching
HOLD = 2.0           # seconds each valve stays open
DEMO_RECIPE = "resonance"


async def find():
    """Short repeated scans: the point is to catch the advertising window."""
    deadline = time.time() + TOTAL_WAIT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=4.0)
        if dev is not None:
            print(f"\n>>> FOUND: {dev.name} @ {dev.address}", flush=True)
            return dev
        if attempt % 5 == 0:
            print(f"    ...nothing yet ({int(deadline - time.time())}s left) "
                  f"- check the side switch", flush=True)
    return None


async def run_test(dev):
    print("Connecting...", flush=True)
    t0 = time.time()
    async with BleakClient(dev, timeout=45.0) as client:
        print(f"Connected in {time.time() - t0:.1f}s\n", flush=True)

        async def write(byte):
            await client.write_gatt_char(CHARACTERISTIC_UUID, bytes([byte]), response=True)

        await write(0)
        state = await client.read_gatt_char(CHARACTERISTIC_UUID)
        print(f"Initial state: {describe_byte(state[0])}\n", flush=True)

        try:
            print("--- Pump alone (2 s): you should hear it run ---", flush=True)
            await write(pack_byte(True, set()))
            await asyncio.sleep(HOLD)
            await write(0)
            await asyncio.sleep(1.0)

            for ing in INGREDIENTS:
                byte = pack_byte(True, {ing.valve})
                print(f"--- Valve {ing.valve}: {ing.name} ({HOLD:.0f} s) "
                      f"-> {describe_byte(byte)}", flush=True)
                await write(byte)
                await asyncio.sleep(HOLD)
                await write(0)
                await asyncio.sleep(1.0)

            recipe = BY_KEY[DEMO_RECIPE]
            print(f"\n--- Recipe {DEMO_RECIPE!r}: {recipe.describe()} ---", flush=True)
            for step in burst_schedule(recipe, 10.0):
                print(f"    valve {sorted(step.valves)} for {step.hold:.1f}s", flush=True)
                await write(pack_byte(True, step.valves))
                await asyncio.sleep(step.hold)
        finally:
            await write(0)
            print("\nAll outputs are shut.", flush=True)


async def main():
    print("Watching for the diffuser (10 min). Slide the switch UP.\n", flush=True)
    dev = await find()
    if dev is None:
        print("\nNever appeared. Switch it off, wait, switch it back on.")
        return
    for attempt in range(1, 4):
        try:
            await run_test(dev)
            return
        except Exception as exc:
            print(f"Connection failed ({type(exc).__name__}: {exc}) "
                  f"- attempt {attempt}/3", flush=True)
            await asyncio.sleep(2)
    print("Could not connect after 3 attempts.")


asyncio.run(main())
