"""
Driving the smell1 diffuser (ESP32-C3, 6 valves + 1 pump) over BLE.

Protocol (taken from github.com/AwuChen/AromaGen_MOI): a single byte written to
one characteristic sets all 7 outputs atomically.
    bit0 = pump, bit1..bit6 = valve1..valve6, bit7 unused

The device cannot meter flow: a valve is either open or shut. Dosing is
therefore done by open time within a burst (see burst_schedule).

Two implementations share one interface: BleAromaDevice (real hardware, with
automatic reconnection) and MockAromaDevice (console, to develop without the
device at hand).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .ingredients import valve_name
from .recipes import Recipe

log = logging.getLogger("aromagen.device")

DEVICE_NAME = "PumpValveController"
CHARACTERISTIC_UUID = "6d7f0002-2e2d-4a63-9a1b-2b2f6a9d0b10"

# Hard limit: outputs are never left active longer than this, whatever the rest
# of the pipeline asks for.
MAX_BURST_SECONDS = 20.0
WATCHDOG_PERIOD = 1.0

# ONE VALVE AT A TIME. Measured on the hardware: pump + 1 valve is rock solid
# (all six valves tested back to back), but pump + 3 valves open together
# collapses the supply rail, the ESP32 reboots and the BLE link drops.
# Dosing is therefore sequential, never parallel.
MAX_SIMULTANEOUS_VALVES = 1

# Below this, a solenoid has no time to pass a useful volume of air: stretch the
# opening rather than emit something inaudible and odourless.
MIN_OPEN_SECONDS = 0.3


def pack_byte(pump: bool, valves) -> int:
    """Build the protocol byte. `valves` holds valve numbers 1..6."""
    byte = 1 if pump else 0
    for v in valves:
        if not 1 <= v <= 6:
            raise ValueError(f"invalid valve number: {v}")
        byte |= 1 << v
    return byte


def describe_byte(byte: int) -> str:
    on = [valve_name(v) for v in range(1, 7) if byte >> v & 1]
    pump = "PUMP " if byte & 1 else "-----"
    return f"0b{byte:08b} [{pump}] " + (", ".join(on) if on else "(no valve)")


@dataclass(frozen=True)
class Step:
    """One burst step: the valve state to hold for `hold` seconds."""
    valves: frozenset
    hold: float


def burst_schedule(recipe: Recipe, duration: float, intensity: float = 1.0) -> list:
    """
    Turn a recipe into a sequence of valve states.

    Valves open ONE AT A TIME (see MAX_SIMULTANEOUS_VALVES), each for a slice of
    the burst proportional to its dose. The burst lasts `duration` in total and
    every ingredient gets its exact share of the air pushed through -- so the
    recipe's proportions hold, without ever browning out the supply.

    Ingredients run in increasing dose order, so the dominant note comes last
    and therefore dominates the immediate impression.
    """
    duration = max(0.0, min(duration * max(0.0, min(intensity, 1.0)), MAX_BURST_SECONDS))
    valves = recipe.valves()
    if not valves or duration <= 0:
        return []

    total_dose = sum(valves.values())
    steps = [Step(frozenset({valve}), max(duration * dose / total_dose, MIN_OPEN_SECONDS))
             for valve, dose in sorted(valves.items(), key=lambda kv: (kv[1], kv[0]))]

    # Openings stretched to the minimum can push past the ceiling: rescale
    # rather than let the total run over.
    span = sum(s.hold for s in steps)
    if span > MAX_BURST_SECONDS:
        steps = [Step(s.valves, s.hold * MAX_BURST_SECONDS / span) for s in steps]
    return steps


class AromaDevice:
    """Shared interface. `connected` says whether a burst can go out now."""

    connected: bool = False

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def apply(self, pump: bool, valves) -> bool:
        raise NotImplementedError

    async def all_off(self) -> bool:
        return await self.apply(False, frozenset())

    async def diffuse(self, recipe: Recipe, duration: float, intensity: float = 1.0) -> bool:
        """
        Play a full burst, returning once everything is shut again.
        Returns False if the device could not take the whole sequence.
        """
        steps = burst_schedule(recipe, duration, intensity)
        if not steps:
            return await self.all_off()
        ok = True
        try:
            for step in steps:
                if not await self.apply(True, step.valves):
                    ok = False
                    break
                await asyncio.sleep(step.hold)
        finally:
            # Always shut down, even if this task is cancelled mid-burst:
            # shield() guarantees the off command still goes out.
            try:
                ok = await asyncio.shield(self.all_off()) and ok
            except Exception:
                ok = False
        return ok


class MockAromaDevice(AromaDevice):
    """Simulated device: prints valve state to the console. No hardware needed."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.connected = True
        self.last_byte = 0
        self.history = []

    async def start(self) -> None:
        log.info("SIMULATED device active (no hardware in use)")

    async def close(self) -> None:
        await self.all_off()

    async def apply(self, pump: bool, valves) -> bool:
        byte = pack_byte(pump, valves)
        self.last_byte = byte
        self.history.append((time.monotonic(), byte))
        if self.verbose:
            bars = "".join("#" if byte >> v & 1 else "." for v in range(1, 7))
            print(f"    [sim] {'P' if pump else '.'}|{bars}  {describe_byte(byte)}")
        return True


class BleAromaDevice(AromaDevice):
    """
    Real device. Reconnects on its own in the background: a BLE dropout never
    brings the pipeline down, bursts are simply skipped until the link is back.
    """

    def __init__(self, name: str = DEVICE_NAME, address=None, scan_timeout: float = 10.0):
        self.name = name
        self.address = address
        self.scan_timeout = scan_timeout
        self.connected = False
        self._client = None
        self._supervisor = None
        self._watchdog = None
        self._lock = asyncio.Lock()
        self._closing = False
        self._last_on_at = None

    # --- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self._closing = False
        self._supervisor = asyncio.create_task(self._keep_connected(), name="ble-supervisor")
        self._watchdog = asyncio.create_task(self._watch(), name="ble-watchdog")

    async def close(self) -> None:
        self._closing = True
        try:
            await self.all_off()
        except Exception:
            pass
        for task in (self._supervisor, self._watchdog):
            if task is not None:
                task.cancel()
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self.connected = False

    async def wait_connected(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connected:
                return True
            await asyncio.sleep(0.2)
        return self.connected

    # --- connection ------------------------------------------------------
    async def _keep_connected(self) -> None:
        from bleak import BleakClient, BleakScanner

        backoff = 1.0
        while not self._closing:
            if self.connected:
                await asyncio.sleep(1.0)
                continue
            try:
                if self.address:
                    target = self.address
                else:
                    log.info("Scanning for BLE device %r...", self.name)
                    dev = await BleakScanner.find_device_by_name(
                        self.name, timeout=self.scan_timeout)
                    if dev is None:
                        raise RuntimeError(
                            f"{self.name!r} not found (powered on? switch up? in range?)")
                    target = dev.address

                client = BleakClient(target, disconnected_callback=self._on_disconnect)
                await client.connect()
                self._client = client
                self.connected = True
                backoff = 1.0
                log.info("Connected to the diffuser (%s)", target)
                await self.all_off()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("BLE connection failed (%s) - retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _on_disconnect(self, _client) -> None:
        if self.connected and not self._closing:
            log.warning("BLE link lost - reconnecting")
        self.connected = False
        self._client = None

    async def _watch(self) -> None:
        """Safety net: cut everything if outputs stay active for too long."""
        while not self._closing:
            await asyncio.sleep(WATCHDOG_PERIOD)
            started = self._last_on_at
            if started and time.monotonic() - started > MAX_BURST_SECONDS + 2:
                log.error("Watchdog: outputs active far too long, forcing off")
                await self.all_off()

    # --- writing ---------------------------------------------------------
    async def apply(self, pump: bool, valves) -> bool:
        byte = pack_byte(pump, valves)
        async with self._lock:
            client = self._client
            if not self.connected or client is None:
                return False
            try:
                await client.write_gatt_char(
                    CHARACTERISTIC_UUID, bytes([byte]), response=True)
            except Exception as exc:
                log.warning("BLE write failed (%s)", exc)
                self.connected = False
                self._client = None
                return False
            self._last_on_at = time.monotonic() if byte else None
            log.debug("-> %s", describe_byte(byte))
            return True


def make_device(mock: bool = False, address=None, name: str = DEVICE_NAME,
                verbose: bool = True) -> AromaDevice:
    if mock:
        return MockAromaDevice(verbose=verbose)
    return BleAromaDevice(name=name, address=address)
