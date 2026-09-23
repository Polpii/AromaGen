# Device notes — from the creator's video

Source: Zoom meeting of 12 August, ~17 min, Awu Chen (MIT).
Raw transcript: `AromaGen_MOI/transcript.txt`.

---

## 1. Operating essentials

### Powering up

1. **Power it**: USB-C cable into the port **on the side of the case**. An LED
   lights up. *Power alone does not start the device.*
2. **Slide the switch UP.** It is a small slide switch on the case. Down = off.

> "The way the device turns on is we have a little switch here... You can turn
> on by sliding it up this angle." / "You can turn it off again by sliding it
> down here."

### If BLE finds nothing

A failure the creator describes himself, with his own remedy:

> "It says, cannot find the BLE thing. So usually when that happens, maybe it's
> not turned on correctly, so we'll turn it off and turn it on again. Now it has
> found the pump."

**Switch off, wait 2 s, switch back on.**

This version has no status LED strip: the only way to know the board booted is
to run a BLE scan.

> "This version doesn't have an LED strip, so you need to be running the demo
> for it to know if it's powered on correctly or not."

### If valves click but no scent comes out

Not a software bug: the air tube has come off the pump.

> "If the device is not working or the smell is not coming up, it's probably
> because this tube is disconnected from this air pump here. Usually double
> check that to make sure it's connected."

---

## 2. Power: two options

| Option | Detail |
|---|---|
| **External USB-C battery** (the creator's preference) | Plugs into the side port and "hangs there". No way to get it wrong. |
| **Small internal battery** | Slides into its slot, **metal side facing up**. |

**Watch out with the internal battery**: it can be inserted either way round.

> "It has like two sides to it. If you accidentally plug in this way, the system
> would crash basically, and it requires like a reboot of the board... this metal
> side needs to be facing up, and then we slide it in right here."

That risk is exactly why the creator favours the external battery. When
travelling, the battery goes in **carry-on luggage** (standard lithium rule).

---

## 3. What the device is made of

- **3D-printed case** (3D files available on request, modifiable).
- **PCB**: this is the second one manufactured; the first had a fault. This one
  is described as fully functional. Solidly soldered.
- **6 solenoid valves** + **1 air pump**, joined by a tube.
- **6 scent cartridges** grouped into a single block (the v0 individual
  cartridges were dropped: the block is easier to swap). They are labelled and
  close with **red caps**.

Principle: the pump pushes air, the open valve decides which cartridge it passes
through, and the scent leaves via the tube.

> "Whichever one is open, the air is going to pump through, and we have the
> smells that we're going to attach right here."

### The cartridges

They are **perfumer's blotter strips** dipped in essential oil, then slid into
the cartridge. So they are refillable and interchangeable per installation.

> "They have a little strip of like paper in it. These are like the testing
> papers for perfumers. We dip them in essential oil and then we put them in
> here."

---

## 4. The project behind the object

The research question, stated outright:

> "We're trying to find if there is an **RGB for smell**. Can we find the base
> smells that have the ability to generate all the smells in the world?"

They started from **12 base scents** and are working down to **6** — hence the
device's format. The six mounted ingredients are therefore not an arbitrary
choice: they are an intended **primary basis**, meant to be recombined.

That directly supports this project's recipe approach: blend the 6 primaries in
varying proportions to cover a space of moods.

Context: the object is going to **Paris** for an installation, with an artist
named **Gigi** involved.

Origin of the project, as told by the creator: his father has chronic sinus
trouble that costs him his sense of smell, except on rare days.

> "Once a year when he can smell, he'd be like, oh, this is the happiest day
> ever... So I've been thinking about how maybe there could be a device that can
> make it easier for him to smell."

Research started less than a year ago.

---

## 5. Assembly and disassembly

The creator takes the object apart on video to show the inside. Worth knowing if
it ever has to be reopened:

- The air tube slides into a hole in the case; let it go all the way down.
- **Closing the case is the fiddly part**: tuck the wires in before clipping it,
  and it holds once clipped.
- Space was deliberately left inside for repairs.
- In normal use there is **no reason to open it**. The video teardown was for
  documentation.

> "This is the hardest, trickiest part... Once it's in this shape, it's pretty
> durable."

---

## 6. Supply limit — measured here, absent from the video

**Only one valve can be open at a time.**

Measured 7 September, device powered over USB-C:

| Load | Result |
|---|---|
| Pump alone | fine |
| Pump + 1 valve, all six in a row | fine, no incident |
| Pump + 3 valves simultaneously | **immediate crash**: BLE link dropped, ESP32 rebooted and stopped advertising |

The inrush current of several solenoids at once drags the rail below the
microcontroller's operating threshold. Recovery requires an off/on cycle of the
switch.

Software consequence: a recipe is dosed **sequentially**, one valve after
another, each for a slice proportional to its dose
(`MAX_SIMULTANEOUS_VALVES = 1` in `aromagen/device.py`). Proportions are
unchanged, since what matters is the volume of air pushed through each cartridge.

To check later: the external battery probably delivers more current than a PC
USB port. If so, two simultaneous valves might hold — but nothing requires it,
the sequential version produces the same blend.

---

## 7. What the video does not say

- No mention of **flashing** the firmware, nor of a programming header. The side
  USB-C port carries power; nothing suggests it exposes data (tests on this
  machine confirm: no USB enumeration whatsoever).
- No schematic. The PCB repository referenced in the README
  (`sosucat/smell1-PCB1`) returns **404**.
- The **URGENEX** brand is never mentioned: most likely the external battery
  brought by the artist, not a component of the device.
