# AromaGen

Diffuses a scent chosen from the conversation happening in the room.

Microphone -> live transcription (Whisper) -> mood vote (a local LLM) ->
olfactory recipe -> smell1 diffuser (ESP32-C3, 6 valves + 1 pump, over BLE).

English is the primary language; French works too. The firmware and reference
BLE client live in `AromaGen_MOI/` (clone of
[AwuChen/AromaGen_MOI](https://github.com/AwuChen/AromaGen_MOI)).

**Everything runs on the laptop, with no internet** -- the exhibition venue has
none. See [The brain](#the-brain-on-this-machine-no-internet) and
`prepare_offline.py`.

## Ingredients and recipes

The palette changes often, so it is not duplicated here -- it would only go
stale. Print the live one:

```bash
python -m aromagen --recipes
```

Cartridge wiring lives in [aromagen/ingredients.py](aromagen/ingredients.py),
blends in [aromagen/recipes.py](aromagen/recipes.py). Those are the only two
files to touch. A recipe calling for a cartridge that is not fitted is refused
outright rather than quietly opening the wrong valve, and the startup log says
how many recipes are actually available.

There are three vibes, **one ingredient each**, so that every vibe is a single
unmistakable material rather than a blend a visitor has to unpick:

| Vibe | Pump | What it is |
|---|---|---|
| `confession` | 1 | anything sad or vulnerable: an admission, a fear, a regret, grief, loneliness, bad news |
| `solitude` | 2 | flat and neutral: logistics, directions, plain greetings, silence |
| `resonance` | 3 | anything happy or connecting: joy, laughter, good news, agreement, feeling the same |

`solitude` doubles as the resting scent: after 90 s without speech it comes back
and is revived every 45 s, quietly, so an empty room never goes odourless. It
takes the room back **slowly** (see
[The room's mood](#the-rooms-mood-not-the-last-sentence)), so confession and
resonance linger.

Which oil sits on which pump is declared in
[aromagen/ingredients.py](aromagen/ingredients.py) and is yours to choose --
recipes address pumps, not perfumes, so swapping a cartridge never touches
`recipes.py`. Pumps 4, 5 and 6 are free.

### Dosing without a flow meter

The device can only open or shut a valve. Dosing is therefore done by **open
time**, with valves running **one at a time**, each held open for a slice
proportional to its dose. A recipe of 3 + 2 + 1 parts over a 10 s burst opens
its three valves for 5 s, 3.3 s and 1.7 s in turn. Notes run in increasing dose
order, so the dominant one comes last; they mix downstream, in the tube and in
the room. `python run_neutral.py` prints the exact plan before it starts.

**Why only one valve at a time:** measured on the hardware, pump + 1 valve is
rock solid, but pump + 3 valves together collapses the supply rail and reboots
the ESP32 (details in [NOTES-DEVICE.md](NOTES-DEVICE.md)). Proportions are
identical either way, since what matters is the volume of air pushed through
each cartridge.

## Install

```bash
pip install bleak sounddevice faster-whisper transformers torch numpy rich
python prepare_offline.py        # once, while online: fetch and verify every model
```

Run everything with the interpreter that has these packages -- `py -3.13` on
this machine. A virtual environment from another project shadowing `python` is a
real trap: the pipeline then dies with `No module named torch`, and only the
commands that need no models keep working.

`torch` must be a CUDA build (this laptop uses `2.6.0+cu124`): both models run on
the GPU, and the local brain refuses the CPU (see below).

`prepare_offline.py` downloads Whisper and the local language model (about 5 GB
in all), then proves the installation starts in a child process that genuinely
cannot reach the network. Run it before leaving for the venue: a pass means the
missing wifi cannot stop the piece from starting.

Claude remains an option where there is internet (`--brain claude`):
`pip install anthropic`, and put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file
at the project root (git-ignored).

## Usage

```bash
python -m aromagen --recipes                     # catalogue
python -m aromagen --list-audio                  # pick a microphone
python -m aromagen --test-valves                 # hardware test, valve by valve
python -m aromagen --text "I've had enough"      # classify one text, no mic
python -m aromagen --replay conversations_test.txt   # replay a conversation
python -m aromagen                               # live
```

`--mock` can be added to any command to run everything **without hardware**:
valve states are printed to the console.

### Language

**English or French, decided per utterance.** Speakers switch languages
mid-conversation, so the language is never locked for a session: `--lang auto`
(the default) lets Whisper judge each utterance, and its verdict is then
constrained to English or French before use.

That constraint matters. On real takes from this room Whisper's raw detection
has returned Japanese, Chinese, Russian and Hebrew -- sometimes with a
probability as low as 0.17. Forcing the wrong language is not a small error:
French audio decoded as English comes back as fluent *invented* English. "Salut,
ça va ?" was transcribed as "Thank you so much for watching."

`--lang en` or `--lang fr` force one and skip the detection pass, saving about
0.35 s per utterance. Only worth it if the room is genuinely monolingual.

### Accuracy

Word error rate against a read reference, on the takes in `samples/`, cut into
utterances exactly as the live pipeline cuts them:

| Setup | French | English | Per utterance |
|---|---|---|---|
| `small` on CPU, 420 ms segments | 26.1% | 6.8% | 1.87 s |
| `medium` on GPU | 17.4% | 9.1% | 0.33 s |
| **`large-v3-turbo` on GPU, 900 ms segments** | **0.0%** | **6.8%** | **0.74 s** |

Two changes account for that, and the second was the surprise:

**The GPU removes the tradeoff.** CTranslate2 runs on CUDA roughly seven times
faster than on this CPU, which makes the *largest* model both the most accurate
and the fastest option available. There is nothing left to trade.

**Segment length matters more than model size.** Whisper leans heavily on
context, and closing a segment after 420 ms of silence left it 1.4 s fragments
to work with. Waiting 900 ms grew them to 4.4 s and took French from 8.7% to
0.0% -- a perfect transcript -- for half a second of extra lag. That is why
"Ils vécurent heureux" used to come back as "Il les cure heureux".

Reproduce any of this with `python bench_asr.py`, or record new takes with
`python check_audio.py --record 25`. The remaining English error is a word-order
slip on "how's it going"; the whole-clip transcript has the same one.

### Quiet and distant voices

Visitors will not lean into the microphone. The original detector judged
loudness against the room's noise floor, and a voice 22 dB quieter than the
reference takes -- someone speaking softly a few metres away -- simply never
opened a segment: nothing reached Whisper at all.

Speech detection is now a hybrid ([SpeechSegmenter](aromagen/transcribe.py)): a
frame is speech if it is loud **or** if Silero, the small neural voice detector
bundled with faster-whisper, hears a voice in it. A segment is then transcribed
only if Silero heard at least 300 ms of speech in it, so knocks and fan noise
that merely were loud never reach Whisper and never turn into invented
sentences. Measured on the takes attenuated, with a faint noise floor added:

| Level | Loudness only | Hybrid, English | Hybrid, French |
|---|---|---|---|
| normal | caught | 7% | 2 - 11% |
| -16 dB | partial (57% / 93% error) | 5% | 67% |
| -22 dB | **nothing** | 23% | 89% |
| -28 dB | **nothing** | 34% | 100% |

Through the live pipeline at -22 dB *without* added noise, both takes come back
essentially word-perfect. So detection is no longer the limit -- **the ratio of
voice to room noise is**, and French suffers from it first. That part is
physical: put the microphone where people talk (a table centre, a ceiling
boundary mic above the bench), not by the diffuser or the fan.

Tried and rejected: a lower loudness floor (the same quiet voices, plus every
clatter), and boosting each segment's gain before Whisper (no improvement at
all). Silero costs about 11 ms of CPU per second of audio. If it cannot load, the
pipeline says so at startup and carries on with loudness alone.

`python check_audio.py` replays the exact same segmenter, so its cuts are the
live ones.

### GPU setup

Nothing to install. CTranslate2 needs cuBLAS and cuDNN, does not ship them, and
PyTorch does -- so `enable_cuda_libraries()` prepends PyTorch's `lib` directory
to `PATH` before ctranslate2 is imported. `os.add_dll_directory` is silently not
enough on Windows. Without a usable GPU the pipeline falls back to `small` on
the CPU on its own; `--device cpu` forces it.

### Latency budget

From the end of a sentence to the vote reaching the tracker, on this machine:

| Stage | Time |
|---|---|
| VAD hangover (deciding the sentence ended) | 0.90 s |
| Whisper `large-v3-turbo` on GPU, with language detection | 0.74 s |
| Engine tick | up to 0.4 s |
| Local brain on GPU, busy 15 s window | 0.16 s |
| **Total** | **~1.5 - 2.2 s** |

With Claude, the classification step alone took 1.5 to 5 s over the network.

## The brain: on this machine, no internet

The venue has no internet, so the mood is judged by
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
running on the laptop's GPU ([aromagen/local_brain.py](aromagen/local_brain.py)).
It is used as a scorer, not a chatbot: it reads the transcript once and compares
how likely it finds each vibe's answer word. No generated text to parse, and the
winning probability is the vote's confidence.

### How it was chosen

`python bench_vibe.py` compares classifiers on 48 labelled snippets
(`samples/vibe_eval.jsonl`: English, French and mixed, a third of them traps --
sarcasm, a warm memory of a funeral, an argument about a spreadsheet):

| Classifier | Accuracy | Traps | Per vote | Where |
|---|---|---|---|---|
| Claude Haiku (reference) | 100% | 100% | 1.5 s | internet |
| **Qwen2.5-1.5B, local** | **98%** | **92%** | **80 ms** | GPU |
| XLM-RoBERTa sentiment | 85% | 75% | 13 ms | GPU |
| multilingual-e5 embeddings | 85% | 67% | 13 ms | GPU |
| Qwen2.5-0.5B | 79% | 50% | 51 ms | GPU |

The dedicated sentiment models are fast but fall for sarcasm every time. Qwen
1.5B misses one French sarcastic line; the prototype without timestamps in its
prompt scored 48/48.

A clean labelled set flatters every model, so the decision was made on a harder
test: a scripted six-minute evening (arrival, reunion, logistics, an argument,
bad news, reconciliation, leaving) fed through the real vote policy:

| Setup | Correct votes | Scent matching the room (40 s grace) |
|---|---|---|
| Claude Haiku, 45 s window (the previous setup) | 72% | 76% |
| Qwen local, 45 s window | 70% | 73% |
| **Qwen local, 15 s window** | **91%** | **84%** |

The small model is excellent on a few lines and drifts on long mixed stretches:
it follows the last lines, or dilutes an emotional moment inside logistics. So
it judges **15 s** of talk, and the vote tracker does the long-term integration
-- which is what the tracker is for. With 45 s windows Haiku never went back to
neutral during the logistics after the reunion; the local brain did. This is one
scripted evening, labelled by the author: a strong signal, not proof. The
rehearsal is the real test.

### It must run on the GPU

| Local brain, busy 15 s window | Per vote |
|---|---|
| RTX 2060, next to Whisper | 0.16 s |
| CPU | ~24 s |

A vote that takes 24 s to judge a 15 s window arrives stale, and every decision
waits for it -- worse than the instant keyword lexicon. So the CPU is never
chosen automatically: without a usable GPU the pipeline says so loudly and runs
on the lexicon. The model fits next to Whisper with little spare: 2.93 GiB at
peak, 3.46 GiB free once Whisper is loaded. **At the venue, close anything else
that uses the graphics card** -- a browser with hardware acceleration can be
enough to push it out.

### The current vibes, measured

The tables above come from an earlier three-way palette (positive, neutral,
negative). The piece now uses `confession`, `solitude` and `resonance`, named by
the work rather than by sentiment, so they were measured again:

```bash
python tests/check_vibes.py --show
```

The vibes follow feeling: **sad or vulnerable is confession, happy or connecting
is resonance, flat and neutral is solitude.** Two sets, English and French,
traps included (sadness about someone else, good news from one voice, plain
agreement with no joy, an admission slipped in lightly):

| Set | Accuracy |
|---|---|
| `samples/vibe_eval_vibes.jsonl`, 32 snippets the prompt was tuned on | 32/32 |
| `samples/vibe_eval_heldout.jsonl`, 18 snippets never tuned on | 17/18 |

```bash
python tests/check_vibes.py --file samples/vibe_eval_heldout.jsonl
```

The one miss is understated sadness ("nobody came to my birthday -- it is fine,
whatever"), read as neutral.

**The model does not answer "solitude".** A small model reads the names
literally, and "solitude" pulled loneliness -- which belongs to confession --
towards itself. It answers `neutral` instead, mapped back to solitude
(`MODEL_WORDS` in local_brain.py). Measured on the 46 snippets: 39/46 answering
"solitude", 45/46 answering "neutral". Plain feeling words everywhere
(`sad`/`neutral`/`happy`) did worse, and `warm` for resonance lost plain
agreement ("right, I agree") to neutral.

Re-run both sets after renaming a vibe or rewriting a `theme`: those strings go
verbatim into the prompt, and a 1.5B model is sensitive to phrasing in a way
Claude is not.

### Choosing the brain

```bash
python -m aromagen                      # local brain (default)
python -m aromagen --brain claude       # Claude, needs internet
python -m aromagen --brain lexicon      # keywords only
```

By default the Hugging Face hub is never contacted, exactly as at the venue: a
missing model fails at once with a clear message instead of a network timeout in
front of visitors. `--online` lifts that.

After changing the recipes, or rewording the prompt in `local_brain.py`, re-run
`python bench_vibe.py --backends local`: a 1.5B model is far more sensitive to
phrasing than Claude is.

## The dashboard

Running the live or replay mode opens a live screen ([aromagen/ui.py](aromagen/ui.py)):
the conversation on the left with the detected language in brackets, the vote on
the right, and under it the scent being expressed with a light that is green
only while air is actually moving through the cartridges.

It reads the engine directly on every frame rather than keeping its own copy of
the state, so the screen cannot drift out of step with what is really happening.

`--plain` gives scrolling lines instead, which is what you want when piping to a
file; it is also chosen automatically when stdout is not a terminal. On a legacy
console that cannot render block characters, the bars fall back to ASCII rather
than crashing on the first frame.

## The room's mood, not the last sentence

A classification says what the last few seconds sounded like. Visitors
experience the room over minutes. So a classification is a **vote**, never an
order: [aromagen/vibe.py](aromagen/vibe.py) keeps a decaying average of them and
*that* is what gets diffused.

Votes fade with a 35 s half-life, so the estimate follows the room without a
window edge cutting evidence off abruptly. **The moment another vibe's bar
overtakes the current one, the piece switches** -- no waiting period:

| Guard | Default | Why |
|---|---|---|
| evidence | 0.8 total weight | about one confident vote, so an empty tracker cannot flip |
| share | 34% of the room | simply leading a three-way split |
| lead | 1.0x the current vibe | overtaking is enough, no head start required |
| dwell | 0 s | a new vibe may replace one expressed a moment ago |

How long a genuine change takes depends on how long the room held the previous
vibe, because its votes have to fade first (votes every 5 s, confidence 0.9):

| Previous vibe held for | Half-life 35 s (default) | Half-life 20 s |
|---|---|---|
| 15 s | 10 s | 5 s |
| 60 s | 25 s | 15 s |
| 5 min | 30 s | 15 s |

A single outlier sentence still cannot move a settled room. If the scent
flickers on site, raise `--half-life` or `--dwell`; if it still feels slow,
`--half-life 20`.

**Solitude comes back slowly.** Almost any uneventful stretch of talk votes for
it, so with symmetric rules it won back confession and resonance within seconds.
Three things slow its return, and only its return -- confession and resonance
still take over from each other, and from solitude, at once:

| Rule | Default | Setting |
|---|---|---|
| a solitude vote counts half | 0.5 | `resting_vote_weight` in `EngineConfig` |
| solitude must be twice as strong as the current vibe | 2.0x | `return_lead` in `EngineConfig` |
| confession or resonance holds at least | 45 s | `--rest-hold` |

| After confession held for | Solitude takes back after (before) |
|---|---|
| 5 s | 40 s (0 s) |
| 60 s | 65 s (25 s) |
| 5 min | 80 s (30 s) |
| confession resurfacing 1 vote in 3 | never (55 s) |

`python tests/check_rest_return.py` replays these.

Silence is explicit: nothing heard for 90 s clears the accumulator and returns
the room to `solitude`, so whatever was said before the room emptied cannot
colour the next conversation. The resting scent is then revived, quietly, every
45 s.

**A change of vibe sprays its pump at once.** If a burst of the previous vibe is
still going out, it is cut short and the new pump opens straight away (measured:
0.3 s). The cooldown between bursts only applies to repeats of the same vibe.
`python tests/check_vibe_switch.py` guards this.

**Classification keeps running while a burst diffuses.** Bursts go out as a
background task; a repeat of the same vibe decided meanwhile is queued for when
the burst ends. Two bursts never overlap -- the valves open one at a time, and
the power supply would not take more. `python tests/check_engine_bursts.py`
guards this (see the traps below for why it matters).

A **saturation guard** caps diffusion at 20 bursts per 10 minutes. Skipping a
burst is correct behaviour, not a failure; the console says when it fires, and
the run summary counts them.

### Tuning

| Option | Default | Effect |
|---|---|---|
| `--window` | 15 s local, 45 s Claude | how much talk one classification judges |
| `--interval` | 5 s | classification cadence |
| `--half-life` | 35 s | how fast old votes fade; lower = follows faster |
| `--dwell` | 0 s | minimum life of an expressed vibe; raise it if the scent flickers |
| `--silence` | 90 s | quiet before returning to `solitude` |
| `--rest-hold` | 45 s | minimum life of confession/resonance before solitude may return |
| `--burst` | 15 s | burst duration (max 20) |
| `--gain` | 1.0 | strength multiplier |
| `--cooldown` | 3 s | rest between bursts |
| `--max-bursts` | 20 | ceiling per 10 minutes |

`--fast` compresses every timing by five. It exists so a two-minute test shows
behaviour that normally unfolds over an hour -- **never run an installation with
it**.

The defaults assume **a fan clearing the air between bursts**: 15 s of diffusion
every 45 s, a 25% duty cycle, with a ceiling of 20 bursts per 10 minutes. Without
ventilation, halve `--burst` and space out `refresh_interval` in `EngineConfig`:
a scent that never clears saturates the room and stops being noticed.

## Running it for an exhibition

**Before leaving, while still online:**

```bash
python prepare_offline.py            # fetch every model, then prove it starts offline
```

**On site:**

```bash
python check_audio.py --record 25    # calibrate the microphone in the real room
python -m aromagen --test-valves     # every valve clicks, every cartridge is right
python supervise.py                  # run the piece, restarting it if it dies
```

**Keep the laptop plugged in**, and close everything else using the graphics
card (see [It must run on the GPU](#it-must-run-on-the-gpu)). Measured: on
battery this Max-Q card drops to 300 MHz of 2100 and a vote takes 1156 ms
instead of 156 ms. Overwolf, a game overlay that starts by itself, holds the GPU
permanently.

[supervise.py](supervise.py) runs `python -m aromagen` as a child process and
restarts it on any non-zero exit, with a backoff that grows from 5 s to 2 min and
resets once a run has survived five minutes. After ten short failures in a row it
stops and says so, rather than flapping in silence all evening. A clean exit is
never restarted. It touches nothing inside the pipeline, so it cannot introduce a
fault of its own. Logs land in `logs/aromagen.log` and rotate at 20 MB. Anything
after `--` is passed through: `python supervise.py -- --gain 0.8`.

Two things it cannot do: restart the diffuser if the ESP32 browns out -- that
needs the side switch toggled by hand -- and fix a microphone that was never
calibrated for the room. The VAD now hears quiet and distant voices (see
[Quiet and distant voices](#quiet-and-distant-voices)), but it was measured on
recordings, not on a crowd at three metres; a badly placed microphone still
fails silently.

### Cost and privacy

With the local brain there is no API bill, and nothing leaves the laptop:
transcripts live in memory, are pruned continuously, and are never written to
disk or sent anywhere. With `--brain claude` each classification is an API call,
one every few seconds for hours; `--interval` scales that bill almost linearly.

## Robustness

| Failure | Behaviour |
|---|---|
| No internet | nothing is lost: every model is local, and the hub is never contacted |
| A model was never downloaded | fails at startup, naming the model and `prepare_offline.py` |
| No usable GPU for the local brain | loud error, keyword lexicon instead -- never a 24 s CPU vote |
| Local brain runs out of GPU memory mid-run | that vote falls back to the lexicon; circuit breaker after 3 failures |
| Diffuser off or out of range | automatic reconnection with backoff; bursts are skipped, never queued |
| BLE link cut mid-burst | valves are shut as soon as it returns; hardware watchdog at 20 s |
| Whisper falls behind | oldest segments are dropped (a late scent is worthless) |
| Microphone unplugged | reopened every 5 s |
| Ctrl+C | a burst in flight shuts its valves, then the device closes |

### Traps hit along the way

**Whisper must be loaded on the main thread, first.** CTranslate2 initialises
badly from a secondary thread once PortAudio and the Windows BLE stack are
loaded: access violation, process killed, no Python traceback at all.
`__main__.py` therefore loads Whisper, then the local brain, before
`asyncio.run()`.

**The engine waits for the BLE link before listening.** `device.start()` only
launches the connection supervisor; without an explicit wait, the first burst
went out before the link existed and was lost. `--mock` hid the problem
entirely, the simulated device being ready instantly.

**Bursts blocked the decision loop.** Diffusion was awaited inline, so the room
went deaf for the whole 15 s of every burst -- while the code comments claimed
the opposite. With a 15 s classification window that is worse than a delay:
speech heard during a burst fell out of the window before being judged, and a
replayed argument passed without a single negative vote. Bursts now run in the
background; `tests/check_engine_bursts.py` failed before the fix and passes
after it.

**transformers contacts the hub even offline.** Version 4.57 calls the Hugging
Face API while loading a tokenizer, even with `local_files_only=True`, and dies
without a network. Models are loaded from their resolved local directory
instead.

**A download error looked like a GPU fault.** Whisper's CPU fallback caught every
exception, so with no network it reported a broken GPU, retried on the CPU --
which needed the network too -- and died with a misleading message. Resolving
model files now happens before, and apart from, device initialisation.

**The "safe" CPU fallback was a trap.** The local brain used to drop to the CPU
below 3.5 GiB of free VRAM. Measured, 3.46 GiB is free in normal use, so it
landed on a CPU where each vote takes about 24 s. It now uses the GPU, or
nothing.

## Status

**Hardware: validated.** The diffuser is found over BLE and all seven outputs
drive cleanly; the supply limit (one valve at a time) comes from that test. The
device only advertises when its side switch is up, and needs an off/on cycle
after a crash. See [NOTES-DEVICE.md](NOTES-DEVICE.md).

**Transcription: validated** on recorded takes -- 0% word error in French, 6.8%
in English, with per-utterance language detection.

**Offline: validated.** `prepare_offline.py` loads Whisper and the local brain in
a process whose network is cut, and classifies correctly there. The live
microphone mode, `--replay` and `--text` all run with every proxy pointing
nowhere.

**Mood classification: validated locally against Claude**, on a labelled set
(47/48) and on a scripted evening through the real vote policy, where the scent
matched the room 84% of the time against 76% for the previous Claude setup.

**Not yet verified:** a long live session with real visitors, the actual smells
in a ventilated room, and the microphone in the exhibition space.
