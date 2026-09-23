"""
Diagnose and tune the transcription chain.

Record once, then try as many settings as you like on that same recording --
comparing configurations against live speech is hopeless, because you never say
the same thing twice.

    python check_audio.py --record 20            # record 20 s, save, analyse
    python check_audio.py --wav samples/take1.wav        # re-analyse a take
    python check_audio.py --wav samples/take1.wav --models tiny,base,small,medium

What it reports:
  1. microphone level -- if the input is too quiet, nothing downstream can work;
  2. how the current VAD would cut that audio into utterances;
  3. what each model transcribes, both on the whole clip and on the VAD
     segments, so the cost of the segmentation itself is visible.

Say a couple of sentences you can check against, in a normal voice, from where
you would normally sit.
"""

from __future__ import annotations

import argparse
import logging
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
SAMPLES_DIR = Path(__file__).parent / "samples"


# --- recording ------------------------------------------------------------
def record(seconds: float, device=None) -> np.ndarray:
    import sounddevice as sd

    name = sd.query_devices(device if device is not None else sd.default.device[0])["name"]
    print(f"Recording {seconds:.0f}s from {name!r}")
    print("Speak normally, from where you would normally sit.\n")
    for i in range(3, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  GO\n", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=device)
    sd.wait()
    print("Done.\n")
    return audio[:, 0]


def save_wav(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes((pcm * 32767).astype("<i2").tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        if fh.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path} is {fh.getframerate()} Hz, expected {SAMPLE_RATE}")
        raw = fh.readframes(fh.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32767.0


# --- level check ----------------------------------------------------------
def report_levels(audio: np.ndarray) -> None:
    def dbfs(x):
        return 20 * np.log10(max(float(x), 1e-9))

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    # Loud frames stand in for speech; quiet ones for the room's noise floor.
    frames = audio[: len(audio) // 480 * 480].reshape(-1, 480)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    speech = float(np.percentile(frame_rms, 90))
    noise = float(np.percentile(frame_rms, 20))
    clipped = int(np.sum(np.abs(audio) > 0.99))

    print("--- microphone level " + "-" * 47)
    print(f"  duration        {len(audio) / SAMPLE_RATE:.1f}s")
    print(f"  peak            {dbfs(peak):+.1f} dBFS")
    print(f"  overall RMS     {dbfs(rms):+.1f} dBFS")
    print(f"  speech (p90)    {dbfs(speech):+.1f} dBFS")
    print(f"  noise floor     {dbfs(noise):+.1f} dBFS")
    print(f"  speech / noise  {dbfs(speech) - dbfs(noise):+.1f} dB")
    if clipped:
        print(f"  CLIPPED SAMPLES {clipped} -- input gain is too high")

    verdict = []
    if dbfs(peak) < -35:
        verdict.append("VERY quiet: raise the Windows input level, or move closer")
    elif dbfs(peak) < -20:
        verdict.append("quiet: raising the Windows input level would help")
    if dbfs(speech) - dbfs(noise) < 12:
        verdict.append("poor speech-to-noise: the room or the mic is the problem, "
                       "not the model")
    print("  verdict         " + ("; ".join(verdict) if verdict else "levels look fine"))
    print()


# --- segmentation ---------------------------------------------------------
def vad_segments(audio: np.ndarray, hangover_frames: int, max_segment_s: float,
                 preroll_frames: int, min_segment_s: float,
                 use_silero: bool = True) -> list:
    """
    Offline replay of the live VAD, so its cuts can be inspected. It runs the
    very same SpeechSegmenter as the live pipeline; use_silero=False gives the
    older energy-only detector, for comparison.
    """
    from aromagen.transcribe import SpeechSegmenter

    return SpeechSegmenter(use_silero=use_silero, hangover_frames=hangover_frames,
                           max_segment_s=max_segment_s, preroll_frames=preroll_frames,
                           min_segment_s=min_segment_s).segments(audio)


# --- transcription --------------------------------------------------------
def transcribe(model, audio, language, beam_size, initial_prompt=None):
    t0 = time.time()
    segs, _info = model.transcribe(
        audio, language=language, beam_size=beam_size,
        condition_on_previous_text=False, vad_filter=False,
        without_timestamps=True, initial_prompt=initial_prompt,
        no_speech_threshold=0.6)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", type=float, metavar="SECONDS",
                   help="record a new take from the microphone")
    p.add_argument("--wav", type=Path, help="analyse an existing 16 kHz take")
    p.add_argument("--mic", type=int, default=None, help="input device index")
    p.add_argument("--models", default="base,small",
                   help="comma-separated model sizes to compare")
    p.add_argument("--lang", default="en", choices=("en", "fr"))
    p.add_argument("--beam", type=int, default=5,
                   help="beam size (1 = greedy, what the live pipeline used)")
    p.add_argument("--hangover", type=int, default=14,
                   help="frames of silence that close a segment (14 = 420 ms)")
    p.add_argument("--max-segment", type=float, default=7.0)
    p.add_argument("--min-segment", type=float, default=0.4)
    p.add_argument("--preroll", type=int, default=10, help="frames kept before onset")
    p.add_argument("--prompt", default=None,
                   help="initial_prompt given to Whisper for context")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    if args.record:
        audio = record(args.record, args.mic)
        path = args.wav or SAMPLES_DIR / f"take-{time.strftime('%H%M%S')}.wav"
        save_wav(audio, path)
        print(f"Saved to {path}\n")
    elif args.wav:
        audio = load_wav(args.wav)
        print(f"Loaded {args.wav}\n")
    else:
        raise SystemExit("give --record SECONDS or --wav PATH")

    report_levels(audio)

    segments = vad_segments(audio, args.hangover, args.max_segment,
                            args.preroll, args.min_segment)
    total = sum(len(s) for _, s in segments) / SAMPLE_RATE
    print("--- how the VAD cuts it " + "-" * 44)
    print(f"  {len(segments)} segment(s), {total:.1f}s of speech kept out of "
          f"{len(audio) / SAMPLE_RATE:.1f}s")
    for start, seg in segments:
        print(f"    {start:5.1f}s  +{len(seg) / SAMPLE_RATE:4.1f}s")
    if segments:
        durations = [len(s) / SAMPLE_RATE for _, s in segments]
        print(f"  shortest {min(durations):.1f}s, median "
              f"{float(np.median(durations)):.1f}s, longest {max(durations):.1f}s")
        if float(np.median(durations)) < 1.5:
            print("  WARNING: very short segments. Whisper has little context to "
                  "work with;\n           this alone can wreck accuracy.")
    print()

    from faster_whisper import WhisperModel
    import os

    threads = max(4, (os.cpu_count() or 4) // 2)
    for size in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"--- model {size!r} (beam={args.beam}, lang={args.lang}) " + "-" * 30)
        model = WhisperModel(size, device="cpu", compute_type="int8",
                             cpu_threads=threads)

        whole, dt = transcribe(model, audio, args.lang, args.beam, args.prompt)
        print(f"  WHOLE CLIP ({dt:.1f}s):")
        print(f"    {whole or '(nothing)'}")

        print(f"  PER VAD SEGMENT (what the live pipeline actually sees):")
        spent = 0.0
        for start, seg in segments:
            text, dt = transcribe(model, seg, args.lang, args.beam, args.prompt)
            spent += dt
            print(f"    {start:5.1f}s {len(seg) / SAMPLE_RATE:4.1f}s "
                  f"[{dt:4.2f}s] {text or '(nothing)'}")
        if segments:
            print(f"    -> {spent / len(segments):.2f}s per utterance on average")
        print()
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
