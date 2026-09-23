"""
Measure transcription accuracy and speed, model by model, on recorded takes.

Accuracy is word error rate (WER) against the reference text that was read
aloud: substitutions + deletions + insertions, divided by the number of
reference words. Lower is better; 0.0 is a perfect transcript.

    python bench_asr.py                                  # all default configs
    python bench_asr.py --models small,large-v3-turbo
    python bench_asr.py --device cpu                     # compare without GPU

Two modes are reported for every model:

    whole    the take transcribed in one go -- Whisper at its best, with full
             context, and the ceiling any segmentation can aspire to;
    segments the take cut by the live VAD and transcribed piece by piece, which
             is what the running pipeline actually does. The gap between the
             two is the price of chopping speech into fragments.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import site
import sys
import time
import unicodedata
import wave

import numpy as np

SAMPLE_RATE = 16_000
HERE = pathlib.Path(__file__).parent


def enable_cuda_libraries() -> bool:
    """
    Put PyTorch's bundled cuBLAS/cuDNN on the DLL search path.

    CTranslate2 needs them and does not ship them; PyTorch does. This MUST run
    before ctranslate2 is imported, and PATH is the only mechanism that works --
    os.add_dll_directory is not enough.
    """
    for base in site.getsitepackages():
        lib = pathlib.Path(base) / "torch" / "lib"
        if (lib / "cublas64_12.dll").is_file() or (lib / "libcublas.so.12").is_file():
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


enable_cuda_libraries()

REFERENCES = {
    "samples/take-en.wav": (
        "en",
        "Hey, how's it going? I just got off the train. "
        "Right, let's look at the project budget before the meeting. "
        "Oh come on, that is ridiculous, this is completely your fault. "
        "Wait, look, it actually works! That's amazing, well done everyone."
    ),
    "samples/take-fr.wav": (
        "fr",
        "Salut, ça va ? Je viens d'arriver, il est quelle heure là ? "
        "Bon, on regarde le budget du projet avant la réunion. "
        "Non mais n'importe quoi, c'est complètement ta faute si on est en retard. "
        "Attends, regarde, ça marche vraiment ! C'est génial, bravo."
    ),
}


# --- scoring --------------------------------------------------------------
def normalise(text: str) -> list:
    """Lowercase, strip accents and punctuation -- compare words, not typography."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0
    # Levenshtein over words, one row at a time.
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (r != h)))  # substitution
        previous = current
    return previous[-1] / len(ref)


# --- audio ----------------------------------------------------------------
def load_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        raw = fh.readframes(fh.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32767.0


def vad_segments(audio: np.ndarray) -> list:
    """Cut the take exactly the way the live VAD would."""
    from check_audio import vad_segments as cut
    from aromagen.transcribe import (HANGOVER_FRAMES, MAX_SEGMENT_S,
                                     MIN_SEGMENT_S, PREROLL_FRAMES)
    return cut(audio, HANGOVER_FRAMES, MAX_SEGMENT_S, PREROLL_FRAMES, MIN_SEGMENT_S)


# --- benchmark ------------------------------------------------------------
def run(model, audio, language, beam):
    t0 = time.time()
    segs, _info = model.transcribe(
        audio, language=language, beam_size=beam,
        condition_on_previous_text=False, vad_filter=False,
        without_timestamps=True, no_speech_threshold=0.6)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, time.time() - t0


def main() -> int:
    import logging
    logging.getLogger("faster_whisper").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="small,medium,large-v3-turbo,large-v3")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--compute", default=None,
                   help="float16 on GPU, int8 on CPU unless overridden")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--show", action="store_true", help="print every transcript")
    args = p.parse_args()

    compute = args.compute or ("float16" if args.device == "cuda" else "int8")
    threads = max(4, (os.cpu_count() or 4) // 2)

    takes = []
    for rel, (lang, reference) in REFERENCES.items():
        path = HERE / rel
        if not path.is_file():
            print(f"missing {rel}, skipping")
            continue
        audio = load_wav(path)
        takes.append((rel, lang, reference, audio, vad_segments(audio)))
    if not takes:
        print("No takes found. Record some with check_audio.py --record 25")
        return 1

    from faster_whisper import WhisperModel

    print(f"\ndevice={args.device} compute={compute} beam={args.beam}\n")
    header = f"{'model':18} {'take':6} {'mode':9} {'WER':>7} {'per utt':>9}"
    print(header)
    print("-" * len(header))

    results = {}
    for size in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            t0 = time.time()
            model = WhisperModel(size, device=args.device, compute_type=compute,
                                 cpu_threads=threads)
            load_time = time.time() - t0
        except Exception as exc:
            print(f"{size:18} could not load: {type(exc).__name__}: {str(exc)[:80]}")
            continue

        for rel, lang, reference, audio, segments in takes:
            take = pathlib.Path(rel).stem.replace("take-", "")

            whole, dt = run(model, audio, lang, args.beam)
            wer = word_error_rate(reference, whole)
            print(f"{size:18} {take:6} {'whole':9} {wer:6.1%} {dt:8.2f}s")
            if args.show:
                print(f"      {whole}")

            texts, spent = [], 0.0
            for _start, seg in segments:
                text, dt = run(model, seg, lang, args.beam)
                texts.append(text)
                spent += dt
            joined = " ".join(t for t in texts if t)
            wer_seg = word_error_rate(reference, joined)
            per_utt = spent / max(len(segments), 1)
            print(f"{size:18} {take:6} {'segments':9} {wer_seg:6.1%} {per_utt:8.2f}s")
            if args.show:
                print(f"      {joined}")
            results[(size, take)] = (wer, wer_seg, per_utt, load_time)
        del model
        print()

    if results:
        print("Best by segment WER (what the live pipeline gets):")
        ranked = sorted(results.items(), key=lambda kv: kv[1][1])
        for (size, take), (_w, wer_seg, per_utt, _l) in ranked[:8]:
            print(f"  {wer_seg:6.1%}  {per_utt:6.2f}s/utt   {size} [{take}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
