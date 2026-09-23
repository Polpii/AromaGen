"""
Live transcription of nearby conversation.

Chain: microphone -> voice activity detection (VAD) -> Whisper -> queue.

The VAD is a hybrid (see SpeechSegmenter): an energy detector with an adaptive
noise floor, OR Silero -- the small neural voice detector bundled with
faster-whisper. Visitors will not be close to the microphone, and energy alone
misses quiet or distant voices entirely; Silero hears them.

Language: English or French, decided per utterance, because speakers switch
between them mid-conversation. Whisper's own detection is unreliable on short
clips -- it has returned Japanese, Chinese, Russian and Hebrew on real takes from
this room -- so its verdict is constrained to English/French before use. Getting
it wrong is not a small error: French audio decoded as English comes back as
fluent invented English, not as a rough transcript.

Robustness contract:
  - the microphone runs in a real-time audio thread that ONLY copies frames;
    no heavy work happens in the callback;
  - Whisper runs in its own thread. If it falls behind, the oldest segments are
    dropped rather than building a backlog that would leave the scent trailing
    well behind what is being said;
  - a failure (unplugged mic, missing model) is logged and does not stop the
    program: the rest of the pipeline keeps running.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("aromagen.audio")

# The only two languages this installation supports. Speakers switch between
# them mid-conversation, so the language is decided PER UTTERANCE, never locked
# for the session: decoding English speech as French does not give a rough
# transcript, it gives invented French.
SUPPORTED_LANGUAGES = ("en", "fr")

# Beam search rather than greedy decoding. Measured on a French take: beam 5 was
# both more accurate ("budget du projet" vs "midi du projet") and no slower than
# beam 1, whose worst case was worse. There is no reason to stay greedy.
BEAM_SIZE = 5

# Best model for each backend, measured on real takes from this room (see
# bench_asr.py). Word error rate on French, cut into utterances:
#     small  on CPU   26.1%   1.87 s per utterance
#     small  on GPU   26.1%   0.14 s
#     medium on GPU   17.4%   0.33 s
#     large-v3-turbo  on GPU    8.7%   0.36 s   <- and 0.0% on longer segments
# The GPU makes the largest model both the most accurate AND the fastest option,
# so there is nothing to trade off.
DEFAULT_MODEL = {"cuda": "large-v3-turbo", "cpu": "small"}

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# --- VAD tuning -----------------------------------------------------------
NOISE_WINDOW_FRAMES = 200        # ~6 s of history to estimate the noise floor
SPEECH_FACTOR = 3.0              # how many times the noise floor counts as speech
ABSOLUTE_FLOOR = 0.006           # hard floor, stops silence from triggering
START_FRAMES = 3                 # consecutive voiced frames to open a segment
# 900 ms, not the 420 ms this started with. Whisper's accuracy depends heavily
# on context: measured on a French take, closing segments after 420 ms gave 8.7%
# word error, while waiting 900 ms gave 0.0% -- a perfect transcript -- because
# the segments grew from 1.4 s to 4.4 s. Half a second of extra lag buys all of
# that back. Waiting longer still adds lag without improving accuracy further.
HANGOVER_FRAMES = 30
PREROLL_FRAMES = 10              # ~300 ms kept from before the trigger
# Fragments shorter than this carry too little context to transcribe reliably.
MIN_SEGMENT_S = 1.0
# A monologue is cut here rather than waiting for a pause: any later and the
# sentence would arrive too late for the scent to still relate to it.
MAX_SEGMENT_S = 10.0

# --- quiet and distant voices ---------------------------------------------
# Measured by attenuating real takes (English and French) and adding a faint
# noise floor, to simulate someone speaking softly a few metres away:
#
#                        normal   -16 dB   -22 dB   -28 dB
#     energy only        caught   caught   NOTHING  nothing
#     energy OR Silero   caught   caught   caught   caught
#
# Word error with the hybrid: English 7 / 11 / 18 / 41%, French 2 / 72 / 87 / 96%.
# Whisper itself struggles with faint French; past that point only a closer or
# better microphone helps. Normal-level accuracy is unchanged.
#
# Lowering the energy floor instead was tried and rejected: it catches the same
# quiet voices but also every clatter in the room, and the digital gain boost
# (normalising each segment) did not improve the transcripts at all.
SILERO_THRESHOLD = 0.5
# A segment is only transcribed if Silero heard at least this much speech in it.
# Energy alone opens segments on clinks, footsteps and fan noise, from which
# Whisper invents sentences; this keeps them from reaching it.
SPEECH_GATE_MS = 300
SILERO_WINDOW = 512              # samples Silero judges at once (32 ms)
SILERO_CONTEXT = 64              # samples it needs from the previous window

# Whisper hallucinates these on silence or noise, in both languages. Drop them.
HALLUCINATIONS = re.compile(
    r"^\W*(sous-titres?.*|merci d.avoir regard.*|abonnez-vous.*|amara\.org.*|"
    r"thank you (so much |very much )?for watching.*|thanks? for watching.*|"
    r"subtitles? by.*|please subscribe.*|sous-titrage.*|"
    r"\.\.\.|musique|music|you|bye|\[.*\])\W*$",
    re.IGNORECASE,
)


@dataclass
class Utterance:
    """One transcribed turn of speech."""
    text: str
    started_at: float          # time.time() at the start of the audio segment
    duration: float
    confidence: float          # 0..1, derived from Whisper's no-speech probability
    language: str = ""         # detected language code, when available

    def __str__(self) -> str:
        return self.text


def enable_cuda_libraries() -> bool:
    """
    Put PyTorch's bundled cuBLAS/cuDNN on the DLL search path.

    CTranslate2 needs those libraries and does not ship them; PyTorch does. This
    has to happen before ctranslate2 is imported, and PATH is the only mechanism
    that works -- os.add_dll_directory is silently not enough on Windows.
    Without it, `device="cuda"` fails with "Library cublas64_12.dll is not found".
    """
    import site

    for base in site.getsitepackages():
        lib = Path(base) / "torch" / "lib"
        if (lib / "cublas64_12.dll").is_file() or (lib / "libcublas.so.12").is_file():
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            return True
    return False


def pick_device() -> tuple:
    """
    Return (device, compute_type): the GPU when one is usable, else the CPU.

    Worth the trouble -- on this hardware the GPU is roughly seven times faster,
    which is what makes the largest model affordable in real time.
    """
    enable_cuda_libraries()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception as exc:
        log.debug("CUDA unavailable (%s)", exc)
    return "cpu", "int8"


def hub_offline() -> bool:
    """True when the Hugging Face hub must not be contacted (always, at the venue)."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def resolve_whisper_files(model_size: str) -> str:
    """
    Local directory holding a Whisper model, downloading it only when allowed.

    With the hub offline this is a pure cache lookup, and a model that was never
    fetched fails immediately with an explicit message -- instead of a network
    timeout halfway through startup, far from any hint of the real cause.
    """
    from faster_whisper.utils import download_model

    try:
        return download_model(model_size, local_files_only=hub_offline())
    except Exception as exc:
        raise RuntimeError(
            f"Whisper model {model_size!r} is not available on this machine. "
            f"Run `python prepare_offline.py` once while connected to the "
            f"internet.") from exc


def load_model(model_size: str = None, device: str = None,
               compute_type: str = None, cpu_threads: int = 0):
    """
    Load the Whisper model. CALL THIS FROM THE MAIN THREAD, before opening the
    microphone and before starting the asyncio loop.

    CTranslate2 (the engine behind faster-whisper) initialises badly from a
    secondary thread once PortAudio and the Windows BLE stack are already
    loaded: the result is an access violation that kills the process with no
    Python traceback. Loading first, on the main thread, avoids it entirely.

    With no model size given, picks the best one the backend can afford.
    """
    if device is None:
        device, compute_type = pick_device()
    elif compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    model_size = model_size or DEFAULT_MODEL[device]

    from faster_whisper import WhisperModel

    # faster-whisper logs one INFO line per segment; far too chatty here.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

    threads = cpu_threads or max(4, (os.cpu_count() or 4) // 2)
    log.info("Loading Whisper %r on %s (%s)...", model_size, device.upper(), compute_type)
    t0 = time.time()
    # Resolving the files is the only step that may touch the network, so it is
    # kept apart from device initialisation. Mixed together, a venue with no
    # internet produced a download error that was caught as a "GPU fault",
    # triggered a CPU fallback that needed the network too, and died with a
    # misleading message.
    path = resolve_whisper_files(model_size)
    try:
        model = WhisperModel(path, device=device, compute_type=compute_type,
                             cpu_threads=threads)
    except Exception as exc:
        if device == "cpu":
            raise
        # A GPU that refuses to load must not take the whole installation down.
        log.warning("GPU unusable (%s) - falling back to the CPU", exc)
        model_size = DEFAULT_MODEL["cpu"]
        path = resolve_whisper_files(model_size)
        log.info("Loading Whisper %r on CPU (int8)...", model_size)
        model = WhisperModel(path, device="cpu", compute_type="int8",
                             cpu_threads=threads)
    log.info("Model ready in %.1fs", time.time() - t0)
    return model


def list_input_devices() -> list:
    import sounddevice as sd

    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            out.append((idx, dev["name"], dev["default_samplerate"]))
    return out


class StreamingSilero:
    """
    Silero speech probability, fed audio as it arrives.

    faster-whisper only exposes Silero for whole recordings. This drives the same
    ONNX session window by window, carrying its recurrent state (h, c) and the
    64-sample context between calls -- verified to give exactly the probabilities
    of the whole-recording call, for ~5 ms of CPU per second of audio.
    """

    def __init__(self):
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self._pending = np.zeros(0, dtype="float32")
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros(SILERO_CONTEXT, dtype="float32")
        self.prob = 0.0

    def push(self, samples: np.ndarray) -> float:
        """Add audio; return the probability of the latest complete window."""
        self._pending = np.concatenate([self._pending, samples.astype("float32")])
        while len(self._pending) >= SILERO_WINDOW:
            window = self._pending[:SILERO_WINDOW]
            self._pending = self._pending[SILERO_WINDOW:]
            out, self._h, self._c = self._session.run(None, {
                "input": np.concatenate([self._context, window])[None, :],
                "h": self._h, "c": self._c})
            self._context = window[-SILERO_CONTEXT:]
            self.prob = float(np.asarray(out).reshape(-1)[-1])
        return self.prob


class SpeechSegmenter:
    """
    Cut a stream of 30 ms frames into utterances. Used both live and by the
    offline tools (check_audio.py, bench_asr.py), so what is measured is exactly
    what runs.

    A frame counts as speech when it is loud against the room's noise floor OR
    when Silero hears a voice in it. Loudness catches clear nearby speech the
    instant it starts; Silero catches the quiet and distant voices that loudness
    misses. A finished segment is kept only if Silero heard at least
    SPEECH_GATE_MS of speech in it, so noise that merely was loud is dropped.

    If Silero cannot be loaded, or fails mid-evening, the segmenter carries on
    with energy alone -- the previous behaviour -- rather than stopping.
    """

    def __init__(self, use_silero: bool = True, hangover_frames: int = HANGOVER_FRAMES,
                 max_segment_s: float = MAX_SEGMENT_S,
                 preroll_frames: int = PREROLL_FRAMES,
                 min_segment_s: float = MIN_SEGMENT_S):
        self.hangover_frames = hangover_frames
        self.max_frames = int(max_segment_s * 1000 / FRAME_MS)
        self.min_samples = int(min_segment_s * SAMPLE_RATE)
        self.silero = None
        if use_silero:
            try:
                self.silero = StreamingSilero()
            except Exception as exc:
                log.warning("Silero voice detector unavailable (%s): quiet voices "
                            "will be missed, loud ones still work", exc)
        self._history = collections.deque(maxlen=NOISE_WINDOW_FRAMES)
        # Preroll entries are (frame, silero_heard_speech) so the speech gate
        # also counts the start of a sentence that arrived before the trigger.
        self._preroll = collections.deque(maxlen=preroll_frames)
        self._buffer = []
        self._speech_frames = 0
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._start = 0
        self.frames_seen = 0
        self.gated_segments = 0

    @property
    def uses_silero(self) -> bool:
        return self.silero is not None

    def _neural_speech(self, frame: np.ndarray) -> bool:
        if self.silero is None:
            return False
        try:
            return self.silero.push(frame) > SILERO_THRESHOLD
        except Exception as exc:
            log.error("Silero failed (%s): continuing with the energy detector only",
                      exc)
            self.silero = None
            return False

    def push(self, frame: np.ndarray):
        """
        Feed one frame. Returns (start_sample, audio) when an utterance ends,
        otherwise None. start_sample counts from the first frame ever pushed.
        """
        index = self.frames_seen
        self.frames_seen += 1

        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-9)
        self._history.append(rms)
        floor = (float(np.percentile(self._history, 20)) if len(self._history) > 30
                 else ABSOLUTE_FLOOR)
        loud = rms > max(floor * SPEECH_FACTOR, ABSOLUTE_FLOOR)
        neural = self._neural_speech(frame)
        voiced = loud or neural

        if not self._in_speech:
            self._preroll.append((frame, neural))
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run >= START_FRAMES:
                self._in_speech = True
                self._silence_run = 0
                self._buffer = [f for f, _n in self._preroll]
                self._speech_frames = sum(n for _f, n in self._preroll)
                self._start = index - len(self._buffer) + 1
                self._preroll.clear()
            return None

        self._buffer.append(frame)
        self._speech_frames += neural
        self._silence_run = 0 if voiced else self._silence_run + 1
        if self._silence_run >= self.hangover_frames or len(self._buffer) >= self.max_frames:
            return self._close()
        return None

    def flush(self):
        """End of stream: the utterance in progress, if any."""
        return self._close() if self._in_speech and self._buffer else None

    def _close(self):
        frames, speech_frames, start = self._buffer, self._speech_frames, self._start
        self._in_speech = False
        self._speech_run = 0
        self._buffer = []
        self._speech_frames = 0
        audio = np.concatenate(frames)
        if len(audio) < self.min_samples:
            return None
        if self.silero is not None and speech_frames * FRAME_MS < SPEECH_GATE_MS:
            self.gated_segments += 1
            log.debug("Dropped %.1fs of noise (Silero heard %d ms of speech)",
                      len(audio) / SAMPLE_RATE, speech_frames * FRAME_MS)
            return None
        return start * FRAME_SAMPLES, audio

    def segments(self, audio: np.ndarray) -> list:
        """Offline: cut a whole recording. Returns [(start_seconds, audio), ...]."""
        found = []
        for i in range(len(audio) // FRAME_SAMPLES):
            seg = self.push(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
            if seg is not None:
                found.append(seg)
        seg = self.flush()
        if seg is not None:
            found.append(seg)
        return [(start / SAMPLE_RATE, a) for start, a in found]


class LiveTranscriber:
    """
    Capture the microphone and push `Utterance` objects into an asyncio.Queue.

    Usage:
        t = LiveTranscriber(loop, out_queue, model=load_model())
        t.start()          # non-blocking
        ...
        t.stop()
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue,
                 model_size: str = None, language: str = "auto",
                 device_index=None, max_pending: int = 4,
                 compute_type: str = "int8", model=None):
        self.loop = loop
        self.out_queue = out_queue
        self.model_size = model_size
        # "auto" (the default) decides per utterance, constrained to English or
        # French. "en" / "fr" force one and skip the detection pass, which is
        # faster but wrong the moment somebody switches language mid-sentence.
        self.requested_language = language or "auto"
        self.forced_language = (None if self.requested_language == "auto"
                                else self.requested_language)
        self.device_index = device_index
        self.compute_type = compute_type

        self._frames: queue.Queue = queue.Queue(maxsize=400)             # mic -> VAD
        self._segments: queue.Queue = queue.Queue(maxsize=max_pending)   # VAD -> Whisper
        self._stop = threading.Event()
        self._threads = []
        self._stream = None
        # Ideally already loaded by the main thread (see load_model).
        self._model = model
        self.model_ready = threading.Event()
        if model is not None:
            self.model_ready.set()
        self.dropped_segments = 0
        self.frames_seen = 0
        # Built here, on the caller's thread, so a missing Silero is reported at
        # startup rather than from inside the audio thread.
        self._segmenter = SpeechSegmenter()

    @property
    def uses_silero(self) -> bool:
        return self._segmenter.uses_silero

    # --- start / stop -----------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        if self._model is None:
            # Fallback if the caller did not preload: less safe (see load_model),
            # but better than no transcription at all.
            self._spawn(self._load_model, "whisper-load")
        self._spawn(self._vad_loop, "vad")
        self._spawn(self._transcribe_loop, "whisper")
        self._spawn(self._capture_loop, "mic")

    def _spawn(self, target, name: str) -> None:
        th = threading.Thread(target=target, name=name, daemon=True)
        th.start()
        self._threads.append(th)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    # --- 1. microphone ----------------------------------------------------
    def _capture_loop(self) -> None:
        import sounddevice as sd

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("audio status: %s", status)
            try:
                # The copy is mandatory: sounddevice reuses the buffer.
                self._frames.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # the VAD fell behind; skip this frame

        while not self._stop.is_set():
            try:
                self._stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=FRAME_SAMPLES, device=self.device_index,
                    callback=callback)
                self._stream.start()
                log.info("Microphone open (%s)", sd.query_devices(
                    self.device_index if self.device_index is not None
                    else sd.default.device[0])["name"])
                while not self._stop.is_set() and self._stream.active:
                    time.sleep(0.25)
                return
            except Exception as exc:
                log.error("Microphone unavailable (%s) - retrying in 5 s", exc)
                time.sleep(5.0)

    # --- 2. voice activity detection --------------------------------------
    def _vad_loop(self) -> None:
        segmenter = self._segmenter

        def emit(seg) -> None:
            start_sample, audio = seg
            # Wall-clock time the utterance began: now, minus the audio pushed
            # since its first sample.
            behind = segmenter.frames_seen * FRAME_SAMPLES - start_sample
            self._emit_segment(audio, time.time() - behind / SAMPLE_RATE)

        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            self.frames_seen += 1
            seg = segmenter.push(frame)
            if seg is not None:
                emit(seg)

        seg = segmenter.flush()
        if seg is not None:
            emit(seg)

    def _emit_segment(self, audio: np.ndarray, started_at: float) -> None:
        duration = len(audio) / SAMPLE_RATE
        item = (audio, started_at, duration)
        try:
            self._segments.put_nowait(item)
        except queue.Full:
            # Whisper is behind: drop the oldest to stay close to the present.
            # A scent that arrives late is worse than no scent at all.
            try:
                self._segments.get_nowait()
                self.dropped_segments += 1
                log.warning("Transcription behind: dropped the oldest segment "
                            "(%d total)", self.dropped_segments)
            except queue.Empty:
                pass
            try:
                self._segments.put_nowait(item)
            except queue.Full:
                pass

    # --- 3. Whisper -------------------------------------------------------
    def _load_model(self) -> None:
        try:
            self._model = load_model(self.model_size, self.compute_type)
            self.model_ready.set()
        except Exception as exc:
            log.error("Could not load the model (%s)", exc)

    @staticmethod
    def _best_supported_language(info) -> str:
        """
        Pick English or French from Whisper's detection, ignoring every other
        language it may have preferred. Whisper happily returns Japanese for a
        second of French breathing; constraining the choice removes that whole
        class of nonsense.
        """
        probs = getattr(info, "all_language_probs", None)
        if probs:
            ranked = {lang: p for lang, p in probs if lang in SUPPORTED_LANGUAGES}
            if ranked:
                return max(ranked, key=ranked.get)
        detected = getattr(info, "language", "") or ""
        return detected if detected in SUPPORTED_LANGUAGES else SUPPORTED_LANGUAGES[0]

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio, started_at, duration = self._segments.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._model is None:
                continue
            try:
                # language=None makes Whisper detect this utterance on its own.
                # Costs about 0.35 s on the GPU, and is what lets a speaker
                # switch between English and French from one sentence to the
                # next. `_best_supported_language` then throws away anything
                # that is not English or French.
                segments, info = self._model.transcribe(
                    audio, language=self.forced_language, beam_size=BEAM_SIZE,
                    condition_on_previous_text=False, vad_filter=False,
                    without_timestamps=True, temperature=0.0,
                    no_speech_threshold=0.6)
                pieces, no_speech = [], []
                for seg in segments:
                    pieces.append(seg.text.strip())
                    no_speech.append(getattr(seg, "no_speech_prob", 0.0))
                language = self.forced_language or self._best_supported_language(info)

                # Detection is one pass, decoding is another: when Whisper
                # picked something outside English/French, its transcript was
                # produced in that wrong language, so redo it in the language we
                # actually allow. Rare, and far cheaper than emitting nonsense.
                detected = getattr(info, "language", "") or ""
                if self.forced_language is None and detected != language:
                    log.debug("Whisper guessed %r, redecoding as %r", detected, language)
                    segments, _info = self._model.transcribe(
                        audio, language=language, beam_size=BEAM_SIZE,
                        condition_on_previous_text=False, vad_filter=False,
                        without_timestamps=True, temperature=0.0,
                        no_speech_threshold=0.6)
                    pieces, no_speech = [], []
                    for seg in segments:
                        pieces.append(seg.text.strip())
                        no_speech.append(getattr(seg, "no_speech_prob", 0.0))
            except Exception as exc:
                log.warning("Transcription failed (%s)", exc)
                continue

            text = " ".join(p for p in pieces if p).strip()
            if not text or HALLUCINATIONS.match(text):
                continue
            confidence = 1.0 - (sum(no_speech) / len(no_speech) if no_speech else 0.0)
            utt = Utterance(text=text, started_at=started_at, duration=duration,
                            confidence=confidence, language=language)
            self.loop.call_soon_threadsafe(self._deliver, utt)

    def _deliver(self, utt: Utterance) -> None:
        try:
            self.out_queue.put_nowait(utt)
        except asyncio.QueueFull:
            log.warning("Output queue full, transcription dropped")
