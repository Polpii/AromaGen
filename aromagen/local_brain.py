"""
Mood classification with a small language model running on this machine.

The exhibition venue has no internet, so the room's mood has to be judged
locally. Measured against Claude Haiku (bench_vibe.py, and an evening simulated
through the real vote policy):

    labelled 2-3 line snippets            Qwen2.5-1.5B 48/48    Haiku 48/48
    correct votes over the evening        Qwen, 15 s   91%      Haiku, 45 s 72%
    scent matching the room (40 s grace)  Qwen, 15 s   84%      Haiku, 45 s 76%

Those numbers were measured on an earlier three-way palette (positive, neutral,
negative). The vibes are now named by the piece, so re-measure the current ones
with tests/check_vibes.py.

The model is used as a SCORER, not a chatbot: it reads the prompt once and
compares how likely it finds each possible answer. One forward pass, no text
generation, so the reply can never be malformed, and the winning probability
doubles as the vote's confidence.

Two findings shaped the design:

  short windows  the model is excellent on a few lines and drifts on long, mixed
                 stretches -- it follows the last lines, or dilutes an emotional
                 moment inside logistics. So it judges about 15 s of talk and the
                 vote tracker does the long-term integration, which is exactly
                 what the tracker is for.
  offline loads  transformers 4.57 contacts the Hugging Face hub while loading a
                 tokenizer, even with local_files_only=True, and fails with no
                 network. Models are therefore always loaded from their resolved
                 local directory.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from .classify import Decision
from .recipes import AVAILABLE_RECIPES, DEFAULT_RECIPE

log = logging.getLogger("aromagen.local")

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# How much talk one local classification judges. On the simulated evening, 45 s
# windows gave 70% correct votes, 15 s gave 91% and 10 s 96%, with identical
# scent trajectories for 10 and 15 s. 15 s keeps enough lines for sarcasm that
# only lands in the reply.
LOCAL_WINDOW_SECONDS = 15.0

# Measured with Whisper loaded and a busy 15 s window (422 prompt tokens): the
# model peaks at 2.93 GiB, and 3.46 GiB is free once Whisper has taken its share.
# Below this a warning is logged, but the GPU is still tried: there is no
# acceptable alternative to fall back to (see NO_GPU_MESSAGE).
EXPECTED_VRAM_BYTES = int(3.1 * 2**30)

# On this laptop's CPU the same model takes ~24 s per vote on a 15 s window: the
# window is stale long before the vote lands, and every decision waits for it.
# That is worse than the instant keyword lexicon, so the CPU is never chosen
# automatically -- only when a caller asks for it explicitly.
NO_GPU_MESSAGE = ("no usable GPU for the local model; on the CPU it takes ~24 s per "
                  "vote, which is unusable live")

# The word the model answers with for a vibe, when its name misleads. The vibes
# are named by the piece, and a small model reads those names literally:
# "solitude" pulled loneliness -- which belongs to confession -- towards itself.
# Measured on 46 snippets (samples/vibe_eval_*.jsonl) and 9 plain agreements:
#     confession / solitude / resonance   39/46
#     sad / neutral / happy               43/46
#     heavy / neutral / warm              45/46   agreement 5/9  ("warm" != agreeing)
#     confession / neutral / resonance    45/46   agreement 9/9
# A vibe absent from this table answers with its own name.
MODEL_WORDS = {
    "solitude": "neutral",
}


def model_word(vibe: str) -> str:
    return MODEL_WORDS.get(vibe, vibe)


# One line per vibe, injected into the prompt. Rewording is fine, but re-run
# tests/check_vibes.py afterwards: a 1.5B model is far more sensitive to phrasing
# than Claude is. A vibe absent from this table falls back to its recipe theme.
MOOD_HINTS = {
    "confession": "anything sad, heavy or vulnerable: an admission, a secret, fear, "
                  "regret, shame, grief, loss, feeling lonely, disappointment, "
                  "bad news.",
    "solitude": "flat, practical and neutral, nothing felt: times, directions, "
                "logistics, plain greetings, silence.",
    "resonance": "anything happy, warm or connecting: joy, excitement, laughter, "
                 "good news, congratulations, affection, compliments, agreement, "
                 "feeling the same.",
}
RULES = ("Judge the feeling in the exchange. Anything sad or vulnerable is "
         "{confession}, even when it is said lightly or is about someone else. "
         "Anything happy, warm or connecting is {resonance}, however small the "
         "subject: good news, congratulations, shared laughter, a cheerful remark "
         "someone picks up, right, same, exactly, me too. {Solitude} only when the "
         "talk is flat and practical, with no feeling either way, or when nothing "
         "is said.")


def build_system_prompt(labels) -> str:
    """The benchmarked prompt, extended from recipe themes for any other mood."""
    themes = {r.key: r.theme for r in AVAILABLE_RECIPES}
    labels = list(labels)
    words = [model_word(k) for k in labels]
    answer = words[0] if len(words) == 1 else \
        ", ".join(words[:-1]) + f", or {words[-1]}"
    lines = [
        "You judge the overall mood of a short stretch of conversation overheard "
        "in a room. It may be in English, French, or both.",
        "",
        f"Answer with exactly one word: {answer}.",
    ]
    lines += [f"- {w}: {MOOD_HINTS.get(k) or themes.get(k, k)}"
              for k, w in zip(labels, words)]
    names = {k: model_word(k) for k in MOOD_HINTS}
    names.update({k.capitalize(): model_word(k).capitalize() for k in MOOD_HINTS})
    lines += ["", RULES.format(**names)]
    return "\n".join(lines)


def local_model_dir(repo: str = DEFAULT_LOCAL_MODEL) -> str:
    """
    The model's cached directory, resolved without touching the network.

    Raises with an actionable message when the model was never fetched, rather
    than letting a connection error surface mid-startup at the venue.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Local model {repo!r} is not on this machine. Run "
            f"`python prepare_offline.py` once while connected to the internet."
        ) from exc


class LocalClassifier:
    """Same interface as ClaudeClassifier: `await classify(window) -> Decision`."""

    name = "local"

    def __init__(self, repo: str = DEFAULT_LOCAL_MODEL, device: str = None,
                 max_prompt_tokens: int = 700, labels=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.torch = torch
        self.repo = repo
        self.device = device or self._pick_device()
        self.max_prompt_tokens = max_prompt_tokens
        # `labels` lets a benchmark score the model on a different set of vibes
        # than the palette currently mounted on the device.
        self.labels = tuple(labels) if labels else tuple(r.key for r in AVAILABLE_RECIPES)
        self.system = build_system_prompt(self.labels)
        # The engine awaits one classification at a time, but the model itself
        # is not re-entrant: never let two calls share it.
        self._lock = threading.Lock()

        path = local_model_dir(repo)
        dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype).to(self.device).eval()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "not enough GPU memory for the local model next to Whisper; close "
                "anything else using the graphics card (browsers, other scripts) and "
                "restart") from exc
        self.label_ids = [self.tok(model_word(k), add_special_tokens=False).input_ids
                          for k in self.labels]
        self.single_token = all(len(ids) == 1 for ids in self.label_ids)
        log.info("Local model %s ready on %s in %.1fs", repo.split("/")[-1],
                 self.device.upper(), time.time() - t0)

    @property
    def description(self) -> str:
        return f"{self.repo.split('/')[-1]} ({self.device})"

    def _pick_device(self) -> str:
        """
        The GPU, or nothing. An explicit device="cpu" still works, for benchmarks.

        This used to fall back to the CPU below a free-VRAM threshold. Measured,
        that threshold sat at 3.50 GiB while 3.46 GiB is free in normal use -- so
        the brain silently landed on a CPU where each vote takes ~24 s.
        """
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError(NO_GPU_MESSAGE)
        free, _total = torch.cuda.mem_get_info()
        if free < EXPECTED_VRAM_BYTES:
            log.warning("Only %.2f GiB of VRAM free, the local model needs about %.2f: "
                        "trying the GPU anyway; close other programs using it if "
                        "loading fails", free / 2**30, EXPECTED_VRAM_BYTES / 2**30)
        return "cuda"

    # --- prompt ----------------------------------------------------------
    @staticmethod
    def _lines(window) -> list:
        now = time.time()
        return [f"[{now - getattr(u, 'started_at', now):4.0f}s ago] {u}" for u in window]

    def _prompt_ids(self, lines) -> list:
        """Tokenised prompt, dropping the oldest lines if it grows too long."""
        while True:
            messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": "Transcript:\n"
                 + "\n".join(f"- {line}" for line in lines)},
            ]
            text = self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
            ids = self.tok(text, add_special_tokens=False).input_ids
            if len(ids) <= self.max_prompt_tokens or len(lines) <= 1:
                return ids
            lines = lines[1:]

    # --- scoring ---------------------------------------------------------
    def _score_single_token(self, ids):
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        # Only the last position's logits are needed; the rest would be ~150 MB
        # of vocabulary-sized activations for nothing.
        logits = self.model(input_ids=x, logits_to_keep=1).logits[0, -1].float()
        return logits.log_softmax(-1)[[label[0] for label in self.label_ids]]

    def _score_multi_token(self, ids):
        """Prompt + each answer in one batch; right padding keeps positions exact."""
        torch = self.torch
        seqs = [ids + label for label in self.label_ids]
        width = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None \
            else self.tok.eos_token_id
        input_ids = torch.full((len(seqs), width), pad, device=self.device)
        mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
            mask[i, :len(seq)] = 1
        hidden = self.model.model(input_ids=input_ids,
                                  attention_mask=mask).last_hidden_state
        totals = []
        for i, label in enumerate(self.label_ids):
            pos = torch.arange(len(ids) - 1, len(ids) - 1 + len(label),
                               device=self.device)
            logprobs = self.model.lm_head(hidden[i, pos]).float().log_softmax(-1)
            totals.append(logprobs[torch.arange(len(label)), torch.tensor(label)].sum())
        return torch.stack(totals)

    def classify_sync(self, window) -> Decision:
        torch = self.torch
        t0 = time.monotonic()
        lines = self._lines(window)
        if not lines:
            return Decision(DEFAULT_RECIPE, 0.25, 0.35, "nothing to judge",
                            self.name, 0.0).clamp()
        ids = self._prompt_ids(lines)
        with self._lock, torch.inference_mode():
            try:
                scores = (self._score_single_token(ids) if self.single_token
                          else self._score_multi_token(ids))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                # Surfaces as a classifier failure: the fallback lexicon takes
                # this vote and the circuit breaker counts it.
                raise RuntimeError("GPU out of memory in the local model") from exc
        probs = scores.softmax(0).tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        spread = " / ".join(f"{k[:3]} {p:.0%}" for k, p in zip(self.labels, probs))
        return Decision(self.labels[best], probs[best], probs[best], spread,
                        self.name, time.monotonic() - t0).clamp()

    async def classify(self, window) -> Decision:
        # ~0.15 s on the GPU, longer on the CPU: run it off the event loop so the
        # dashboard, the BLE link and the audio queue keep moving meanwhile.
        return await asyncio.to_thread(self.classify_sync, list(window))
