"""
Compare mood classifiers on a labelled set of overheard room snippets.

The venue has no internet, so the question is whether a model running entirely
on this machine can classify the room as well as Claude does. This measures it
instead of guessing:

    python bench_vibe.py
    python bench_vibe.py --backends claude:claude-haiku-4-5,qwen-1.5b-fewshot
    python bench_vibe.py --show-errors

Backends (append @cpu to force the CPU, e.g. qwen-3b@cpu):

    claude:<model>     the online reference -- needs the network
    local              the pipeline's own LocalClassifier, exactly as it runs
    xlmr               multilingual sentiment model fine-tuned for exactly
                       positive / neutral / negative (XLM-RoBERTa base)
    e5                 multilingual embeddings compared against descriptions of
                       each mood and a handful of worked examples
    qwen-1.5b, qwen-3b small instruction-tuned LLMs, zero-shot
    ...-fewshot        the same, shown six worked examples first

The LLM backends never generate free text. They score the three possible
answers and pick the likeliest, so the output can never be malformed and the
probability of the winner doubles as the vote's confidence.

Local backends are loaded with the Hugging Face hub forced OFFLINE -- the
benchmark doubles as proof that they will load at the venue.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Simulate the venue before anything touches the hub: no network, cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).parent
EVAL_SET = HERE / "samples" / "vibe_eval.jsonl"
LABELS = ("positive", "neutral", "negative")

LLM_SYSTEM = """You judge the overall mood of a short stretch of conversation \
overheard in a room. It may be in English, French, or both.

Answer with exactly one word: positive, neutral, or negative.
- positive: warmth, laughter, enthusiasm, good news, affection, gratitude, relief.
- negative: conflict, blame, irritation, stress, sarcasm, sadness, grief, worry, \
bad news.
- neutral: logistics, practical or technical talk, small talk, nothing \
emotionally charged.

Judge the tone, not the subject: an argument about a spreadsheet is negative; a \
warm, laughing memory of someone who died is positive. Judge the whole stretch, \
not its sharpest line. Sarcasm is negative even when it uses positive words."""

# Worked examples for the few-shot variants. Deliberately NOT drawn from the
# evaluation set, so a few-shot score is not a memorisation score.
FEWSHOT = (
    (["honestly this was the best evening we've had in ages", "thank you both"],
     "positive"),
    (["on a fini à temps, je suis trop contente", "merci pour ton aide"], "positive"),
    (["oh wonderful, the printer is jammed again", "just what I needed today"],
     "negative"),
    (["laisse tomber, tu comprends jamais rien", "c'est toujours pareil avec toi"],
     "negative"),
    (["the files are in the shared folder", "second tab from the left"], "neutral"),
    (["on se retrouve devant la gare", "vers dix-huit heures"], "neutral"),
)

MOOD_DESCRIPTIONS = {
    "positive": "people enjoying themselves: laughter, warmth, enthusiasm, good "
                "news, affection, gratitude, relief",
    "neutral": "ordinary talk with no emotion: logistics, directions, schedules, "
               "practical or technical discussion, small talk",
    "negative": "tension or sorrow: conflict, blame, irritation, stress, sarcasm, "
                "sadness, grief, worry, bad news",
}


def local_path(repo: str) -> str:
    """
    The cached snapshot directory for `repo`, resolved without any network.

    Loading by repo id is NOT offline-safe in transformers 4.57: the tokenizer's
    `_patch_mistral_regex` calls the hub's model_info API even with
    local_files_only=True, and dies without a connection. Handing it a local
    directory skips that check entirely.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


def render(lines) -> str:
    return "Transcript:\n" + "\n".join(f"- {line}" for line in lines)


# --- backends -------------------------------------------------------------
class ClaudeBackend:
    """The online reference, through the pipeline's own classifier."""

    def __init__(self, model: str):
        from aromagen.classify import ClaudeClassifier
        from aromagen.env import load_env_file

        load_env_file()
        self.inner = ClaudeClassifier(model=model)
        # One loop for the whole run: the async HTTP client is bound to it.
        self.loop = asyncio.new_event_loop()

    def classify(self, lines):
        class _Line:
            def __init__(self, text):
                self.text, self.started_at = text, time.time()

            def __str__(self):
                return self.text

        decision = self.loop.run_until_complete(
            self.inner.classify([_Line(t) for t in lines]))
        return decision.recipe, decision.confidence

    def close(self):
        self.loop.close()


class XlmrBackend:
    repo = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        path = local_path(self.repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path).to(device).eval()
        if device == "cuda":
            self.model.half()
        names = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        # Older checkpoints only say LABEL_0..2; Cardiff's order is neg, neu, pos.
        if not set(LABELS) <= set(names.values()):
            names = {0: "negative", 1: "neutral", 2: "positive"}
        self.names = names
        self.torch = torch

    def classify(self, lines):
        enc = self.tok(" ".join(lines), return_tensors="pt", truncation=True,
                       max_length=512).to(self.device)
        with self.torch.inference_mode():
            probs = self.model(**enc).logits.float().softmax(-1)[0]
        best = int(probs.argmax())
        return self.names[best], float(probs[best])


class E5Backend:
    repo = "intfloat/multilingual-e5-base"
    temperature = 0.02

    def __init__(self, device: str):
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.model = SentenceTransformer(local_path(self.repo), device=device)
        # One prototype per mood: the mean of its description and its examples.
        prototypes = []
        for label in LABELS:
            texts = [f"passage: {MOOD_DESCRIPTIONS[label]}"]
            texts += [f"passage: {' '.join(lines)}"
                      for lines, lab in FEWSHOT if lab == label]
            emb = self.model.encode(texts, convert_to_tensor=True,
                                    normalize_embeddings=True)
            prototypes.append(torch.nn.functional.normalize(emb.mean(0), dim=0))
        self.prototypes = torch.stack(prototypes)

    def classify(self, lines):
        query = self.model.encode(f"query: {' '.join(lines)}", convert_to_tensor=True,
                                  normalize_embeddings=True)
        sims = self.prototypes @ query
        probs = (sims / self.temperature).softmax(0)
        best = int(probs.argmax())
        return LABELS[best], float(probs[best])


class LlmBackend:
    """
    A small instruction-tuned LLM used as a scorer, not a generator.

    It reads the prompt once and compares how likely it finds each of the three
    answers. That is one forward pass, cannot produce a malformed reply, and
    yields a proper probability for the confidence.
    """

    def __init__(self, repo: str, device: str, fewshot: bool):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.fewshot = fewshot
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        path = local_path(repo)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype).to(device).eval()
        self.label_ids = [self.tok(label, add_special_tokens=False).input_ids
                          for label in LABELS]
        self.single_token = all(len(ids) == 1 for ids in self.label_ids)

    def _prompt_ids(self, lines):
        messages = [{"role": "system", "content": LLM_SYSTEM}]
        if self.fewshot:
            for example, label in FEWSHOT:
                messages.append({"role": "user", "content": render(example)})
                messages.append({"role": "assistant", "content": label})
        messages.append({"role": "user", "content": render(lines)})
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False).input_ids

    def classify(self, lines):
        torch = self.torch
        prompt = self._prompt_ids(lines)
        with torch.inference_mode():
            if self.single_token:
                ids = torch.tensor([prompt], device=self.device)
                logits = self.model(input_ids=ids).logits[0, -1].float()
                scores = logits.log_softmax(-1)[[ids[0] for ids in self.label_ids]]
            else:
                scores = self._score_multi_token(prompt)
        probs = scores.softmax(0)
        best = int(probs.argmax())
        return LABELS[best], float(probs[best])

    def _score_multi_token(self, prompt):
        """Batch prompt+answer for every label; right padding keeps positions exact."""
        torch = self.torch
        seqs = [prompt + ids for ids in self.label_ids]
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
        start = len(prompt)
        totals = []
        for i, ids in enumerate(self.label_ids):
            positions = torch.arange(start - 1, start - 1 + len(ids), device=self.device)
            logprobs = self.model.lm_head(hidden[i, positions]).float().log_softmax(-1)
            totals.append(logprobs[torch.arange(len(ids)), torch.tensor(ids)].sum())
        return torch.stack(totals)


class ProductionLocalBackend:
    """The pipeline's own LocalClassifier, prompt and timestamps exactly as live."""

    def __init__(self, device: str):
        from aromagen.local_brain import LocalClassifier

        self.inner = LocalClassifier(device=device, labels=LABELS)

    def classify(self, lines):
        class _Line:
            def __init__(self, text):
                self.text, self.started_at = text, time.time()

            def __str__(self):
                return self.text

        decision = self.inner.classify_sync([_Line(t) for t in lines])
        return decision.recipe, decision.confidence


REGISTRY = {
    "local": lambda dev: ProductionLocalBackend(dev),
    "xlmr": lambda dev: XlmrBackend(dev),
    "e5": lambda dev: E5Backend(dev),
    "qwen-0.5b": lambda dev: LlmBackend("Qwen/Qwen2.5-0.5B-Instruct", dev, False),
    "qwen-0.5b-fewshot": lambda dev: LlmBackend("Qwen/Qwen2.5-0.5B-Instruct", dev, True),
    "qwen-1.5b": lambda dev: LlmBackend("Qwen/Qwen2.5-1.5B-Instruct", dev, False),
    "qwen-1.5b-fewshot": lambda dev: LlmBackend("Qwen/Qwen2.5-1.5B-Instruct", dev, True),
    "qwen-3b": lambda dev: LlmBackend("Qwen/Qwen2.5-3B-Instruct", dev, False),
    "qwen-3b-fewshot": lambda dev: LlmBackend("Qwen/Qwen2.5-3B-Instruct", dev, True),
}


def build(name: str, default_device: str):
    base, _, device = name.partition("@")
    device = device or default_device
    if base.startswith("claude:"):
        return ClaudeBackend(base.split(":", 1)[1]), "network"
    if base not in REGISTRY:
        raise SystemExit(f"unknown backend {base!r}; known: claude:<model>, "
                         + ", ".join(REGISTRY))
    return REGISTRY[base](device), device


# --- evaluation -----------------------------------------------------------
def load_items():
    with EVAL_SET.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(name, backend, items, show_errors):
    backend.classify(items[0]["lines"])          # warm-up, not timed
    results, latencies = [], []
    for item in items:
        t0 = time.perf_counter()
        got, confidence = backend.classify(item["lines"])
        latencies.append(time.perf_counter() - t0)
        results.append((item, got, confidence))

    def accuracy(subset):
        subset = list(subset)
        return sum(got == item["label"] for item, got, _ in subset) / max(len(subset), 1)

    row = {
        "backend": name,
        "all": accuracy(results),
        "en": accuracy(r for r in results if r[0]["lang"] == "en"),
        "fr": accuracy(r for r in results if r[0]["lang"] == "fr"),
        "mix": accuracy(r for r in results if r[0]["lang"] == "mix"),
        "hard": accuracy(r for r in results if r[0].get("hard")),
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
    }
    errors = [(item, got, conf) for item, got, conf in results if got != item["label"]]
    # A confident mistake moves the room; a hesitant one barely does.
    row["confident_errors"] = sum(conf >= 0.6 for _, _, conf in errors)

    if show_errors and errors:
        for item, got, conf in errors:
            tag = f" [{item['hard']}]" if item.get("hard") else ""
            print(f"    {item['id']} want {item['label']:8} got {got:8} "
                  f"conf {conf:.2f}{tag}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", default=(
        "claude:claude-haiku-4-5,xlmr,e5,qwen-1.5b,qwen-1.5b-fewshot,"
        "qwen-3b,qwen-3b-fewshot"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--out", type=Path, help="write the table as JSON")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("transformers", "sentence_transformers", "httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    items = load_items()
    print(f"{len(items)} labelled snippets from {EVAL_SET.name}\n")

    rows = []
    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"--- {name}")
        vram = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            torch = None
        try:
            t0 = time.perf_counter()
            backend, where = build(name, args.device)
            load = time.perf_counter() - t0
            row = evaluate(name, backend, items, args.show_errors)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"    skipped: {type(exc).__name__}: {str(exc)[:160]}\n")
            continue
        if torch is not None and torch.cuda.is_available() and where == "cuda":
            vram = torch.cuda.max_memory_allocated() / 2**20
        row.update(load=load, where=where, vram=vram)
        rows.append(row)
        print(f"    accuracy {row['all']:.0%}  hard {row['hard']:.0%}  "
              f"p50 {row['p50'] * 1000:.0f} ms  load {load:.1f}s"
              + (f"  VRAM {vram:.0f} MiB" if vram else "") + "\n")
        if hasattr(backend, "close"):
            backend.close()
        del backend
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        return 1
    print(f"{'backend':26} {'all':>5} {'en':>5} {'fr':>5} {'mix':>5} {'hard':>5} "
          f"{'conf.err':>8} {'p50':>7} {'p95':>7} {'VRAM':>8}  where")
    print("-" * 104)
    for r in sorted(rows, key=lambda r: (-r["all"], r["p50"])):
        vram = f"{r['vram']:.0f}M" if r["vram"] else "-"
        print(f"{r['backend']:26} {r['all']:5.0%} {r['en']:5.0%} {r['fr']:5.0%} "
              f"{r['mix']:5.0%} {r['hard']:5.0%} {r['confident_errors']:8d} "
              f"{r['p50'] * 1000:6.0f}ms {r['p95'] * 1000:6.0f}ms {vram:>8}  {r['where']}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
