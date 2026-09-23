"""
Fetch every model the installation needs, then prove it all runs with no network.

Run this ONCE, while connected to the internet, on the laptop that will run the
exhibition -- before leaving for the venue:

    python prepare_offline.py
    python prepare_offline.py --verify-only      # re-check without downloading

After downloading, it starts a separate process with the Hugging Face hub forced
offline AND every proxy pointed at nowhere, loads each model there and
classifies two sentences. A pass means the missing wifi at the venue cannot stop
the installation from starting. A failure names what is missing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# GPU default, plus the CPU fallback in case the GPU refuses to load on the day.
WHISPER_SIZES = ("large-v3-turbo", "small")
# Formats nothing here loads; they would only double the download.
IGNORE = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "*.gguf",
          "*.tflite", "coreml/*", "*.mlmodel"]

VERIFY = '''
import asyncio, logging, time
logging.basicConfig(level=logging.WARNING)
from aromagen.transcribe import load_model
from aromagen.local_brain import LocalClassifier

t0 = time.time()
load_model()
print(f"  ok   Whisper loads with no network ({time.time() - t0:.1f}s)", flush=True)

t0 = time.time()
brain = LocalClassifier()
print(f"  ok   local model loads with no network, on {brain.device} "
      f"({time.time() - t0:.1f}s)", flush=True)

class Line:
    def __init__(self, text):
        self.text, self.started_at = text, time.time()
    def __str__(self):
        return self.text

# An infrastructure check, not an accuracy one: what matters here is that the
# model loads with no network and answers with a vibe the device can actually
# diffuse. How well it tells the vibes apart is tests/check_vibes.py.
from aromagen.recipes import AVAILABLE_RECIPES
allowed = {r.key for r in AVAILABLE_RECIPES}
bad = 0
for lines in (["honestly, I have never told anyone this", "I was terrified"],
              ["exactly, yes, that is it", "I feel the same way"]):
    d = asyncio.run(brain.classify([Line(t) for t in lines]))
    ok = d.recipe in allowed
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'}  answers {d.recipe:12} [{d.reason}] "
          f"{d.latency * 1000:.0f} ms", flush=True)
raise SystemExit(bad)
'''


def download() -> bool:
    from faster_whisper.utils import download_model
    from huggingface_hub import snapshot_download

    from aromagen.local_brain import DEFAULT_LOCAL_MODEL

    ok = True
    for size in WHISPER_SIZES:
        t0 = time.time()
        try:
            download_model(size)
            print(f"  ok   Whisper {size:28} {time.time() - t0:5.0f}s")
        except Exception as exc:
            ok = False
            print(f"  FAIL Whisper {size}: {type(exc).__name__}: {str(exc)[:160]}")
    t0 = time.time()
    try:
        snapshot_download(DEFAULT_LOCAL_MODEL, ignore_patterns=IGNORE)
        print(f"  ok   {DEFAULT_LOCAL_MODEL:36} {time.time() - t0:5.0f}s")
    except Exception as exc:
        ok = False
        print(f"  FAIL {DEFAULT_LOCAL_MODEL}: {type(exc).__name__}: {str(exc)[:160]}")
    return ok


def verify() -> bool:
    """Run the checks in a child process that genuinely cannot reach the network."""
    nowhere = "http://127.0.0.1:9"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HTTP_PROXY=nowhere, HTTPS_PROXY=nowhere,
               http_proxy=nowhere, https_proxy=nowhere, NO_PROXY="", no_proxy="")
    result = subprocess.run([sys.executable, "-c", VERIFY], cwd=str(HERE), env=env)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify-only", action="store_true",
                        help="skip downloading, only check offline startup")
    args = parser.parse_args()

    if not args.verify_only:
        print("Downloading models (skipped if already present)...")
        if not download():
            print("\nSome downloads failed. Check the connection and run again.")
            return 1

    print("\nChecking that everything starts with the network cut off...")
    if verify():
        print("\nREADY: the installation will start without internet.")
        return 0
    print("\nNOT READY: see the failures above. Fix, then run this again while "
          "still connected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
