"""
Loading the project's `.env` file.

Keeps the API key out of the source and out of shell history: it lives in a
git-ignored file, is never printed and never logged.

Expected format, one variable per line:

    ANTHROPIC_API_KEY=sk-ant-...

Blank lines and lines starting with # are ignored. Quotes around the value are
stripped. A variable already present in the environment is never overwritten:
the real environment wins.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("aromagen.env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def candidate_paths() -> list:
    """Where a .env is accepted, in priority order."""
    return [
        PROJECT_ROOT / ".env",                     # the recommended location
        Path.cwd() / ".env",                       # launched from elsewhere
        PROJECT_ROOT / "AromaGen_MOI" / ".env",    # next to the firmware clone
    ]


def load_env_file(path: Path = None) -> int:
    """
    Load the first .env found. Returns how many variables were set.

    Several locations are searched rather than demanding one exact path: a key
    dropped in the wrong folder is a silent failure and a miserable one to
    diagnose.
    """
    if path is None:
        for candidate in candidate_paths():
            if candidate.is_file():
                path = candidate
                break
    if path is None or not path.is_file():
        return 0
    log.debug("environment file: %s", path)

    defined = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            log.warning("%s line %d ignored (no '=')", path.name, lineno)
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            defined += 1
    return defined


def describe_credentials() -> str:
    """Credential status, without ever revealing the key itself."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return "ANTHROPIC_API_KEY missing - classifying with the local lexicon"
    return f"ANTHROPIC_API_KEY present ({len(key)} chars, ...{key[-4:]})"
