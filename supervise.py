"""
Keep AromaGen running for the length of an exhibition.

Launches `python -m aromagen` as a child process, restarts it if it ever exits,
and writes a log of what happened. Nothing here touches the pipeline itself --
it only starts, watches and restarts it -- so it cannot introduce a failure into
the running installation.

    python supervise.py                       # default live settings
    python supervise.py -- --mock --lang fr   # anything after -- goes to aromagen
    python supervise.py --log logs/expo.log

Ctrl+C stops the supervisor and the child together, cleanly.

Design notes, all of them learned the hard way in unattended systems:

  backoff       a process that dies instantly and is restarted instantly burns
                a CPU core and fills the disk with logs. Restart delay grows
                from 5 s to 2 min while failures keep coming, and resets once
                the child has survived a while;
  crash budget  if it cannot stay up at all, stop and say so loudly rather than
                flapping in silence all evening;
  clean stop    exit code 0 means the child was asked to quit; do not restart it;
  log rotation  a day-long run must not fill the disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Restart pacing. Slow enough not to hammer the machine, fast enough that a
# visitor does not notice the gap.
FIRST_DELAY = 5.0
MAX_DELAY = 120.0
# Surviving this long counts as a healthy run and resets the backoff.
HEALTHY_SECONDS = 300.0
# Give up after this many failures that never reached HEALTHY_SECONDS.
MAX_CONSECUTIVE_FAILURES = 10
MAX_LOG_BYTES = 20 * 1024 * 1024


class Log:
    """Timestamped log to a file and to the console, with simple rotation."""

    def __init__(self, path: Path = None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_large()
            self.fh = open(path, "a", encoding="utf-8", buffering=1)
        else:
            self.fh = None

    def _rotate_if_large(self) -> None:
        if self.path.is_file() and self.path.stat().st_size > MAX_LOG_BYTES:
            self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))

    def __call__(self, message: str) -> None:
        line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] supervisor: {message}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")

    def close(self) -> None:
        if self.fh:
            self.fh.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=HERE / "logs" / "aromagen.log",
                        help="where to write the supervisor log")
    parser.add_argument("--no-log-file", action="store_true",
                        help="console only")
    parser.add_argument("child", nargs="*",
                        help="arguments passed through to aromagen (after --)")
    args = parser.parse_args()

    log = Log(None if args.no_log_file else args.log)
    command = [sys.executable, "-u", "-m", "aromagen", *args.child]

    log(f"starting: {' '.join(command[2:])}")
    if log.path:
        log(f"logging to {log.path}")

    delay = FIRST_DELAY
    failures = 0
    runs = 0
    child = None
    stopping = False

    def handle_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            # Let aromagen shut the valves itself rather than killing it.
            log("stop requested, asking the child to finish")
            try:
                child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt"
                                  else signal.SIGINT)
            except Exception:
                child.terminate()

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    try:
        while not stopping:
            runs += 1
            started = time.monotonic()
            log(f"run #{runs} starting")
            try:
                child = subprocess.Popen(command, cwd=str(HERE),
                                         creationflags=creation)
            except Exception as exc:
                log(f"could not start the child: {exc!r}")
                return 1

            code = child.wait()
            lived = time.monotonic() - started

            if stopping:
                log(f"child exited with {code} after a stop request")
                break
            if code == 0:
                log(f"child exited cleanly after {lived:.0f}s, not restarting")
                break

            if lived >= HEALTHY_SECONDS:
                # It ran fine for a long while, so this is a fresh problem
                # rather than a crash loop: start the backoff over.
                failures = 0
                delay = FIRST_DELAY
                log(f"child exited with {code} after {lived / 60:.1f} min")
            else:
                failures += 1
                log(f"child exited with {code} after only {lived:.0f}s "
                    f"({failures}/{MAX_CONSECUTIVE_FAILURES} short failures)")

            if failures >= MAX_CONSECUTIVE_FAILURES:
                log("giving up: the child cannot stay running. "
                    "Run it by hand to see the error.")
                return 1

            log(f"restarting in {delay:.0f}s")
            waited = 0.0
            while waited < delay and not stopping:
                time.sleep(0.5)
                waited += 0.5
            delay = min(delay * 2, MAX_DELAY)
    finally:
        if child and child.poll() is None:
            try:
                child.wait(timeout=20)
            except Exception:
                log("child did not stop in time, terminating")
                child.terminate()
        log(f"supervisor finished after {runs} run(s)")
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
