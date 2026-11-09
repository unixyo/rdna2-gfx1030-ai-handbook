"""Read-only KFD compute-overlap audit plus a research-folder process lock.

This detects KFD compute clients, not graphics/display activity. A PID merely
present in KFD is conservatively considered a possible competing compute job.
An unavailable audit or any observed foreign PID invalidates timing evidence.
No process, service, operating configuration, or GPU clock is modified.
"""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import threading
import time


class GpuAudit:
    def __init__(self, allowed_pids=(), interval_s=0.25, lock_path=None):
        self.allowed_pids = {os.getpid(), *(int(p) for p in allowed_pids)}
        self.interval_s = float(interval_s)
        if self.interval_s <= 0:
            raise ValueError("Positive audit interval required")
        self.lock_path = Path(lock_path or Path(__file__).with_name(".gpu_bench_v06.lock"))
        self.kfd_path = Path("/sys/class/kfd/kfd/proc")
        self.foreign = {}
        self.errors = []
        self.samples = 0
        self.transitions = []
        self._last_state = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = None
        self.started = None
        self.ended = None

    def allow_pid(self, pid):
        """Allow an explicitly owned child before it opens a GPU context."""
        self.allowed_pids.add(int(pid))

    def sample(self):
        now = time.time()
        try:
            pids = sorted(int(p.name) for p in self.kfd_path.iterdir() if p.name.isdigit())
            foreign = [p for p in pids if p not in self.allowed_pids]
            state = tuple(foreign)
            for pid in foreign:
                if pid not in self.foreign:
                    try:
                        name = (Path("/proc") / str(pid) / "comm").read_text().strip()
                    except OSError:
                        name = None
                    self.foreign[pid] = {"pid": pid, "comm": name, "first_seen_unix_s": now}
        except OSError as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message not in self.errors:
                self.errors.append(message)
            state = ("unavailable",)
            pids = []
        self.samples += 1
        if state != self._last_state:
            self.transitions.append({"unix_s": now, "kfd_pids": pids,
                                     "foreign_or_error": list(state)})
            self._last_state = state

    def ensure_clear(self):
        self.sample()
        if self.errors or self.foreign:
            raise RuntimeError("GPU exclusivity not established; timing is invalid: "
                               + str(self.report()))

    def _poll(self):
        while not self._stop.wait(self.interval_s):
            self.sample()

    def __enter__(self):
        self._lock = self.lock_path.open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise RuntimeError("Another v0.6 research GPU job holds " + str(self.lock_path))
        self.started = time.time()
        self.sample()
        self._thread = threading.Thread(target=self._poll, name="gpu-kfd-audit", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_s))
        self.sample()
        self.ended = time.time()
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
        return False

    def report(self):
        return {"available": not self.errors, "valid_for_timing": not self.errors and not self.foreign,
                "method": "KFD client PID snapshots and periodic polling; research flock",
                "limitation": "Detects observed KFD clients, not graphics load or sub-poll transient clients",
                "allowed_pids": sorted(self.allowed_pids), "interval_s": self.interval_s,
                "sample_count": self.samples, "started_unix_s": self.started,
                "ended_unix_s": self.ended, "foreign_clients": list(self.foreign.values()),
                "errors": list(self.errors), "transitions": list(self.transitions)}
