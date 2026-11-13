"""The segments of recordings waiting to be analysed, shared by this computer's workers and by workers on other
computers (phase 5).

Rules:
- A recording cut into segments (video_segments) publishes them here as one job. This computer's worker threads
  take segments of that job only; a worker on another computer takes a segment of any job that allows it.
- Another computer helps only while the operator has turned "other computers may help" on and the worker shows a
  pairing code that is still valid. It is sent only what one segment needs: the time range, the detector settings
  and the model's checksum. It reads that range of the video from this server and posts back the tracks it found.
- A segment is leased to one worker at a time. A worker on another computer renews its lease while it works; a lease
  not renewed for LEASE_S seconds ends, and the segment goes back to the front of the queue for anyone to take. A
  late result from a lease that ended is ignored, so every segment is counted once.
- A job withdrawn (finished, cancelled or failed) ends its leases; workers asking about them are told to stop.
"""
from __future__ import annotations

import itertools
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

LEASE_S = 120.0
SEEN_S = 90.0          # a worker counts as connected when it asked for work this recently


@dataclass
class Job:
    id: str
    mission_id: str
    run_item_id: int
    version_id: int
    spec: dict                         # what a worker needs besides the segment (sent to other computers)
    remote: bool                       # whether other computers may take its segments
    todo: deque = field(default_factory=deque)
    total: int = 0
    done: dict = field(default_factory=dict)          # idx -> (result, worker)
    fresh: list = field(default_factory=list)         # (segment, result, worker) not yet stored by the processor
    error: str | None = None


@dataclass
class Lease:
    id: str
    job: Job
    seg: Any
    worker: str
    remote: bool
    expires: float


@dataclass
class Helper:
    name: str
    code_id: int | None
    info: dict = field(default_factory=dict)
    last_seen: float = 0.0
    segments: int = 0
    seconds: float = 0.0
    lost: int = 0


class SegmentHub:
    def __init__(self, lease_s: float = LEASE_S):
        self.lease_s = lease_s
        self.cond = threading.Condition()
        self.jobs: dict[str, Job] = {}
        self.leases: dict[str, Lease] = {}
        self.helpers: dict[str, Helper] = {}
        self.enabled = False
        self._ids = itertools.count(1)

    # ---------- the processor's side ----------
    def publish(self, mission_id: str, run_item_id: int, version_id: int, segments: list, spec: dict) -> Job:
        with self.cond:
            job = Job(f"j{next(self._ids)}-{secrets.token_hex(4)}", mission_id, run_item_id, version_id, spec,
                      self.enabled, deque(segments), len(segments))
            self.jobs[job.id] = job
            self.cond.notify_all()
            return job

    def withdraw(self, job: Job) -> None:
        with self.cond:
            self.jobs.pop(job.id, None)
            for k in [k for k, ls in self.leases.items() if ls.job is job]:
                del self.leases[k]
            self.cond.notify_all()

    def finished(self, job: Job) -> bool:
        with self.cond:
            return len(job.done) >= job.total or job.error is not None

    def collect(self, job: Job, timeout: float) -> list:
        """Results that came in since the last call (waits up to ``timeout`` for one)."""
        with self.cond:
            self._expire()
            if not job.fresh and len(job.done) < job.total and job.error is None:
                self.cond.wait(timeout)
                self._expire()
            out, job.fresh = job.fresh, []
            return out

    def helpers_ready(self) -> bool:
        """Whether another computer is connected and could take segments now."""
        now = time.monotonic()
        with self.cond:
            return self.enabled and any(now - h.last_seen < SEEN_S for h in self.helpers.values())

    # ---------- taking and returning segments ----------
    def take(self, worker: str, job: Job | None = None, remote: bool = False, can: set[str] | None = None) -> Lease | None:
        """A segment for this worker: of ``job`` (this computer's threads), or of any job open to other computers
        whose detector this worker can run (``can``: the detector kinds it has, e.g. {"standard", "sam3"})."""
        with self.cond:
            self._expire()
            jobs = [job] if job is not None else [j for j in self.jobs.values() if j.remote and self.enabled]
            for j in jobs:
                if j.id not in self.jobs or not j.todo or j.error:
                    continue
                if remote and can is not None and not set(j.spec.get("needs", ["standard"])) <= can:
                    continue
                seg = j.todo.popleft()
                ls = Lease(f"l{next(self._ids)}-{secrets.token_hex(8)}", j, seg, worker, remote,
                           time.monotonic() + self.lease_s)
                self.leases[ls.id] = ls
                return ls
            return None

    def lease(self, lease_id: str) -> Lease | None:
        with self.cond:
            self._expire()
            return self.leases.get(lease_id)

    def renew(self, lease_id: str) -> bool:
        with self.cond:
            self._expire()
            ls = self.leases.get(lease_id)
            if ls is None:
                return False
            ls.expires = time.monotonic() + self.lease_s
            if ls.worker in self.helpers:
                self.helpers[ls.worker].last_seen = time.monotonic()
            return True

    def complete(self, lease_id: str, result) -> bool:
        """Store a segment's result. False when the lease had already ended (the segment went to someone else)."""
        with self.cond:
            self._expire()
            ls = self.leases.pop(lease_id, None)
            if ls is None or ls.job.id not in self.jobs or ls.seg.idx in ls.job.done:
                return False
            ls.job.done[ls.seg.idx] = (result, ls.worker)
            ls.job.fresh.append((ls.seg, result, ls.worker))
            h = self.helpers.get(ls.worker)
            if h:
                h.segments += 1
                h.seconds += float(getattr(result, "seconds", 0.0) or 0.0)
                h.last_seen = time.monotonic()
            self.cond.notify_all()
            return True

    def give_back(self, lease_id: str, error: str | None = None) -> None:
        """A worker could not finish: the segment goes back to the queue (this computer's errors fail the job)."""
        with self.cond:
            ls = self.leases.pop(lease_id, None)
            if ls is None:
                return
            if ls.remote:
                ls.job.todo.appendleft(ls.seg)
            else:
                ls.job.error = error or "a segment failed"
            self.cond.notify_all()

    def _expire(self) -> None:
        now = time.monotonic()
        for k, ls in list(self.leases.items()):
            if ls.expires < now:
                del self.leases[k]
                if ls.job.id in self.jobs and ls.seg.idx not in ls.job.done:
                    ls.job.todo.appendleft(ls.seg)
                if ls.worker in self.helpers:
                    self.helpers[ls.worker].lost += 1
                self.cond.notify_all()

    # ---------- other computers ----------
    def seen(self, name: str, code_id: int | None, info: dict) -> None:
        with self.cond:
            h = self.helpers.setdefault(name, Helper(name, code_id))
            h.code_id, h.info, h.last_seen = code_id, info, time.monotonic()

    def forget_code(self, code_id: int) -> None:
        """A revoked pairing code: its workers' leases end now and their segments go back to the queue."""
        with self.cond:
            names = {h.name for h in self.helpers.values() if h.code_id == code_id}
            for k, ls in list(self.leases.items()):
                if ls.worker in names:
                    ls.expires = 0.0
            self._expire()

    def status(self) -> dict:
        now = time.monotonic()
        with self.cond:
            self._expire()
            working = {ls.worker: ls for ls in self.leases.values()}
            return {
                "enabled": self.enabled,
                "helpers": [{"name": h.name, "connected": now - h.last_seen < SEEN_S,
                             "seen_s_ago": round(now - h.last_seen, 1), "segments": h.segments,
                             "seconds": round(h.seconds, 1), "lost": h.lost, "info": h.info,
                             "working_on": ({"mission_id": working[h.name].job.mission_id,
                                             "segment": working[h.name].seg.idx}
                                            if h.name in working else None)}
                            for h in sorted(self.helpers.values(), key=lambda h: h.name)],
                "queue": [{"mission_id": j.mission_id, "version_id": j.version_id, "segments": j.total,
                           "done": len(j.done), "waiting": len(j.todo), "open_to_others": j.remote}
                          for j in self.jobs.values()],
            }
