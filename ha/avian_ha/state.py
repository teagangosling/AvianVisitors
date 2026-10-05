"""Today's visitors, persisted so a restart doesn't blank the collage."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from .detection import Detection


@dataclass
class Tally:
    sci: str
    com: str
    count: int
    first_seen: str
    last_seen: str
    best_confidence: float


class DayState:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.day = date.today()
        self.species: dict[str, Tally] = {}
        self.latest: dict | None = None
        self.seen_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.latest = raw.get("latest")
        if raw.get("day") == date.today().isoformat():
            self.species = {k: Tally(**v) for k, v in raw.get("species", {}).items()}
            self.seen_ids = list(raw.get("seen_ids", []))

    def save(self) -> None:
        with self._lock:
            data = {
                "day": self.day.isoformat(),
                "species": {k: asdict(v) for k, v in self.species.items()},
                "latest": self.latest,
                "seen_ids": self.seen_ids[-200:],
            }
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def roll(self, today: date | None = None) -> bool:
        """Start a new day. Returns True if the day changed."""
        today = today or date.today()
        with self._lock:
            if today == self.day:
                return False
            self.day, self.species, self.seen_ids = today, {}, []
            return True

    def add(self, det: Detection) -> bool:
        """Count a detection. Returns False for a repeat (e.g. the retained
        message BirdNET-Go's broker replays on every reconnect)."""
        when = det.when or datetime.now()
        if when.date() != self.day:
            # A detection stamped another day (late replay, or just before
            # midnight) shouldn't reopen or pollute today's tally.
            return False
        key = det.detection_id or f"{det.sci}|{when.isoformat()}"
        with self._lock:
            if key in self.seen_ids:
                return False
            self.seen_ids.append(key)
            t = self.species.get(det.slug)
            stamp = when.isoformat(timespec="seconds")
            if t is None:
                self.species[det.slug] = Tally(det.sci, det.com, 1, stamp, stamp, det.confidence)
            else:
                t.count += 1
                t.last_seen = stamp
                t.best_confidence = max(t.best_confidence, det.confidence)
            return True

    def visitors(self) -> list[Tally]:
        with self._lock:
            return sorted(self.species.values(), key=lambda t: (-t.count, t.com))
