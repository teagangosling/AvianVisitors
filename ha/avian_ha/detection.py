"""Turn an MQTT detection payload into a Detection.

The primary producer is BirdNET-Go, whose payload is a JSON
`NoteWithBirdImage`: datastore.Note fields in PascalCase (CommonName,
ScientificName, Confidence 0-1, Date, Time, Unlikely, ...) plus
detectionId, sourceId, sourceName and BirdImage {URL, AuthorName,
LicenseName, SourceProvider, ...}. Those names are BirdNET-Go's documented
public MQTT contract.

The lowercase shape that AvianVisitors' own forwarding/mqtt-bridge.py emits
(sci, com, best_conf, last_seen) is accepted too, so the bridge also works
behind a BirdNET-Pi station.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

# Same binomial/trinomial gate as avian/api/cutout.php. Non-bird classes
# BirdNET-Go can report ("Dog", "Human vocal", "Engine") fail it and are
# dropped before any filesystem or network lookup.
SCI_RE = re.compile(r"^[A-Za-z]{2,40}(?: [a-z]{2,40}){1,3}$")


def slugify(sci: str) -> str:
    """Matches cutout.php: lowercase, runs of non-alphanumerics -> '-'."""
    return re.sub(r"[^a-z0-9]+", "-", sci.lower()).strip("-")


@dataclass(frozen=True)
class Detection:
    sci: str
    com: str
    confidence: float  # 0-1
    when: datetime | None
    unlikely: bool = False
    source: str = ""
    detection_id: str = ""
    photo_url: str = ""
    photo_author: str = ""
    photo_license: str = ""

    @property
    def slug(self) -> str:
        return slugify(self.sci)


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _parse_when(payload: dict) -> datetime | None:
    date = _first(payload, "Date", "date")
    time = _first(payload, "Time", "time")
    if date and time:
        try:
            return datetime.fromisoformat(f"{date}T{time}")
        except ValueError:
            pass
    for key in ("BeginTime", "last_seen", "timestamp"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
    return None


def parse(raw: bytes | str) -> Detection | None:
    """Return a Detection, or None for anything that isn't a bird."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    sci = str(_first(payload, "ScientificName", "scientificName", "sci", default="")).strip()
    if not SCI_RE.match(sci):
        return None
    com = str(_first(payload, "CommonName", "commonName", "com", default=sci)).strip()

    try:
        conf = float(_first(payload, "Confidence", "confidence", "best_conf", default=0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if conf > 1.0:  # percent
        conf /= 100.0

    image = payload.get("BirdImage") if isinstance(payload.get("BirdImage"), dict) else {}
    when = _parse_when(payload)
    return Detection(
        sci=sci,
        com=com,
        confidence=max(0.0, min(conf, 1.0)),
        # Day rollover compares against local wall-clock dates (container TZ).
        when=when.astimezone().replace(tzinfo=None) if when and when.tzinfo else when,
        unlikely=bool(payload.get("Unlikely", False)),
        source=str(_first(payload, "sourceName", "sourceId", default="")),
        detection_id=str(_first(payload, "detectionId", default="")),
        photo_url=str(image.get("URL") or ""),
        photo_author=str(image.get("AuthorName") or ""),
        photo_license=str(image.get("LicenseName") or ""),
    )
