import io
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avian_ha import collage  # noqa: E402
from avian_ha.detection import parse, slugify  # noqa: E402
from avian_ha.images import ImageResolver  # noqa: E402
from avian_ha.state import DayState  # noqa: E402

# Trimmed from a real BirdNET-Go NoteWithBirdImage MQTT payload.
BIRDNET_GO = {
    "SourceNode": "birdnet-go", "Date": "2026-10-05", "Time": "07:41:12",
    "BeginTime": "2026-10-05T07:41:12-07:00", "EndTime": "2026-10-05T07:41:15-07:00",
    "ScientificName": "Calypte anna", "CommonName": "Anna's Hummingbird",
    "Confidence": 0.8731, "Latitude": 49.28, "Longitude": -123.12, "Threshold": 0.8,
    "Sensitivity": 1, "ClipName": "clips/2026/10/calypte_anna_87p_20261005T074112Z.wav",
    "ProcessingTime": 512000000, "Unlikely": False, "Results": None, "Review": None,
    "Comments": None, "Lock": None, "detectionId": 4211, "sourceId": "rtsp_8a1f",
    "sourceName": "birdmic",
    "BirdImage": {"URL": "https://upload.wikimedia.org/x/Calypte_anna.jpg",
                  "ScientificName": "Calypte anna", "LicenseName": "CC BY-SA 4.0",
                  "LicenseURL": "", "AuthorName": "A. Photographer", "AuthorURL": "",
                  "CachedAt": "2026-10-05T07:00:00Z", "SourceProvider": "wikimedia"},
    "speciesFirstDetectedAt": None, "speciesLastDetectedAt": None,
}


def test_parse_birdnet_go():
    d = parse(json.dumps(BIRDNET_GO))
    assert d.sci == "Calypte anna" and d.com == "Anna's Hummingbird"
    assert d.slug == "calypte-anna"
    assert d.confidence == pytest.approx(0.8731)
    assert d.when == datetime(2026, 10, 5, 7, 41, 12)
    assert d.detection_id == "4211" and d.source == "birdmic"
    assert d.photo_url.startswith("https://upload.wikimedia.org/")


@pytest.mark.parametrize("sci", ["Dog", "Human vocal_Human vocal", "Engine", "", "../../etc passwd"])
def test_parse_rejects_non_birds(sci):
    assert parse(json.dumps({**BIRDNET_GO, "ScientificName": sci})) is None


def test_parse_avianvisitors_shape_and_percent():
    d = parse(json.dumps({"sci": "Turdus migratorius", "com": "American Robin", "best_conf": 91}))
    assert d.slug == "turdus-migratorius" and d.confidence == pytest.approx(0.91)


def test_parse_garbage():
    assert parse(b"\xff\x00") is None
    assert parse("[1,2]") is None


def test_slug_matches_cutout_php():
    assert slugify("Setophaga coronata auduboni") == "setophaga-coronata-auduboni"


def test_state_dedupes_retained_replay_and_rolls(tmp_path):
    s = DayState(tmp_path / "state.json")
    s.roll(date(2026, 10, 5))
    d = parse(json.dumps(BIRDNET_GO))
    assert s.add(d) is True
    assert s.add(d) is False  # same detectionId replayed on reconnect
    assert s.add(parse(json.dumps({**BIRDNET_GO, "detectionId": 4212}))) is True
    assert s.visitors()[0].count == 2
    s.save()
    yesterday = parse(json.dumps({**BIRDNET_GO, "Date": "2026-10-04", "detectionId": 1}))
    assert s.add(yesterday) is False
    assert s.roll(date(2026, 10, 6)) and s.visitors() == []


def _png(path: Path, size=(200, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    im.paste((120, 60, 20, 255), (40, 20, 160, 100))
    # Noise so the file clears the >1 KB "usable" floor like real art does.
    im.putdata([(x % 251, (x * 7) % 253, 20, 255 if 40 <= x % 200 < 160 else 0)
                for x in range(size[0] * size[1])])
    im.save(path)
    return path


def test_resolver_chain(tmp_path):
    assets, data = tmp_path / "assets", tmp_path / "data"
    _png(assets / "illustrations" / "calypte-anna.png")
    _png(assets / "illustrations" / "calypte-anna-2.png")
    _png(assets / "cutouts" / "pica-nuttalli.png")
    r = ImageResolver(assets, data, wikipedia=False, cutout_model=None)

    hb = parse(json.dumps(BIRDNET_GO))
    assert r.resolve(hb).path.name == "calypte-anna.png"
    assert r.resolve(hb, pose=2).path.name == "calypte-anna-2.png"
    assert r.resolve(hb).kind == "illustration"

    magpie = parse(json.dumps({**BIRDNET_GO, "ScientificName": "Pica nuttalli", "BirdImage": {}}))
    assert r.resolve(magpie, pose=2).kind == "cutout"  # no illustration -> bundled cutout

    # Unknown species, no network: a miss, remembered so it isn't retried every detection.
    odd = parse(json.dumps({**BIRDNET_GO, "ScientificName": "Rara avis",
                            "BirdImage": {"URL": "https://evil.example.com/x.jpg"}}))
    assert r.resolve(odd) is None
    assert "rara-avis" in json.loads((data / "image-misses.json").read_text())


def test_collage_renders(tmp_path):
    imgs = [_png(tmp_path / f"b{i}.png", (180 + i * 30, 140)) for i in range(5)]
    vs = [collage.Visitor(f"Bird {i}", f"Avis b{i}", p, i + 1) for i, p in enumerate(imgs)]
    for visitors in (vs, []):
        png = collage.render(visitors, day=date(2026, 10, 5), width=900, height=600)
        im = Image.open(io.BytesIO(png))
        assert im.size == (900, 600)
