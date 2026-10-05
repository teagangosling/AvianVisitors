"""Species -> image, the same lookup chain as avian/api/cutout.php:

  1. assets/illustrations/<slug>[-2].png   bundled art, perched (1) / flight (2)
  2. assets/cutouts/<slug>.png             bundled background-removed photo
  3. <data>/cutouts/<slug>.png             cached cutout from an earlier run
  4. fresh photo -> background removal -> cache

Step 4 tries the photo BirdNET-Go attached to the detection (BirdImage.URL,
usually a curated Avicommons portrait) first, then the Wikipedia lead image,
and only follows URLs on Wikimedia/Wikipedia/Avicommons hosts. Background removal runs a U^2-Net
ONNX model directly through onnxruntime, the same model family rembg uses
on the Pi, minus rembg's dependency tree. Without the model the plain photo
is cached and used instead, so a species never ends up with no image.
"""
from __future__ import annotations

import html
import io
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from .detection import Detection, slugify

log = logging.getLogger(__name__)

ALLOWED_HOSTS = re.compile(r"(?:^|\.)(?:wikimedia\.org|wikipedia\.org|avicommons\.org)$", re.I)
MAX_DOWNLOAD = 12 * 1024 * 1024
MISS_RETRY_SECONDS = 24 * 3600
ILLUSTRATION_CREDIT = "AvianVisitors illustration (CC BY-NC-SA 4.0)"


@dataclass(frozen=True)
class ResolvedImage:
    path: Path
    kind: str  # illustration | cutout | photo-cutout | photo
    credit: str = ""


def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 1024
    except OSError:
        return False


class Cutter:
    """U^2-Net salient-object mask -> RGBA cutout tightly cropped to the bird."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_path: Path):
        import numpy as np
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # shared host; a cutout is a once-per-species job
        self._np = np
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def cut(self, image: Image.Image) -> Image.Image | None:
        np = self._np
        rgb = image.convert("RGB")
        a = np.asarray(rgb.resize((320, 320), Image.LANCZOS), dtype=np.float32)
        a = a / max(float(a.max()), 1e-6)
        a = (a - np.array(self.MEAN, dtype=np.float32)) / np.array(self.STD, dtype=np.float32)
        pred = self._session.run(None, {self._input: a.transpose(2, 0, 1)[None].astype(np.float32)})[0]
        pred = pred[0, 0]
        pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-8)
        mask = Image.fromarray((pred * 255).astype(np.uint8), "L").resize(rgb.size, Image.LANCZOS)

        # rembg's post-process, softened: open away specks, blur, then a short
        # linear ramp instead of a hard threshold so the edge stays antialiased.
        mask = mask.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        mask = mask.point(lambda v: 0 if v < 96 else 255 if v > 160 else (v - 96) * 255 // 64)

        bbox = mask.point(lambda v: 255 if v > 16 else 0).getbbox()
        if not bbox:
            return None
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if bw * bh < 0.02 * rgb.width * rgb.height:
            return None  # model found nothing bird-sized; better the plain photo
        out = rgb.convert("RGBA")
        out.putalpha(mask)
        return out.crop(bbox)


class ImageResolver:
    def __init__(self, assets_dir: Path, data_dir: Path, *, wikipedia: bool = True,
                 cutout_model: Path | None = None, user_agent: str = "AvianVisitors-HA/1.0"):
        self.assets = assets_dir
        self.cutouts = data_dir / "cutouts"
        self.photos = data_dir / "photos"
        self.cutouts.mkdir(parents=True, exist_ok=True)
        self.photos.mkdir(parents=True, exist_ok=True)
        self._misses_path = data_dir / "image-misses.json"
        self._misses: dict[str, float] = self._load_json(self._misses_path)
        self.wikipedia = wikipedia
        self.user_agent = user_agent
        self._lock = threading.Lock()  # one cold fetch + model run at a time
        # The model is loaded per cold fetch and dropped afterwards: it costs
        # ~600 MB resident, and a cutout is a once-per-species event.
        self._model = cutout_model if cutout_model and cutout_model.is_file() else None
        log.info("background removal %s", f"enabled ({self._model.name})" if self._model else "disabled")

    # ---- bundled / cached -------------------------------------------------

    def bundled(self, sci: str, pose: int = 1) -> ResolvedImage | None:
        slug = slugify(sci)
        suffix = "" if pose == 1 else f"-{pose}"
        for path in (self.assets / "illustrations" / f"{slug}{suffix}.png",
                     self.assets / "illustrations" / f"{slug}.png"):
            if _usable(path):
                return ResolvedImage(path, "illustration", ILLUSTRATION_CREDIT)
        path = self.assets / "cutouts" / f"{slug}.png"
        if _usable(path):
            return ResolvedImage(path, "cutout", "AvianVisitors bundled cutout")
        return None

    def cached(self, sci: str) -> ResolvedImage | None:
        slug = slugify(sci)
        for path, kind in ((self.cutouts / f"{slug}.png", "photo-cutout"),
                           (self.photos / f"{slug}.png", "photo")):
            if _usable(path):
                meta = self._load_json(path.with_suffix(".json"))
                return ResolvedImage(path, kind, meta.get("credit", ""))
        return None

    def resolve(self, det: Detection, pose: int = 1, fetch: bool = True) -> ResolvedImage | None:
        found = self.bundled(det.sci, pose) or self.cached(det.sci)
        if found or not fetch:
            return found
        with self._lock:
            found = self.cached(det.sci)  # another thread may have just made it
            if found:
                return found
            last_miss = self._misses.get(det.slug, 0)
            if time.time() - last_miss < MISS_RETRY_SECONDS:
                return None
            try:
                found = self._fetch_and_cut(det)
            except Exception as exc:  # noqa: BLE001 - network/model errors are per-species
                log.warning("image fetch failed for %s: %s", det.sci, exc)
                found = None
            if found is None:
                self._misses[det.slug] = time.time()
                self._save_json(self._misses_path, self._misses)
            return found

    # ---- cold path --------------------------------------------------------

    def _candidates(self, det: Detection):
        """Photo sources, best first. BirdNET-Go's own BirdImage comes first:
        its default provider, Avicommons, is a curated set of single-bird
        portraits, while a Wikipedia lead image can be anything (the Snowy
        Owl lead is an owl carrying off a duck)."""
        if det.photo_url:
            credit = ", ".join(x for x in (det.photo_author, det.photo_license) if x)
            # BirdNET-Go asks Avicommons for 320 px; 900 px exists for every entry.
            big = re.sub(r"(static\.avicommons\.org/[^?#]+)-(?:240|320|480)\.jpg$", r"\1-900.jpg", det.photo_url)
            if big != det.photo_url:
                yield big, credit
            yield det.photo_url, credit
        if self.wikipedia:
            wiki = self._wikipedia_image(det.sci)
            if wiki:
                yield wiki

    def _fetch_and_cut(self, det: Detection) -> ResolvedImage | None:
        photo = None
        for url, credit in self._candidates(det):
            host = urllib.parse.urlparse(url).hostname or ""
            if not url.startswith("https://") or not ALLOWED_HOSTS.search(host):
                log.info("skipping image host %s for %s", host, det.sci)
                continue
            try:
                photo = Image.open(io.BytesIO(self._get(url)))
                photo.load()
                break
            except Exception as exc:  # noqa: BLE001 - try the next source
                log.info("image source failed for %s (%s): %s", det.sci, host, exc)
                photo = None
        if photo is None:
            return None
        if self._model:
            try:
                cut = Cutter(self._model).cut(photo)  # session freed when it goes out of scope
            except Exception as exc:  # noqa: BLE001 - fall back to the plain photo
                log.warning("background removal failed for %s: %s", det.sci, exc)
                cut = None
            if cut is not None:
                cut.thumbnail((800, 800), Image.LANCZOS)
                return self._store(self.cutouts / f"{det.slug}.png", cut, "photo-cutout", credit, url)
        flat = photo.convert("RGB")
        flat.thumbnail((800, 800), Image.LANCZOS)
        return self._store(self.photos / f"{det.slug}.png", flat, "photo", credit, url)

    def _store(self, path: Path, image: Image.Image, kind: str, credit: str, url: str) -> ResolvedImage:
        tmp = path.with_name(f".{path.name}.tmp")
        image.save(tmp, format="PNG", optimize=True)
        tmp.replace(path)  # atomic: readers never see a half-written PNG
        self._save_json(path.with_suffix(".json"), {"credit": credit, "source": url, "kind": kind})
        log.info("cached %s image for %s (%s)", kind, path.stem, credit or "no credit")
        return ResolvedImage(path, kind, credit)

    def _wikipedia_image(self, sci: str) -> tuple[str, str] | None:
        data = self._get_json("https://en.wikipedia.org/w/api.php", {
            "action": "query", "format": "json", "formatversion": "2", "redirects": "1",
            "prop": "pageimages", "piprop": "thumbnail|name", "pithumbsize": "1200",
            "titles": sci,
        })
        pages = (data.get("query") or {}).get("pages") or []
        page = pages[0] if pages else {}
        url = (page.get("thumbnail") or {}).get("source")
        if not url:
            return None
        return url, self._wikimedia_credit(page.get("pageimage", ""))

    def _wikimedia_credit(self, filename: str) -> str:
        if not filename:
            return "Wikipedia"
        try:
            data = self._get_json("https://en.wikipedia.org/w/api.php", {
                "action": "query", "format": "json", "formatversion": "2",
                "prop": "imageinfo", "iiprop": "extmetadata", "titles": f"File:{filename}",
            })
            meta = data["query"]["pages"][0]["imageinfo"][0]["extmetadata"]
        except Exception:  # noqa: BLE001 - attribution is best effort
            return "Wikimedia Commons"
        artist = re.sub(r"<[^>]+>", "", html.unescape((meta.get("Artist") or {}).get("value", ""))).strip()
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        return ", ".join(x for x in (artist, lic, "via Wikimedia Commons") if x)

    # ---- http / json helpers ---------------------------------------------

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=15) as resp:
            final_host = urllib.parse.urlparse(resp.geturl()).hostname or ""
            if not ALLOWED_HOSTS.search(final_host):
                raise ValueError(f"redirected off allowlist to {final_host}")
            body = resp.read(MAX_DOWNLOAD + 1)
        if len(body) > MAX_DOWNLOAD:
            raise ValueError("image larger than 12 MB")
        return body

    def _get_json(self, url: str, params: dict) -> dict:
        return json.loads(self._get(f"{url}?{urllib.parse.urlencode(params)}"))

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(path)


def png_bytes(path: Path, max_edge: int) -> bytes:
    """Downscaled PNG for MQTT: retained payloads stay well under 1 MB."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

