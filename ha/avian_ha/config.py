"""Environment-driven settings. Every knob has a default that matches a
stock BirdNET-Go + Mosquitto + Home Assistant install."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _secret(name: str) -> str:
    """NAME, or the contents of the file named by NAME_FILE (Docker secrets)."""
    path = os.environ.get(f"{name}_FILE", "").strip()
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return os.environ.get(name, "")


@dataclass(frozen=True)
class Settings:
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_user: str = ""
    mqtt_password: str = ""
    mqtt_client_id: str = "avian-visitors-ha"

    # BirdNET-Go's realtime.mqtt.topic. It publishes every detection here.
    detection_topic: str = "birdnet"
    # Where this bridge publishes its own state, images and events.
    base_topic: str = "avian"
    discovery_prefix: str = "homeassistant"
    device_name: str = "Avian Visitors"

    min_confidence: float = 0.0
    skip_unlikely: bool = True

    assets_dir: Path = Path("/app/assets")
    data_dir: Path = Path("/data")
    font_path: Path = Path("/app/fonts/Caveat.ttf")

    # Fallback chain beyond the bundled art.
    wikipedia_fallback: bool = True
    cutout_model: Path | None = Path("/app/models/cutout.onnx")
    user_agent: str = "AvianVisitors-HA/1.0 (+https://github.com/teagangosling/AvianVisitors)"

    collage_width: int = 1200
    collage_height: int = 800
    collage_title: str = "Avian Visitors"
    collage_dark: bool = False
    image_max_edge: int = 640

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.environ.get("CUTOUT_MODEL", "/app/models/cutout.onnx").strip()
        return cls(
            mqtt_host=os.environ.get("MQTT_HOST", cls.mqtt_host),
            mqtt_port=_int("MQTT_PORT", cls.mqtt_port),
            mqtt_user=os.environ.get("MQTT_USER", ""),
            mqtt_password=_secret("MQTT_PASSWORD"),
            mqtt_client_id=os.environ.get("MQTT_CLIENT_ID", cls.mqtt_client_id),
            detection_topic=os.environ.get("DETECTION_TOPIC", cls.detection_topic),
            base_topic=os.environ.get("BASE_TOPIC", cls.base_topic).rstrip("/"),
            discovery_prefix=os.environ.get("DISCOVERY_PREFIX", cls.discovery_prefix).rstrip("/"),
            device_name=os.environ.get("DEVICE_NAME", cls.device_name),
            min_confidence=_float("MIN_CONFIDENCE", cls.min_confidence),
            skip_unlikely=_bool("SKIP_UNLIKELY", cls.skip_unlikely),
            assets_dir=Path(os.environ.get("ASSETS_DIR", str(cls.assets_dir))),
            data_dir=Path(os.environ.get("DATA_DIR", str(cls.data_dir))),
            font_path=Path(os.environ.get("FONT_PATH", str(cls.font_path))),
            wikipedia_fallback=_bool("WIKIPEDIA_FALLBACK", cls.wikipedia_fallback),
            cutout_model=Path(model) if model else None,
            user_agent=os.environ.get("USER_AGENT", cls.user_agent),
            collage_width=_int("COLLAGE_WIDTH", cls.collage_width),
            collage_height=_int("COLLAGE_HEIGHT", cls.collage_height),
            collage_title=os.environ.get("COLLAGE_TITLE", cls.collage_title),
            collage_dark=_bool("COLLAGE_DARK", cls.collage_dark),
            image_max_edge=_int("IMAGE_MAX_EDGE", cls.image_max_edge),
        )
