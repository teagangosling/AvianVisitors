"""AvianVisitors -> Home Assistant bridge.

Listens for BirdNET-Go detections on MQTT, finds each species' image
the way AvianVisitors does (bundled illustration -> bundled cutout ->
cached/fresh background-removed photo), and publishes, all over MQTT with
Home Assistant discovery:

  image.<device>_latest_bird      the detected bird's illustration
  image.<device>_today            today's collage of every species heard
  sensor.<device>_latest_bird     common name; attributes carry the rest
  sensor.<device>_species_today / _detections_today
  event.<device>_visitor          fires 'detection' and 'new_species'

Images are MQTT payloads, not URLs, so they reach HA's frontend and the
companion app off-LAN through HA itself; nothing here listens on a port.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw

from . import collage
from .config import Settings
from .detection import Detection, parse
from .images import ImageResolver, png_bytes
from .state import DayState

log = logging.getLogger("avian_ha")

HEALTH_FILE = Path("/tmp/avian-ha.healthy")
COLLAGE_MIN_INTERVAL = 60  # s; a busy dawn chorus shouldn't re-render every few seconds


class Bridge:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.resolver = ImageResolver(cfg.assets_dir, cfg.data_dir, wikipedia=cfg.wikipedia_fallback,
                                      cutout_model=cfg.cutout_model, user_agent=cfg.user_agent)
        self.state = DayState(cfg.data_dir / "state.json")
        self.queue: queue.Queue[Detection | None] = queue.Queue(maxsize=500)
        self.stop = threading.Event()
        self._collage_due = True
        self._collage_at = 0.0

        self.t = {k: f"{cfg.base_topic}/{k}" for k in
                  ("status", "latest/state", "latest/image", "today/state", "today/image", "event")}
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cfg.mqtt_client_id)
        if cfg.mqtt_user:
            self.client.username_pw_set(cfg.mqtt_user, cfg.mqtt_password)
        self.client.will_set(self.t["status"], "offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(1, 60)

    # ---- MQTT callbacks (network thread: keep them fast) ------------------

    def _on_connect(self, client, _userdata, _flags, reason, _props):
        if reason.is_failure:
            log.error("MQTT connect refused: %s", reason)
            return
        log.info("connected to %s:%s", self.cfg.mqtt_host, self.cfg.mqtt_port)
        client.subscribe([(self.cfg.detection_topic, 0), (f"{self.cfg.discovery_prefix}/status", 0)])
        self._publish_discovery()
        client.publish(self.t["status"], "online", qos=1, retain=True)
        HEALTH_FILE.touch()

    def _on_disconnect(self, _client, _userdata, _flags, reason, _props):
        HEALTH_FILE.unlink(missing_ok=True)
        if not self.stop.is_set():
            log.warning("MQTT disconnected (%s); paho will reconnect", reason)

    def _on_message(self, _client, _userdata, msg):
        if msg.topic == f"{self.cfg.discovery_prefix}/status":
            if msg.payload == b"online":  # HA restarted: re-announce
                self._publish_discovery()
                self.client.publish(self.t["status"], "online", qos=1, retain=True)
            return
        det = parse(msg.payload)
        if det is None:
            return
        try:
            self.queue.put_nowait(det)
        except queue.Full:
            log.warning("queue full, dropping %s", det.sci)

    # ---- discovery --------------------------------------------------------

    def _publish_discovery(self) -> None:
        c, avail = self.cfg, [{"topic": self.t["status"]}]
        node = "avian_visitors"
        device = {"identifiers": [node], "name": c.device_name, "manufacturer": "AvianVisitors",
                  "model": "BirdNET-Go image bridge",
                  "sw_version": "1.0"}
        common = {"availability": avail, "device": device}
        configs = {
            ("image", "latest_bird"): {
                "name": "Latest bird", "image_topic": self.t["latest/image"],
                "content_type": "image/png", "icon": "mdi:bird"},
            ("image", "today"): {
                "name": "Today", "image_topic": self.t["today/image"],
                "content_type": "image/png", "icon": "mdi:image-multiple"},
            ("sensor", "latest_bird"): {
                "name": "Latest bird", "state_topic": self.t["latest/state"],
                "value_template": "{{ value_json.common_name }}",
                "json_attributes_topic": self.t["latest/state"], "icon": "mdi:bird"},
            ("sensor", "species_today"): {
                "name": "Species today", "state_topic": self.t["today/state"],
                "value_template": "{{ value_json.species }}", "unit_of_measurement": "species",
                "state_class": "measurement", "json_attributes_topic": self.t["today/state"],
                "json_attributes_template": "{{ {'visitors': value_json.visitors} | tojson }}",
                "icon": "mdi:feather"},
            ("sensor", "detections_today"): {
                "name": "Detections today", "state_topic": self.t["today/state"],
                "value_template": "{{ value_json.detections }}", "unit_of_measurement": "detections",
                "state_class": "measurement", "icon": "mdi:waveform"},
            ("event", "visitor"): {
                "name": "Visitor", "state_topic": self.t["event"],
                "event_types": ["detection", "new_species"], "icon": "mdi:bird"},
        }
        for (component, key), cfg in configs.items():
            uid = f"{node}_{key}"
            # default_entity_id replaced the deprecated object_id (HA 2025.10+).
            payload = {**common, **cfg, "unique_id": uid, "default_entity_id": f"{component}.{uid}"}
            self.client.publish(f"{c.discovery_prefix}/{component}/{node}/{key}/config",
                                json.dumps(payload), qos=1, retain=True)

    # ---- worker -----------------------------------------------------------

    def _handle(self, det: Detection) -> None:
        if det.unlikely and self.cfg.skip_unlikely:
            log.info("skip unlikely %s", det.sci)
            return
        if det.confidence < self.cfg.min_confidence:
            log.info("skip %s at %.0f%% (< %.0f%%)", det.com, det.confidence * 100, self.cfg.min_confidence * 100)
            return
        self.state.roll()
        new_today = det.slug not in self.state.species
        counted = self.state.add(det)
        if not counted and self.state.latest:
            return  # replayed retained message we've already shown

        image = self.resolver.resolve(det)
        when = det.when or datetime.now()
        tally = self.state.species.get(det.slug)
        latest = {
            "common_name": det.com,
            "scientific_name": det.sci,
            "confidence": round(det.confidence * 100, 1),
            "time": when.isoformat(timespec="seconds"),
            "source": det.source,
            "image_kind": image.kind if image else None,
            "image_credit": image.credit if image else None,
            "count_today": tally.count if tally else 0,
            "first_today": tally.first_seen if tally else None,
            "wikipedia": f"https://en.wikipedia.org/wiki/{det.sci.replace(' ', '_')}",
        }
        self.state.latest = latest
        self.state.save()

        payload = png_bytes(image.path, self.cfg.image_max_edge) if image else self._name_card(det.com)
        self.client.publish(self.t["latest/image"], payload, qos=1, retain=True)
        self.client.publish(self.t["latest/state"], json.dumps(latest), qos=1, retain=True)
        if counted:
            event = {**latest, "event_type": "new_species" if new_today else "detection"}
            self.client.publish(self.t["event"], json.dumps(event), qos=1, retain=False)
            self._publish_today()
            self._collage_due = True
        log.info("%s %s (%.0f%%) image=%s", "NEW" if new_today and counted else "   ",
                 det.com, det.confidence * 100, image.kind if image else "none")

    def _publish_today(self) -> None:
        visitors = self.state.visitors()
        self.client.publish(self.t["today/state"], json.dumps({
            "day": self.state.day.isoformat(),
            "species": len(visitors),
            "detections": sum(v.count for v in visitors),
            "visitors": [{"common_name": v.com, "scientific_name": v.sci, "count": v.count,
                          "last_seen": v.last_seen} for v in visitors],
        }), qos=1, retain=True)

    def _render_collage(self) -> None:
        visitors = []
        for t in self.state.visitors():
            det = Detection(t.sci, t.com, t.best_confidence, None)
            img = self.resolver.resolve(det, fetch=False)
            if img:
                visitors.append(collage.Visitor(t.com, t.sci, img.path, t.count))
        started = time.monotonic()
        png = collage.render(visitors, day=self.state.day, width=self.cfg.collage_width,
                             height=self.cfg.collage_height, title=self.cfg.collage_title,
                             font_path=self.cfg.font_path, dark=self.cfg.collage_dark)
        self.client.publish(self.t["today/image"], png, qos=1, retain=True)
        log.info("collage: %d species, %d KB, %.1fs", len(visitors), len(png) // 1024,
                 time.monotonic() - started)

    def _name_card(self, name: str) -> bytes:
        """Last resort when no image exists anywhere: the name, handwritten."""
        im = Image.new("RGBA", (self.cfg.image_max_edge, self.cfg.image_max_edge // 2), (0, 0, 0, 0))
        font = collage._font(self.cfg.font_path, self.cfg.image_max_edge // 9)
        draw = ImageDraw.Draw(im)
        w = draw.textlength(name, font=font)
        draw.text(((im.width - w) / 2, im.height / 2 - font.size / 2), name, font=font,
                  fill=collage.LIGHT["ink"] + (255,))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def worker(self) -> None:
        while not self.stop.is_set():
            try:
                det = self.queue.get(timeout=5)
            except queue.Empty:
                det = None
            try:
                if det is not None:
                    self._handle(det)
                if self.state.roll():
                    log.info("new day %s", self.state.day)
                    self.state.save()
                    self._publish_today()
                    self._collage_due = True
                if self._collage_due and time.monotonic() - self._collage_at >= COLLAGE_MIN_INTERVAL \
                        and self.queue.empty():
                    self._collage_due = False
                    self._collage_at = time.monotonic()
                    self._render_collage()
            except Exception:  # noqa: BLE001 - one bad detection mustn't kill the bridge
                log.exception("worker error")

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        self.client.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port, keepalive=60)
        self.client.loop_start()
        worker = threading.Thread(target=self.worker, name="worker", daemon=True)
        worker.start()
        # Re-publish today's tally on start so HA shows it before the first bird.
        while not self.stop.wait(1):
            if self.client.is_connected():
                self._publish_today()
                break
        self.stop.wait()
        log.info("shutting down")
        self.client.publish(self.t["status"], "offline", qos=1, retain=True).wait_for_publish(3)
        self.client.disconnect()
        self.client.loop_stop()
        worker.join(timeout=10)
        HEALTH_FILE.unlink(missing_ok=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Bridge(Settings.from_env()).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
