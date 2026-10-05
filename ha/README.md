# AvianVisitors for Home Assistant (BirdNET-Go bridge)

A small container that turns [BirdNET-Go](https://github.com/tphakala/birdnet-go)
detections into AvianVisitors art inside Home Assistant. It needs no Raspberry
Pi image and no web server, so it suits any host that already runs BirdNET-Go
and an MQTT broker.

```
BirdNET-Go ──MQTT birdnet──► avian-ha ──MQTT (discovery)──► Home Assistant
                              │
                              ├─ bundled illustration (666, 333 species)
                              ├─ bundled photo cutout (157 species)
                              └─ Wikipedia photo → U²-Net cutout → cached
```

## What shows up in Home Assistant

All of it arrives by MQTT discovery as one **Avian Visitors** device:

| Entity | What |
|---|---|
| `image.avian_visitors_latest_bird` | The species just heard, as its AvianVisitors illustration |
| `image.avian_visitors_today` | Today's collage: every species heard today, with frequent visitors drawn larger |
| `sensor.avian_visitors_latest_bird` | Common name. Attributes: `scientific_name`, `confidence`, `time`, `count_today`, `first_today`, `image_kind`, `image_credit`, `wikipedia` |
| `sensor.avian_visitors_species_today` | Distinct species today. The `visitors` attribute lists them with counts |
| `sensor.avian_visitors_detections_today` | Detections today |
| `event.avian_visitors_visitor` | Fires `new_species` for the first detection of a species each day, then `detection` |

The images are MQTT payloads, not URLs, so HA serves them over its own
connection. That means they work off-LAN through HA's own access, and in
companion-app notifications:

```yaml
triggers:
  - trigger: state
    entity_id: event.avian_visitors_visitor
conditions:
  - condition: template
    value_template: "{{ trigger.to_state.attributes.event_type == 'new_species' }}"
actions:
  - action: notify.mobile_app_phone
    data:
      title: "{{ trigger.to_state.attributes.common_name }}"
      message: "First visit today ({{ trigger.to_state.attributes.confidence }}%)"
      data:
        image: /api/image_proxy/image.avian_visitors_latest_bird
```

Dashboard:

```yaml
type: vertical-stack
cards:
  - type: picture-entity
    entity: image.avian_visitors_today
    show_name: false
    show_state: false
  - type: picture-entity
    entity: image.avian_visitors_latest_bird
    name: Latest
```

## How it picks an image

The lookup chain matches `avian/api/cutout.php`, slugged by scientific name
(`Calypte anna` → `calypte-anna`):

1. `avian/assets/illustrations/<slug>.png`, the bundled AvianVisitors art. The
   collage and latest-bird image use the perched pose.
2. `avian/assets/cutouts/<slug>.png`, a bundled background-removed photo.
3. `/data/cutouts/<slug>.png`, a cutout this container made earlier.
4. A fresh cutout. It takes the photo BirdNET-Go attached to the detection
   (`BirdImage.URL`), usually a curated Avicommons portrait, fetched at
   900 px rather than BirdNET-Go's 320. Failing that, it uses the species'
   Wikipedia lead image, which is less reliable as a portrait (the Snowy Owl
   lead shows an owl carrying a dead duck). It follows only `wikimedia.org`,
   `wikipedia.org` and `avicommons.org`. The U²-Net model (the one rembg
   uses, run directly through onnxruntime) removes the background, then the
   result is cropped and cached together with its attribution in `/data`.
   The model loads only for this step, so the container idles at a fraction
   of the RAM.
   If no cutout comes out, the plain photo is cached instead. A species with
   no photo anywhere gets a handwritten name card, and it isn't looked up
   again for 24 h.

Non-bird classes that BirdNET-Go can report (dog, human, engine…) are dropped
because they aren't binomial names. BirdNET-Go's retained message, which
replays on every reconnect, is de-duplicated by `detectionId`.

## Run it

```bash
docker build -f ha/Dockerfile -t avian-visitors-ha .      # from the repo root
```

See [`docker-compose.example.yaml`](docker-compose.example.yaml). Configuration
is by environment:

| Variable | Default | |
|---|---|---|
| `MQTT_HOST` / `MQTT_PORT` | `localhost` / `1883` | Broker |
| `MQTT_USER` / `MQTT_PASSWORD` (or `MQTT_PASSWORD_FILE`) | none | |
| `DETECTION_TOPIC` | `birdnet` | BirdNET-Go's `realtime.mqtt.topic` |
| `BASE_TOPIC` | `avian` | Where this bridge publishes |
| `DISCOVERY_PREFIX` | `homeassistant` | |
| `MIN_CONFIDENCE` | `0` | 0–1. BirdNET-Go already applies its own threshold |
| `SKIP_UNLIKELY` | `true` | Ignore detections BirdNET-Go flagged `Unlikely` |
| `WIKIPEDIA_FALLBACK` | `true` | Step 4 above |
| `CUTOUT_MODEL` | `/app/models/cutout.onnx` | Empty disables background removal |
| `COLLAGE_WIDTH` / `COLLAGE_HEIGHT` | `1200` / `800` | |
| `COLLAGE_TITLE` | `Avian Visitors` | |
| `COLLAGE_DARK` | `false` | Charcoal paper, as in the web UI's dark theme |
| `IMAGE_MAX_EDGE` | `640` | Size of the latest-bird image |
| `TZ` | `UTC` | Sets when "today" rolls over. Set it |

The build argument `CUTOUT_MODEL=u2netp` swaps the 176 MB model for the
4.5 MB lite one.

The broker user needs to read `birdnet` and `homeassistant/status`, and to
write `avian/#` and `homeassistant/+/avian_visitors/#`.

## Tests

```bash
cd ha && pip install -r requirements.txt pytest && python -m pytest -q tests
```

## Licence

Same as the rest of the repo: CC-BY-NC-SA-4.0. The bundled illustrations are
AvianVisitors'. Wikipedia-derived cutouts carry their photographer's credit in
`sensor.avian_visitors_latest_bird`'s `image_credit` attribute.
