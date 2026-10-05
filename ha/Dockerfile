# AvianVisitors -> Home Assistant bridge for BirdNET-Go.
# Build from the repo root so the bundled art is in context:
#   docker build -f ha/Dockerfile -t avian-visitors-ha .
# ha/Dockerfile.dockerignore keeps the BirdNET-Pi tree out of the context.

FROM python:3.12-slim AS model
# U^2-Net salient-object model, the one rembg ships as "u2net". Pinned by
# hash; pass CUTOUT_MODEL=u2netp for the 4.5 MB lite model instead.
ARG CUTOUT_MODEL=u2net
ARG U2NET_SHA256=8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491
ARG U2NETP_SHA256=309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8
ADD https://github.com/danielgatis/rembg/releases/download/v0.0.0/${CUTOUT_MODEL}.onnx /model.onnx
RUN case "$CUTOUT_MODEL" in \
      u2net)  want=$U2NET_SHA256 ;; \
      u2netp) want=$U2NETP_SHA256 ;; \
      *) echo "unknown CUTOUT_MODEL $CUTOUT_MODEL" >&2; exit 1 ;; \
    esac \
 && echo "$want  /model.onnx" | sha256sum -c - \
 && chmod 0644 /model.onnx

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=UTC
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin avian \
 && mkdir /data && chown avian:avian /data
WORKDIR /app
COPY ha/requirements.txt .
RUN pip install -r requirements.txt

COPY --from=model /model.onnx /app/models/cutout.onnx
COPY avian/frontend/fonts/Caveat.ttf avian/frontend/fonts/OFL-Caveat.txt /app/fonts/
COPY avian/assets/cutouts /app/assets/cutouts
COPY avian/assets/illustrations /app/assets/illustrations
COPY ha/avian_ha /app/avian_ha

USER avian
VOLUME /data
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
  CMD ["test", "-f", "/tmp/avian-ha.healthy"]
CMD ["python", "-m", "avian_ha"]
