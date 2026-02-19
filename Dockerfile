# Ubuntu 22.04 amd64: PDF from images + score extraction (Audiveris only).
# Build with: docker build --platform linux/amd64 -t scorextract .
# On Apple Silicon Mac, run the amd64 image; Docker will use emulation.
ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    poppler-utils \
    libheif-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Audiveris OMR: install deps then extract .deb (skip post-install to avoid xdg-desktop-menu/xdg-mime errors in Docker).
# Linux .deb installs to /opt/audiveris/bin/Audiveris — must be on PATH.
ENV AUDIVERIS_VERSION=5.8.1
ENV PATH="/opt/audiveris/bin:$PATH"
# Audiveris needs legacy Tesseract traineddata (4.x); apt ships LSTM-only, so OCR would not init and TEXTS/lyrics are skipped.
# Download legacy eng+chi_sim from tessdata 4.0.0 and set TESSDATA_PREFIX to that dir.
ENV TESSDATA_PREFIX=/opt/audiveris-tessdata
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libasound2 libasound2-data libxi6 libxtst6 x11-common xdg-utils \
        xvfb \
    && wget -q "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu22.04-x86_64.deb" -O /tmp/audiveris.deb \
    && dpkg-deb -x /tmp/audiveris.deb /tmp/audiveris-extract \
    && cp -a /tmp/audiveris-extract/* / \
    && rm -rf /tmp/audiveris.deb /tmp/audiveris-extract \
    && Audiveris -batch -help 2>/dev/null || true \
    && mkdir -p "$TESSDATA_PREFIX/tessdata" \
    && wget -q -O "$TESSDATA_PREFIX/tessdata/eng.traineddata" "https://github.com/tesseract-ocr/tessdata/raw/4.0.0/eng.traineddata" \
    && wget -q -O "$TESSDATA_PREFIX/tessdata/chi_sim.traineddata" "https://github.com/tesseract-ocr/tessdata/raw/4.0.0/chi_sim.traineddata" \
    && test -f "$TESSDATA_PREFIX/tessdata/eng.traineddata" && test -f "$TESSDATA_PREFIX/tessdata/chi_sim.traineddata" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ app/

# Verify app code is in the image (run: sha256sum app/image_pipeline.py locally to compare)
RUN echo "App code fingerprint:" && sha256sum app/image_pipeline.py app/main.py 2>/dev/null || true

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "--threads", "4", "app.main:create_app()"]
