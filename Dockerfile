FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    xvfb \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-cursor0 \
    libegl1 \
    libopengl0 \
    libgl1 \
    libfuse2 \
    && rm -rf /var/lib/apt/lists/*

# MuseScore 4 AppImage — extract for Docker (no FUSE in containers)
RUN wget -q "https://cdn.jsdelivr.net/musescore/v4.6.5/MuseScore-Studio-4.6.5.253511702-x86_64.AppImage" \
        -O /tmp/MuseScore.AppImage \
    && chmod +x /tmp/MuseScore.AppImage \
    && cd /opt \
    && /tmp/MuseScore.AppImage --appimage-extract > /dev/null 2>&1 \
    && ln -sf /opt/squashfs-root/AppRun /usr/local/bin/mscore \
    && rm /tmp/MuseScore.AppImage

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "--threads", "4", "app.main:create_app()"]
