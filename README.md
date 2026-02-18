# Audio/Video → MusicXML Transcriber

A containerized tool that extracts melody or lead instrument from audio/video files and produces a MusicXML transcription. Built for publishers who need a first-pass transcription to manually verify and include in SoloBand bundles.

## Pipeline

```
Upload → FFmpeg → Demucs (source separation) → Noise reduction → BasicPitch (→MIDI) → music21 (→MusicXML) → Download
```

## Quick Start

```bash
# Build the container (first build downloads ~2 GB of ML models)
docker build -t transcriber .

# Run it
docker run -p 8080:8080 transcriber

# Open http://localhost:8080 in your browser
```

To persist outputs to your host machine:

```bash
docker run -p 8080:8080 -v $(pwd)/output:/tmp/transcriber transcriber
```

## How It Works

1. **Upload** — drag-and-drop or browse for an audio/video file; choose what to extract
2. **FFmpeg** — extracts audio at full quality (44.1 kHz) for source separation
3. **Demucs** (Meta) — AI source separation splits audio into 4 stems: vocals, bass, drums, other instruments
4. **Stem selection** — combines the stems you chose (e.g. vocals + lead instrument for "Auto")
5. **Noise reduction** — spectral gating on the isolated stem removes residual room noise / reverb
6. **BasicPitch** (Spotify) — neural MIDI transcription with raised thresholds to suppress phantom notes
7. **MIDI filter** — strips notes that are too short or too quiet (likely artifacts)
8. **music21** (MIT) — quantizes and converts clean MIDI into readable MusicXML notation
9. **Download** — grab `melody.musicxml`, `melody.mid`, and `metadata.json`

## Stem Selection

The web UI lets you choose what to extract:

| Option | What it does |
|--------|-------------|
| **Auto** (default) | Vocals + lead instrument combined — best for most music |
| **Vocals only** | Singing voice isolated from everything else |
| **Lead instrument only** | Non-vocal melodic instruments (guitar solo, sax, synth lead, etc.) |
| **Bass line** | Isolated bass part |
| **Full mix** | Skip separation entirely — use for solo instrument recordings |

## Output Files

| File | Description |
|------|-------------|
| `melody.musicxml` | Notation file — open in MuseScore, Dorico, Finale, etc. |
| `melody.mid` | Extracted melody as standard MIDI |
| `metadata.json` | Tempo, duration, stem mode, timestamps |

## Supported Input Formats

**Audio:** MP3, WAV, FLAC, OGG, M4A, AAC, WMA
**Video:** MP4, MKV, MOV, AVI, WebM

## Project Structure

```
producer/
├── .github/
│   └── workflows/
│       └── docker-publish.yml   # Multi-arch build + Docker Hub push
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── main.py          # Flask web server + endpoints
    ├── pipeline.py      # Full transcription pipeline
    └── static/
        └── index.html   # Single-page web UI
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | Upload file + stem choice, returns `{ job_id }` |
| `GET` | `/status?job_id=X` | Poll job progress |
| `GET` | `/download/<job_id>/<file>` | Download output file |

## Tech Stack

- **Python 3.11** + Flask + Gunicorn
- **FFmpeg** — audio extraction & frequency filtering
- **Demucs** (Meta Research) — AI source separation into stems
- **noisereduce** — spectral-gated noise reduction
- **BasicPitch** (Spotify) — neural audio-to-MIDI transcription
- **music21** (MIT) — MIDI quantization & MusicXML export
- **Docker** — single-container deployment (works on both arm64 and x86_64)

## CI / Docker Hub Publishing

A GitHub Actions workflow builds multi-arch images (`linux/amd64` + `linux/arm64`) and pushes to Docker Hub on every push to `main` or version tag.

**Setup:** Add these secrets to your GitHub repo (Settings → Secrets → Actions):

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | A Docker Hub [access token](https://hub.docker.com/settings/security) |

**Triggers:**

- Push to `main` → tagged as `latest`
- Push a tag like `v1.0.0` → tagged as `1.0.0`, `1.0`, and the git SHA

**Pull the published image:**

```bash
docker pull <your-dockerhub-username>/transcriber:latest
docker run -p 8080:8080 <your-dockerhub-username>/transcriber:latest
```

## Limitations

- Single-user tool (no auth, no database, no queue)
- Not for real-time use — run once per piece
- Source separation quality depends on the recording; very dense mixes may have some bleed between stems
- Output should always be manually verified in a notation editor before publishing
