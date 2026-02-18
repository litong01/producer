# Audio/Video → MusicXML Transcriber

A containerized tool that converts audio or video files into melody-only MusicXML using open-source tools. Built for publishers who need a quick first-pass transcription to manually verify and include in SoloBand bundles.

## Pipeline

```
Upload → FFmpeg (extract/normalize audio) → BasicPitch (audio→MIDI) → MuseScore (MIDI→MusicXML) → Download
```

## Quick Start

```bash
# Build the container
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

1. **Upload** — drag-and-drop or browse for an audio/video file in the web UI
2. **FFmpeg** — extracts audio, downmixes to mono, normalizes loudness, outputs WAV
3. **BasicPitch** — Spotify's neural MIDI transcription converts WAV → MIDI
4. **Melody extraction** — keeps only the most active track from the MIDI
5. **MuseScore** — converts the melody MIDI into clean MusicXML notation
6. **Download** — grab `melody.musicxml`, `melody.mid`, and `metadata.json`

## Output Files

| File | Description |
|------|-------------|
| `melody.musicxml` | Notation file — open in MuseScore, Dorico, Finale, etc. |
| `melody.mid` | Extracted melody as standard MIDI |
| `metadata.json` | Tempo, duration, timestamps |

## Supported Input Formats

**Audio:** MP3, WAV, FLAC, OGG, M4A, AAC, WMA  
**Video:** MP4, MKV, MOV, AVI, WebM

## Project Structure

```
producer/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── main.py          # Flask web server + endpoints
    ├── pipeline.py      # FFmpeg → BasicPitch → MuseScore pipeline
    └── static/
        └── index.html   # Single-page web UI
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | Upload file, returns `{ job_id }` |
| `GET` | `/status?job_id=X` | Poll job progress |
| `GET` | `/download/<job_id>/<file>` | Download output file |

## Tech Stack

- **Python 3.11** + Flask + Gunicorn
- **FFmpeg** — audio extraction & normalization
- **BasicPitch** (Spotify) — neural audio-to-MIDI transcription
- **MuseScore 4** — MIDI-to-MusicXML conversion
- **Docker** — single-container deployment

## Limitations

- Single-user tool (no auth, no database, no queue)
- Not for real-time use — run once per piece
- Melody extraction picks the most active MIDI track, which works well for solo/lead parts but may need manual cleanup for complex arrangements
