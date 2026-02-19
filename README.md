# PDF & Score Extractor

Generate a PDF from images and extract music scores to MusicXML. Upload images (JPG, PNG, HEIC, etc.) or a PDF; get one PDF and one or more MusicXML files.

## Pipeline

```
Images or PDF → PDF (image-only) + page images → preprocess → OMR (Audiveris) → MusicXML
```

## Quick Start

The image is **amd64 only** (Audiveris). On Apple Silicon Mac, Docker runs it via emulation.

```bash
docker build --platform linux/amd64 -t transcriber .
docker run -p 8080:8080 transcriber
# Open http://localhost:8080
```

Or use the helper script:

```bash
./trans.sh build
./trans.sh start
./trans.sh stop
```

## CLI

```bash
# Images → PDF + MusicXML
./trans.sh run page1.png page2.png -o ./output

# PDF → MusicXML (PDF is kept as-is; scores extracted from pages)
./trans.sh run scores.pdf -o ./output
```

Or with Docker:

```bash
docker run --rm -v /path/to/images:/input:ro -v /path/to/output:/output transcriber \
  python -m app.cli /input/page1.png /input/page2.png -o /output
```

## How It Works

1. **PDF** — If you upload images, they are combined into one PDF (image-only). If you upload a PDF, it is kept as-is.
2. **Pages** — Each page (from the PDF or from the image list) is preprocessed (deskew, perspective) and resized for speed.
3. **OMR** — Each page is run through Audiveris to produce MusicXML (image is amd64 only). **Lyrics**: if Audiveris does not attach lyrics, the pipeline runs Tesseract on the sheet image (chi_sim+eng) and injects text onto notes (one character per note for Chinese).
4. **Pieces** — Sections are grouped by final barlines; multi-page pieces are merged into one MusicXML each.
5. **Download** — You get the PDF plus `base.musicxml` (or `base_1.musicxml`, `base_2.musicxml`, …).

## Supported Input

- **Images:** JPG, PNG, BMP, TIFF, GIF, WebP, HEIC/HEIF
- **Document:** PDF

## Output

| File | Description |
|------|-------------|
| `{name}.pdf` | Generated from images, or your uploaded PDF |
| `{name}.musicxml` | One score (or `_1`, `_2`, … per piece) — open in MuseScore, Dorico, etc. |

## Tests

Use **童年.pdf** in the project root as the example file. Run unit tests (no Audiveris):

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Run the **integration test** (full pipeline on 童年.pdf; requires Audiveris, ~several minutes):

```bash
./trans.sh build   # if not already built
./trans.sh test    # runs pytest -m integration in Docker; needs 童年.pdf in repo root
```

The integration test checks that the pipeline produces MusicXML and that the output has title **童年**. When lyrics extraction works, you can add an assertion on `<lyric>` in `tests/test_pipeline.py`.

## Project Structure

```
producer/
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
├── trans.sh
├── 童年.pdf          # optional: example for tests
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py
└── app/
    ├── __init__.py
    ├── __main__.py   # python -m app → CLI
    ├── cli.py
    ├── main.py       # Flask: upload, status, download
    ├── image_pipeline.py   # PDF + OMR
    └── static/
        └── index.html
```

## Tech Stack

- **Python 3** + Flask + Gunicorn
- **Pillow** — image handling, PDF from images
- **PyPDF2** / **pdf2image** + Poppler — PDF handling
- **OpenCV** — deskew, perspective correction
- **Audiveris** — OMR (pitched and percussion; amd64 only)
- **music21** — merge multi-page MusicXML

## Limitations

- Single-user, no auth
- Score extraction requires the amd64 image (use `--platform linux/amd64` when building/running; on Mac, Docker uses emulation)
- Output should be verified in a notation editor
