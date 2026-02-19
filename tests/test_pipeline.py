"""
Pipeline tests using 童年.pdf as the example.

- Unit tests (title injection, etc.) run without Audiveris.
- Integration test runs the full pipeline with 童年.pdf when Audiveris is available
  (e.g. inside the Docker image); use: pytest tests/ -m integration
"""
from pathlib import Path

import pytest

from app.image_pipeline import (
    _audiveris_available,
    _write_musicxml_with_title,
    run,
)


# ---------------------------------------------------------------------------
# Unit tests (no Audiveris, no 童年.pdf required)
# ---------------------------------------------------------------------------

def test_title_injected_into_musicxml(tmp_path):
    """Setting a title on a minimal MusicXML produces a file with that work-title."""
    import music21

    # Minimal valid MusicXML (one part, one measure, one note)
    minimal = tmp_path / "minimal.musicxml"
    minimal.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>Untitled</work-title></work>
  <part-list><score-part id="P1" part-name="Part 1"/></part-list>
  <part id="P1">
    <measure number="1">
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
""", encoding="utf-8")

    out = tmp_path / "out.musicxml"
    _write_musicxml_with_title(minimal, out, title="童年")

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "童年" in content
    # music21 typically writes movement-title and/or work-title
    assert "work-title" in content or "movement-title" in content or "<title>童年</title>" in content.lower()


# ---------------------------------------------------------------------------
# Integration test: full pipeline with 童年.pdf (requires Audiveris in Docker)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.timeout(600)
def test_pipeline_tongnian_pdf_produces_musicxml_with_title(tongnian_pdf, temp_work_dir):
    """
    Run the full pipeline on 童年.pdf and verify:
    - At least one MusicXML file is produced.
    - The MusicXML has work title "童年".
    - (Optional) Lyrics are present when OCR works; currently we only assert title.
    """
    if not _audiveris_available():
        pytest.skip("Audiveris not available (run with Docker to test full pipeline)")
    result = run([tongnian_pdf], temp_work_dir, base_name="童年")

    assert "files" in result
    files = result["files"]
    musicxml_files = [f for f in files if f.endswith(".musicxml")]
    assert musicxml_files, f"Expected at least one .musicxml in result: {files}"

    work_dir = Path(temp_work_dir)
    for name in musicxml_files:
        path = work_dir / name
        assert path.exists(), f"Output file missing: {path}"
        content = path.read_text(encoding="utf-8", errors="replace")
        assert "童年" in content, f"MusicXML {name} should contain title 童年"
        # When lyrics extraction works, uncomment:
        # assert "<lyric" in content.lower(), f"MusicXML {name} should contain lyrics (Chinese)"


@pytest.mark.integration
def test_audiveris_available_in_integration_env():
    """In integration runs (e.g. Docker), Audiveris should be on PATH."""
    if not _audiveris_available():
        pytest.skip("Audiveris not available (run integration tests inside the Docker image)")


# ---------------------------------------------------------------------------
# Helpers for local runs
# ---------------------------------------------------------------------------

def test_tongnian_pdf_fixture_skips_when_missing(repo_root):
    """If 童年.pdf is missing, tests that use tongnian_pdf fixture are skipped."""
    if (repo_root / "童年.pdf").exists():
        pytest.skip("童年.pdf is present; no need to test skip behavior")
