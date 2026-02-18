"""
Audio/Video → MIDI → MusicXML transcription pipeline.

Steps:
  1. FFmpeg: extract audio, downmix mono, normalize → WAV
  2. BasicPitch: WAV → MIDI
  3. MuseScore CLI: MIDI → MusicXML
"""

import json
import subprocess
import time
from pathlib import Path

from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH
from mido import MidiFile, tempo2bpm


def run(input_path: str, work_dir: str, on_progress=None) -> dict:
    """Run the full pipeline. Returns metadata dict."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = Path(input_path)

    def _progress(step: str, pct: int):
        if on_progress:
            on_progress(step, pct)

    # --- Step 1: Extract audio with FFmpeg ---
    _progress("Extracting audio", 10)
    wav_path = work / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-vn",                # drop video
        "-ac", "1",           # mono
        "-ar", "22050",       # 22 kHz (BasicPitch expects this)
        "-sample_fmt", "s16",
        "-af", "loudnorm",    # EBU R128 normalization
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-2000:]}")

    # --- Step 2: BasicPitch → MIDI ---
    _progress("Transcribing audio to MIDI", 40)
    predict_and_save(
        audio_path_list=[str(wav_path)],
        output_directory=str(work),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )
    raw_midi = work / "audio_basic_pitch.mid"
    melody_midi = work / "melody.mid"
    if not raw_midi.exists():
        raise RuntimeError("BasicPitch produced no MIDI output")

    _progress("Extracting melody", 60)
    _extract_melody(raw_midi, melody_midi)

    # --- Step 3: MuseScore CLI → MusicXML ---
    _progress("Converting MIDI to MusicXML", 75)
    musicxml_path = work / "melody.musicxml"
    cmd = [
        "mscore", "--no-webview", "-o", str(musicxml_path), str(melody_midi),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        env=_musescore_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"MuseScore failed:\n{result.stderr[-2000:]}")

    # --- Step 4: Build metadata ---
    _progress("Generating metadata", 90)
    meta = _build_metadata(melody_midi, wav_path)
    meta_path = work / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    _progress("Done", 100)
    return meta


def _extract_melody(src: Path, dst: Path):
    """Keep only the track with the most notes (likely the melody)."""
    mid = MidiFile(str(src))
    best_track = None
    best_count = -1
    for track in mid.tracks:
        count = sum(1 for msg in track if msg.type == "note_on" and msg.velocity > 0)
        if count > best_count:
            best_count = count
            best_track = track

    out = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    out.tracks.append(best_track)
    out.save(str(dst))


def _build_metadata(midi_path: Path, wav_path: Path) -> dict:
    mid = MidiFile(str(midi_path))
    duration = mid.length

    tempo = 120
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = round(tempo2bpm(msg.tempo))
                break

    return {
        "source_file": wav_path.name,
        "tempo_bpm": tempo,
        "duration_seconds": round(duration, 2),
        "midi_ticks_per_beat": mid.ticks_per_beat,
        "num_tracks": len(mid.tracks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _musescore_env():
    """MuseScore needs a display; use virtual framebuffer."""
    import os
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env
