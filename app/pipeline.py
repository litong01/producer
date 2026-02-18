"""
Audio/Video → melody MusicXML transcription pipeline.

Pipeline order matters — each step feeds the next cleaner data:

  1. FFmpeg:      extract audio at full quality (44.1 kHz) for Demucs
  2. Demucs:      source separation → isolate vocals / lead / bass stem
  3. Prepare:     downmix isolated stem to mono 22 kHz WAV for BasicPitch
  4. Denoise:     spectral-gated noise reduction on the isolated stem
  5. BasicPitch:  clean mono stem → MIDI (conservative thresholds)
  6. MIDI filter: strip residual phantom notes (too short / too quiet)
  7. music21:     MIDI → quantized MusicXML
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from basic_pitch.inference import predict, ICASSP_2022_MODEL_PATH
from mido import MidiFile, MidiTrack, tempo2bpm

# Demucs outputs at this rate
DEMUCS_SR = 44100
# BasicPitch expects this rate
BASIC_PITCH_SR = 22050

# BasicPitch thresholds — raised from defaults (0.5 / 0.3) to suppress
# phantom notes from any residual bleed after source separation.
ONSET_THRESHOLD = 0.6
FRAME_THRESHOLD = 0.4
MINIMUM_NOTE_LEN_MS = 80
MINIMUM_VELOCITY = 40

# Which Demucs stems map to each user-facing choice.
# htdemucs produces: drums, bass, other, vocals
STEM_PRESETS = {
    "auto":       ["vocals", "other"],  # melody is usually voice or lead instrument
    "vocals":     ["vocals"],
    "instrument": ["other"],
    "bass":       ["bass"],
    "full":       None,                 # skip separation entirely
}


def run(input_path: str, work_dir: str, stem: str = "auto",
        on_progress=None) -> dict:
    """Run the full pipeline. Returns metadata dict."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = Path(input_path)
    stem = stem if stem in STEM_PRESETS else "auto"

    def _progress(step: str, pct: int):
        if on_progress:
            on_progress(step, pct)

    # --- Step 1: Extract audio at full quality for Demucs ---
    _progress("Extracting audio", 5)
    full_wav = work / "full.wav"
    _ffmpeg_extract(inp, full_wav)

    # --- Step 2: Source separation with Demucs ---
    stems_to_use = STEM_PRESETS[stem]
    if stems_to_use is not None:
        _progress("Separating sources (Demucs)", 10)
        stem_dir = _demucs_separate(full_wav, work)
        _progress("Mixing selected stems", 30)
        isolated_wav = work / "isolated.wav"
        _mix_stems(stem_dir, stems_to_use, isolated_wav)
    else:
        isolated_wav = full_wav

    # --- Step 3: Prepare for BasicPitch (mono, 22 kHz) ---
    _progress("Preparing audio", 35)
    clean_input = work / "bp_input.wav"
    _prepare_for_basicpitch(isolated_wav, clean_input)

    # --- Step 4: Noise reduction on isolated stem ---
    _progress("Reducing noise", 40)
    denoised = work / "denoised.wav"
    _denoise(clean_input, denoised)

    # --- Step 5: BasicPitch → MIDI ---
    _progress("Transcribing to MIDI", 50)
    raw_midi = work / "raw.mid"
    _transcribe(denoised, raw_midi)

    # --- Step 6: Filter phantom notes ---
    _progress("Filtering MIDI", 70)
    melody_midi = work / "melody.mid"
    _filter_midi(raw_midi, melody_midi)

    # --- Step 7: music21 → MusicXML ---
    _progress("Converting to MusicXML", 80)
    musicxml_path = work / "melody.musicxml"
    _midi_to_musicxml(melody_midi, musicxml_path)

    # --- Metadata ---
    _progress("Generating metadata", 95)
    meta = _build_metadata(melody_midi, full_wav, stem)
    (work / "metadata.json").write_text(json.dumps(meta, indent=2))

    _progress("Done", 100)
    return meta


# ---------------------------------------------------------------------------
# Step 1 — FFmpeg: extract audio at full quality
# ---------------------------------------------------------------------------

def _ffmpeg_extract(inp: Path, out: Path):
    """Extract audio preserving stereo and sample rate for best Demucs results.
    Only apply a high-pass to remove sub-bass rumble."""
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-vn",
        "-ar", str(DEMUCS_SR),
        "-af", "highpass=f=30",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Step 2 — Demucs: source separation
# ---------------------------------------------------------------------------

def _demucs_separate(wav_path: Path, work: Path) -> Path:
    """Run Demucs htdemucs model. Returns path to the stems directory."""
    out_dir = work / "separated"
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs",
        "--device", "cpu",
        "--out", str(out_dir),
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr[-2000:]}")

    stem_dir = out_dir / "htdemucs" / wav_path.stem
    if not stem_dir.exists():
        raise RuntimeError(f"Demucs produced no output in {stem_dir}")
    return stem_dir


# ---------------------------------------------------------------------------
# Step 2b — Combine selected stems
# ---------------------------------------------------------------------------

def _mix_stems(stem_dir: Path, stem_names: list[str], out_path: Path):
    """Load and sum the selected Demucs stems into a single WAV."""
    mixed = None
    sr = None
    for name in stem_names:
        path = stem_dir / f"{name}.wav"
        if not path.exists():
            continue
        audio, sr = sf.read(str(path))
        if mixed is None:
            mixed = audio.copy()
        else:
            mixed += audio

    if mixed is None:
        raise RuntimeError(f"No stems found for {stem_names} in {stem_dir}")

    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95

    sf.write(str(out_path), mixed, sr)


# ---------------------------------------------------------------------------
# Step 3 — Prepare for BasicPitch (mono, 22 kHz)
# ---------------------------------------------------------------------------

def _prepare_for_basicpitch(inp: Path, out: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-ac", "1",
        "-ar", str(BASIC_PITCH_SR),
        "-sample_fmt", "s16",
        "-af", ",".join([
            "highpass=f=80",
            "lowpass=f=8000",
            "loudnorm",
        ]),
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg (prepare) failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Step 4 — Spectral noise reduction
# ---------------------------------------------------------------------------

def _denoise(inp: Path, out: Path):
    audio, sr = sf.read(str(inp))
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.8,
        n_fft=2048,
        freq_mask_smooth_hz=200,
    )
    sf.write(str(out), reduced, sr)


# ---------------------------------------------------------------------------
# Step 5 — BasicPitch with conservative thresholds
# ---------------------------------------------------------------------------

def _transcribe(wav_path: Path, midi_out: Path):
    model_output, midi_data, note_events = predict(
        audio_path=str(wav_path),
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=ONSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        minimum_note_length=MINIMUM_NOTE_LEN_MS,
    )
    midi_data.write(str(midi_out))


# ---------------------------------------------------------------------------
# Step 6 — MIDI post-filter (velocity + duration)
# ---------------------------------------------------------------------------

def _filter_midi(src: Path, dst: Path):
    """Remove notes that are too quiet or too short — likely residual bleed."""
    mid = MidiFile(str(src))

    best_track = None
    best_count = -1
    for track in mid.tracks:
        count = sum(1 for m in track if m.type == "note_on" and m.velocity > 0)
        if count > best_count:
            best_count = count
            best_track = track

    filtered = _filter_track(best_track, mid.ticks_per_beat)

    out = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    out.tracks.append(filtered)
    out.save(str(dst))


def _filter_track(track: MidiTrack, tpb: int) -> MidiTrack:
    min_ticks = int(tpb * (MINIMUM_NOTE_LEN_MS / 1000) * 2)

    note_starts: dict[int, tuple[int, int]] = {}
    bad_notes: set[tuple[int, int]] = set()
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_starts[msg.note] = (abs_tick, msg.velocity)
        elif msg.type in ("note_off", "note_on") and (
            msg.type == "note_off" or msg.velocity == 0
        ):
            if msg.note in note_starts:
                start_tick, vel = note_starts.pop(msg.note)
                duration = abs_tick - start_tick
                if vel < MINIMUM_VELOCITY or duration < min_ticks:
                    bad_notes.add((msg.note, start_tick))

    out = MidiTrack()
    active_bad: dict[int, bool] = {}
    abs_tick = 0
    pending_delta = 0
    for msg in track:
        abs_tick += msg.time
        pending_delta += msg.time

        is_note_on = msg.type == "note_on" and msg.velocity > 0
        is_note_off = msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        )

        if is_note_on and (msg.note, abs_tick) in bad_notes:
            active_bad[msg.note] = True
            continue
        if is_note_off and active_bad.pop(msg.note, False):
            continue

        out.append(msg.copy(time=pending_delta))
        pending_delta = 0

    return out


# ---------------------------------------------------------------------------
# Step 7 — music21 MIDI → MusicXML
# ---------------------------------------------------------------------------

def _midi_to_musicxml(midi_path: Path, out_path: Path):
    import music21

    score = music21.converter.parse(str(midi_path), quantizePost=True)

    score.quantize(
        quarterLengthDivisors=(4, 3),
        inPlace=True,
    )

    for part in score.parts:
        part.makeRests(fillGaps=True, inPlace=True)
        part.makeMeasures(inPlace=True)

    score.write("musicxml", fp=str(out_path))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(midi_path: Path, wav_path: Path, stem: str) -> dict:
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
        "stem_mode": stem,
        "tempo_bpm": tempo,
        "duration_seconds": round(duration, 2),
        "midi_ticks_per_beat": mid.ticks_per_beat,
        "num_tracks": len(mid.tracks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
