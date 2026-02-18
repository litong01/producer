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
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

import numpy as np
import noisereduce as nr
import soundfile as sf
import torch
from basic_pitch.inference import predict, ICASSP_2022_MODEL_PATH
from demucs.apply import apply_model
from demucs.pretrained import get_model
from mido import MidiFile, MidiTrack, tempo2bpm

# Demucs outputs at this rate
DEMUCS_SR = 44100
# BasicPitch expects this rate
BASIC_PITCH_SR = 22050

# BasicPitch thresholds — balance between phantom notes and missing real notes.
# Slightly lower than very conservative values so melody recall is better.
ONSET_THRESHOLD = 0.55
FRAME_THRESHOLD = 0.35
MINIMUM_NOTE_LEN_MS = 60
MINIMUM_VELOCITY = 35

# Pitch ranges per stem mode.  Notes outside are stripped.
# Default range covers most melody instruments (C3–C7).
# Vocal range is tighter: E2–C6 covers bass to soprano with some headroom.
# Bass range: E1–E3.
PITCH_RANGES = {
    "auto":       (48, 96),   # C3–C7
    "vocals":     (40, 84),   # E2–C6  (bass voice low E to soprano high C)
    "instrument": (48, 96),   # C3–C7
    "bass":       (28, 52),   # E1–E3
    "full":       (48, 96),   # C3–C7
}

# Which Demucs stems map to each user-facing choice.
# htdemucs produces: drums, bass, other, vocals
STEM_PRESETS = {
    "auto":       ["vocals", "other"],
    "vocals":     ["vocals"],
    "instrument": ["other"],
    "bass":       ["bass"],
    "full":       None,
}


def run(input_path: str, work_dir: str, stem: str = "auto",
        original_name: str = "", on_progress=None) -> dict:
    """Run the full pipeline. Returns metadata dict."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = Path(input_path)
    stem = stem if stem in STEM_PRESETS else "auto"
    base_name = Path(original_name).stem if original_name else inp.stem

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
    melody_midi = work / f"{base_name}.mid"
    _filter_midi(raw_midi, melody_midi, stem=stem)

    # --- Step 7: music21 → MusicXML ---
    _progress("Converting to MusicXML", 80)
    musicxml_path = work / f"{base_name}.musicxml"
    _midi_to_musicxml(melody_midi, musicxml_path, title=base_name)

    # --- Metadata ---
    _progress("Generating metadata", 95)
    meta = _build_metadata(melody_midi, full_wav, stem)
    meta["base_name"] = base_name
    (work / f"{base_name}.json").write_text(json.dumps(meta, indent=2))

    _progress("Done", 100)
    return meta


# ---------------------------------------------------------------------------
# Step 1 — FFmpeg: extract audio at full quality
# ---------------------------------------------------------------------------

def _ffmpeg_extract(inp: Path, out: Path):
    """Extract audio at high quality for Demucs: stereo, 44.1 kHz, 16-bit.
    Use best audio stream, high-pass to remove rumble, avoid negative timestamps
    for clean video sync."""
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-vn", "-sn", "-dn",
        "-map", "0:a:0",
        "-c:a", "pcm_s16le",
        "-ar", str(DEMUCS_SR),
        "-ac", "2",
        "-af", "highpass=f=30",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Step 2 — Demucs: source separation (Python API, avoids torchaudio.save)
# ---------------------------------------------------------------------------

_demucs_model = None

def _get_demucs_model():
    global _demucs_model
    if _demucs_model is None:
        _demucs_model = get_model("htdemucs")
        _demucs_model.eval()
    return _demucs_model


# htdemucs source order
_STEM_NAMES = ["drums", "bass", "other", "vocals"]


def _demucs_separate(wav_path: Path, work: Path) -> Path:
    """Run Demucs htdemucs model. Returns path to the stems directory."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)  # mono → stereo
    # soundfile returns [samples, channels], torch wants [channels, samples]
    wav = torch.from_numpy(audio.T).float()

    model = _get_demucs_model()

    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu")[0]
    # sources shape: [num_sources, channels, samples]

    stem_dir = work / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(_STEM_NAMES):
        stem_audio = sources[i].cpu().numpy().T  # → [samples, channels]
        sf.write(str(stem_dir / f"{name}.wav"), stem_audio, sr)

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
    """Mono 22 kHz for BasicPitch. Keep more high end (10 kHz) for clarity;
    gentle peak normalize in code instead of loudnorm to avoid over-compression."""
    cmd = [
        "ffmpeg", "-y", "-i", str(inp),
        "-ac", "1",
        "-ar", str(BASIC_PITCH_SR),
        "-sample_fmt", "s16",
        "-af", "highpass=f=80,lowpass=f=10000",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg (prepare) failed:\n{result.stderr[-2000:]}")
    # Peak normalize so BasicPitch gets consistent level without loudnorm dynamics
    audio, sr = sf.read(str(out), dtype="float64")
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    sf.write(str(out), audio.astype("float32"), sr)


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

def _filter_midi(src: Path, dst: Path, stem: str = "auto"):
    """Remove notes that are too quiet, too short, or out of pitch range."""
    mid = MidiFile(str(src))

    best_track = None
    best_count = -1
    for track in mid.tracks:
        count = sum(1 for m in track if m.type == "note_on" and m.velocity > 0)
        if count > best_count:
            best_count = count
            best_track = track

    note_low, note_high = PITCH_RANGES.get(stem, PITCH_RANGES["auto"])
    filtered = _filter_track(best_track, mid.ticks_per_beat, note_low, note_high)

    out = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    out.tracks.append(filtered)
    out.save(str(dst))


def _filter_track(track: MidiTrack, tpb: int,
                  note_low: int, note_high: int) -> MidiTrack:
    min_ticks = int(tpb * (MINIMUM_NOTE_LEN_MS / 1000) * 2)

    note_starts: dict[int, tuple[int, int]] = {}
    bad_notes: set[tuple[int, int]] = set()
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note < note_low or msg.note > note_high:
                bad_notes.add((msg.note, abs_tick))
            else:
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

MIN_QUARTER_LENGTH = 0.25  # 16th note — nothing shorter in the output


def _midi_to_musicxml(midi_path: Path, out_path: Path, title: str = "Melody"):
    import music21

    score = music21.converter.parse(str(midi_path), quantizePost=True)

    score.quantize(
        quarterLengthDivisors=(4, 3),
        inPlace=True,
    )

    # Set score metadata so the title reflects the uploaded file name
    # instead of music21's auto-generated "Music32 Fragment" etc.
    if score.metadata is None:
        score.metadata = music21.metadata.Metadata()
    score.metadata.title = title
    score.metadata.movementName = title

    # Flatten to a single part — melody should never be a grand staff.
    # music21 sometimes splits overlapping notes into multiple parts.
    if len(score.parts) > 1:
        merged = score.parts[0]
        for extra_part in list(score.parts[1:]):
            for el in extra_part.recurse().getElementsByClass(music21.note.GeneralNote):
                merged.insert(el.getOffsetInHierarchy(extra_part), el)
            score.remove(extra_part)

    part = score.parts[0]
    part.partName = "Melody"

    if not part.getElementsByClass(music21.meter.TimeSignature):
        part.insert(0, music21.meter.TimeSignature("4/4"))
    if not part.getElementsByClass(music21.key.KeySignature):
        part.insert(0, music21.key.KeySignature(0))
    if not part.getElementsByClass(music21.clef.Clef):
        part.insert(0, music21.clef.TrebleClef())

    # First makeNotation pass — establish proper measure structure
    score = score.makeNotation()

    # Remove tiny notes within valid measure boundaries
    for part in score.parts:
        _remove_tiny_notes(part)

    # Detect and collapse repeated sections
    score = _detect_and_collapse_repeats(score)

    # Final makeNotation pass — fix any gaps left by modifications
    for part in score.parts:
        for measure in part.getElementsByClass(music21.stream.Measure):
            measure.makeRests(fillGaps=True, inPlace=True)

    # Write and post-process for MuseScore compatibility
    score.write("musicxml", fp=str(out_path))
    _fix_musicxml_for_musescore(out_path)


def _remove_tiny_notes(part):
    """Replace notes shorter than a 16th note with rests of the same
    duration, preserving measure timing.  32nd and 64th notes in a melody
    transcription are artifacts, not real musical content."""
    import music21

    for measure in part.getElementsByClass(music21.stream.Measure):
        to_replace = []
        for n in measure.recurse().getElementsByClass(music21.note.GeneralNote):
            if (float(n.quarterLength) < MIN_QUARTER_LENGTH
                    and not isinstance(n, music21.note.Rest)
                    and float(n.quarterLength) > 0):
                to_replace.append(n)

        for n in to_replace:
            r = music21.note.Rest(quarterLength=n.quarterLength)
            measure.replace(n, r)


def _fix_musicxml_for_musescore(out_path: Path):
    """Post-process the MusicXML file to fix known MuseScore compatibility
    issues: ensure correct XML declaration, DOCTYPE, and version attribute."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(out_path))
    root = tree.getroot()

    # Ensure the root element has the correct MusicXML version
    if root.tag == "score-partwise":
        root.set("version", "4.0")

    # Write back with XML declaration
    tree.write(
        str(out_path),
        xml_declaration=True,
        encoding="UTF-8",
    )

    # Prepend the DOCTYPE that MuseScore expects
    content = out_path.read_text(encoding="utf-8")
    if "<!DOCTYPE" not in content:
        content = content.replace(
            '<?xml version=\'1.0\' encoding=\'UTF-8\'?>',
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"\n'
            '  "http://www.musicxml.org/dtds/partwise.dtd">',
        )
        out_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Repeat detection — fingerprint, detect, collapse
# ---------------------------------------------------------------------------

_rep_log = logging.getLogger("pipeline.repeats")

MIN_SECTION_LEN = 4       # minimum measures in a repeated section
MIN_REPEATS = 2           # section must appear at least this many times
# Lower thresholds so repeats still match when BasicPitch transcribes
# the same passage slightly differently each time (extra/missing notes, timing).
MEASURE_SIM_THRESHOLD = 0.58   # SequenceMatcher ratio for two measures
SECTION_SIM_THRESHOLD = 0.58   # fraction of measures that must pass
MAX_COLLAPSE_PASSES = 5        # more passes to catch nested repeats


def _fingerprint_measure(measure) -> tuple:
    """Ordered pitch-class sequence (rests omitted).  Preserves melodic
    contour so that measures with the same pitches in different order
    (e.g. verse vs chorus) are distinguished, while ignoring octave and
    duration — the two things BasicPitch varies most between repetitions."""
    import music21
    pcs = []
    for n in measure.recurse().getElementsByClass(music21.note.GeneralNote):
        if isinstance(n, music21.note.Note):
            pcs.append(n.pitch.pitchClass)
        elif isinstance(n, music21.chord.Chord):
            for p in sorted(n.pitches, key=lambda p: p.midi):
                pcs.append(p.pitchClass)
    return tuple(pcs)


def _measure_similarity(fp_a: tuple, fp_b: tuple) -> float:
    """Sequence similarity via difflib.SequenceMatcher (0.0–1.0).
    Handles extra/missing notes (insertions/deletions) gracefully while
    requiring that the melodic order of pitch classes be similar."""
    from difflib import SequenceMatcher
    if fp_a == fp_b:
        return 1.0
    if not fp_a and not fp_b:
        return 1.0
    if not fp_a or not fp_b:
        return 0.0
    return SequenceMatcher(None, fp_a, fp_b).ratio()


def _sections_similar(fps: list[tuple], start_a: int, start_b: int,
                      length: int) -> bool:
    """True if >= SECTION_SIM_THRESHOLD of measure pairs individually
    exceed MEASURE_SIM_THRESHOLD."""
    if length == 0:
        return False
    matching = 0
    for i in range(length):
        if _measure_similarity(fps[start_a + i], fps[start_b + i]) >= MEASURE_SIM_THRESHOLD:
            matching += 1
    return matching / length >= SECTION_SIM_THRESHOLD


def _section_has_content(fps: list[tuple], start: int, length: int) -> bool:
    """At least half the measures must contain notes to avoid collapsing
    stretches of silence."""
    non_empty = sum(1 for i in range(length) if len(fps[start + i]) > 0)
    return non_empty >= length * 0.5


def _find_best_repeat(fps: list[tuple]) -> tuple[int, int, int] | None:
    """Find the consecutively-repeated section that saves the most measures.
    Returns (start_index, section_length, repeat_count) or None."""
    n = len(fps)
    best = None
    best_score = 0

    for sec_len in range(MIN_SECTION_LEN, n // MIN_REPEATS + 1):
        for start in range(n - sec_len * MIN_REPEATS + 1):
            if not _section_has_content(fps, start, sec_len):
                continue
            count = 1
            next_start = start + sec_len
            while next_start + sec_len <= n:
                if _sections_similar(fps, start, next_start, sec_len):
                    count += 1
                    next_start += sec_len
                else:
                    break
            if count >= MIN_REPEATS:
                score = sec_len * (count - 1)
                if score > best_score:
                    best_score = score
                    best = (start, sec_len, count)

    return best


def _detect_and_collapse_repeats(score):
    """Iteratively detect and collapse consecutively repeated sections.
    Each pass finds the single best (most-measures-saved) repeat, collapses
    it with repeat barlines, then re-scans the shorter score."""
    import music21

    for part in score.parts:
        for pass_num in range(1, MAX_COLLAPSE_PASSES + 1):
            measures = list(part.getElementsByClass(music21.stream.Measure))
            if len(measures) < MIN_SECTION_LEN * MIN_REPEATS:
                break

            fps = [_fingerprint_measure(m) for m in measures]
            _rep_log.info("Pass %d: %d measures", pass_num, len(measures))

            result = _find_best_repeat(fps)
            if result is None:
                _rep_log.info("No more repeated sections found")
                break

            start_idx, sec_len, repeat_count = result
            saved = sec_len * (repeat_count - 1)
            _rep_log.info(
                "Collapsing measures %d–%d (x%d), removing %d measures",
                start_idx + 1, start_idx + sec_len,
                repeat_count, saved,
            )

            first_measure = measures[start_idx]
            last_measure = measures[start_idx + sec_len - 1]
            first_measure.leftBarline = music21.bar.Repeat(direction="start")
            last_measure.rightBarline = music21.bar.Repeat(
                direction="end", times=repeat_count,
            )

            to_remove = []
            for r in range(1, repeat_count):
                for i in range(sec_len):
                    to_remove.append(measures[start_idx + r * sec_len + i])
            for m in to_remove:
                part.remove(m)

            for i, m in enumerate(
                part.getElementsByClass(music21.stream.Measure), 1,
            ):
                m.number = i

    return score


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
