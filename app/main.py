"""Flask app — upload, transcribe / process images, download."""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from app.pipeline import run as run_audio_pipeline
from app.image_pipeline import run as run_image_pipeline, IMAGE_EXTENSIONS, is_pdf

app = Flask(__name__, static_folder="static", static_url_path="/static")

UPLOAD_MAX = 500 * 1024 * 1024  # 500 MB
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "/tmp/transcriber"))

_jobs: dict[str, dict] = {}
_lock = threading.Lock()

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _extension_or_heic_from_file(f) -> str:
    """Return allowed extension from filename, or .heic from MIME type for HEIC/HEIF."""
    ext = (Path(f.filename).suffix or "").strip().lower()
    if ext and ext in ALLOWED_EXTENSIONS:
        return ext
    ct = (f.content_type or "").strip().lower()
    if ct in ("image/heic", "image/heif"):
        return ".heic"
    return ext


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    if not files or not files[0].filename:
        return jsonify(error="No file provided"), 400

    first_ext = _extension_or_heic_from_file(files[0])
    if first_ext not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type: {first_ext or 'unknown'}"), 400

    mode = "image" if (first_ext in IMAGE_EXTENSIONS or first_ext in PDF_EXTENSIONS) else "audio"
    stem = request.form.get("stem", "auto")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    original_names: list[str] = []

    for i, f in enumerate(files):
        ext = _extension_or_heic_from_file(f)
        if ext not in ALLOWED_EXTENSIONS:
            continue
        dest = job_dir / f"input_{i}{ext}"
        f.save(str(dest))
        saved_paths.append(str(dest))
        original_names.append(f.filename)

    if not saved_paths:
        return jsonify(error="No valid files uploaded"), 400

    base_name = Path(original_names[0]).stem

    with _lock:
        _jobs[job_id] = {
            "status": "queued", "step": "", "progress": 0,
            "error": None, "base_name": base_name, "mode": mode,
            "output_type": None, "output_files": [],
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, saved_paths, str(job_dir), stem,
              original_names, base_name, mode),
    )
    t.daemon = True
    t.start()

    return jsonify(job_id=job_id, base_name=base_name, mode=mode)


@app.route("/status")
def status():
    job_id = request.args.get("job_id", "")
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    return jsonify(**job)


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    if not filename.endswith((".mid", ".musicxml", ".json", ".pdf")):
        return jsonify(error="Invalid file type"), 400
    job_dir = OUTPUT_ROOT / job_id
    path = job_dir / filename
    if not path.exists():
        return jsonify(error="File not found"), 404
    return send_from_directory(str(job_dir), filename, as_attachment=True)


def _run_job(job_id: str, paths: list[str], work_dir: str, stem: str,
             original_names: list[str], base_name: str, mode: str):
    def on_progress(step, pct):
        with _lock:
            _jobs[job_id].update(status="processing", step=step, progress=pct)

    try:
        with _lock:
            _jobs[job_id]["status"] = "processing"

        if mode == "image":
            result = run_image_pipeline(
                paths, work_dir,
                base_name=base_name, on_progress=on_progress,
            )
            with _lock:
                _jobs[job_id].update(
                    status="done", progress=100, step="Done",
                    output_type="image",
                    output_files=result["files"],
                    message=result.get("message"),
                )
        else:
            run_audio_pipeline(
                paths[0], work_dir, stem=stem,
                original_name=original_names[0], on_progress=on_progress,
            )
            with _lock:
                _jobs[job_id].update(
                    status="done", progress=100, step="Done",
                    output_type="transcription",
                    output_files=[
                        f"{base_name}.musicxml",
                        f"{base_name}.mid",
                        f"{base_name}.json",
                    ],
                )
    except Exception as e:
        with _lock:
            _jobs[job_id].update(status="error", error=str(e))


def create_app():
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8080, debug=False)
