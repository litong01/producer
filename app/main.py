"""Flask app — upload, transcribe, download."""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from app.pipeline import run as run_pipeline

app = Flask(__name__, static_folder="static", static_url_path="/static")

UPLOAD_MAX = 500 * 1024 * 1024  # 500 MB
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "/tmp/transcriber"))

# In-memory job store (single-user tool, no DB needed)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file provided"), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type: {ext}"), 400

    stem = request.form.get("stem", "auto")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{ext}"
    f.save(str(input_path))

    with _lock:
        _jobs[job_id] = {"status": "queued", "step": "", "progress": 0, "error": None}

    t = threading.Thread(
        target=_run_job,
        args=(job_id, str(input_path), str(job_dir), stem),
    )
    t.daemon = True
    t.start()

    return jsonify(job_id=job_id)


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
    allowed = {"melody.mid", "melody.musicxml", "metadata.json"}
    if filename not in allowed:
        return jsonify(error="Invalid file"), 400
    job_dir = OUTPUT_ROOT / job_id
    path = job_dir / filename
    if not path.exists():
        return jsonify(error="File not found"), 404
    return send_from_directory(str(job_dir), filename, as_attachment=True)


def _run_job(job_id: str, input_path: str, work_dir: str, stem: str):
    def on_progress(step, pct):
        with _lock:
            _jobs[job_id].update(status="processing", step=step, progress=pct)

    try:
        with _lock:
            _jobs[job_id]["status"] = "processing"
        run_pipeline(input_path, work_dir, stem=stem, on_progress=on_progress)
        with _lock:
            _jobs[job_id].update(status="done", progress=100, step="Done")
    except Exception as e:
        with _lock:
            _jobs[job_id].update(status="error", error=str(e))


def create_app():
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8080, debug=False)
