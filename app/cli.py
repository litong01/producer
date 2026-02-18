"""
CLI interface for the transcription / image / PDF processing pipeline.

Usage:
  python -m app.cli input.mp4 -o ./output
  python -m app.cli input.mp4 -o ./output --stem vocals
  python -m app.cli page1.png page2.png -o ./output     # images → PDF + MusicXML
  python -m app.cli scores.pdf -o ./output               # PDF → MusicXML
  python -m app.cli *.mp4 -o ./output                   # batch audio
"""

import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from app.pipeline import run as run_audio_pipeline, STEM_PRESETS
from app.image_pipeline import is_image, is_pdf, is_document, run as run_image_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video or process images",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Input file(s) — audio, video, or image",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--stem", default="auto", choices=list(STEM_PRESETS.keys()),
        help="What to extract for audio/video (default: auto)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Working directory for intermediate files (default: temp)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    existing = [p for p in args.inputs if p.exists()]
    missing = [p for p in args.inputs if not p.exists()]
    for p in missing:
        print(f"SKIP {p} — file not found", file=sys.stderr)

    if not existing:
        print("No valid input files.", file=sys.stderr)
        sys.exit(1)

    doc_files = [p for p in existing if is_document(str(p))]
    audio_files = [p for p in existing if not is_document(str(p))]

    failed = []

    # --- Process images/PDFs → PDF + MusicXML per score ---
    if doc_files:
        base = doc_files[0].stem
        kind = "PDF" if is_pdf(str(doc_files[0])) else "image"
        print(f"Processing {len(doc_files)} {kind} file(s)...")
        t0 = time.time()
        work = args.work_dir or Path(f"/tmp/transcriber-cli-{uuid.uuid4().hex[:8]}")
        work.mkdir(parents=True, exist_ok=True)

        try:
            result = run_image_pipeline(
                [str(p) for p in doc_files], str(work),
                base_name=base,
                on_progress=lambda step, pct: print(f"  {pct:3d}% {step}"),
            )
            for fname in result["files"]:
                src = work / fname
                if src.exists():
                    shutil.copy2(str(src), str(args.output / fname))
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s → {args.output}")
            print(f"  Output: {', '.join(result['files'])}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.extend(doc_files)
        finally:
            if not args.work_dir:
                shutil.rmtree(str(work), ignore_errors=True)

    # --- Process audio/video files individually ---
    for i, input_path in enumerate(audio_files, 1):
        name = input_path.stem
        print(f"[{i}/{len(audio_files)}] Processing: {input_path.name}")
        t0 = time.time()
        work = args.work_dir or Path(f"/tmp/transcriber-cli-{uuid.uuid4().hex[:8]}")
        work.mkdir(parents=True, exist_ok=True)

        try:
            run_audio_pipeline(
                str(input_path), str(work),
                stem=args.stem,
                original_name=input_path.name,
                on_progress=lambda step, pct: print(f"  {pct:3d}% {step}"),
            )
            out_dir = args.output if len(audio_files) == 1 else args.output / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname in [f"{name}.musicxml", f"{name}.mid", f"{name}.json"]:
                src = work / fname
                if src.exists():
                    shutil.copy2(str(src), str(out_dir / fname))
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s → {out_dir}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(input_path)
        finally:
            if not args.work_dir:
                shutil.rmtree(str(work), ignore_errors=True)

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
