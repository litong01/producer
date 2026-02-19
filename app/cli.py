"""
CLI: images or PDF → PDF + MusicXML.

Usage:
  python -m app.cli page1.png page2.png -o ./output
  python -m app.cli scores.pdf -o ./output
"""

import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

from app.image_pipeline import is_document, is_pdf, run as run_image_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF from images and extract scores to MusicXML",
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Image file(s) or one PDF file",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Working directory for intermediate files (default: temp)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    existing = [p for p in args.inputs if p.exists()]
    for p in args.inputs:
        if not p.exists():
            print(f"SKIP {p} — file not found", file=sys.stderr)

    if not existing:
        print("No valid input files.", file=sys.stderr)
        sys.exit(1)

    doc_files = [p for p in existing if is_document(str(p))]
    if not doc_files:
        print("No image or PDF files. Supported: image (jpg, png, heic, …) or PDF.", file=sys.stderr)
        sys.exit(1)

    base = doc_files[0].stem
    kind = "PDF" if is_pdf(str(doc_files[0])) else "image"
    print(f"Processing {len(doc_files)} {kind} file(s)...")
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
        print(f"  Done → {args.output}")
        print(f"  Output: {', '.join(result['files'])}")
        if result.get("message"):
            print(f"  Note: {result['message']}")
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.work_dir:
            shutil.rmtree(str(work), ignore_errors=True)


if __name__ == "__main__":
    main()
