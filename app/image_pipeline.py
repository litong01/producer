"""
Image / PDF pipeline: generate PDF from images and extract scores to MusicXML.

Accepts either:
  A) One or more image files (sorted by stem), or
  B) A single PDF file.

Steps:
  1. PDF: For images → combine into one PDF. For PDF input → keep as-is.
  2. Extract page images (from PDF) or use uploaded images.
  3. Preprocess (deskew, perspective). Resize for speed.
  4. Run OMR (Audiveris) per page.
  5. Group by final barlines; merge multi-page pieces.
  6. Output one MusicXML per piece.

Output:
  - {base_name}.pdf
  - {base_name}.musicxml (or _1, _2, …)
"""

import logging
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

log = logging.getLogger("pipeline.image")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".heic", ".heif",
}

# Skew angles beyond this (degrees) are corrected; smaller is noise.
MIN_SKEW_ANGLE = 0.5
MAX_SKEW_ANGLE = 15.0

# Max long edge (px) for the whole pipeline (resize → preprocess → PDF → OMR).
# Audiveris needs sufficient resolution (e.g. ~300 DPI equivalent) or it flags "no staves / resolution too low"
# and may not run export; 2400 gives a better chance of completing transcription and getting .mxl.
MAX_WORKING_LONG_EDGE = 2400


def is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_pdf(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def is_document(path: str) -> bool:
    """Return True for any file type handled by this pipeline."""
    return is_image(path) or is_pdf(path)


def _resize_by_long_edge(img: Image.Image, max_pixels: int) -> Image.Image:
    """Return image resized so the longer side is at most max_pixels (unchanged if already smaller)."""
    w, h = img.size
    if w <= max_pixels and h <= max_pixels:
        return img
    if w >= h:
        new_w = max_pixels
        new_h = int(round(h * max_pixels / w))
    else:
        new_h = max_pixels
        new_w = int(round(w * max_pixels / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


# ------------------------------------------------------------------
# PDF → page images
# ------------------------------------------------------------------

def _pdf_to_images(pdf_path: Path, work_dir: Path, *, dpi: int = 200,
                   _progress=None) -> list[Path]:
    """Render each page of a PDF to a PNG image. Returns sorted paths."""
    from pdf2image import convert_from_path

    if _progress:
        _progress("Extracting pages from PDF", 8)

    pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")
    image_paths: list[Path] = []

    for i, page_img in enumerate(pages):
        out = work_dir / f"_pdfpage_{i + 1:04d}.png"
        page_img.save(str(out), "PNG")
        image_paths.append(out)
        if _progress:
            _progress(f"Extracted page {i + 1}/{len(pages)}",
                      8 + int(20 * (i + 1) / len(pages)))

    log.info("Extracted %d page(s) from %s", len(pages), pdf_path.name)
    return image_paths


# ------------------------------------------------------------------
# Image preprocessing — auto-rotate, deskew, perspective correction
# ------------------------------------------------------------------

def _preprocess_image(image_path: Path, work_dir: Path, *,
                      page_idx: int = 0) -> Path:
    """Straighten a scanned image: fix rotation, skew, and mild
    perspective distortion.  Returns the path to the corrected image
    (may be the original if no correction was needed)."""
    try:
        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            return image_path

        corrected = img_cv
        changed = False

        # 1. Auto-rotate (90°/180°/270°) via Tesseract OSD
        rotated, did_rotate = _auto_rotate(corrected, image_path)
        if did_rotate:
            corrected = rotated
            changed = True

        # 2. Deskew small angles
        deskewed, did_deskew = _deskew(corrected)
        if did_deskew:
            corrected = deskewed
            changed = True

        # 3. Perspective correction (straighten trapezoid pages)
        flattened, did_flatten = _perspective_correct(corrected)
        if did_flatten:
            corrected = flattened
            changed = True

        if not changed:
            return image_path

        out_path = work_dir / f"_clean_p{page_idx + 1}.png"
        cv2.imwrite(str(out_path), corrected)
        log.info("Page %d preprocessed (rotate=%s, deskew=%s, perspective=%s)",
                 page_idx + 1, did_rotate, did_deskew, did_flatten)
        return out_path

    except Exception as e:
        log.warning("Preprocessing failed for page %d: %s", page_idx + 1, e)
        return image_path


def _auto_rotate(img_cv: np.ndarray, image_path: Path) -> tuple[np.ndarray, bool]:
    """Placeholder: no OCR-based rotation. Returns image unchanged."""
    return img_cv, False


def _deskew(img_cv: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect and correct small skew angles using Hough line detection."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Edge detection to find lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines — staff lines in music are long and horizontal
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200,
                            minLineLength=gray.shape[1] // 4,
                            maxLineGap=10)
    if lines is None or len(lines) < 3:
        return img_cv, False

    # Compute angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if abs(dx) < 10:
            continue  # skip near-vertical lines
        angle = np.degrees(np.arctan2(y2 - y1, dx))
        if abs(angle) < MAX_SKEW_ANGLE:
            angles.append(angle)

    if not angles:
        return img_cv, False

    median_angle = float(np.median(angles))
    if abs(median_angle) < MIN_SKEW_ANGLE:
        return img_cv, False

    log.info("Deskew: correcting %.2f° skew", median_angle)
    h, w = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Compute new bounding size so nothing is clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img_cv, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, True


def _perspective_correct(img_cv: np.ndarray) -> tuple[np.ndarray, bool]:
    """Detect the page quadrilateral and warp to a flat rectangle.
    Only applies if a clear page boundary is found (e.g. a photo
    of a page on a dark background)."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_cv, False

    h, w = img_cv.shape[:2]
    img_area = h * w

    # Find the largest quadrilateral contour that covers most of the image
    best_quad = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx
            best_area = area

    if best_quad is None:
        return img_cv, False

    # Check if the quadrilateral is significantly non-rectangular
    # (if it's already rectangular, no correction needed)
    pts = best_quad.reshape(4, 2).astype(np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = _order_points(pts)
    tl, tr, br, bl = pts

    # Compute if the distortion is significant enough to correct
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    w_ratio = min(top_w, bot_w) / max(top_w, bot_w) if max(top_w, bot_w) > 0 else 1
    h_ratio = min(left_h, right_h) / max(left_h, right_h) if max(left_h, right_h) > 0 else 1

    if w_ratio > 0.95 and h_ratio > 0.95:
        return img_cv, False  # already nearly rectangular

    new_w = int(max(top_w, bot_w))
    new_h = int(max(left_h, right_h))
    dst = np.array([
        [0, 0], [new_w - 1, 0],
        [new_w - 1, new_h - 1], [0, new_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_cv, M, (new_w, new_h),
                                 flags=cv2.INTER_CUBIC)
    log.info("Perspective corrected (w_ratio=%.2f, h_ratio=%.2f)",
             w_ratio, h_ratio)
    return warped, True


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
    return rect


def run(image_paths: list[str], work_dir: str, *, base_name: str = "images",
        on_progress=None):
    """Process a list of image or PDF files.

    Returns {"files": ["name.pdf", "piece_1.musicxml", ...]}
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    def _progress(step, pct):
        log.info("%d%% %s", pct, step)
        if on_progress:
            on_progress(step, pct)

    output_files: list[str] = []
    pdf_path = work / f"{base_name}.pdf"
    extracted_images: list[Path] = []  # temp images from PDF (need cleanup)

    # Detect whether input is a PDF or images
    pdf_inputs = [p for p in image_paths if is_pdf(p)]

    if pdf_inputs:
        # --- PDF input: keep original, extract pages as images ---
        src_pdf = Path(pdf_inputs[0])
        _progress("Saving PDF", 2)
        shutil.copy2(str(src_pdf), str(pdf_path))
        output_files.append(pdf_path.name)

        paths = _pdf_to_images(src_pdf, work, _progress=_progress)
        extracted_images = list(paths)
    else:
        # --- Image input: we'll create PDF after preprocessing (so PDF shows straightened pages) ---
        paths = sorted([Path(p) for p in image_paths], key=lambda p: p.stem)

    n = len(paths)

    # --- Resize large images once so preprocessing, split, and OMR are fast ---
    work_resized_temps: list[Path] = []
    work_paths: list[Path] = []
    for i, p in enumerate(paths):
        img = Image.open(str(p))
        if img.width <= MAX_WORKING_LONG_EDGE and img.height <= MAX_WORKING_LONG_EDGE:
            work_paths.append(p)
        else:
            img = _resize_by_long_edge(img, MAX_WORKING_LONG_EDGE)
            out = work / f"_work_p{i + 1}.png"
            img.save(str(out))
            work_paths.append(out)
            work_resized_temps.append(out)
    paths = work_paths

    # --- Preprocess images (deskew, rotate, flatten) ---
    clean_paths: list[Path] = []
    for i, img_path in enumerate(paths):
        _progress(f"Straightening page {i + 1}/{n}", 25 + int(15 * i / max(n, 1)))
        clean = _preprocess_image(img_path, work, page_idx=i)
        clean_paths.append(clean)

    # --- For image input: create PDF from preprocessed pages ---
    if not pdf_inputs:
        _progress("Creating PDF", 42)
        _images_to_pdf(clean_paths, pdf_path, _progress)
        output_files.append(pdf_path.name)

    # --- Split pages that contain multiple pieces, then OMR ---
    score_sections: list[tuple[int, Path]] = []
    seq = 0
    last_omr_failure: str = ""
    no_lyrics_any = False

    for i, img_path in enumerate(clean_paths):
        page_label = f"page {i + 1}/{n}" if n > 1 else "image"
        pct_base = 35 + int(40 * i / max(n, 1))
        _progress(f"Analyzing {page_label}", pct_base)

        sections = _split_page(img_path, work, page_idx=i)
        if len(sections) > 1:
            log.info("%s split into %d sections", page_label, len(sections))

        found_any = False
        for sec_idx, sec_path in enumerate(sections):
            sec_label = (f"{page_label} section {sec_idx + 1}"
                         if len(sections) > 1 else page_label)
            _progress(f"OMR on {sec_label}", pct_base + 2)

            mxml = work / f"_omr_p{i + 1}_s{sec_idx}.musicxml"
            ok = False
            reason = ""
            if _audiveris_available():
                ok, reason = _try_audiveris(sec_path, mxml)
                if not ok:
                    last_omr_failure = reason
                elif reason == "no_lyrics":
                    no_lyrics_any = True
                    # Audiveris did not attach lyrics; run Tesseract on the image and inject into MusicXML
                    if _inject_lyrics_from_image(sec_path, mxml):
                        no_lyrics_any = False
            else:
                if not ok:
                    last_omr_failure = "Audiveris not available (use amd64 image: build/run with --platform linux/amd64)"
            if ok:
                score_sections.append((seq, mxml))
                found_any = True
                log.info("Score found in %s", sec_label)
            else:
                log.info("No score in %s", sec_label)
            seq += 1

            # Clean up cropped section images
            if sec_path != img_path:
                sec_path.unlink(missing_ok=True)

        if not found_any:
            seq += 1  # keep a gap so next page isn't merged with previous

    if not score_sections:
        for p in work_resized_temps:
            p.unlink(missing_ok=True)
        _progress("Done", 100)
        msg = "No scores were detected."
        if last_omr_failure:
            msg += " Last OMR reason: " + last_omr_failure
        else:
            msg += " Single-line percussion and non-standard staves may not be supported."
        return {
            "files": output_files,
            "message": msg,
        }

    # --- Step 3: Group sections into pieces using final barlines ---
    _progress("Grouping pieces", 80)
    pieces = _group_into_pieces(score_sections)
    log.info("Detected %d piece(s) across %d score section(s)",
             len(pieces), len(score_sections))

    # --- Step 4: Produce one MusicXML per piece ---
    _progress("Building MusicXML files", 88)
    single_piece = len(pieces) == 1

    for piece_idx, section_group in enumerate(pieces, 1):
        section_mxmls = [p for _, p in section_group]

        if single_piece:
            piece_name = f"{base_name}.musicxml"
        else:
            piece_name = f"{base_name}_{piece_idx}.musicxml"

        piece_path = work / piece_name
        title = base_name if single_piece else f"{base_name} ({piece_idx})"

        if len(section_mxmls) == 1:
            _write_musicxml_with_title(section_mxmls[0], piece_path, title=title)
        else:
            _merge_musicxml(section_mxmls, piece_path, title=title)

        output_files.append(piece_path.name)
        log.info("Piece %d → %s (%d section(s))",
                 piece_idx, piece_name, len(section_mxmls))

    # Clean up temp files
    for _, p in score_sections:
        p.unlink(missing_ok=True)
    for p in clean_paths:
        if p not in paths:
            p.unlink(missing_ok=True)
    for p in work_resized_temps:
        p.unlink(missing_ok=True)
    for p in extracted_images:
        p.unlink(missing_ok=True)

    _progress("Done", 100)
    out = {"files": output_files}
    if no_lyrics_any:
        out["message"] = (
            "No lyrics were detected in the score. "
            "Audiveris expects lyrics below the staves; ensure the image resolution is sufficient (e.g. 300 DPI) and that OCR is configured (TESSDATA_PREFIX with chi_sim+eng)."
        )
    return out


# ------------------------------------------------------------------
# Page splitting — detect multiple pieces on one page
# ------------------------------------------------------------------

def _split_page(image_path: Path, work_dir: Path, *,
                page_idx: int = 0) -> list[Path]:
    """Return the page as a single section (no splitting)."""
    return [image_path]


# ------------------------------------------------------------------
# Piece grouping — final barline detection
# ------------------------------------------------------------------

def _has_final_barline(mxml_path: Path) -> bool:
    """Check whether a MusicXML file ends with a final barline,
    indicating the piece is complete on this page."""
    import music21

    try:
        score = music21.converter.parse(str(mxml_path))
        for part in score.parts:
            measures = list(part.getElementsByClass(music21.stream.Measure))
            if not measures:
                continue
            last = measures[-1]
            rb = last.rightBarline
            if rb is not None and rb.type in ("final", "light-heavy"):
                return True
    except Exception as e:
        log.warning("Could not check barline in %s: %s", mxml_path.name, e)

    return False


def _group_into_pieces(
    score_sections: list[tuple[int, Path]],
) -> list[list[tuple[int, Path]]]:
    """Group consecutive score sections into pieces.

    A section that ends with a final barline marks the end of the
    current piece.  A gap in sequence indices (from a non-score page
    or section in between) also starts a new piece.
    """
    if not score_sections:
        return []

    pieces: list[list[tuple[int, Path]]] = []
    current_piece: list[tuple[int, Path]] = []

    for seq_idx, mxml_path in score_sections:
        if current_piece:
            prev_seq = current_piece[-1][0]
            if seq_idx != prev_seq + 1:
                pieces.append(current_piece)
                current_piece = []

        current_piece.append((seq_idx, mxml_path))

        if _has_final_barline(mxml_path):
            pieces.append(current_piece)
            current_piece = []

    if current_piece:
        pieces.append(current_piece)

    return pieces


# ------------------------------------------------------------------
# PDF creation (images → single PDF, no OCR)
# ------------------------------------------------------------------

def _images_to_pdf(image_paths: list[Path], output_path: Path, _progress):
    """Combine images into a single PDF (image-only, not searchable)."""
    images = []
    for i, p in enumerate(image_paths):
        _progress(f"Adding page {i + 1}/{len(image_paths)} to PDF", 10 + int(25 * i / max(len(image_paths), 1)))
        img = Image.open(str(p)).convert("RGB")
        images.append(img)
    if not images:
        return
    first = images[0]
    first.save(str(output_path), "PDF", resolution=150, save_all=True,
               append_images=images[1:])


# ------------------------------------------------------------------
# OMR — Audiveris (x86_64 only)
# ------------------------------------------------------------------

def _audiveris_available() -> bool:
    """True if Audiveris binary is on PATH or in standard location."""
    try:
        r = subprocess.run(
            ["Audiveris", "-help"],
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0 or b"Audiveris" in (r.stdout or b"") + (r.stderr or b"")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _tesseract_langs_available(prefix: str) -> bool:
    """True if tesseract --list-langs with TESSDATA_PREFIX lists chi_sim or eng (OCR for lyrics)."""
    try:
        r = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "TESSDATA_PREFIX": prefix},
        )
        out = (r.stdout or "") + (r.stderr or "")
        return "chi_sim" in out or "eng" in out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _try_audiveris(image_path: Path, output_path: Path) -> tuple[bool, str]:
    """Run Audiveris on a single image. Returns (True, "") or (False, reason).
    Audiveris supports pitched and percussion notation (x86_64 only).
    """
    work_dir = output_path.parent
    book_name = image_path.stem
    out_book = work_dir / book_name
    # Use absolute path so Audiveris has a clear output target (book folder = input stem).
    abs_image = image_path.resolve()
    abs_work = work_dir.resolve()

    try:
        # TESSDATA_PREFIX so Audiveris can run Tesseract OCR for lyrics (TEXTS step).
        # Prefer /opt/audiveris-tessdata (legacy 4.0 traineddata); apt tessdata is LSTM-only and Audiveris needs legacy.
        audiveris_env = os.environ.copy()
        _tess_prefix = None
        for prefix in ("/opt/audiveris-tessdata", "/usr/share/tesseract-ocr/5", "/usr/share/tesseract-ocr/4.00", "/usr/share/tesseract-ocr"):
            if (Path(prefix) / "tessdata").exists():
                _tess_prefix = prefix
                audiveris_env["TESSDATA_PREFIX"] = prefix
                break
        if not _tess_prefix:
            _tess_prefix = audiveris_env.get("TESSDATA_PREFIX")
        if _tess_prefix:
            log.info("Audiveris TESSDATA_PREFIX=%s (lyrics)", _tess_prefix)
            if not _tesseract_langs_available(_tess_prefix):
                log.warning(
                    "tesseract --list-langs did not show chi_sim/eng; Audiveris may need legacy tessdata (e.g. /opt/audiveris-tessdata)"
                )
        else:
            log.warning("No tessdata dir found; TEXTS/lyrics may be skipped")
        # Set OCR language before -batch so it's applied at startup (TEXTS step needs it for lyrics).
        # Some docs use -constant; we pass both in case the CLI accepts only one.
        lang_opt = "org.audiveris.omr.text.Language.defaultSpecification=chi_sim+eng"
        result = subprocess.run(
            [
                "xvfb-run", "-a",
                "Audiveris",
                "-option", lang_opt,
                "-batch",
                "-save",
                "-transcribe",
                "-export",
                "-output", str(abs_work),
                "--",
                str(abs_image),
            ],
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            timeout=300,
            env=audiveris_env,
        )

        out_txt = (result.stdout or "").strip()
        err_txt = (result.stderr or "").strip()
        err_msg = (err_txt or out_txt)[:800]
        if out_txt:
            log.info("Audiveris stdout: %s", out_txt[:1500])
        if err_txt:
            # Log full stderr to spot OCR/TEXTS errors (e.g. "Failed loading language", "Could not initialize Tesseract").
            log.info("Audiveris stderr: %s", err_txt[:2000])

        if result.returncode != 0:
            log.info("Audiveris exited %d: %s", result.returncode, err_msg[:500])
            return False, f"Audiveris exit {result.returncode}: {err_msg[:400]}"

        # Audiveris creates a subfolder under -output named from the input stem (e.g. _clean_p1).
        candidates = list(out_book.glob("*.mxl")) + list(out_book.glob("*.xml")) if out_book.exists() else []
        if not candidates:
            candidates = [p for p in (list(work_dir.glob("*/*.mxl")) + list(work_dir.glob("*/*.xml"))) if not p.name.startswith("_")]
        if not candidates:
            for ext in ("*.mxl", "*.xml"):
                candidates.extend(work_dir.rglob(ext))
            candidates = [p for p in candidates if not p.name.startswith("_") and p.is_file()]
        if candidates and out_book.exists():
            in_book = [p for p in candidates if str(p).startswith(str(out_book))]
            if in_book:
                candidates = in_book + [p for p in candidates if p not in in_book]

        # If transcribe produced .omr but no .mxl, run export separately.
        if not candidates:
            omr_files = list(work_dir.glob(f"{book_name}.omr")) or list(work_dir.glob("*.omr"))
            if omr_files:
                omr_path = omr_files[0].resolve()
                log.info("Running Audiveris -export on %s", omr_path.name)
                # Request uncompressed .xml so we have a single predictable output file.
                # Pass .omr path relative to work_dir so Audiveris finds the book in the output folder.
                omr_rel = omr_files[0].name
                # Force baseFolder so export writes under work_dir (Audiveris may otherwise use default XDG_DATA_HOME).
                env = os.environ.copy()
                env["XDG_DATA_HOME"] = str(abs_work)
                _tp = env.get("TESSDATA_PREFIX")
                if not _tp or not (Path(_tp) / "tessdata").exists():
                    for prefix in ("/opt/audiveris-tessdata", "/usr/share/tesseract-ocr/5", "/usr/share/tesseract-ocr/4.00", "/usr/share/tesseract-ocr"):
                        if (Path(prefix) / "tessdata").exists():
                            env["TESSDATA_PREFIX"] = prefix
                            break
                lang_opt = "org.audiveris.omr.text.Language.defaultSpecification=chi_sim+eng"
                result2 = subprocess.run(
                    [
                        "xvfb-run", "-a",
                        "Audiveris",
                        "-option", lang_opt,
                        "-batch",
                        "-export",
                        "-output", str(abs_work),
                        "-option", "org.audiveris.omr.sheet.BookManager.baseFolder=" + str(abs_work),
                        "-option", "org.audiveris.omr.sheet.BookManager.useCompression=false",
                        "--",
                        omr_rel,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(work_dir),
                    timeout=120,
                    env=env,
                )
                export_err = (result2.stderr or result2.stdout or "").strip()[:500]
                if result2.stderr:
                    log.info("Audiveris export stderr: %s", (result2.stderr or "")[:400])
                # Search for any .mxl or .xml regardless of returncode (export may write then exit non-zero).
                stem = omr_files[0].stem
                for ext in ("*.mxl", "*.xml"):
                    candidates.extend(work_dir.glob(ext))
                    candidates.extend(work_dir.rglob(ext))
                # Audiveris may write to work_dir/stem/stem.mxl, work_dir/stem.mxl, or XDG base (e.g. work_dir/AudiverisLtd/audiveris/...).
                for f in (work_dir / stem / f"{stem}.mxl", work_dir / stem / f"{stem}.xml",
                          work_dir / f"{stem}.mxl", work_dir / f"{stem}.xml"):
                    if f.is_file():
                        candidates.append(f.resolve())
                audiveris_base = work_dir / "AudiverisLtd" / "audiveris"
                if audiveris_base.exists():
                    for f in audiveris_base.rglob("*.mxl"):
                        if f.is_file():
                            candidates.append(f.resolve())
                    for f in audiveris_base.rglob("*.xml"):
                        if f.is_file():
                            candidates.append(f.resolve())
                candidates = list({p.resolve() for p in candidates if p.is_file()})
                if omr_files and candidates:
                    stem = omr_files[0].stem
                    by_stem = [p for p in candidates if p.stem == stem]
                    if by_stem:
                        candidates = by_stem + [p for p in candidates if p not in by_stem]
                if not candidates and export_err:
                    err_msg = export_err  # surface export error in final message

        if not candidates:
            # Surface Audiveris output so user can see why no file was produced.
            detail = err_msg[:350].strip() or " (no stderr/stdout)"
            try:
                contents = list(work_dir.iterdir())
                detail += "; work_dir contents: " + ", ".join(p.name for p in contents[:15])
            except OSError:
                pass
            log.info("Audiveris produced no MusicXML output (book_dir=%s). %s", out_book, detail)
            return False, "Audiveris produced no MusicXML output. " + detail

        src = candidates[0]
        if src.suffix.lower() == ".mxl":
            with zipfile.ZipFile(src, "r") as z:
                names = [n for n in z.namelist() if n.lower().endswith(".xml")]
                if not names:
                    return False, "Audiveris MXL had no XML"
                content = z.read(names[0]).decode("utf-8", errors="replace")
        else:
            content = src.read_text(encoding="utf-8", errors="replace")

        lower = content.lower()
        if "<note" not in lower and "<unpitched" not in lower:
            log.info("Audiveris output has no note/unpitched elements (percussion may need manual editing)")
        has_lyrics = "<lyric" in lower
        if not has_lyrics:
            log.info("Audiveris output has no <lyric> elements (TEXTS/OCR may be skipped or lyrics not attached)")
        output_path.write_text(content, encoding="utf-8")
        if out_book.exists():
            shutil.rmtree(out_book, ignore_errors=True)
        return True, "" if has_lyrics else "no_lyrics"

    except subprocess.TimeoutExpired:
        log.warning("Audiveris timed out after 5 min")
        return False, "Audiveris timed out"
    except FileNotFoundError:
        return False, "Audiveris not installed"
    except Exception as e:
        log.warning("Audiveris failed: %s", e)
        return False, str(e)


# ------------------------------------------------------------------
# Lyric injection — Tesseract OCR on image, attach text to notes (workaround when Audiveris TEXTS gives no lyrics)
# ------------------------------------------------------------------

def _tessdata_prefix() -> str | None:
    """Return TESSDATA_PREFIX if a tessdata dir exists (same order as Audiveris)."""
    for prefix in ("/opt/audiveris-tessdata", "/usr/share/tesseract-ocr/5", "/usr/share/tesseract-ocr/4.00", "/usr/share/tesseract-ocr"):
        if (Path(prefix) / "tessdata").exists():
            return prefix
    return os.environ.get("TESSDATA_PREFIX") or None


def _inject_lyrics_from_image(image_path: Path, mxml_path: Path) -> bool:
    """
    Run Tesseract OCR on the sheet image and attach recognized text to notes in the MusicXML.
    Used when Audiveris does not produce lyrics (TEXTS step skipped or not attached).
    Assigns one character per note for Chinese-style lyrics; multiple parts get lines split by newline.
    Returns True if any lyrics were added.
    """
    import music21

    prefix = _tessdata_prefix()
    env = os.environ.copy()
    if prefix:
        env["TESSDATA_PREFIX"] = prefix
    try:
        r = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(image_path.parent),
        )
        text = (r.stdout or "").strip() if r.returncode == 0 else ""
        # Also try stderr for some tesseract versions
        if not text and r.stderr:
            text = (r.stderr or "").strip()
        if not text or not text.replace("\n", "").replace(" ", ""):
            log.debug("Tesseract produced no text for %s", image_path.name)
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("Tesseract not run for lyrics: %s", e)
        return False

    try:
        score = music21.converter.parse(str(mxml_path))
    except Exception as e:
        log.warning("Could not parse MusicXML for lyric injection: %s", e)
        return False

    # Collect note "slots" per part (one per Note or Chord, in order)
    parts_slots: list[list] = []
    for part in score.parts:
        slots = list(part.flat.notes)  # Note and Chord objects in order
        slots.sort(key=lambda n: n.getOffsetInHierarchy(score))
        parts_slots.append(slots)

    if not parts_slots or all(not s for s in parts_slots):
        return False

    # Split OCR text by newlines for multiple parts; otherwise use single block
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        lines = [text]
    # One line per part, or repeat first line
    while len(lines) < len(parts_slots):
        lines.append(lines[0] if lines else "")

    added = 0
    for part_idx, slots in enumerate(parts_slots):
        if not slots:
            continue
        line = lines[part_idx] if part_idx < len(lines) else lines[0]
        # Remove spaces for Chinese; then assign one character per note
        chars = [c for c in line if c.strip()] or list(line)
        for i, n in enumerate(slots):
            if i < len(chars):
                n.lyric = chars[i]
                added += 1

    if not added:
        return False
    try:
        score.write("musicxml", fp=str(mxml_path))
        log.info("Injected %d lyric(s) from Tesseract into %s", added, mxml_path.name)
    except Exception as e:
        log.warning("Could not write MusicXML after lyric injection: %s", e)
        return False
    return True


# ------------------------------------------------------------------
# MusicXML — set title, instrument (piano), and optionally merge
# ------------------------------------------------------------------

def _set_score_instrument_piano(score):
    """Set every part in the score to Piano (so output is not vocal).
    Remove any existing instrument (e.g. vocal), set part name, and insert Piano at offset 0."""
    import music21
    for part in score.parts:
        for inst in part.getElementsByClass(music21.instrument.Instrument):
            part.remove(inst)
        piano = music21.instrument.Piano()
        part.instrument = piano
        part.partName = "Piano"
        part.insert(0, piano)


def _write_musicxml_with_title(source_path: Path, output_path: Path, *, title: str):
    """Parse one MusicXML file, set work title and instrument to piano, and write to output_path."""
    import music21

    try:
        s = music21.converter.parse(str(source_path))
        if s.metadata is None:
            s.metadata = music21.metadata.Metadata()
        s.metadata.title = title
        s.metadata.movementName = title
        _set_score_instrument_piano(s)
        _fix_clefs_at_end_of_measure(s)
        _fill_measure_number_gaps(s)
        s.write("musicxml", fp=str(output_path))
        # Post-pass: move clefs that ended up at end of measure in XML to start of measure
        try:
            xml = output_path.read_text(encoding="utf-8")
            output_path.write_text(_move_clefs_from_end_to_start_of_measure_xml(xml), encoding="utf-8")
        except Exception as e:
            log.warning("Clef post-pass failed for %s: %s", output_path.name, e)
    except Exception as e:
        log.warning("Could not set title on %s: %s; copying as-is", source_path.name, e)
        shutil.copy2(str(source_path), str(output_path))


def _count_measures_per_part(score) -> list[int]:
    """Return list of measure counts per part (for verification)."""
    import music21
    return [
        len(list(part.getElementsByClass(music21.stream.Measure)))
        for part in score.parts
    ]


def _local_tag(el: ET.Element) -> str:
    """Return tag name without namespace (MusicXML uses default ns)."""
    return el.tag.split("}")[-1] if "}" in el.tag else el.tag


def _attributes_has_clef(attr_el: ET.Element) -> bool:
    """True if this <attributes> element contains a <clef> child."""
    return attr_el.find(".//{*}clef") is not None or attr_el.find(".//clef") is not None


def _move_clefs_from_end_to_start_of_measure_xml(xml_content: str) -> str:
    """MusicXML post-pass: move <attributes> blocks that contain <clef> and appear after
    a note/rest to the start of that measure (clef must be inside <attributes>)."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        log.warning("Clef post-pass: could not parse MusicXML: %s", e)
        return xml_content
    # MusicXML 3.0 uses default ns; 2.0 may use partwise ns or no ns
    ns = {"m": "http://www.musicxml.org/ns/3.0", "p": "http://www.musicxml.org/ns/partwise"}
    parts = (
        list(root.iterfind(".//m:part", ns))
        or list(root.iterfind(".//p:part", ns))
        or list(root.iterfind(".//{*}part"))
        or [e for e in root.iter() if _local_tag(e) == "part"]
    )
    if not parts:
        parts = [root]
    moved_count = 0
    for part in parts:
        for measure in list(part):
            if _local_tag(measure) != "measure":
                continue
            children = list(measure)
            first_note_idx = None
            for i, child in enumerate(children):
                if _local_tag(child) == "note":
                    first_note_idx = i
                    break
            if first_note_idx is None:
                continue
            to_move = []
            for i, child in enumerate(children):
                if _local_tag(child) != "attributes":
                    continue
                if not _attributes_has_clef(child):
                    continue
                if i > first_note_idx:
                    to_move.append((i, child))
            if not to_move:
                continue
            insert_idx = 0
            for i in range(first_note_idx):
                if _local_tag(children[i]) == "attributes":
                    insert_idx = i + 1
            for i, el in sorted(to_move, reverse=True):
                measure.remove(el)
                moved_count += 1
            for el in [el for _, el in to_move]:
                measure.insert(insert_idx, el)
                insert_idx += 1
    if moved_count:
        log.info("Clef post-pass: moved %d attributes-with-clef from end to start of measure", moved_count)
    out = ET.tostring(root, encoding="unicode", default_namespace="http://www.musicxml.org/ns/3.0")
    if xml_content.lstrip().startswith("<?xml"):
        decl = xml_content[: xml_content.find("?>") + 2]
        if not out.lstrip().startswith("<?xml"):
            out = decl + "\n" + out
    return out


def _fix_clefs_at_end_of_measure(score):
    """Move any clef that appears at the end of a measure to the start of the next measure.
    If there is no next measure (OMR dropped it), insert a rest measure and put the clef there.
    This fixes wrong MusicXML where treble clefs appear at the end of measures."""
    import music21
    for part in score.parts:
        measures = list(part.getElementsByClass(music21.stream.Measure))
        if not measures:
            continue
        measures.sort(key=lambda m: m.getOffsetInHierarchy(part))

        # Collect (measure, clef) where clef is misplaced (at end of measure).
        # Detect by: clef offset > 0, OR clef is not the first element in the measure (there are notes before it).
        to_move = []
        for m in measures:
            clefs = list(m.getElementsByClass(music21.clef.Clef))
            for clef in clefs:
                try:
                    off = m.elementOffset(clef) if hasattr(m, "elementOffset") else clef.getOffsetBySite(m)
                except Exception:
                    off = getattr(clef, "offset", 0)
                m_len = m.quarterLength or 4.0
                # Clef at end: either offset in second half, or offset > 0 (not at start)
                if off >= 0.5 * m_len:
                    to_move.append((m, clef))
                    continue
                # Also: if in stream order a note/rest appears before this clef, the clef is at the end (wrong)
                seen_note_or_rest_before_clef = False
                for el in m.iter:
                    if el is clef:
                        if seen_note_or_rest_before_clef:
                            to_move.append((m, clef))
                        break
                    if isinstance(el, (music21.note.Note, music21.note.Rest, music21.chord.Chord)):
                        seen_note_or_rest_before_clef = True

        for m, clef in to_move:
            try:
                m.remove(clef)
            except Exception:
                continue
            m_offset = m.getOffsetInHierarchy(part)
            m_end = m_offset + m.quarterLength
            # Find next measure (start offset >= m_end, smallest first)
            next_measure = None
            for other in measures:
                other_off = other.getOffsetInHierarchy(part)
                if other_off >= m_end - 0.001:
                    next_measure = other
                    break
            if next_measure is not None:
                next_measure.insert(0, clef)
            else:
                # No next measure: insert a rest measure and put clef at start
                new_m = music21.stream.Measure()
                new_m.insert(0, clef)
                new_m.append(music21.note.Rest(quarterLength=m.quarterLength))
                part.insert(m_end, new_m)
                measures.append(new_m)
                measures.sort(key=lambda x: x.getOffsetInHierarchy(part))


def _is_repeat_barline(measure) -> bool:
    """True if measure has a repeat-style right barline."""
    rb = getattr(measure, "rightBarline", None)
    if rb is None:
        return False
    t = getattr(rb, "type", None) or ""
    return t in ("light-heavy", "heavy-light", "heavy-light-heavy") or "repeat" in str(t).lower()


def _measure_has_notes(measure) -> bool:
    """True if measure contains at least one Note or Chord."""
    import music21
    return bool(measure.getElementsByClass(music21.note.Note)) or bool(
        measure.getElementsByClass(music21.chord.Chord)
    )


def _fill_measure_number_gaps(score):
    """Insert rest measures so numbering is continuous and so a missing measure before a repeat is filled.

    1) Numerical gaps: if measure numbers jump (e.g. 21 then 23), insert rest measure(s).
    2) Repeat heuristic: if a measure has a repeat barline and is empty/short (OMR often drops the measure
       before a repeat), insert one full rest measure before it.
    """
    import music21
    for part in score.parts:
        measures = list(part.getElementsByClass(music21.stream.Measure))
        if not measures:
            continue
        measures.sort(key=lambda m: m.getOffsetInHierarchy(part))

        # Typical measure length (median of non-zero lengths) for repeat heuristic
        lengths = [m.quarterLength for m in measures if m.quarterLength > 0]
        typical_q = lengths[len(lengths) // 2] if lengths else 4.0

        inserts = []

        # 1) Numerical gaps
        for i in range(len(measures) - 1):
            prev_m = measures[i]
            next_num = measures[i + 1].number
            for missing in range(measures[i].number + 1, next_num):
                insert_offset = prev_m.offset + prev_m.quarterLength * (missing - measures[i].number)
                inserts.append((insert_offset, prev_m.quarterLength))

        # 2) Measure before repeat that looks missing (repeat barline but no/short content)
        for m in measures:
            if not _is_repeat_barline(m):
                continue
            q = m.quarterLength or 0
            no_notes = not _measure_has_notes(m)
            short = q < 0.5 * typical_q
            if no_notes or short:
                insert_offset = m.offset
                inserts.append((insert_offset, typical_q))
                break  # at most one insert per part for repeat heuristic

        # Insert in descending offset order so earlier offsets are not shifted
        for insert_offset, qlen in sorted(inserts, key=lambda x: -x[0]):
            new_measure = music21.stream.Measure()
            new_measure.append(music21.note.Rest(quarterLength=qlen))
            part.insert(insert_offset, new_measure)

        # Renumber 1..N
        measures = list(part.getElementsByClass(music21.stream.Measure))
        measures.sort(key=lambda m: m.getOffsetInHierarchy(part))
        for j, m in enumerate(measures, 1):
            m.number = j


def _merge_musicxml(page_paths: list[Path], output_path: Path, *,
                    title: str = "Score"):
    """Merge multiple per-page MusicXML files into one continuous score.

    Pages are assumed to be in order.  Parts are matched by index
    (part 0 of page 2 appends to part 0 of page 1, etc.).
    Verifies that no measures are dropped (logs warning if counts don't match).
    """
    import music21

    scores = []
    for p in page_paths:
        try:
            s = music21.converter.parse(str(p))
            scores.append(s)
        except Exception as e:
            log.warning("Could not parse %s: %s", p.name, e)

    if not scores:
        return

    base = scores[0]

    # Expected total measures per part (sum across all pages)
    n_parts = max(len(base.parts), 1)
    expected_per_part: list[int] = [0] * n_parts
    for s in scores:
        counts = _count_measures_per_part(s)
        for i, c in enumerate(counts):
            while len(expected_per_part) <= i:
                expected_per_part.append(0)
            expected_per_part[i] += c

    if base.metadata is None:
        base.metadata = music21.metadata.Metadata()
    base.metadata.title = title
    base.metadata.movementName = title
    _set_score_instrument_piano(base)

    base_parts = list(base.parts)

    for page_score in scores[1:]:
        page_parts = list(page_score.parts)

        for part_idx, base_part in enumerate(base_parts):
            if part_idx >= len(page_parts):
                break

            page_part = page_parts[part_idx]
            page_measures = list(
                page_part.getElementsByClass(music21.stream.Measure),
            )
            # Ensure chronological order (by offset) so no measure is skipped or duplicated
            page_measures.sort(key=lambda m: m.getOffsetInHierarchy(page_part))

            existing = list(
                base_part.getElementsByClass(music21.stream.Measure),
            )
            existing.sort(key=lambda m: m.getOffsetInHierarchy(base_part))
            next_num = existing[-1].number + 1 if existing else 1
            last_offset = (
                existing[-1].offset + existing[-1].quarterLength
                if existing else 0.0
            )

            for m in page_measures:
                m.number = next_num
                next_num += 1
                base_part.insert(last_offset, m)
                last_offset += m.quarterLength

    for part in base.parts:
        measures = list(part.getElementsByClass(music21.stream.Measure))
        measures.sort(key=lambda m: m.getOffsetInHierarchy(part))
        for i, m in enumerate(measures, 1):
            m.number = i

    # Verify no measures were dropped
    actual_per_part = _count_measures_per_part(base)
    for part_idx, (expected, actual) in enumerate(zip(expected_per_part, actual_per_part)):
        if actual < expected:
            log.warning(
                "Merge: part %d has %d measures but expected %d (from %d page(s)); some measures may be missing",
                part_idx + 1, actual, expected, len(scores),
            )
    if len(actual_per_part) < len(expected_per_part):
        log.warning(
            "Merge: output has %d parts but input had %d; extra part(s) may have been dropped",
            len(actual_per_part), len(expected_per_part),
        )

    _fix_clefs_at_end_of_measure(base)
    _fill_measure_number_gaps(base)
    base.write("musicxml", fp=str(output_path))
    try:
        xml = output_path.read_text(encoding="utf-8")
        output_path.write_text(_move_clefs_from_end_to_start_of_measure_xml(xml), encoding="utf-8")
    except Exception as e:
        log.warning("Clef post-pass failed for merged %s: %s", output_path.name, e)
    log.info("Merged %d pages into %s (%d parts)",
             len(scores), output_path.name, len(base_parts))
