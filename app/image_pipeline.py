"""
Image / PDF processing pipeline.

Accepts either:
  A) One or more image files (sorted by filename), or
  B) A single PDF file.

Steps:
  1. Produce a searchable PDF:
     - For images: combine into one PDF with OCR text layers.
     - For PDF input: keep the original PDF as-is.
  2. Extract page images (from PDF input, or use the uploaded images).
  3. Preprocess images — auto-rotate, deskew, perspective correction.
  4. Split pages that contain multiple pieces (detected by text between
     score sections — e.g. a new title between two scores).
  5. Run OMR (oemer) on each section.
  6. Group sections into pieces using final-barline detection:
     - A section whose score ends with a final barline = piece ends.
     - No final barline = piece continues into the next section.
  7. Produce one MusicXML file per piece, merging multi-page pieces.

Output:
  - {base_name}.pdf                   — always produced
  - {base_name}.musicxml              — if exactly one piece found
  - {base_name}_1.musicxml, _2, …     — if multiple pieces found
"""

import logging
import shutil
import subprocess
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

MIN_GAP_HEIGHT = 30
MIN_TITLE_CHARS = 3

# Skew angles beyond this (degrees) are corrected; smaller is noise.
MIN_SKEW_ANGLE = 0.5
MAX_SKEW_ANGLE = 15.0

# Max long edge (px) for the whole pipeline (resize → preprocess → PDF/OCR → OMR).
# Lower = faster; 1200 is a good balance for one-page speed vs quality.
MAX_WORKING_LONG_EDGE = 1200


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
    """Detect page orientation with Tesseract OSD and rotate if needed."""
    import pytesseract

    try:
        pil_img = Image.open(str(image_path))
        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle == 0:
            return img_cv, False

        log.info("OSD detected rotation: %d°", angle)

        # OpenCV rotates counter-clockwise; Tesseract reports the angle
        # the image needs to be rotated to be upright.
        if angle == 90:
            return cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE), True
        elif angle == 180:
            return cv2.rotate(img_cv, cv2.ROTATE_180), True
        elif angle == 270:
            return cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE), True

        return img_cv, False
    except Exception as e:
        log.debug("OSD rotation detection failed: %s", e)
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

    # --- For image input: create searchable PDF from preprocessed (straightened) pages ---
    if not pdf_inputs:
        _progress("Creating PDF", 42)
        _images_to_searchable_pdf(clean_paths, pdf_path, _progress)
        output_files.append(pdf_path.name)

    # --- Split pages that contain multiple pieces, then OMR ---
    score_sections: list[tuple[int, Path]] = []
    seq = 0

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
            if _try_omr(sec_path, mxml):
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
        return {
            "files": output_files,
            "message": "No scores were detected. Single-line percussion (e.g. snare drum) and non-standard staves may not be supported by the OMR engine.",
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

        if len(section_mxmls) == 1:
            shutil.copy2(str(section_mxmls[0]), str(piece_path))
        else:
            title = base_name if single_piece else f"{base_name} ({piece_idx})"
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
    return {"files": output_files}


# ------------------------------------------------------------------
# Page splitting — detect multiple pieces on one page
# ------------------------------------------------------------------

def _split_page(image_path: Path, work_dir: Path, *,
                page_idx: int = 0) -> list[Path]:
    """Split a page image into separate piece sections.

    Uses horizontal projection to find gaps between score systems,
    then runs OCR on each gap.  Gaps that contain text (a new title)
    mark piece boundaries.  Returns a list of image paths — the
    original if no split is needed, or cropped section images.
    """
    import pytesseract

    img = Image.open(str(image_path))
    gray = img.convert("L")
    pixels = np.array(gray)
    h, w = pixels.shape

    # Count "ink" pixels per row (darker than threshold)
    ink_per_row = np.sum(pixels < 180, axis=1)
    quiet_threshold = w * 0.01  # less than 1% of width has ink = quiet row

    # Find contiguous quiet regions (gaps between systems)
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0
    for row in range(h):
        if ink_per_row[row] <= quiet_threshold:
            if not in_gap:
                gap_start = row
                in_gap = True
        else:
            if in_gap:
                gap_h = row - gap_start
                if gap_h >= MIN_GAP_HEIGHT:
                    gaps.append((gap_start, row))
                in_gap = False

    if not gaps:
        return [image_path]

    # Check each gap for text (a piece title between two scores)
    split_rows: list[int] = []
    for gap_top, gap_bottom in gaps:
        # Skip gaps at the very top or bottom of the page
        if gap_top < h * 0.05 or gap_bottom > h * 0.95:
            continue

        # Crop the gap region with a small margin
        margin = 5
        crop_top = max(0, gap_top - margin)
        crop_bottom = min(h, gap_bottom + margin)
        gap_img = gray.crop((0, crop_top, w, crop_bottom))

        try:
            text = pytesseract.image_to_string(
                gap_img, lang="chi_sim+chi_tra+eng",
            ).strip()
        except Exception:
            text = ""

        clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(clean) >= MIN_TITLE_CHARS:
            mid = (gap_top + gap_bottom) // 2
            split_rows.append(mid)
            log.info("Page %d: title detected in gap y=%d–%d: '%s'",
                     page_idx + 1, gap_top, gap_bottom,
                     text.replace("\n", " ")[:60])

    if not split_rows:
        return [image_path]

    # Crop into sections
    y_starts = [0] + split_rows
    y_ends = split_rows + [h]
    sections: list[Path] = []

    for sec_idx, (y0, y1) in enumerate(zip(y_starts, y_ends)):
        if y1 - y0 < MIN_GAP_HEIGHT * 2:
            continue  # too thin to contain a score
        section = img.crop((0, y0, w, y1))
        sec_path = work_dir / f"_crop_p{page_idx + 1}_s{sec_idx}.png"
        section.save(str(sec_path))
        sections.append(sec_path)

    return sections if sections else [image_path]


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
# PDF creation
# ------------------------------------------------------------------

def _images_to_searchable_pdf(image_paths: list[Path], output_path: Path,
                              _progress):
    """Combine images into a single searchable PDF using Tesseract.

    Each page gets an OCR text layer (Chinese + English) so the PDF
    is searchable, while the original image is preserved visually.
    """
    import pytesseract

    if len(image_paths) == 1:
        _progress("Running OCR on image", 10)
        img = Image.open(str(image_paths[0])).convert("RGB")
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            img, lang="chi_sim+chi_tra+eng", extension="pdf",
        )
        output_path.write_bytes(pdf_bytes)
        return

    page_pdfs: list[Path] = []
    for i, p in enumerate(image_paths):
        _progress(f"OCR page {i + 1}/{len(image_paths)}", 10 + int(25 * i / len(image_paths)))
        img = Image.open(str(p)).convert("RGB")
        page_path = output_path.parent / f"_ocr_page_{i}.pdf"
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            img, lang="chi_sim+chi_tra+eng", extension="pdf",
        )
        page_path.write_bytes(pdf_bytes)
        page_pdfs.append(page_path)

    _progress("Merging PDF pages", 35)
    _merge_pdfs(page_pdfs, output_path)

    for p in page_pdfs:
        p.unlink(missing_ok=True)


def _merge_pdfs(input_paths: list[Path], output_path: Path):
    """Merge multiple single-page PDFs into one multi-page PDF."""
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        for p in input_paths:
            merger.append(str(p))
        merger.write(str(output_path))
        merger.close()
        return
    except ImportError:
        pass

    log.warning("PyPDF2 not installed; using image-only PDF fallback")
    images = []
    for p in input_paths:
        img = Image.open(str(p))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        images.append(img)
    first = images[0]
    first.save(str(output_path), "PDF", resolution=150, save_all=True,
               append_images=images[1:])


# ------------------------------------------------------------------
# OMR
# ------------------------------------------------------------------

def _try_omr(image_path: Path, output_path: Path) -> bool:
    """Run oemer on a single image.  Returns True if valid MusicXML produced.
    Images larger than MAX_WORKING_LONG_EDGE are downscaled first to keep runtime reasonable.
    """
    try:
        work_dir = output_path.parent
        omr_input = image_path
        img = Image.open(str(image_path))
        w, h = img.size
        if w > MAX_WORKING_LONG_EDGE or h > MAX_WORKING_LONG_EDGE:
            img = _resize_by_long_edge(img, MAX_WORKING_LONG_EDGE)
            omr_input = work_dir / f"_omr_in_{output_path.stem}.png"
            img.save(str(omr_input), "PNG")
            log.info("Downscaled to %dx%d for OMR", img.width, img.height)

        result = subprocess.run(
            ["oemer", str(omr_input)],
            capture_output=True, text=True,
            cwd=str(work_dir),
            timeout=300,
        )

        if omr_input != image_path:
            omr_input.unlink(missing_ok=True)

        if result.returncode != 0:
            log.info("oemer exited %d: %s", result.returncode,
                     result.stderr[:500])
            return False

        candidates = [f for f in work_dir.glob("*.musicxml")
                      if not f.name.startswith("_")]
        if not candidates:
            log.info("oemer produced no MusicXML output")
            return False

        omr_output = candidates[0]
        if omr_output.resolve() != output_path.resolve():
            shutil.move(str(omr_output), str(output_path))

        content = output_path.read_text(encoding="utf-8", errors="replace")
        lower = content.lower()
        # Accept both pitched notes and percussion (unpitched) so percussion output isn't rejected
        if "<note" not in lower and "<unpitched" not in lower:
            log.info("oemer MusicXML has no note/unpitched elements (single-line percussion may be unsupported)")
            output_path.unlink(missing_ok=True)
            return False

        return True

    except subprocess.TimeoutExpired:
        log.warning("oemer timed out after 5 min")
        return False
    except FileNotFoundError:
        log.warning("oemer binary not found — OMR unavailable")
        return False
    except Exception as e:
        log.warning("OMR failed: %s", e)
        return False


# ------------------------------------------------------------------
# MusicXML merge — concatenate measures from multiple sections
# ------------------------------------------------------------------

def _merge_musicxml(page_paths: list[Path], output_path: Path, *,
                    title: str = "Score"):
    """Merge multiple per-page MusicXML files into one continuous score.

    Pages are assumed to be in order.  Parts are matched by index
    (part 0 of page 2 appends to part 0 of page 1, etc.).
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

    if base.metadata is None:
        base.metadata = music21.metadata.Metadata()
    base.metadata.title = title
    base.metadata.movementName = title

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

            existing = list(
                base_part.getElementsByClass(music21.stream.Measure),
            )
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
        for i, m in enumerate(
            part.getElementsByClass(music21.stream.Measure), 1,
        ):
            m.number = i

    base.write("musicxml", fp=str(output_path))
    log.info("Merged %d pages into %s (%d parts)",
             len(scores), output_path.name, len(base_parts))
