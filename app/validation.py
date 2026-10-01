"""Validation of everything the user gives to the application.

Every file is checked before it reaches the model. A file is REJECTED (with a clear
message) when it has an unsupported extension, is empty or too large, is not a
decodable image, is too small, or has an extreme aspect ratio. It is ACCEPTED WITH
A WARNING when it is usable but not ideal (e.g. grey-scale, transparent, animated,
low resolution) so the user knows the prediction may be less reliable.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_FILE_MB = 20                 # larger uploads are rejected
MIN_SIDE_PX = 32                 # smaller images carry too little information
LOW_RES_PX = 128                 # below this a warning is shown
MAX_ASPECT_RATIO = 6.0           # e.g. 3000 x 400 panoramas are rejected
MAX_PIXELS = 60_000_000          # protection against "decompression bomb" files
MAX_FOLDER_IMAGES = 500          # batch limit for the folder option

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@dataclass
class ValidationResult:
    name: str
    ok: bool
    image: Image.Image | None = None
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0


def validate_image_bytes(name: str, data: bytes) -> ValidationResult:
    """Validate one file given its name and raw bytes (used for uploads and files on disk)."""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
        return ValidationResult(name, False, error=f"Unsupported file type '{ext or 'none'}'. Allowed: {allowed}.")
    if len(data) == 0:
        return ValidationResult(name, False, error="The file is empty (0 bytes).")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return ValidationResult(name, False, error=f"The file is {size_mb:.1f} MB; the maximum is {MAX_FILE_MB} MB.")

    warnings: list[str] = []
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                      # detects truncated / corrupted files
        img = Image.open(io.BytesIO(data))
        n_frames = getattr(img, "n_frames", 1)
        img.load()
    except Image.DecompressionBombError:
        return ValidationResult(name, False, error="The image is far too large (possible decompression bomb).")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return ValidationResult(name, False, error="The file is not a valid or readable image: it may be damaged, "
                                                   f"incomplete or not really a picture ({type(exc).__name__}). "
                                                   "Please choose another photo.")

    if n_frames > 1:
        warnings.append("Animated/multi-page image: only the first frame is used.")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        warnings.append("Transparent areas were filled with white.")
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    elif img.mode in ("L", "1", "I", "I;16", "F"):
        warnings.append("Grey-scale image: colour is an important cue, so confidence may be lower.")
    img = ImageOps.exif_transpose(img).convert("RGB")

    w, h = img.size
    if min(w, h) < MIN_SIDE_PX:
        return ValidationResult(name, False, error=f"The image is too small ({w}x{h} px); at least {MIN_SIDE_PX} px per side is needed.")
    if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
        return ValidationResult(name, False, error=f"Extreme aspect ratio ({w}x{h}); photograph a single item instead.")
    if min(w, h) < LOW_RES_PX:
        warnings.append(f"Low resolution ({w}x{h} px): the prediction may be less reliable.")
    if max(w, h) / min(w, h) > 2:
        warnings.append("Very elongated image: only the central square part is analysed.")
    return ValidationResult(name, True, image=img, warnings=warnings, width=w, height=h)


def validate_image_file(path: str | Path) -> ValidationResult:
    path = Path(path)
    if not path.exists():
        return ValidationResult(path.name, False, error=f"File not found: {path}")
    if not path.is_file():
        return ValidationResult(path.name, False, error=f"Not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ValidationResult(path.name, False, error=f"The file cannot be read ({exc.strerror}).")
    return validate_image_bytes(path.name, data)


def list_folder_images(folder: str | Path) -> tuple[list[Path], str]:
    """Return (image paths, error message). The error is empty when the folder is usable."""
    folder = Path(str(folder).strip().strip('"').strip("'")).expanduser()
    if not str(folder) or str(folder) == ".":
        return [], "Please enter a folder path."
    if not folder.exists():
        return [], f"The folder does not exist: {folder}"
    if not folder.is_dir():
        return [], f"This is a file, not a folder: {folder}"
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        return [], "The folder contains no supported images (jpg, jpeg, png, bmp, webp, tif, tiff)."
    if len(files) > MAX_FOLDER_IMAGES:
        return files[:MAX_FOLDER_IMAGES], ""
    return files, ""
