"""Input validation: invalid files must be rejected with a message, odd files accepted with a warning."""
import io

from PIL import Image

from validation import list_folder_images, validate_image_bytes, validate_image_file


def _png(size=(256, 256), mode="RGB", color=(200, 100, 50)):
    buf = io.BytesIO()
    Image.new(mode, size, color if mode != "L" else 128).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_is_accepted():
    r = validate_image_bytes("bottle.png", _png())
    assert r.ok and r.image.mode == "RGB" and r.warnings == []


def test_wrong_extension_is_rejected():
    r = validate_image_bytes("notes.txt", b"hello")
    assert not r.ok and "Unsupported file type" in r.error


def test_empty_file_is_rejected():
    r = validate_image_bytes("empty.jpg", b"")
    assert not r.ok and "empty" in r.error


def test_corrupted_file_is_rejected():
    r = validate_image_bytes("broken.jpg", b"definitely not a jpeg")
    assert not r.ok and "not a valid" in r.error


def test_too_small_image_is_rejected():
    r = validate_image_bytes("tiny.png", _png(size=(20, 20)))
    assert not r.ok and "too small" in r.error


def test_extreme_aspect_ratio_is_rejected():
    r = validate_image_bytes("strip.png", _png(size=(1400, 100)))
    assert not r.ok and "aspect ratio" in r.error


def test_greyscale_and_transparent_images_get_warnings():
    grey = validate_image_bytes("grey.png", _png(mode="L"))
    rgba = validate_image_bytes("alpha.png", _png(mode="RGBA", color=(0, 0, 0, 0)))
    assert grey.ok and any("Grey-scale" in w for w in grey.warnings)
    assert rgba.ok and any("Transparent" in w for w in rgba.warnings)
    assert rgba.image.getpixel((5, 5)) == (255, 255, 255)   # filled with white


def test_low_resolution_gets_warning():
    r = validate_image_bytes("small.png", _png(size=(64, 64)))
    assert r.ok and any("Low resolution" in w for w in r.warnings)


def test_missing_file_and_folder(tmp_path):
    assert not validate_image_file(tmp_path / "missing.jpg").ok
    files, error = list_folder_images(tmp_path / "nope")
    assert files == [] and "does not exist" in error


def test_folder_listing_keeps_only_images(tmp_path):
    (tmp_path / "a.png").write_bytes(_png())
    (tmp_path / "b.txt").write_text("x")
    files, error = list_folder_images(tmp_path)
    assert error == "" and [f.name for f in files] == ["a.png"]
