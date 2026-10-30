import numpy as np
from PIL import Image

from marker.utils.image import is_blank_image


def _dense_widow_crop() -> Image.Image:
    """A tightly-cropped, fully-inked line only 7 px tall.

    Mimics the short final line of a paragraph (a one- or two-word widow, or
    the tail of a hyphenated word) as ``filter_blank_lines`` crops it from the
    96-dpi lowres page image: small in both dimensions and dominated by ink.
    Mid-gray vertical strokes reproduce the antialiased text of a real crop.
    """
    gray = np.full((7, 47), 255, np.uint8)
    gray[1:6, ::2] = 90  # ~37% ink, matching the reported line crops
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return Image.fromarray(rgb)


def test_is_blank_image_keeps_small_inked_line():
    # A tiny crop that is plainly full of text must not be judged blank; the
    # 7x7 blur used to be as tall as the crop and averaged the strokes away,
    # so filter_blank_lines silently deleted the line (issue #1071).
    assert is_blank_image(_dense_widow_crop()) is False


def test_is_blank_image_still_reports_blank_crops():
    # Small and full-size white crops must stay blank.
    assert is_blank_image(Image.fromarray(np.full((7, 47, 3), 255, np.uint8))) is True
    assert (
        is_blank_image(Image.fromarray(np.full((12, 347, 3), 255, np.uint8))) is True
    )


def test_is_blank_image_detects_normal_line():
    # A normal-height inked line is detected as not blank, unchanged.
    gray = np.full((12, 347), 255, np.uint8)
    gray[3:9, ::6] = 60
    assert is_blank_image(Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2))) is False


def test_is_blank_image_keeps_noisy_paper_blank():
    # Scanned paper is never a flat 255: it carries a few grey levels of
    # sensor/JPEG noise. The blur is what stops that noise from thresholding
    # as ink, so shrinking the kernel for a small crop must not remove it --
    # dropping to a 1x1 (i.e. no) blur reports blank paper as text and
    # defeats filter_blank_lines on exactly the crops this function guards.
    rng = np.random.default_rng(0)
    for height, width in ((2, 20), (3, 47), (5, 90), (7, 340)):
        paper = np.clip(rng.normal(183, 6, (height, width)), 0, 255).astype(np.uint8)
        crop = Image.fromarray(np.repeat(paper[:, :, None], 3, axis=2))
        assert is_blank_image(crop) is True, f"{width}x{height} noisy paper"
