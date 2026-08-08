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
