import numpy as np
from PIL import Image

from marker.utils.image import is_blank_image


def _rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def test_small_text_crop_is_not_blank():
    """Widow-line crops are smaller than the adaptive-threshold window."""
    crop = np.full((8, 22, 3), 255, dtype=np.uint8)
    crop[2:6, 3:7] = 20
    crop[2:6, 10:14] = 20
    crop[2:6, 16:20] = 20
    assert is_blank_image(_rgb(crop)) is False


def test_small_white_crop_is_blank():
    crop = np.full((8, 22, 3), 255, dtype=np.uint8)
    assert is_blank_image(_rgb(crop)) is True


def test_large_text_crop_still_detected():
    crop = np.full((64, 64, 3), 255, dtype=np.uint8)
    crop[16:48, 16:48] = 0
    assert is_blank_image(_rgb(crop)) is False


def test_large_white_crop_is_blank():
    crop = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert is_blank_image(_rgb(crop)) is True
