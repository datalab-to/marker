from typing import List, Optional, Sequence
import cv2
import numpy as np
from PIL import Image


def is_blank_gray(gray: np.ndarray) -> bool:
    if gray is None or gray.size == 0 or gray.shape[0] == 0 or gray.shape[1] == 0:
        return True
    # In adaptiveThreshold(..., THRESH_BINARY_INV, 31, 15), a pixel is foreground
    # iff blurred < local_mean - 15. Since both lie in [min, max], if max - min <= 15
    # no pixel can ever be foreground.
    if int(gray.max()) - int(gray.min()) <= 15:
        return True
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    binarized = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return not bool(binarized.any())


def is_blank_bbox(gray_page: np.ndarray, bbox: Sequence[float]) -> bool:
    h, w = gray_page.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2]))
    y1 = min(h, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return True
    return is_blank_gray(gray_page[y0:y1, x0:x1])


def is_blank_image(
    image: Image.Image, polygon: Optional[List[List[int]]] = None
) -> bool:
    arr = np.asarray(image)
    if arr is None or arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return True

    if polygon is not None:
        rounded_polys = [[int(corner[0]), int(corner[1])] for corner in polygon]
        if (
            rounded_polys[0] == rounded_polys[1]
            and rounded_polys[2] == rounded_polys[3]
        ):
            return True

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    return is_blank_gray(gray)
