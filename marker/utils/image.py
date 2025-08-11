from PIL import Image
import numpy as np
import cv2

def is_blank_image(image: Image.Image) -> bool:
    image = np.asarray(image)
    if (
        image is None
        or image.size == 0
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        # Handle empty image case
        return True

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold (inverse for text as white)
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binarized, connectivity=8
    )
    cleaned = np.zeros_like(binarized)
    for i in range(1, num_labels):  # skip background
        cleaned[labels == i] = 255

    kernel = np.ones((1, 5), np.uint8)
    dilated = cv2.dilate(cleaned, kernel, iterations=3)
    b = dilated / 255
    return b.sum() == 0