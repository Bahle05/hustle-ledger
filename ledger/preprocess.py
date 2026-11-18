"""Image clean-up before OCR.

Phone photos of receipts are rarely flat or straight. This module:
1. finds the receipt (the largest four-sided shape) and warps it flat, like a document scanner,
2. upscales small images so characters are big enough for OCR,
3. converts to greyscale, removes noise and binarises with an adaptive threshold
   so uneven lighting and shadows don't wipe out text.
"""

import cv2
import numpy as np


def load_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes (JPEG/PNG) into an OpenCV BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image. Upload a JPEG or PNG photo.")
    return img


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_document(img: np.ndarray):
    """Locate the receipt outline. Returns its 4 corners, or None if no clear outline exists."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    scale = min(scale, 1.0)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        # Ignore tiny shapes and shapes that are just the photo border
        if len(approx) == 4 and 0.15 * area_total < cv2.contourArea(approx) < 0.98 * area_total:
            return _order_corners(approx) / scale
    return None


def flatten(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-warp the four corners into a straight, top-down rectangle."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def clean_for_ocr(img: np.ndarray) -> np.ndarray:
    """Greyscale, upscale if small, denoise, then adaptive threshold to black text on white."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.shape[1] < 1000:
        f = 1000 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def prepare(data: bytes):
    """Full pipeline. Returns (clean image for OCR, whether a receipt outline was found)."""
    img = load_image(data)
    corners = find_document(img)
    if corners is not None:
        img = flatten(img, corners)
    return clean_for_ocr(img), corners is not None
