"""Text recognition with Tesseract, keeping only reasonably confident words."""

import os
import shutil

import numpy as np
import pytesseract

# Find Tesseract even when it isn't on PATH: an explicit TESSERACT_CMD wins, then the default Windows install.
_cmd = os.getenv("TESSERACT_CMD")
if not _cmd and not shutil.which("tesseract"):
    for candidate in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                      r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(candidate):
            _cmd = candidate
            break
if _cmd:
    pytesseract.pytesseract.tesseract_cmd = _cmd


class OCRUnavailable(RuntimeError):
    """Raised when the Tesseract program itself can't be found."""


import re

MONEY_TOKEN = re.compile(r"\d[.,]\d{2}\b")


def _read(img: np.ndarray, psm: int, min_conf: float):
    try:
        data = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise OCRUnavailable(
            "Tesseract OCR isn't installed or can't be found. Install it (see README step 1) "
            "or set TESSERACT_CMD to the full path of tesseract.exe, then restart the app.") from e
    lines, confs = {}, []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < min_conf:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0), len(confs)


def read_text(img: np.ndarray, min_conf: float = 30.0):
    """Return (text with original line breaks, mean word confidence 0..100).

    Receipts mix left-aligned item names with right-aligned prices. Tesseract's page segmentation
    modes handle that differently: --psm 4 (single column) sometimes drops the right-hand prices,
    --psm 6 (uniform block) sometimes merges columns. Both are run and the reading that recovers
    more words and more money values, weighted by confidence, is kept.
    """
    best = None
    for psm in (6, 4):
        text, conf, n_words = _read(img, psm, min_conf)
        score = (n_words + 5 * len(MONEY_TOKEN.findall(text))) * conf
        if best is None or score > best[0]:
            best = (score, text, conf)
    return best[1], best[2]
