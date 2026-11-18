"""Generate realistic test photos of South African receipts and payment slips.

Each receipt is rendered in a till-printer font, then placed on a darker 'table' background with a
random tilt, perspective, uneven lighting and sensor noise, imitating a phone photo. These are fictional
receipts for testing only. The expected values are written to samples/expected.json so tests can
check the pipeline end to end.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

RECEIPTS = [
    {"file": "shoprite.jpg", "merchant": "Shoprite", "date": "2026-11-03", "amount": 287.45, "kind": "expense",
     "lines": ["SHOPRITE", "Bellville Mall", "TAX INVOICE", "VAT No 4100104461", "", "03/11/2026  14:22",
               "", "ALBANY BREAD 700G        18.99", "CLOVER MILK 2L           32.49", "WHITE STAR MAIZE 5KG     79.99",
               "SUNFOIL OIL 2L           69.99", "CHICKEN PIECES 1KG       85.99", "", "SUBTOTAL                287.45",
               "VAT 15%                  37.49", "TOTAL               R   287.45", "CASH                    300.00",
               "CHANGE                   12.55", "", "THANK YOU FOR SHOPPING"]},
    {"file": "kfc.jpg", "merchant": "KFC", "date": "2026-11-07", "amount": 89.90, "kind": "expense",
     "lines": ["KFC", "Cape Town CBD", "Tax Invoice", "", "Date: 07-11-2026 18:41", "Order 1043", "",
               "1 Streetwise 2 + Chips   59.90", "1 Coke 500ml             30.00", "", "Total Due       R89.90",
               "Card                     89.90", "", "Finger lickin' good"]},
    {"file": "clicks.jpg", "merchant": "Clicks", "date": "2026-11-12", "amount": 154.97, "kind": "expense",
     "lines": ["CLICKS", "Tyger Valley", "TAX INVOICE", "12 Nov 2026", "", "PANADO 24S               42.99",
               "VASELINE LOTION          64.99", "COLGATE 100ML            46.99", "", "TOTAL SAVINGS             8.00",
               "AMOUNT DUE         R 154.97", "CLUBCARD POINTS EARNED   15", "", "Be well"]},
    {"file": "engen.jpg", "merchant": "Engen", "date": "2026-11-15", "amount": 450.00, "kind": "expense",
     "lines": ["ENGEN", "Voortrekker Rd Service Stn", "TAX INVOICE", "2026/11/15 07:55", "", "PUMP 4  ULP95",
               "LITRES    20.93", "PRICE/L   21.50", "", "TOTAL           R 450.00", "", "Have a safe trip"]},
    {"file": "vodacom.jpg", "merchant": "Vodacom", "date": "2026-11-16", "amount": 99.00, "kind": "expense",
     "lines": ["VODACOM", "Prepaid Recharge", "", "Date 16/11/2026", "", "Data Bundle 2GB", "", "Amount     R99.00",
               "Ref  VDC88231", "", "Stay connected"]},
    {"file": "payment_received.jpg", "merchant": "Capitec", "date": "2026-11-10", "amount": 1500.00, "kind": "income",
     "lines": ["CAPITEC", "Payment Notification", "", "You have received a payment", "", "Date:   10 Nov 2026",
               "From:   M NDLOVU", "Ref:    MATHS TUTORING", "", "Amount:  R1 500.00", "", "Available balance updated"]},
]


def render(lines):
    font, bold = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT_BOLD, 34)
    w, line_h = 560, 38
    img = Image.new("L", (w, 60 + line_h * len(lines) + 40), 248)
    d = ImageDraw.Draw(img)
    y = 40
    for i, line in enumerate(lines):
        f = bold if i == 0 else font
        tw = d.textlength(line, font=f)
        x = (w - tw) / 2 if i < 2 else 30
        d.text((x, y), line, fill=30, font=f)
        y += line_h + (8 if i == 0 else 0)
    return cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)


def photograph(receipt, rng):
    """Place the receipt on a table with tilt, perspective, a lighting gradient and noise."""
    h, w = receipt.shape[:2]
    H, W = int(h * 1.35), int(w * 1.9)
    table = np.full((H, W, 3), (70, 85, 105), np.uint8)
    table = cv2.add(table, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = W / 2, H / 2
    angle = np.deg2rad(rng.uniform(-8, 8))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = (src - [w / 2, h / 2]) @ rot.T + [cx, cy]
    corners += rng.uniform(-25, 25, corners.shape)                     # perspective wobble
    M = cv2.getPerspectiveTransform(src, np.float32(corners))
    warped = cv2.warpPerspective(receipt, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    photo = np.where(mask[..., None] > 0, warped, table)
    shade = np.linspace(rng.uniform(0.75, 0.9), 1.05, W)[None, :, None]   # uneven light across the shot
    photo = np.clip(photo * shade + rng.normal(0, 6, photo.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(photo, (3, 3), 0)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    expected = {}
    for r in RECEIPTS:
        cv2.imwrite(str(OUT / r["file"]), photograph(render(r["lines"]), rng), [cv2.IMWRITE_JPEG_QUALITY, 88])
        expected[r["file"]] = {k: r[k] for k in ("merchant", "date", "amount", "kind")}
    (OUT / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Wrote {len(RECEIPTS)} sample photos to {OUT}")


if __name__ == "__main__":
    main()
