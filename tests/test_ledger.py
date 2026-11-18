import json
import os
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
TODAY = date(2026, 11, 18)


# ---------- parser ----------
from ledger import categorise, ocr, parser, preprocess


@pytest.mark.parametrize("token, value", [
    ("R1 299,99", 1299.99), ("1,299.99", 1299.99), ("R 45.50", 45.50), ("TOTAL R89.90", 89.90), ("no money here", None),
])
def test_parse_money_handles_sa_formats(token, value):
    assert parser.parse_money(token) == value


def test_total_line_beats_cash_and_change():
    text = "SUBTOTAL 100.00\nVAT 15.00\nTOTAL R 115.00\nCASH 200.00\nCHANGE 85.00"
    assert parser.find_amount(text) == 115.00


def test_amount_on_line_after_total_label():
    assert parser.find_amount("Total Due\nR 42.00") == 42.00


def test_ocr_digit_slips_are_repaired():
    assert parser.find_amount("Amount: Ril 500.00") == 1500.00


@pytest.mark.parametrize("text, iso", [
    ("03/11/2026 14:22", "2026-11-03"), ("2026/11/15 07:55", "2026-11-15"),
    ("Date: 10 Nov 2026", "2026-11-10"), ("07-11-26", "2026-11-07"),
])
def test_dates_are_read_day_first(text, iso):
    assert parser.find_date(text, today=TODAY) == iso


def test_impossible_and_future_dates_are_rejected():
    assert parser.find_date("31/02/2026", today=TODAY) is None
    assert parser.find_date("01/12/2027", today=TODAY) is None


def test_merchant_survives_ocr_typo():
    assert parser.find_merchant("SH0PRITE\nBellville") == "Shoprite"


def test_income_detection():
    assert parser.find_kind("You have received a payment") == "income"
    assert parser.find_kind("TAX INVOICE\nTOTAL 20.00") == "expense"


# ---------- full pipeline on sample photos ----------
EXPECTED = json.loads((SAMPLES / "expected.json").read_text())


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_sample_photo_end_to_end(filename):
    clean, outline = preprocess.prepare((SAMPLES / filename).read_bytes())
    assert outline, "receipt outline should be detected and flattened"
    text, conf = ocr.read_text(clean)
    draft = parser.parse(text, today=TODAY)
    expected = EXPECTED[filename]
    assert draft.merchant == expected["merchant"]
    assert draft.amount == pytest.approx(expected["amount"])
    assert draft.date == expected["date"]
    assert draft.kind == expected["kind"]


def test_unreadable_upload_is_rejected():
    with pytest.raises(ValueError):
        preprocess.load_image(b"not an image")


# ---------- categoriser ----------
def test_categoriser_knows_common_merchants():
    model = categorise.build()
    assert model.predict("KFC")[0] == "Eating out"
    assert model.predict("Engen")[0] == "Transport"


def test_categoriser_learns_from_a_correction():
    before = categorise.build().predict("Uber Eats")[0]
    after = categorise.build(["Uber Eats"], ["Eating out"]).predict("Uber Eats")[0]
    assert before == "Transport" and after == "Eating out"


# ---------- API ----------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGER_DB", str(tmp_path / "test.db"))
    import importlib
    from ledger import api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def test_scan_then_save_then_summary(client):
    with open(SAMPLES / "kfc.jpg", "rb") as f:
        draft = client.post("/api/scan", files={"file": ("kfc.jpg", f, "image/jpeg")}).json()
    assert draft["amount"] == 89.90 and draft["category"] == "Eating out"
    client.post("/api/transactions", json={"date": "2026-11-07", "merchant": "KFC", "amount": 89.90,
                                           "kind": "expense", "category": "Eating out"})
    client.post("/api/transactions", json={"date": "2026-11-10", "merchant": "M Ndlovu", "amount": 1500,
                                           "kind": "income", "category": "Income", "source": "manual"})
    s = client.get("/api/summary?month=2026-11").json()
    assert s == {"month": "2026-11", "income": 1500.0, "spent": 89.9, "net": 1410.1,
                 "by_category": {"Eating out": 89.9}}


def test_bad_transaction_is_refused(client):
    r = client.post("/api/transactions", json={"date": "7 Nov", "merchant": "", "amount": -5,
                                               "kind": "gift", "category": "X"})
    assert r.status_code == 422


def test_non_image_upload_gets_clear_error(client):
    r = client.post("/api/scan", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "JPEG or PNG" in r.json()["detail"]


def test_missing_tesseract_gives_clear_503(client, monkeypatch):
    import pytesseract
    from ledger import ocr as ocr_mod
    def boom(*a, **k):
        raise pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(ocr_mod.pytesseract, "image_to_data", boom)
    with open(SAMPLES / "kfc.jpg", "rb") as f:
        r = client.post("/api/scan", files={"file": ("kfc.jpg", f, "image/jpeg")})
    assert r.status_code == 503 and "Tesseract" in r.json()["detail"]
