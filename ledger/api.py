"""FastAPI app: scan a receipt into a draft, let the user confirm it, then report on the month.

Nothing from a scan is saved automatically. OCR makes mistakes, so the user always reviews
and edits the draft first, and their category edits are fed back into the classifier.
"""

import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import categorise, ocr, parser, preprocess
from .store import Store

STATIC = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB

app = FastAPI(title="Hustle Ledger")
store = Store(os.getenv("LEDGER_DB", "ledger.db"))
model = categorise.build(*store.corrections())


def classifier_text(merchant: str | None, raw_text: str = "") -> str:
    top = " ".join(raw_text.splitlines()[:6])
    return f"{merchant or ''} {top}".strip()


class Transaction(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    merchant: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, lt=10_000_000)
    kind: str = Field(pattern="^(expense|income)$")
    category: str = Field(min_length=1, max_length=40)
    suggested_category: str | None = None
    source: str = "scan"


@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 8 MB. Take the photo at a lower resolution.")
    try:
        clean, found_outline = preprocess.prepare(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        text, confidence = ocr.read_text(clean)
    except ocr.OCRUnavailable as e:
        raise HTTPException(503, str(e))
    if not text.strip():
        raise HTTPException(422, "No text found. Retake the photo in good light with the whole receipt in frame.")
    draft = parser.parse(text).to_dict()
    if draft["kind"] == "income":
        category, cat_conf = "Income", 1.0
    else:
        category, cat_conf = model.predict(classifier_text(draft["merchant"], text))
    return {**draft, "category": category, "category_confidence": round(cat_conf, 2),
            "ocr_confidence": round(confidence, 1), "receipt_outline_found": found_outline}


@app.post("/api/transactions", status_code=201)
def save(tx: Transaction):
    global model
    tx_id = store.add(tx.date, tx.merchant, tx.amount, tx.kind, tx.category, tx.source)
    # Learn from corrections: if the user changed the suggested category, retrain with their label
    if tx.suggested_category and tx.suggested_category != tx.category and tx.kind == "expense":
        store.add_correction(tx.merchant, tx.category)
        model = categorise.build(*store.corrections())
    return {"id": tx_id}


@app.get("/api/transactions")
def list_transactions(month: str | None = None):
    return store.list(month)


@app.delete("/api/transactions/{tx_id}", status_code=204)
def delete(tx_id: int):
    if not store.delete(tx_id):
        raise HTTPException(404, "Transaction not found")


@app.get("/api/summary")
def summary(month: str | None = None):
    return store.summary(month or date.today().strftime("%Y-%m"))


@app.get("/api/months")
def months():
    return store.months()


@app.get("/api/categories")
def categories():
    return sorted(set(model.categories) | {"Income"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
