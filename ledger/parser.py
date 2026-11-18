"""Pull the useful fields out of OCR text from South African receipts and payment slips.

OCR text is messy, so every extractor is defensive: it returns None rather than guessing wildly,
and the user confirms every field before anything is saved.
"""

import re
from dataclasses import dataclass, asdict
from datetime import date
from difflib import get_close_matches

KNOWN_MERCHANTS = [
    "Shoprite", "Checkers", "Pick n Pay", "Spar", "Woolworths", "Boxer", "Food Lover's Market", "Usave",
    "KFC", "McDonald's", "Nando's", "Steers", "Debonairs", "Wimpy", "Chicken Licken", "Roman's Pizza",
    "Fishaways", "Galito's", "Burger King", "Spur",
    "Clicks", "Dis-Chem", "Medirite",
    "Pep", "Ackermans", "Mr Price", "Truworths", "Edgars", "Jet", "Cotton On", "Markham", "Sportscene",
    "Engen", "Shell", "BP", "Sasol", "TotalEnergies", "Caltex", "Uber", "Bolt", "Golden Arrow", "MyCiTi",
    "Vodacom", "MTN", "Cell C", "Telkom", "Rain",
    "Ster-Kinekor", "Nu Metro", "Netflix", "Showmax", "DStv", "Spotify",
    "Van Schaik", "CNA", "Takealot", "Game", "Makro",
    "Capitec", "FNB", "Absa", "Standard Bank", "Nedbank", "TymeBank",
]

# Lines that contain money but are NOT the amount the customer actually paid
NOT_TOTAL = re.compile(r"sub\s*-?\s*total|vat|tax|saving|discount|change|tendered|cash\s|rounding|points|balance\s+b/f", re.I)
TOTAL_WORDS = re.compile(r"\b(total|amount\s+(due|paid)|balance\s+due|amount|paid|payment)\b", re.I)
MONEY = re.compile(r"(?<![\d.,])R?\s?(\d{1,3}(?:[ ,]\d{3})+|\d+)([.,]\d{2})(?!\d)")
INCOME_WORDS = re.compile(r"payment\s+received|you\s+(have\s+)?received|deposit|salary|credited|credit\s+notification|stipend|allowance", re.I)

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


@dataclass
class Draft:
    merchant: str | None
    amount: float | None
    date: str | None          # ISO yyyy-mm-dd
    kind: str                  # "expense" or "income"
    raw_text: str

    def to_dict(self):
        return asdict(self)


def parse_money(token: str) -> float | None:
    """'R1 299,99' / '1,299.99' / '45.50' -> float."""
    m = MONEY.search(token)
    if not m:
        return None
    whole = re.sub(r"[ ,]", "", m.group(1))
    return round(float(f"{whole}.{m.group(2)[1:]}"), 2)


def fix_ocr_digits(line: str) -> str:
    """Repair common OCR slips inside amounts: 'Ril 500.00' -> 'R1 500.00', 'R1O5.00' -> 'R105.00'."""
    line = re.sub(r"\bR[iIl|]{1,2}(?=\s?\d)", "R1", line)
    return re.sub(r"(?<=\d)[oO](?=[\d.,])|(?<=[\d.,])[oO](?=\d)", "0", line)


def find_amount(text: str) -> float | None:
    """Prefer the amount on a 'TOTAL' style line; fall back to the largest plausible amount."""
    lines = [fix_ocr_digits(l) for l in text.splitlines()]
    for i, line in enumerate(lines):
        if TOTAL_WORDS.search(line) and not NOT_TOTAL.search(line):
            value = parse_money(line)
            if value is None and i + 1 < len(lines):        # amount printed on the next line
                value = parse_money(lines[i + 1])
            if value is not None and value > 0:
                return value
    amounts = [parse_money(l) for l in lines if not NOT_TOTAL.search(l)]
    amounts = [a for a in amounts if a]
    return max(amounts) if amounts else None


def find_date(text: str, today: date | None = None) -> str | None:
    """Day-first South African formats, plus ISO and '18 Nov 2026'. Rejects impossible or future dates."""
    today = today or date.today()
    candidates = []
    for d, mth, y in re.findall(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", text):
        candidates.append((int(y), int(mth), int(d)))
    for y, mth, d in re.findall(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b", text):
        candidates.append((int(y), int(mth), int(d)))
    for d, mon, y in re.findall(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{2,4})\b", text):
        if mon.lower() in MONTHS:
            candidates.append((int(y), MONTHS[mon.lower()], int(d)))
    for y, mth, d in candidates:
        y = y + 2000 if y < 100 else y
        try:
            found = date(y, mth, d)
        except ValueError:
            continue
        if date(2000, 1, 1) <= found <= today:
            return found.isoformat()
    return None


def find_merchant(text: str) -> str | None:
    """Match a known South African merchant near the top, else use the first name-like line."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:10]:
        for name in KNOWN_MERCHANTS:
            if re.search(rf"\b{re.escape(name)}\b", line, re.I):
                return name
        words = re.sub(r"[^A-Za-z' -]", " ", line).split()
        for size in (3, 2, 1):                                  # tolerate OCR slips like 'SH0PRITE'
            for j in range(len(words) - size + 1):
                hit = get_close_matches(" ".join(words[j:j + size]).lower(),
                                        [n.lower() for n in KNOWN_MERCHANTS], n=1, cutoff=0.85)
                if hit:
                    return next(n for n in KNOWN_MERCHANTS if n.lower() == hit[0])
    skip = re.compile(r"tax\s+invoice|receipt|vat\s+no|tel|www\.|@|^\d", re.I)
    for line in lines[:5]:
        if not skip.search(line) and len(re.sub(r"[^A-Za-z]", "", line)) >= 3:
            return line.title()[:40]
    return None


def find_kind(text: str) -> str:
    return "income" if INCOME_WORDS.search(text) else "expense"


def parse(text: str, today: date | None = None) -> Draft:
    return Draft(find_merchant(text), find_amount(text), find_date(text, today), find_kind(text), text)
