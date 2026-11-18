"""SQLite storage for confirmed transactions and the user's category corrections."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    date      TEXT NOT NULL,
    merchant  TEXT NOT NULL,
    amount    REAL NOT NULL CHECK (amount > 0),
    kind      TEXT NOT NULL CHECK (kind IN ('expense', 'income')),
    category  TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'scan',
    created   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS corrections (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    text      TEXT NOT NULL,
    category  TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = "ledger.db"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add(self, date, merchant, amount, kind, category, source="scan") -> int:
        cur = self.conn.execute(
            "INSERT INTO transactions (date, merchant, amount, kind, category, source) VALUES (?, ?, ?, ?, ?, ?)",
            (date, merchant, amount, kind, category, source))
        self.conn.commit()
        return cur.lastrowid

    def delete(self, tx_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def list(self, month: str | None = None):
        sql, args = "SELECT * FROM transactions", ()
        if month:
            sql, args = sql + " WHERE substr(date, 1, 7) = ?", (month,)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY date DESC, id DESC", args)]

    def summary(self, month: str):
        """Income, spending and net for one month, plus spending by category."""
        rows = self.conn.execute(
            "SELECT kind, category, SUM(amount) AS total FROM transactions "
            "WHERE substr(date, 1, 7) = ? GROUP BY kind, category", (month,)).fetchall()
        income = sum(r["total"] for r in rows if r["kind"] == "income")
        spent = sum(r["total"] for r in rows if r["kind"] == "expense")
        by_cat = {r["category"]: round(r["total"], 2) for r in rows if r["kind"] == "expense"}
        return {"month": month, "income": round(income, 2), "spent": round(spent, 2),
                "net": round(income - spent, 2), "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1]))}

    def months(self):
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT substr(date, 1, 7) FROM transactions ORDER BY 1 DESC")]

    def add_correction(self, text: str, category: str):
        self.conn.execute("INSERT INTO corrections (text, category) VALUES (?, ?)", (text, category))
        self.conn.commit()

    def corrections(self):
        rows = self.conn.execute("SELECT text, category FROM corrections").fetchall()
        return [r["text"] for r in rows], [r["category"] for r in rows]
