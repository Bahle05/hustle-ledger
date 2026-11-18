"""Spending category classifier that learns from the user's own corrections.

Model: TF-IDF over word and character n-grams, then multinomial logistic regression.
Character n-grams make it robust to OCR typos ("SH0PRITE" still shares most n-grams with "Shoprite").
It starts from a small hand-labelled seed set of South African merchants and is retrained whenever
the user corrects a category, so it adapts to where that person actually shops.
"""

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

SEED = Path(__file__).resolve().parent.parent / "data" / "seed_labels.csv"


def load_seed(path: Path = SEED):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


class Categoriser:
    def __init__(self):
        self.model = None

    def fit(self, texts, labels):
        features = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)),
        ])
        self.model = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=10))])
        self.model.fit(texts, labels)
        return self

    def predict(self, text: str):
        """Return (category, confidence 0..1)."""
        probs = self.model.predict_proba([text])[0]
        best = probs.argmax()
        return self.model.classes_[best], float(probs[best])

    @property
    def categories(self):
        return list(self.model.classes_)


def build(extra_texts=(), extra_labels=()) -> Categoriser:
    """Train on the seed set plus any user corrections. Corrections are repeated so they outweigh the seed."""
    texts, labels = load_seed()
    texts += list(extra_texts) * 3
    labels += list(extra_labels) * 3
    return Categoriser().fit(texts, labels)
