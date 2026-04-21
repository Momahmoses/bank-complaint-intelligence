"""
Multi-class complaint routing: assigns each complaint to the correct department.
Also performs NER to extract branch names, product names, and amounts.
"""
import pandas as pd
import numpy as np
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
import os

logger = logging.getLogger(__name__)

DEPARTMENTS = {
    "cards": ["card", "debit", "credit", "atm", "pos", "declined", "blocked card", "swipe"],
    "loans": ["loan", "credit", "borrow", "repayment", "interest", "collateral", "overdraft"],
    "transfers": ["transfer", "transaction", "send money", "receive", "beneficiary", "failed transfer", "wrong account"],
    "account_opening": ["open account", "kyc", "bvn", "nin", "verification", "documents", "account number"],
    "mobile_banking": ["app", "mobile banking", "ussd", "online banking", "login", "password", "otp"],
    "customer_service": ["branch", "staff", "officer", "service", "wait", "queue", "rude"],
    "fraud_security": ["fraud", "hack", "unauthorized", "scam", "stolen", "phishing", "suspicious"],
    "charges_fees": ["charge", "fee", "deduction", "sms alert", "maintenance fee", "commission", "hidden"],
}

NIGERIAN_AMOUNTS_PATTERN = re.compile(
    r"(?:₦|NGN|naira)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:naira|NGN|₦)?",
    re.IGNORECASE,
)

BRANCH_NAMES = [
    "Victoria Island", "Lekki", "Ikeja", "Surulere", "Yaba", "Ajah",
    "Abuja", "Kano", "Port Harcourt", "Ibadan", "Enugu",
    "Marina", "Ikorodu", "Festac", "Agege",
]

PRODUCT_NAMES = [
    "Current Account", "Savings Account", "Fixed Deposit", "Mortgage",
    "Personal Loan", "Business Loan", "Mastercard", "Visa Card",
    "USSD", "Mobile App", "Internet Banking", "POS Terminal",
]


def extract_entities(text: str) -> dict:
    text_upper = text.upper()

    amounts = NIGERIAN_AMOUNTS_PATTERN.findall(text)
    parsed_amounts = []
    for a in amounts:
        try:
            parsed_amounts.append(float(a.replace(",", "")))
        except ValueError:
            pass

    branches = [b for b in BRANCH_NAMES if b.lower() in text.lower()]
    products = [p for p in PRODUCT_NAMES if p.lower() in text.lower()]

    account_match = re.search(r"\b(\d{10})\b", text)
    account_number = account_match.group(1) if account_match else None

    return {
        "amounts": parsed_amounts,
        "branches": branches,
        "products": products,
        "account_number": account_number,
    }


def rule_based_route(text: str) -> tuple[str, float]:
    text_lower = text.lower()
    scores = {}
    for dept, keywords in DEPARTMENTS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[dept] = score
    if scores:
        best = max(scores, key=scores.get)
        confidence = min(0.95, 0.5 + scores[best] * 0.1)
        return best, confidence
    return "customer_service", 0.4


class ComplaintClassifier:
    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
        ])
        self.classes_ = list(DEPARTMENTS.keys())
        self._is_trained = False

    def fit(self, texts: list[str], labels: list[str]):
        self.model.fit(texts, labels)
        self.classes_ = list(self.model.classes_)
        self._is_trained = True
        scores = cross_val_score(self.model, texts, labels, cv=5, scoring="f1_weighted")
        logger.info(f"Classifier CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
        return self

    def predict(self, text: str) -> dict:
        if self._is_trained:
            probas = self.model.predict_proba([text])[0]
            pred_idx = probas.argmax()
            return {
                "department": self.classes_[pred_idx],
                "confidence": round(probas[pred_idx], 3),
                "all_scores": {self.classes_[i]: round(p, 3) for i, p in enumerate(probas)},
            }
        dept, conf = rule_based_route(text)
        return {"department": dept, "confidence": conf, "all_scores": {dept: conf}}

    def save(self, path: str = "models/complaint_classifier.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "models/complaint_classifier.pkl"):
        return joblib.load(path)


def full_complaint_analysis(text: str, sentiment_analyzer=None, classifier=None) -> dict:
    sentiment = {}
    if sentiment_analyzer:
        sentiment = sentiment_analyzer.predict(text)

    routing = classifier.predict(text) if classifier else {"department": rule_based_route(text)[0]}
    entities = extract_entities(text)

    return {
        "original_text": text[:500],
        "department": routing["department"],
        "routing_confidence": routing.get("confidence"),
        "sentiment": sentiment.get("sentiment", "UNKNOWN"),
        "is_critical": sentiment.get("is_critical", False),
        "entities": entities,
        "priority": "HIGH" if sentiment.get("is_critical") else ("MEDIUM" if sentiment.get("sentiment") == "NEGATIVE" else "LOW"),
    }
