"""
Sentiment analysis fine-tuned for Nigerian English and Pidgin.
Handles "this bank don finish me" as very negative.
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Union

logger = logging.getLogger(__name__)

NIGERIAN_PIDGIN_LEXICON = {
    "don finish me": -3, "cheat": -2, "thief": -3, "steal": -3,
    "rubbish": -2, "useless": -2, "terrible": -2, "horrible": -2,
    "scatter": -1, "mess up": -2, "wahala": -1, "suffer": -2,
    "block": -1, "freeze": -1, "hold": -1, "delay": -1,
    "sweet": 2, "perfect": 3, "excellent": 3, "wonderful": 2,
    "fast": 1, "quick": 1, "easy": 1, "helpful": 2, "good": 1,
    "bad": -2, "poor": -2, "slow": -1, "long": -1, "frustrat": -2,
    "disappoint": -2, "angry": -2, "upset": -2, "annoyed": -2,
    "happy": 2, "pleased": 2, "satisfied": 2, "recommend": 2,
    "problem": -1, "issue": -1, "error": -1, "fail": -2, "wrong": -1,
    "scam": -3, "fraud": -3, "hack": -3, "unauthorized": -2,
}


def normalize_nigerian_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    replacements = {
        r"\bdon\b": "have",
        r"\bdey\b": "is/are",
        r"\bna\b": "is",
        r"\bwetin\b": "what",
        r"\bno go\b": "won't",
        r"\bfit\b": "can",
        r"\bpikin\b": "child",
        r"\bchop\b": "eat/take",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def lexicon_sentiment_score(text: str) -> float:
    text_lower = text.lower()
    score = 0
    for phrase, weight in NIGERIAN_PIDGIN_LEXICON.items():
        if phrase in text_lower:
            score += weight
    return np.clip(score / 3.0, -1, 1)


class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        self.transformer_available = False
        self.pipeline = None
        self._load_transformer(model_name)

    def _load_transformer(self, model_name: str):
        try:
            from transformers import pipeline as hf_pipeline
            self.pipeline = hf_pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1,
                truncation=True,
                max_length=512,
            )
            self.transformer_available = True
            logger.info(f"Loaded transformer: {model_name}")
        except Exception as e:
            logger.warning(f"Transformer not available: {e}. Using lexicon fallback.")

    def predict(self, text: str) -> dict:
        normalized = normalize_nigerian_text(text)
        lexicon_score = lexicon_sentiment_score(text)

        if self.transformer_available and self.pipeline:
            try:
                result = self.pipeline(normalized[:512])[0]
                label_map = {"LABEL_2": "POSITIVE", "LABEL_1": "NEUTRAL", "LABEL_0": "NEGATIVE",
                             "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL"}
                transformer_label = label_map.get(result["label"], "NEUTRAL")
                transformer_score = result["score"]
                if lexicon_score < -0.3:
                    sentiment = "NEGATIVE"
                    confidence = max(transformer_score, abs(lexicon_score))
                else:
                    sentiment = transformer_label
                    confidence = transformer_score
            except Exception:
                sentiment = "NEGATIVE" if lexicon_score < -0.1 else ("POSITIVE" if lexicon_score > 0.1 else "NEUTRAL")
                confidence = abs(lexicon_score)
        else:
            if lexicon_score < -0.15:
                sentiment = "NEGATIVE"
            elif lexicon_score > 0.15:
                sentiment = "POSITIVE"
            else:
                sentiment = "NEUTRAL"
            confidence = min(0.99, 0.5 + abs(lexicon_score) * 0.5)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "lexicon_score": round(lexicon_score, 3),
            "is_critical": sentiment == "NEGATIVE" and confidence > 0.8,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "complaint_text") -> pd.DataFrame:
        results = self.predict_batch(df[text_col].fillna("").tolist())
        sentiment_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
