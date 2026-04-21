"""
Real-time alert system: triggers PR team if negative sentiment spikes 3x in 1 hour.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import logging
import threading
import time

logger = logging.getLogger(__name__)


class SentimentSpike:
    def __init__(self, topic: str, current_rate: float, baseline_rate: float,
                 spike_ratio: float, window_minutes: int, complaint_count: int):
        self.topic = topic
        self.current_rate = current_rate
        self.baseline_rate = baseline_rate
        self.spike_ratio = spike_ratio
        self.window_minutes = window_minutes
        self.complaint_count = complaint_count
        self.detected_at = datetime.now()

    def to_alert(self) -> dict:
        return {
            "alert_type": "SENTIMENT_SPIKE",
            "severity": "HIGH" if self.spike_ratio >= 5 else "MEDIUM",
            "topic": self.topic,
            "spike_ratio": round(self.spike_ratio, 1),
            "current_negative_rate": round(self.current_rate * 100, 1),
            "baseline_negative_rate": round(self.baseline_rate * 100, 1),
            "complaint_count": self.complaint_count,
            "window_minutes": self.window_minutes,
            "detected_at": self.detected_at.isoformat(),
            "message": (
                f"ALERT: {self.topic} — Negative sentiment spiked {self.spike_ratio:.1f}x "
                f"({self.current_negative_rate:.0f}% vs baseline {self.baseline_negative_rate:.0f}%) "
                f"in last {self.window_minutes} min ({self.complaint_count} complaints)"
            ),
        }


class RealTimeComplaintMonitor:
    def __init__(
        self,
        window_minutes: int = 60,
        baseline_hours: int = 24,
        spike_threshold: float = 3.0,
        min_complaints: int = 10,
    ):
        self.window_minutes = window_minutes
        self.baseline_hours = baseline_hours
        self.spike_threshold = spike_threshold
        self.min_complaints = min_complaints
        self._buffer: deque = deque()
        self._alerts: list = []
        self._lock = threading.Lock()
        self._handlers: list = []

    def on_alert(self, handler):
        self._handlers.append(handler)
        return handler

    def ingest(self, complaint: dict):
        complaint["ingested_at"] = datetime.now()
        with self._lock:
            self._buffer.append(complaint)
            cutoff = datetime.now() - timedelta(hours=self.baseline_hours + 1)
            while self._buffer and self._buffer[0]["ingested_at"] < cutoff:
                self._buffer.popleft()

    def _check_spikes(self) -> list[SentimentSpike]:
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=self.window_minutes)
        baseline_cutoff = now - timedelta(hours=self.baseline_hours)

        with self._lock:
            recent = [c for c in self._buffer if c["ingested_at"] >= recent_cutoff]
            baseline = [c for c in self._buffer if baseline_cutoff <= c["ingested_at"] < recent_cutoff]

        if len(recent) < self.min_complaints:
            return []

        def negative_rate_by_topic(complaints: list) -> dict:
            topic_counts = {}
            topic_negative = {}
            for c in complaints:
                topic = c.get("department", "general")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                if c.get("sentiment") == "NEGATIVE":
                    topic_negative[topic] = topic_negative.get(topic, 0) + 1
            return {
                t: topic_negative.get(t, 0) / topic_counts[t]
                for t in topic_counts
            }

        recent_rates = negative_rate_by_topic(recent)
        baseline_rates = negative_rate_by_topic(baseline) if baseline else {}

        spikes = []
        for topic, rate in recent_rates.items():
            baseline_rate = baseline_rates.get(topic, 0.2)
            baseline_rate = max(baseline_rate, 0.05)
            ratio = rate / baseline_rate
            if ratio >= self.spike_threshold:
                topic_count = sum(1 for c in recent if c.get("department") == topic)
                spikes.append(SentimentSpike(
                    topic=topic,
                    current_rate=rate,
                    baseline_rate=baseline_rate,
                    spike_ratio=ratio,
                    window_minutes=self.window_minutes,
                    complaint_count=topic_count,
                ))
        return spikes

    def check_and_alert(self):
        spikes = self._check_spikes()
        for spike in spikes:
            alert = spike.to_alert()
            self._alerts.append(alert)
            logger.warning(f"SPIKE DETECTED: {alert['message']}")
            for handler in self._handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
        return spikes

    def get_recent_alerts(self, hours: int = 24) -> list:
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self._alerts if datetime.fromisoformat(a["detected_at"]) >= cutoff]


def pr_team_alert_handler(alert: dict):
    print(f"\n🚨 PR TEAM ALERT 🚨")
    print(f"  {alert['message']}")
    print(f"  Action: Notify communications team immediately")


if __name__ == "__main__":
    monitor = RealTimeComplaintMonitor(window_minutes=60, spike_threshold=3.0)
    monitor.on_alert(pr_team_alert_handler)

    for i in range(50):
        monitor.ingest({
            "complaint_id": f"C{i:04d}",
            "department": "transfers",
            "sentiment": "NEGATIVE" if i > 20 else np.random.choice(["POSITIVE", "NEUTRAL", "NEGATIVE"]),
            "complaint_text": f"Sample complaint {i}",
        })

    spikes = monitor.check_and_alert()
    print(f"\nSpikes detected: {len(spikes)}")
