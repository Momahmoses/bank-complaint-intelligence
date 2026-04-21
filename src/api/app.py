"""
FastAPI application: complaint intake, analysis, routing, and real-time monitoring.
"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)
app = FastAPI(title="Bank Complaint Intelligence System", version="2.0.0")

complaint_monitor = None
search_index = None
sentiment_analyzer = None
classifier = None


@app.on_event("startup")
async def startup():
    global complaint_monitor, search_index, sentiment_analyzer, classifier

    from src.monitoring.alerts import RealTimeComplaintMonitor, pr_team_alert_handler
    from src.nlp.sentiment import SentimentAnalyzer
    from src.nlp.classifier import ComplaintClassifier
    from src.rag.search import ComplaintSearchIndex

    complaint_monitor = RealTimeComplaintMonitor()
    complaint_monitor.on_alert(pr_team_alert_handler)
    sentiment_analyzer = SentimentAnalyzer()
    classifier = ComplaintClassifier()
    search_index = ComplaintSearchIndex()

    index_path = "models/complaint_index"
    if os.path.exists(index_path):
        search_index = ComplaintSearchIndex.load(index_path)
        logger.info("Search index loaded")

    logger.info("All services initialized")


class ComplaintRequest(BaseModel):
    complaint_text: str
    channel: str = "app"
    customer_id: str = None
    language: str = "english"


class ComplaintResponse(BaseModel):
    complaint_id: str
    department: str
    priority: str
    sentiment: str
    is_critical: bool
    routing_confidence: float
    entities: dict
    similar_cases: list
    estimated_resolution_hours: int
    rag_context: str = None


@app.post("/complaint", response_model=ComplaintResponse)
async def submit_complaint(request: ComplaintRequest, background_tasks: BackgroundTasks):
    from src.nlp.classifier import full_complaint_analysis
    from src.rag.search import build_rag_context

    complaint_id = f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(request.complaint_text) % 9999:04d}"

    analysis = full_complaint_analysis(
        request.complaint_text,
        sentiment_analyzer=sentiment_analyzer,
        classifier=classifier,
    )

    similar_cases = search_index.search(request.complaint_text, top_k=3) if search_index.complaints else []
    rag_context = build_rag_context(request.complaint_text, search_index) if search_index.complaints else None

    complaint_record = {
        "complaint_id": complaint_id,
        "complaint_text": request.complaint_text,
        "department": analysis["department"],
        "sentiment": analysis["sentiment"],
        "channel": request.channel,
        "timestamp": datetime.now().isoformat(),
        **analysis,
    }

    if complaint_monitor:
        background_tasks.add_task(complaint_monitor.ingest, complaint_record)
        background_tasks.add_task(complaint_monitor.check_and_alert)

    priority = analysis["priority"]
    resolution_hours = {"HIGH": 2, "MEDIUM": 12, "LOW": 48}.get(priority, 24)

    return ComplaintResponse(
        complaint_id=complaint_id,
        department=analysis["department"],
        priority=priority,
        sentiment=analysis["sentiment"],
        is_critical=analysis["is_critical"],
        routing_confidence=analysis.get("routing_confidence", 0.5),
        entities=analysis["entities"],
        similar_cases=[{
            "text": c.get("complaint_text", "")[:100],
            "resolution": c.get("resolution", "N/A"),
            "similarity": c.get("similarity_score"),
        } for c in similar_cases],
        estimated_resolution_hours=resolution_hours,
        rag_context=rag_context,
    )


@app.get("/alerts")
async def get_alerts():
    if complaint_monitor:
        return {"alerts": complaint_monitor.get_recent_alerts(hours=24)}
    return {"alerts": []}


@app.get("/stats")
async def get_stats():
    if complaint_monitor:
        buffer = list(complaint_monitor._buffer)
        sentiment_counts = {}
        for c in buffer:
            s = c.get("sentiment", "UNKNOWN")
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        return {"total_buffered": len(buffer), "sentiment_distribution": sentiment_counts}
    return {"status": "monitor not initialized"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "services": {
            "sentiment": sentiment_analyzer is not None,
            "classifier": classifier is not None,
            "search_index": search_index is not None and bool(search_index.complaints),
            "monitor": complaint_monitor is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
