"""FastAPI application orchestrating the analysis pipeline."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .config import settings
from .pipeline import classify, extract, features, metrics
from .schemas import AnalyzeResponse, Report, StrokePrediction, VideoUploadResponse

app = FastAPI(title="Table Tennis Weakness Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _video_path(video_id: str) -> Path:
    return settings.raw_dir / f"{video_id}.mp4"


@app.post("/api/videos", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)) -> VideoUploadResponse:
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 videos are supported")
    video_id = uuid.uuid4().hex
    db.ensure_directories()
    destination = _video_path(video_id)
    with destination.open("wb") as f:
        content = await file.read()
        f.write(content)
    return VideoUploadResponse(video_id=video_id)


@app.post("/api/videos/{video_id}/analyze", response_model=AnalyzeResponse)
def analyze_video(video_id: str) -> AnalyzeResponse:
    video_file = _video_path(video_id)
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    keypoints = extract.extract_keypoints(video_file, video_id=video_id)
    feature_df = features.keypoints_to_features(keypoints)
    labels_df = db.load_labels()
    feature_df = features.merge_labels(feature_df, labels_df)
    db.save_features(video_id, feature_df)

    classifier = classify.train_or_load(feature_df)
    predictions = classifier.predict(feature_df)
    aggregates = metrics.aggregate_metrics(feature_df, predictions)

    report_dict = {
        "video_id": video_id,
        "summary": aggregates["summary"],
        "stroke_predictions": [
            StrokePrediction(
                event_id=int(row.event_id),
                stroke_hand=row.stroke_hand_pred,
                stroke_good=int(row.stroke_good_pred),
                confidence=float(row.confidence),
            ).dict()
            for row in aggregates["merged"].itertuples()
        ],
        "aggregates": {
            "player_a": aggregates["player_a"],
            "player_b": aggregates["player_b"],
        },
        "radar": aggregates["radar"],
        "heatmap": aggregates["heatmap"],
        "feature_importances": [
            {"feature": name, "importance": value}
            for name, value in classifier.feature_importances().items()
        ],
        "recommendations": aggregates["recommendations"],
        "raw_events_path": str(
            settings.processed_dir / f"{settings.features_prefix}{video_id}{settings.features_suffix}"
        ),
    }
    db.save_report(video_id, report_dict)
    return AnalyzeResponse(
        video_id=video_id,
        events=len(feature_df),
        report_path=str(settings.reports_dir / f"{video_id}{settings.report_suffix}"),
    )


@app.get("/api/videos/{video_id}/report", response_model=Report)
def get_report(video_id: str) -> Report:
    report_data = db.load_report(video_id)
    if report_data is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return Report(**report_data)
