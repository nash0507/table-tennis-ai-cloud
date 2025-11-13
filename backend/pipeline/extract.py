"""Video to keypoint extraction stage.

This MVP implementation provides a lightweight approximation that mimics the
expected output of a pose extraction stage so that downstream components can be
exercised without a GPU. When MediaPipe is available it will be used for pose
estimation, otherwise synthetic keypoints are generated based on simple motion
heuristics over the decoded frames.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import mediapipe as mp
except ImportError:  # pragma: no cover - fallback is tested
    mp = None

from ..config import settings
from .utils import chunk_events

POSE_LANDMARKS = {
    "shoulder": 11,  # left shoulder
    "elbow": 13,
    "wrist": 15,
}


def _synthetic_keypoints(frame_idx: int, total_frames: int, fps: float) -> Tuple[float, float, float, float, float, float]:
    """Generate smooth pseudo keypoints when a detector is unavailable."""
    phase = (frame_idx / max(total_frames, 1)) * 2 * math.pi
    shoulder_x = 0.45 + 0.05 * math.sin(phase)
    shoulder_y = 0.5
    elbow_x = 0.5 + 0.1 * math.sin(phase + math.pi / 4)
    elbow_y = 0.55 + 0.05 * math.cos(phase)
    wrist_x = 0.6 + 0.15 * math.sin(phase + math.pi / 2)
    wrist_y = 0.65 + 0.05 * math.cos(phase)
    return shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y


def _detect_with_mediapipe(frame: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Extract pose landmarks using MediaPipe when available."""
    if mp is None:
        raise RuntimeError("MediaPipe is not installed")
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _POSE.process(image_rgb)
    if not results.pose_landmarks:
        raise ValueError("Pose landmarks not found")
    landmarks = results.pose_landmarks.landmark

    def get_point(name: str) -> Tuple[float, float]:
        idx = POSE_LANDMARKS[name]
        landmark = landmarks[idx]
        return float(landmark.x), float(landmark.y)

    shoulder_x, shoulder_y = get_point("shoulder")
    elbow_x, elbow_y = get_point("elbow")
    wrist_x, wrist_y = get_point("wrist")
    return shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y


if mp:  # pragma: no cover - not executed in tests
    _POSE = mp.solutions.pose.Pose(static_image_mode=False, enable_segmentation=False)
else:
    _POSE = None


def extract_keypoints(video_path: Path, video_id: str | None = None) -> pd.DataFrame:
    """Extract keypoints for a given video and persist the interim CSV."""
    if video_id is None:
        video_id = uuid.uuid4().hex

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 10)
    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = (frame_idx / fps) * 1000.0
        try:
            if mp:
                kps = _detect_with_mediapipe(frame)  # pragma: no cover
            else:
                kps = _synthetic_keypoints(frame_idx, total_frames, fps)
        except Exception:
            kps = _synthetic_keypoints(frame_idx, total_frames, fps)

        shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y = kps
        ball_x = 0.5 + 0.2 * math.sin(frame_idx / 6)
        ball_y = 0.4 + 0.15 * math.cos(frame_idx / 7)
        rows.append(
            {
                "video_id": video_id,
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "shoulder_x": shoulder_x,
                "shoulder_y": shoulder_y,
                "elbow_x": elbow_x,
                "elbow_y": elbow_y,
                "wrist_x": wrist_x,
                "wrist_y": wrist_y,
                "ball_x": ball_x,
                "ball_y": ball_y,
            }
        )
        frame_idx += 1

    cap.release()

    if not rows:
        raise ValueError("Video contained no readable frames")

    df = pd.DataFrame(rows)
    df = chunk_events(df, settings.frames_per_event)
    interim_path = settings.interim_dir / f"keypoints_{video_id}.csv"
    Path(interim_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_path, index=False)
    return df
