"""Classification stage using scikit-learn decision trees."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..model_store import MODEL_STORE

FEATURE_COLUMNS = [
    "shoulder_angle",
    "elbow_angle",
    "wrist_angle",
    "body_rot",
    "approach_vec",
    "serve_flag",
    "landing_x",
    "landing_y",
    "contact_speed_proxy",
    "ball_speed_proxy",
]


class StrokeClassifier:
    """Wraps the hand and quality classifiers."""

    def __init__(self, hand_model: DecisionTreeClassifier, quality_model: DecisionTreeClassifier):
        self.hand_model = hand_model
        self.quality_model = quality_model

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[FEATURE_COLUMNS]
        hand_pred = self.hand_model.predict(X)
        quality_pred = self.quality_model.predict(X)
        proba = self.quality_model.predict_proba(X)
        confidences = proba[:, 1] if proba.shape[1] > 1 else np.ones(len(X))
        return pd.DataFrame(
            {
                "event_id": features["event_id"],
                "stroke_hand_pred": np.where(hand_pred == 1, "forehand", "backhand"),
                "stroke_good_pred": quality_pred,
                "confidence": confidences,
            }
        )

    def feature_importances(self) -> Dict[str, float]:
        importances = self.quality_model.feature_importances_
        return {name: float(val) for name, val in zip(FEATURE_COLUMNS, importances)}


def _prepare_training_data(features: pd.DataFrame):
    X = features[FEATURE_COLUMNS]
    y_hand = (features["stroke_hand"] == "forehand").astype(int)
    y_quality = features["stroke_good"].astype(int)
    return X, y_hand.values, y_quality.values


def _train_tree(X: pd.DataFrame, y: np.ndarray) -> DecisionTreeClassifier:
    if len(np.unique(y)) < 2:
        X = pd.concat([X, X.iloc[[0]]], ignore_index=True)
        y = np.concatenate([y, np.array([1 - y[0]])])
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    model.fit(X, y)
    return model


def train_classifier(features: pd.DataFrame) -> StrokeClassifier:
    X, y_hand, y_quality = _prepare_training_data(features)
    hand_model = _train_tree(X, y_hand)
    quality_model = _train_tree(X, y_quality)
    return StrokeClassifier(hand_model, quality_model)


def save_classifier(classifier: StrokeClassifier) -> None:
    MODEL_STORE.save("stroke_hand_tree.pkl", classifier.hand_model)
    MODEL_STORE.save("decision_tree.pkl", classifier.quality_model)


def load_classifier() -> StrokeClassifier | None:
    hand_model = MODEL_STORE.load("stroke_hand_tree.pkl")
    quality_model = MODEL_STORE.load("decision_tree.pkl")
    if isinstance(hand_model, DecisionTreeClassifier) and isinstance(
        quality_model, DecisionTreeClassifier
    ):
        return StrokeClassifier(hand_model, quality_model)
    return None


def train_with_history(features: pd.DataFrame | None) -> StrokeClassifier:
    if features is None or features.empty:
        existing = load_classifier()
        if existing is None:
            raise ValueError("No training data available to build classifier")
        return existing

    required = set(FEATURE_COLUMNS + ["stroke_hand", "stroke_good"])
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing columns for training: {sorted(missing)}")

    features = features.copy()
    features.sort_values(["video_id", "event_id"], inplace=True)
    features = features.drop_duplicates(subset=["video_id", "event_id"], keep="last")

    classifier = train_classifier(features)
    save_classifier(classifier)
    return classifier
