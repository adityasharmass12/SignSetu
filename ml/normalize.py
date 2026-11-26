import math
import urllib.request
from pathlib import Path

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
HAND_FEATURE_SIZE = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
TOTAL_FEATURE_SIZE = HAND_FEATURE_SIZE * 2

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def ensure_hand_landmarker_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def _write_normalized_hand(hand, out: np.ndarray, offset: int) -> None:
    if len(hand) < LANDMARKS_PER_HAND:
        return

    wrist = hand[0]
    max_dist = 0.0

    for i in range(LANDMARKS_PER_HAND):
        dx = hand[i].x - wrist.x
        dy = hand[i].y - wrist.y
        dz = hand[i].z - wrist.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > max_dist:
            max_dist = dist

    scale = max_dist if max_dist > 1e-6 else 1.0

    for i in range(LANDMARKS_PER_HAND):
        base = offset + i * COORDS_PER_LANDMARK
        out[base] = (hand[i].x - wrist.x) / scale
        out[base + 1] = (hand[i].y - wrist.y) / scale
        out[base + 2] = (hand[i].z - wrist.z) / scale


def normalize_landmarks(hand_landmarks, handedness) -> np.ndarray:
    features = np.zeros(TOTAL_FEATURE_SIZE, dtype=np.float32)
    filled = [False, False]
    count = min(len(hand_landmarks), 2)

    for i in range(count):
        label = None
        if i < len(handedness) and len(handedness[i]) > 0:
            label = handedness[i][0].category_name

        slot = 1 if label == "Right" else 0
        if filled[slot]:
            slot = 0 if slot == 1 else 1

        filled[slot] = True
        _write_normalized_hand(hand_landmarks[i], features, slot * HAND_FEATURE_SIZE)

    return features
