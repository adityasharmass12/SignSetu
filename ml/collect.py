import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from normalize import TOTAL_FEATURE_SIZE, ensure_hand_landmarker_model, normalize_landmarks

ML_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = ML_DIR / "data" / "own_samples.csv"
FEATURE_COLS = [f"f{i}" for i in range(TOTAL_FEATURE_SIZE)]
CSV_HEADER = ["label", "source_file", "n_hands", *FEATURE_COLS]
SAMPLE_INTERVAL_SEC = 0.1


def load_counts(csv_path: Path) -> Counter:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return Counter()
    df = pd.read_csv(csv_path, usecols=["label"])
    return Counter(df["label"].astype(str))


def draw_landmarks(frame, hand_landmarks) -> None:
    h, w, _ = frame.shape
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for hand in hand_landmarks:
        for conn in connections:
            a = hand[conn.start]
            b = hand[conn.end]
            cv2.line(
                frame,
                (int(a.x * w), int(a.y * h)),
                (int(b.x * w), int(b.y * h)),
                (210, 120, 255),
                2,
            )
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (180, 70, 255), -1)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    counts = load_counts(OUTPUT_CSV)

    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    current_label = "A"
    recording = False
    last_record_time = 0.0
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    with (
        vision.HandLandmarker.create_from_options(options) as landmarker,
        OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f,
    ):
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)

        start_ms = time.monotonic() * 1000.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000.0 - start_ms)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            n_hands = min(len(result.hand_landmarks), 2)
            draw_landmarks(frame, result.hand_landmarks)

            now = time.monotonic()
            if recording and n_hands > 0 and (now - last_record_time) >= SAMPLE_INTERVAL_SEC:
                features = normalize_landmarks(result.hand_landmarks, result.handedness)
                sample_id = f"webcam_{int(time.time() * 1000)}"
                writer.writerow([current_label, sample_id, n_hands, *(f"{v:.6f}" for v in features)])
                f.flush()
                counts[current_label] += 1
                last_record_time = now

            display = cv2.flip(frame, 1)
            status_text = "REC" if recording else "PAUSED"
            status_color = (0, 200, 120) if recording else (0, 180, 255)

            cv2.putText(
                display,
                f"Label: {current_label} | {status_text} | Hands: {n_hands}",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2,
            )
            cv2.putText(
                display,
                "Keys: A-Z/1-9 = set label | SPACE = rec/pause | ESC = quit",
                (16, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )

            y = 104
            for lbl in sorted(counts.keys()):
                if counts[lbl] > 0:
                    cv2.putText(
                        display,
                        f"{lbl}: {counts[lbl]}",
                        (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (230, 210, 255),
                        1,
                    )
                    y += 24
                    if y > display.shape[0] - 20:
                        break

            cv2.imshow("SignSetu - Collect Own Samples", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):
                recording = not recording
            elif ord("a") <= key <= ord("z"):
                current_label = chr(key).upper()
            elif ord("A") <= key <= ord("Z"):
                current_label = chr(key)
            elif ord("1") <= key <= ord("9"):
                current_label = chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
