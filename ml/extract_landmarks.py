import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from tqdm import tqdm

from normalize import TOTAL_FEATURE_SIZE, ensure_hand_landmarker_model, normalize_landmarks

ML_DIR = Path(__file__).resolve().parent
CANDIDATE_DATASETS = [
    ML_DIR / "data" / "raw" / "ISL_Dataset",
    ML_DIR / "ISL_Dataset",
]
DEFAULT_OUTPUT_CSV = ML_DIR / "data" / "landmarks.csv"
DEFAULT_REPORT_CSV = ML_DIR / "reports" / "detection_report.csv"

FEATURE_COLS = [f"f{i}" for i in range(TOTAL_FEATURE_SIZE)]
CSV_HEADER = ["label", "source_file", "n_hands", *FEATURE_COLS]


def resolve_dataset_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        return path

    for candidate in CANDIDATE_DATASETS:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Dataset directory not found in ml/data/raw/ISL_Dataset or ml/ISL_Dataset"
    )


def collect_images(dataset_dir: Path) -> list[tuple[str, Path, str]]:
    items: list[tuple[str, Path, str]] = []
    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = f"{label}/{img_path.name}"
                items.append((label, img_path, rel))
    return items


def load_processed_files(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=["source_file"])
    return set(df["source_file"].astype(str))


def generate_report(landmarks_csv: Path, report_csv: Path) -> None:
    df = pd.read_csv(landmarks_csv)
    rows = []

    for label, group in df.groupby("label", sort=True):
        total = len(group)
        h0 = int((group["n_hands"] == 0).sum())
        h1 = int((group["n_hands"] == 1).sum())
        h2 = int((group["n_hands"] == 2).sum())
        detected = total - h0
        rate = (detected / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "label": label,
                "total_images": total,
                "detected_images": detected,
                "detection_rate": round(rate, 2),
                "hands_0": h0,
                "hands_1": h1,
                "hands_2": h2,
                "flagged_below_60": rate < 60.0,
            }
        )

    total_all = len(df)
    h0_all = int((df["n_hands"] == 0).sum())
    h1_all = int((df["n_hands"] == 1).sum())
    h2_all = int((df["n_hands"] == 2).sum())
    detected_all = total_all - h0_all
    rate_all = (detected_all / total_all * 100.0) if total_all > 0 else 0.0

    rows.append(
        {
            "label": "OVERALL",
            "total_images": total_all,
            "detected_images": detected_all,
            "detection_rate": round(rate_all, 2),
            "hands_0": h0_all,
            "hands_1": h1_all,
            "hands_2": h2_all,
            "flagged_below_60": rate_all < 60.0,
        }
    )

    report_df = pd.DataFrame(rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print("\nDetection Report:")
    print(report_df.to_string(index=False))

    flagged = report_df[
        (report_df["label"] != "OVERALL") & (report_df["flagged_below_60"])
    ]
    if not flagged.empty:
        print("\nFlagged classes (< 60% detection):")
        for _, r in flagged.iterrows():
            print(f"  {r['label']}: {r['detection_rate']}% ({r['detected_images']}/{r['total_images']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from ISL dataset.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to ISL_Dataset directory")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_CSV))
    args = parser.parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset)
    output_csv = Path(args.output)
    report_csv = Path(args.report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    images = collect_images(dataset_dir)
    processed = load_processed_files(output_csv)
    pending = [item for item in images if item[2] not in processed]

    print(f"Dataset: {dataset_dir}")
    print(f"Total images: {len(images)} | Already processed: {len(processed)} | Remaining: {len(pending)}")

    if pending:
        model_path = ensure_hand_landmarker_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
        )

        write_header = not output_csv.exists() or output_csv.stat().st_size == 0
        with (
            vision.HandLandmarker.create_from_options(options) as landmarker,
            output_csv.open("a", newline="", encoding="utf-8") as f,
        ):
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_HEADER)

            for label, img_path, rel_path in tqdm(pending, desc="Extracting landmarks"):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                upscaled = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=upscaled)
                result = landmarker.detect(mp_image)

                n_hands = min(len(result.hand_landmarks), 2)
                features = normalize_landmarks(result.hand_landmarks, result.handedness)
                writer.writerow([label, rel_path, n_hands, *(f"{v:.6f}" for v in features)])
                f.flush()

    generate_report(output_csv, report_csv)


if __name__ == "__main__":
    main()
