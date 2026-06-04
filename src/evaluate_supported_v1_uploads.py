from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _reexec_with_project_venv_if_needed() -> None:
    """Tự chuyển sang `.venv/bin/python` trước khi import TensorFlow.

    Nếu chạy bằng Anaconda/base Python, TensorFlow trên macOS có thể bị
    segmentation fault ngay khi import. Vì vậy script tự dùng interpreter trong
    `.venv` của project, giống pipeline train Stage B.
    """

    if os.environ.get("FARMLINK_DISABLE_VENV_REEXEC") == "1":
        return

    project_root = Path(__file__).resolve().parents[1]
    venv_root = project_root / ".venv"
    venv_python = venv_root / "bin" / "python"
    if not venv_python.exists():
        return

    if Path(sys.prefix).resolve() == venv_root.resolve():
        return

    if os.environ.get("FARMLINK_UPLOAD_EVAL_REEXECED") == "1":
        return

    os.environ["FARMLINK_UPLOAD_EVAL_REEXECED"] = "1"
    print(
        f"Upload eval: dang chuyen interpreter tu {sys.executable} sang {venv_python}",
        file=sys.stderr,
        flush=True,
    )
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv_if_needed()

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None
    sns = None

try:
    from src.utils import open_image_safely, resize_with_padding, validate_image_quality
except ImportError:  # pragma: no cover
    from utils import open_image_safely, resize_with_padding, validate_image_quality


SUPPORTED_EXPERIMENT_PREFIX = "stage_b_supported_v1_resnet50_"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
UNSUPPORTED_CLASSES_V1 = frozenset(
    {
        "bell_pepper",
        "coffee",
        "lime",
        "longan",
        "mango",
        "gourd",
        "canistel",
        "burmese_grape",
    }
)
OUT_OF_SCOPE_LABELS = frozenset({"other", "hard", "unknown", "manual_review"})


@dataclass(frozen=True)
class UploadEvalConfig:
    """Config đánh giá ảnh upload có nhãn thật."""

    uploads_root: Path = Path("test_uploads_labeled")
    experiment_dir: Path | None = None
    output_dir: Path | None = None
    top_k: int = 3
    strict_quality_check: bool = False
    confidence_threshold: float = 0.0
    valid_extensions: tuple[str, ...] = VALID_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI."""

    parser = argparse.ArgumentParser(
        description="Evaluate latest Stage B supported v1 model on labeled upload images."
    )
    parser.add_argument("--uploads-root", type=Path, default=Path("test_uploads_labeled"))
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--strict-quality-check", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UploadEvalConfig:
    """Ghép CLI args thành config."""

    return UploadEvalConfig(
        uploads_root=args.uploads_root,
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        confidence_threshold=args.confidence_threshold,
        strict_quality_check=args.strict_quality_check,
    )


def find_latest_supported_v1_experiment(root: Path = Path("experiments")) -> Path:
    """Tìm experiment Stage B supported v1 mới nhất theo tên timestamp."""

    candidates = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(SUPPORTED_EXPERIMENT_PREFIX)
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Khong tim thay experiment {SUPPORTED_EXPERIMENT_PREFIX}* trong {root}"
        )
    return candidates[-1]


def load_label_manifest(experiment_dir: Path) -> dict[str, Any]:
    """Đọc labels.json của experiment."""

    labels_path = experiment_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"Khong tim thay labels.json: {labels_path}")
    manifest = json.loads(labels_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"labels.json khong co class_names hop le: {labels_path}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"labels.json class_names bi trung: {labels_path}")
    return manifest


def normalize_label_token(value: str) -> str:
    """Chuẩn hóa tên folder upload thành snake_case không dấu."""

    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def map_upload_label(raw_label: str, class_names: list[str]) -> tuple[str | None, str, str]:
    """Map folder label upload sang label của model nếu có thể.

    Trả về:
    - expected_label: label dùng để tính accuracy, None nếu không thuộc 32 class.
    - normalized_label: label đã chuẩn hóa để ghi report.
    - status: supported / unsupported_v1 / out_of_scope / unknown_label
    """

    normalized_label = normalize_label_token(raw_label)
    alias_map = {
        "jackfruit": "jackfruit_cempedak",
        "cempedak": "jackfruit_cempedak",
        "suger_apple": "sugar_apple",
        "sugarapple": "sugar_apple",
        "sugar_apple": "sugar_apple",
        "caimito": "caimito",
        "pomegranate": "pomegranate",
    }
    mapped_label = alias_map.get(normalized_label, normalized_label)
    class_name_set = set(class_names)

    if mapped_label in class_name_set:
        return mapped_label, normalized_label, "supported"
    if mapped_label in UNSUPPORTED_CLASSES_V1:
        return None, normalized_label, "unsupported_v1"
    if mapped_label in OUT_OF_SCOPE_LABELS:
        return None, normalized_label, "out_of_scope"
    return None, normalized_label, "unknown_label"


def list_upload_images(uploads_root: Path, valid_extensions: tuple[str, ...]) -> list[tuple[Path, str]]:
    """Liệt kê ảnh upload theo cấu trúc `<label_folder>/<image>`."""

    if not uploads_root.exists():
        raise FileNotFoundError(f"Khong tim thay uploads_root: {uploads_root}")
    if not uploads_root.is_dir():
        raise NotADirectoryError(f"uploads_root khong phai thu muc: {uploads_root}")

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    items: list[tuple[Path, str]] = []
    for label_dir in sorted([path for path in uploads_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower()):
        for image_path in sorted(label_dir.rglob("*"), key=lambda p: str(p).lower()):
            if (
                image_path.is_file()
                and not image_path.name.startswith(".")
                and image_path.suffix.lower() in valid_extension_set
            ):
                items.append((image_path, label_dir.name))
    if not items:
        raise ValueError(f"Khong tim thay anh upload hop le trong: {uploads_root}")
    return items


def prepare_image_tensor(
    image_path: Path,
    manifest: dict[str, Any],
    strict_quality_check: bool,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess ảnh upload giống Stage B train.

    Flow:
    - mở ảnh an toàn, sửa EXIF, RGB;
    - resize/pad đen về image_size;
    - áp dụng `tf.keras.applications.resnet.preprocess_input`;
    - thêm batch dimension.
    """

    warnings: list[str] = []
    image = open_image_safely(image_path)
    quality_ok, quality_reason = validate_image_quality(
        image=image,
        min_image_side=100,
        max_image_side=5000,
        max_aspect_ratio=4.0,
    )
    if not quality_ok and quality_reason is not None:
        if strict_quality_check:
            raise ValueError(f"Anh upload khong dat chat luong: {quality_reason}")
        warnings.append(quality_reason)

    image_size = manifest.get("image_size", [320, 320])
    target_height = int(image_size[0])
    target_width = int(image_size[1])
    standardized_image = resize_with_padding(
        image=image,
        target_size=(target_width, target_height),
        background_color=(0, 0, 0),
    )
    image_array = np.asarray(standardized_image, dtype=np.float32)
    image_array = tf.keras.applications.resnet.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0), warnings


def top_k_predictions(probabilities: np.ndarray, class_names: list[str], top_k: int) -> list[dict[str, Any]]:
    """Format top-k prediction."""

    top_k = max(1, min(top_k, len(class_names)))
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "class_index": int(index),
            "label": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in indices
    ]


def predict_uploads(config: UploadEvalConfig) -> tuple[Path, dict[str, Any]]:
    """Chạy prediction toàn bộ upload folder và lưu report."""

    experiment_dir = config.experiment_dir or find_latest_supported_v1_experiment()
    manifest = load_label_manifest(experiment_dir)
    class_names = [str(item) for item in manifest["class_names"]]
    model_path = experiment_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model.keras: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir or experiment_dir / f"upload_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    model = tf.keras.models.load_model(str(model_path), compile=False)
    upload_items = list_upload_images(config.uploads_root, config.valid_extensions)

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    unreadable: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter()
    unsupported_prediction_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for image_path, raw_label in upload_items:
        raw_label_counts[raw_label.strip()] += 1
        expected_label, normalized_label, status = map_upload_label(raw_label, class_names)
        status_counts[status] += 1

        try:
            image_tensor, warnings = prepare_image_tensor(
                image_path=image_path,
                manifest=manifest,
                strict_quality_check=config.strict_quality_check,
            )
            probabilities = model.predict(image_tensor, verbose=0)[0]
            predictions = top_k_predictions(probabilities, class_names, config.top_k)
            best_prediction = predictions[0]
            predicted_label = best_prediction["label"]
            confidence = float(best_prediction["confidence"])
            correct = expected_label is not None and predicted_label == expected_label

            if expected_label is not None:
                y_true.append(class_names.index(expected_label))
                y_pred.append(class_names.index(predicted_label))
            else:
                unsupported_prediction_counter[normalized_label][predicted_label] += 1

            rows.append(
                {
                    "image_path": str(image_path),
                    "raw_label": raw_label,
                    "normalized_label": normalized_label,
                    "expected_label": expected_label or "",
                    "status": status,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "low_confidence": confidence < config.confidence_threshold,
                    "correct": bool(correct) if expected_label is not None else "",
                    "top_k": predictions,
                    "warnings": warnings,
                }
            )
        except Exception as error:  # noqa: BLE001
            unreadable.append(
                {
                    "image_path": str(image_path),
                    "raw_label": raw_label,
                    "normalized_label": normalized_label,
                    "status": status,
                    "error": str(error),
                }
            )

    supported_total = len(y_true)
    supported_correct = int(sum(int(a == b) for a, b in zip(y_true, y_pred)))
    supported_accuracy = float(supported_correct / supported_total) if supported_total else 0.0

    report_dict: dict[str, Any] | None = None
    confusion: list[list[int]] | None = None
    if supported_total > 0:
        labels = np.arange(len(class_names))
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
            digits=4,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        confusion = matrix.astype(int).tolist()
        save_confusion_matrix_plot(matrix, class_names, output_dir / "upload_confusion_matrix.png")
        (output_dir / "upload_classification_report.txt").write_text(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=class_names,
                zero_division=0,
                digits=4,
            ),
            encoding="utf-8",
        )

    save_predictions_csv(output_dir / "upload_predictions.csv", rows)
    save_predictions_jsonl(output_dir / "upload_predictions.jsonl", rows)

    unsupported_summary = {
        label: {
            "count": sum(counter.values()),
            "predicted_as": dict(counter.most_common()),
        }
        for label, counter in sorted(unsupported_prediction_counter.items())
    }
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir),
        "model_path": str(model_path),
        "labels_path": str(experiment_dir / "labels.json"),
        "uploads_root": str(config.uploads_root),
        "output_dir": str(output_dir),
        "num_model_classes": len(class_names),
        "class_names": class_names,
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "status_counts": dict(status_counts),
        "total_images_found": len(upload_items),
        "total_images_predicted": len(rows),
        "unreadable_count": len(unreadable),
        "unreadable": unreadable,
        "supported_eval": {
            "evaluated_images": supported_total,
            "correct": supported_correct,
            "incorrect": supported_total - supported_correct,
            "accuracy": supported_accuracy,
            "macro_f1": (
                float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
                if report_dict
                else 0.0
            ),
            "macro_recall": (
                float(report_dict.get("macro avg", {}).get("recall", 0.0))
                if report_dict
                else 0.0
            ),
        },
        "unsupported_or_out_of_scope": unsupported_summary,
        "classification_report": report_dict,
        "confusion_matrix": confusion,
        "note": (
            "Accuracy chi tinh tren anh co nhan nam trong 32 supported_classes_v1. "
            "Cac folder unsupported/out_of_scope nhu mango, longan, other, hard "
            "duoc thong ke rieng vi model Stage B production v1 khong co output cho chung."
        ),
    }
    (output_dir / "upload_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir, report


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Lưu confusion matrix PNG cho subset supported upload."""

    if plt is None or sns is None:
        return
    figure_width = max(14, len(class_names) * 0.45)
    figure_height = max(12, len(class_names) * 0.40)
    plt.figure(figsize=(figure_width, figure_height))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=False,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Upload Supported V1 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_predictions_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Lưu prediction từng ảnh dạng CSV để dễ đọc nhanh."""

    fieldnames = [
        "image_path",
        "raw_label",
        "normalized_label",
        "expected_label",
        "status",
        "predicted_label",
        "confidence",
        "low_confidence",
        "correct",
        "top_k",
        "warnings",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["top_k"] = json.dumps(row["top_k"], ensure_ascii=False)
            csv_row["warnings"] = json.dumps(row["warnings"], ensure_ascii=False)
            writer.writerow(csv_row)


def save_predictions_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Lưu prediction từng ảnh dạng JSONL để máy đọc lại dễ hơn."""

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Entry point: test model Stage B supported v1 trên ảnh upload có nhãn."""

    args = parse_args()
    config = build_config(args)
    output_dir, report = predict_uploads(config)
    supported = report["supported_eval"]
    print(f"Output: {output_dir}")
    print(
        "Supported accuracy: "
        f"{supported['accuracy']:.4f} "
        f"({supported['correct']}/{supported['evaluated_images']})"
    )
    print(f"Status counts: {report['status_counts']}")


if __name__ == "__main__":
    main()
