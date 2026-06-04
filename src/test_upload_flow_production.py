from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _reexec_with_project_venv_if_needed() -> None:
    """Chạy lại script bằng `.venv/bin/python` trước khi import TensorFlow.

    Lý do: trên macOS, nếu chạy nhầm bằng Python base/Anaconda thì TensorFlow
    có thể bị segmentation fault ngay lúc import. Script này chỉ test inference,
    nhưng vẫn cần cùng interpreter ổn định với pipeline train.
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

    if os.environ.get("FARMLINK_UPLOAD_PRODUCTION_REEXECED") == "1":
        return

    os.environ["FARMLINK_UPLOAD_PRODUCTION_REEXECED"] = "1"
    print(
        f"Upload production test: chuyen interpreter tu {sys.executable} sang {venv_python}",
        file=sys.stderr,
        flush=True,
    )
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv_if_needed()

import numpy as np
import tensorflow as tf

try:
    from src.utils import open_image_safely, resize_with_padding, validate_image_quality
except ImportError:  # pragma: no cover
    from utils import open_image_safely, resize_with_padding, validate_image_quality


STAGE_A_PREFIX = "stage_a_resnet50_"
STAGE_B_PREFIX = "stage_b_supported_v1_resnet50_"
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
class ProductionFlowConfig:
    """Cấu hình test flow upload production."""

    uploads_root: Path
    stage_a_experiment: Path | None
    stage_b_experiment: Path | None
    stage_a_fruit_threshold: float
    stage_a_other_threshold: float
    stage_b_confidence_threshold: float
    stage_b_margin_threshold: float
    logs_dir: Path
    top_k: int
    valid_extensions: tuple[str, ...] = VALID_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI để test flow production."""

    parser = argparse.ArgumentParser(
        description="Test full upload production flow: Stage A fruit/other -> Stage B supported v1."
    )
    parser.add_argument("--uploads-root", type=Path, default=Path("test_uploads_labeled"))
    parser.add_argument("--stage-a-experiment", type=Path, default=None)
    parser.add_argument("--stage-b-experiment", type=Path, default=None)
    parser.add_argument("--stage-a-fruit-threshold", type=float, default=0.52)
    parser.add_argument("--stage-a-other-threshold", type=float, default=0.01)
    parser.add_argument("--stage-b-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--stage-b-margin-threshold", type=float, default=0.15)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ProductionFlowConfig:
    """Ghép CLI args thành config có kiểm tra threshold cơ bản."""

    if args.stage_a_other_threshold < 0:
        raise ValueError("--stage-a-other-threshold phai >= 0.")
    if args.stage_a_fruit_threshold <= args.stage_a_other_threshold:
        raise ValueError(
            "--stage-a-fruit-threshold phai lon hon --stage-a-other-threshold."
        )
    if not 0 <= args.stage_b_confidence_threshold <= 1:
        raise ValueError("--stage-b-confidence-threshold phai nam trong [0, 1].")
    if not 0 <= args.stage_b_margin_threshold <= 1:
        raise ValueError("--stage-b-margin-threshold phai nam trong [0, 1].")

    return ProductionFlowConfig(
        uploads_root=args.uploads_root,
        stage_a_experiment=args.stage_a_experiment,
        stage_b_experiment=args.stage_b_experiment,
        stage_a_fruit_threshold=float(args.stage_a_fruit_threshold),
        stage_a_other_threshold=float(args.stage_a_other_threshold),
        stage_b_confidence_threshold=float(args.stage_b_confidence_threshold),
        stage_b_margin_threshold=float(args.stage_b_margin_threshold),
        logs_dir=args.logs_dir,
        top_k=int(args.top_k),
    )


def setup_logger(log_path: Path) -> logging.Logger:
    """Tạo logger ghi chi tiết từng ảnh vào file log production test."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("upload_flow_production")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def find_latest_experiment(prefix: str, root: Path = Path("experiments")) -> Path:
    """Tìm experiment mới nhất theo prefix timestamp."""

    if not root.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc experiments: {root}")

    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"Khong tim thay experiment {prefix}* trong {root}")
    return candidates[-1]


def load_json(path: Path) -> dict[str, Any]:
    """Đọc JSON và raise lỗi rõ nếu thiếu file."""

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_names(stage_b_experiment: Path) -> list[str]:
    """Đọc class_names của Stage B supported v1."""

    manifest = load_json(stage_b_experiment / "labels.json")
    class_names = manifest.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"labels.json khong co class_names hop le: {stage_b_experiment}")

    class_names = [str(class_name) for class_name in class_names]
    if len(set(class_names)) != len(class_names):
        raise ValueError("Stage B class_names bi trung.")
    if "other" in class_names:
        raise ValueError("Stage B supported v1 khong duoc chua class other.")
    leaked_unsupported = sorted(set(class_names) & UNSUPPORTED_CLASSES_V1)
    if leaked_unsupported:
        raise ValueError(
            "Stage B supported v1 van chua unsupported class: "
            + ", ".join(leaked_unsupported)
        )
    return class_names


def normalize_label_token(value: str) -> str:
    """Chuẩn hóa tên folder upload về snake_case không dấu."""

    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def map_upload_label(raw_label: str, class_names: list[str]) -> tuple[str | None, str, str]:
    """Map folder label của ảnh upload sang nhóm đánh giá production.

    Return:
    - true_label: label thật nếu thuộc supported v1, ngược lại None;
    - normalized_label: tên folder sau chuẩn hóa;
    - group: supported / unsupported_v1 / out_of_scope / unknown_label.
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


def list_upload_images(
    uploads_root: Path,
    valid_extensions: tuple[str, ...],
) -> list[tuple[Path, str]]:
    """Liệt kê ảnh trong `test_uploads_labeled/<true_label>/<image>`."""

    if not uploads_root.exists():
        raise FileNotFoundError(f"Khong tim thay uploads_root: {uploads_root}")
    if not uploads_root.is_dir():
        raise NotADirectoryError(f"uploads_root khong phai thu muc: {uploads_root}")

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    items: list[tuple[Path, str]] = []
    for label_dir in sorted(
        [path for path in uploads_root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    ):
        for image_path in sorted(label_dir.rglob("*"), key=lambda path: str(path).lower()):
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
) -> tuple[np.ndarray, list[str]]:
    """Tiền xử lý ảnh upload giống ResNet50 inference.

    Flow:
    - mở ảnh an toàn, fix EXIF orientation, convert RGB;
    - kiểm tra chất lượng cơ bản;
    - resize/pad về image_size của experiment;
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


def predict_stage_a(model: tf.keras.Model, image_tensor: np.ndarray) -> float:
    """Predict Stage A và trả về xác suất fruit."""

    output = model.predict(image_tensor, verbose=0)
    return float(np.asarray(output).reshape(-1)[0])


def predict_stage_b(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Predict Stage B và trả về top1/top2/margin/top_k."""

    probabilities = np.asarray(model.predict(image_tensor, verbose=0))[0]
    top_k = max(2, min(top_k, len(class_names)))
    indices = np.argsort(probabilities)[::-1][:top_k]
    predictions = [
        {
            "class_index": int(index),
            "label": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in indices
    ]
    top1 = predictions[0]
    top2 = predictions[1]
    return {
        "top1_label": top1["label"],
        "top1_confidence": float(top1["confidence"]),
        "top2_label": top2["label"],
        "top2_confidence": float(top2["confidence"]),
        "margin": float(top1["confidence"] - top2["confidence"]),
        "top_k": predictions,
    }


def route_stage_a(
    fruit_probability: float,
    fruit_threshold: float,
    other_threshold: float,
) -> str:
    """Route Stage A theo 3 vùng: fruit / other / uncertain."""

    if fruit_probability >= fruit_threshold:
        return "stage_a_fruit"
    if fruit_probability <= other_threshold:
        return "stage_a_other"
    return "manual_review_stage_a_uncertain"


def route_stage_b(
    top1_confidence: float,
    margin: float,
    confidence_threshold: float,
    margin_threshold: float,
) -> str:
    """Route Stage B theo confidence và khoảng cách top1-top2."""

    if top1_confidence < confidence_threshold:
        return "manual_review_low_confidence"
    if margin < margin_threshold:
        return "manual_review_ambiguous"
    return "stage_b_supported_auto_accept"


def is_manual_review_route(final_route: str) -> bool:
    """Kiểm tra final route có phải manual review không."""

    return final_route.startswith("manual_review")


def evaluate_result(
    group: str,
    true_label: str | None,
    predicted_label: str,
    final_route: str,
) -> tuple[str, str, bool, bool]:
    """Đánh giá kết quả production theo nhóm nhãn thật.

    Return:
    - evaluation_result: nhãn đánh giá chi tiết;
    - reason: giải thích vì sao kết quả đúng/sai/review;
    - is_correct_flow: flow xử lý đúng hoặc chấp nhận được;
    - is_wrong_flow: flow xử lý sai rõ ràng.
    """

    if final_route == "error":
        return "error", "image_error_or_unreadable", False, False

    if group == "unsupported_v1":
        if final_route == "manual_review_unsupported":
            return (
                "correct_flow",
                "unsupported_class_routed_to_manual_review",
                True,
                False,
            )
        if final_route == "stage_b_supported_auto_accept":
            return (
                "wrong_flow",
                "unsupported_class_auto_accepted_as_supported",
                False,
                True,
            )
        return "manual_review", "unsupported_class_manual_review", True, False

    if group == "out_of_scope":
        if final_route == "stage_a_other":
            return "correct_flow", "out_of_scope_blocked_by_stage_a", True, False
        if is_manual_review_route(final_route):
            return "acceptable_manual_review", "out_of_scope_sent_to_manual_review", True, False
        if final_route == "stage_b_supported_auto_accept":
            return (
                "wrong_flow",
                "out_of_scope_auto_accepted_as_supported",
                False,
                True,
            )
        return "unknown_evaluation", "unhandled_out_of_scope_case", False, False

    if group == "supported":
        if is_manual_review_route(final_route):
            return "manual_review", "supported_class_sent_to_manual_review", False, False
        if final_route == "stage_a_other":
            return (
                "wrong_flow",
                "supported_class_blocked_as_other_by_stage_a",
                False,
                True,
            )
        if final_route == "stage_b_supported_auto_accept":
            if predicted_label == true_label:
                return "correct_auto", "supported_class_auto_accepted_correctly", True, False
            return "wrong_auto", "supported_class_auto_accepted_wrong_class", False, True
        return "unknown_evaluation", "unhandled_supported_case", False, False

    if group == "unknown_label":
        if is_manual_review_route(final_route):
            return "manual_review", "unknown_label_sent_to_manual_review", False, False
        if final_route == "stage_b_supported_auto_accept":
            return "unknown_evaluation", "unknown_label_auto_accepted", False, False
        return "unknown_evaluation", "unknown_label_case", False, False

    return "unknown_evaluation", "unknown_group_case", False, False


def build_empty_row(
    image_path: Path,
    raw_label: str,
    normalized_label: str,
    true_label: str | None,
    group: str,
) -> dict[str, Any]:
    """Tạo row mặc định cho một ảnh trước khi chạy inference."""

    return {
        "image_path": str(image_path),
        "raw_label": raw_label,
        "normalized_label": normalized_label,
        "true_label": true_label or "",
        "group": group,
        "stage_a_fruit_probability": "",
        "stage_a_route": "",
        "stage_b_top1_label": "",
        "stage_b_top1_confidence": "",
        "stage_b_top2_label": "",
        "stage_b_top2_confidence": "",
        "stage_b_margin": "",
        "final_route": "",
        "evaluation_result": "",
        "reason": "",
        "is_correct_flow": "",
        "is_wrong_flow": "",
        "stage_b_ran": False,
        "top_k": [],
        "warnings": [],
        "error": "",
    }


def run_single_image(
    image_path: Path,
    raw_label: str,
    class_names: list[str],
    stage_a_model: tf.keras.Model,
    stage_b_model: tf.keras.Model,
    stage_a_manifest: dict[str, Any],
    stage_b_manifest: dict[str, Any],
    config: ProductionFlowConfig,
) -> dict[str, Any]:
    """Chạy đúng flow production cho một ảnh upload đã có nhãn thật."""

    true_label, normalized_label, group = map_upload_label(raw_label, class_names)
    row = build_empty_row(image_path, raw_label, normalized_label, true_label, group)

    try:
        stage_a_tensor, warnings = prepare_image_tensor(image_path, stage_a_manifest)
        row["warnings"] = warnings
        fruit_probability = predict_stage_a(stage_a_model, stage_a_tensor)
        stage_a_route = route_stage_a(
            fruit_probability=fruit_probability,
            fruit_threshold=config.stage_a_fruit_threshold,
            other_threshold=config.stage_a_other_threshold,
        )
        row["stage_a_fruit_probability"] = fruit_probability
        row["stage_a_route"] = stage_a_route

        # Vì đây là test có nhãn thật, unsupported v1 được chặn theo label để
        # mô phỏng chính sách production v1: không ép Stage B nhận class chưa hỗ trợ.
        if group == "unsupported_v1":
            final_route = "manual_review_unsupported"
            predicted_label = ""
        elif group == "unknown_label":
            final_route = "manual_review_unknown_label"
            predicted_label = ""
        elif stage_a_route == "stage_a_other":
            final_route = "stage_a_other"
            predicted_label = ""
        elif stage_a_route == "manual_review_stage_a_uncertain":
            final_route = "manual_review_stage_a_uncertain"
            predicted_label = ""
        else:
            stage_b_size = stage_b_manifest.get("image_size", [320, 320])
            stage_a_size = stage_a_manifest.get("image_size", [320, 320])
            if stage_b_size == stage_a_size:
                stage_b_tensor = stage_a_tensor
            else:
                stage_b_tensor, stage_b_warnings = prepare_image_tensor(
                    image_path,
                    stage_b_manifest,
                )
                row["warnings"] = warnings + stage_b_warnings

            stage_b_result = predict_stage_b(
                model=stage_b_model,
                image_tensor=stage_b_tensor,
                class_names=class_names,
                top_k=config.top_k,
            )
            row["stage_b_ran"] = True
            row["stage_b_top1_label"] = stage_b_result["top1_label"]
            row["stage_b_top1_confidence"] = stage_b_result["top1_confidence"]
            row["stage_b_top2_label"] = stage_b_result["top2_label"]
            row["stage_b_top2_confidence"] = stage_b_result["top2_confidence"]
            row["stage_b_margin"] = stage_b_result["margin"]
            row["top_k"] = stage_b_result["top_k"]

            predicted_label = str(stage_b_result["top1_label"])
            final_route = route_stage_b(
                top1_confidence=float(stage_b_result["top1_confidence"]),
                margin=float(stage_b_result["margin"]),
                confidence_threshold=config.stage_b_confidence_threshold,
                margin_threshold=config.stage_b_margin_threshold,
            )

        evaluation_result, reason, is_correct_flow, is_wrong_flow = evaluate_result(
            group=group,
            true_label=true_label,
            predicted_label=predicted_label,
            final_route=final_route,
        )
        row["final_route"] = final_route
        row["evaluation_result"] = evaluation_result
        row["reason"] = reason
        row["is_correct_flow"] = is_correct_flow
        row["is_wrong_flow"] = is_wrong_flow
    except Exception as error:  # noqa: BLE001
        row["final_route"] = "error"
        row["evaluation_result"] = "error"
        row["reason"] = "image_error_or_unreadable"
        row["error"] = str(error)
        row["is_correct_flow"] = False
        row["is_wrong_flow"] = False

    return row


def csv_safe_value(value: Any) -> Any:
    """Chuyển list/dict sang JSON string để ghi CSV."""

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Ghi CSV với field cố định."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_safe_value(row.get(field, "")) for field in fieldnames})


def build_per_class_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tổng hợp kết quả theo từng class/folder thật."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label_key = row["true_label"] or row["normalized_label"]
        buckets[label_key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for label_key in sorted(buckets):
        label_rows = buckets[label_key]
        total = len(label_rows)
        group = label_rows[0]["group"]
        auto_rows = [row for row in label_rows if row["final_route"] == "stage_b_supported_auto_accept"]
        auto_correct = [
            row for row in auto_rows if row["evaluation_result"] == "correct_auto"
        ]
        auto_wrong = [
            row
            for row in auto_rows
            if row["evaluation_result"] in {"wrong_auto", "wrong_flow"}
        ]
        manual_rows = [row for row in label_rows if is_manual_review_route(row["final_route"])]
        stage_a_other_rows = [row for row in label_rows if row["final_route"] == "stage_a_other"]
        error_rows = [row for row in label_rows if row["final_route"] == "error"]
        summary_rows.append(
            {
                "label": label_key,
                "group": group,
                "total": total,
                "auto_accepted": len(auto_rows),
                "auto_correct": len(auto_correct),
                "auto_wrong": len(auto_wrong),
                "manual_review": len(manual_rows),
                "stage_a_other": len(stage_a_other_rows),
                "error": len(error_rows),
                "auto_accuracy": (
                    len(auto_correct) / len(auto_rows) if auto_rows else ""
                ),
            }
        )
    return summary_rows


def build_error_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lấy các case sai rõ ràng hoặc ảnh lỗi để kiểm tra thủ công."""

    return [
        row
        for row in rows
        if row["evaluation_result"] in {"wrong_auto", "wrong_flow", "error"}
    ]


def build_confused_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tổng hợp các cặp class bị nhầm khi Stage B auto accept."""

    counter: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        if row["final_route"] != "stage_b_supported_auto_accept":
            continue
        if row["group"] != "supported":
            continue
        true_label = row["true_label"]
        predicted_label = row["stage_b_top1_label"]
        if not true_label or not predicted_label or true_label == predicted_label:
            continue
        key = (true_label, predicted_label)
        counter[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(row["image_path"])

    confused_rows: list[dict[str, Any]] = []
    for (true_label, predicted_label), count in counter.most_common():
        confused_rows.append(
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": count,
                "examples": examples[(true_label, predicted_label)],
            }
        )
    return confused_rows


def safe_rate(numerator: int, denominator: int) -> float:
    """Chia an toàn để không lỗi khi mẫu số bằng 0."""

    return float(numerator / denominator) if denominator else 0.0


def build_summary(
    rows: list[dict[str, Any]],
    confused_pairs: list[dict[str, Any]],
    config: ProductionFlowConfig,
    stage_a_experiment: Path,
    stage_b_experiment: Path,
) -> dict[str, Any]:
    """Tạo summary JSON theo đúng các metric production cần xem."""

    total = len(rows)
    group_counts = Counter(row["group"] for row in rows)
    final_route_counts = Counter(row["final_route"] for row in rows)
    evaluation_counts = Counter(row["evaluation_result"] for row in rows)

    supported_rows = [row for row in rows if row["group"] == "supported"]
    unsupported_rows = [row for row in rows if row["group"] == "unsupported_v1"]
    out_of_scope_rows = [row for row in rows if row["group"] == "out_of_scope"]
    auto_rows = [row for row in rows if row["final_route"] == "stage_b_supported_auto_accept"]
    manual_rows = [row for row in rows if is_manual_review_route(row["final_route"])]
    supported_auto_rows = [
        row for row in supported_rows if row["final_route"] == "stage_b_supported_auto_accept"
    ]
    supported_auto_correct = [
        row for row in supported_auto_rows if row["evaluation_result"] == "correct_auto"
    ]
    supported_auto_wrong = [
        row for row in supported_auto_rows if row["evaluation_result"] == "wrong_auto"
    ]
    supported_manual_review = [
        row for row in supported_rows if is_manual_review_route(row["final_route"])
    ]
    supported_stage_a_blocked_wrong = [
        row for row in supported_rows if row["final_route"] == "stage_a_other"
    ]
    unsupported_correct_manual = [
        row for row in unsupported_rows if row["final_route"] == "manual_review_unsupported"
    ]
    out_scope_block_or_review = [
        row
        for row in out_of_scope_rows
        if row["final_route"] == "stage_a_other"
        or is_manual_review_route(row["final_route"])
    ]

    wrong_supported_by_class = Counter(
        row["true_label"] for row in supported_auto_wrong if row["true_label"]
    )
    class_sai_nhieu_nhat = (
        wrong_supported_by_class.most_common(1)[0][0]
        if wrong_supported_by_class
        else None
    )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "uploads_root": str(config.uploads_root),
        "stage_a_experiment": str(stage_a_experiment),
        "stage_b_experiment": str(stage_b_experiment),
        "thresholds": {
            "stage_a_fruit_threshold": config.stage_a_fruit_threshold,
            "stage_a_other_threshold": config.stage_a_other_threshold,
            "stage_b_confidence_threshold": config.stage_b_confidence_threshold,
            "stage_b_margin_threshold": config.stage_b_margin_threshold,
        },
        "total_images": total,
        "supported_count": len(supported_rows),
        "unsupported_v1_count": len(unsupported_rows),
        "out_of_scope_count": len(out_of_scope_rows),
        "unknown_label_count": group_counts.get("unknown_label", 0),
        "auto_accepted_count": len(auto_rows),
        "manual_review_count": len(manual_rows),
        "manual_review_rate": safe_rate(len(manual_rows), total),
        "supported_auto_accuracy": safe_rate(
            len(supported_auto_correct),
            len(supported_auto_rows),
        ),
        "supported_auto_coverage": safe_rate(
            len(supported_auto_rows),
            len(supported_rows),
        ),
        "supported_manual_review_count": len(supported_manual_review),
        "unsupported_correct_manual_review_rate": safe_rate(
            len(unsupported_correct_manual),
            len(unsupported_rows),
        ),
        "out_of_scope_block_or_review_rate": safe_rate(
            len(out_scope_block_or_review),
            len(out_of_scope_rows),
        ),
        "stage_a_blocked_supported_wrong_count": len(supported_stage_a_blocked_wrong),
        "stage_b_wrong_supported_count": len(supported_auto_wrong),
        "top_confused_pairs": confused_pairs[:10],
        "class_sai_nhieu_nhat": class_sai_nhieu_nhat,
        "wrong_supported_by_class": dict(wrong_supported_by_class.most_common()),
        "group_counts": dict(group_counts),
        "final_route_counts": dict(final_route_counts),
        "evaluation_result_counts": dict(evaluation_counts),
        "note": (
            "Manual review khong tinh la model doan sai production. "
            "Unsupported v1 duoc dua manual_review_unsupported va khong chay Stage B."
        ),
    }


def log_row(logger: logging.Logger, index: int, total: int, row: dict[str, Any]) -> None:
    """Ghi log một ảnh với đầy đủ field theo yêu cầu."""

    logger.info(
        "[%03d/%03d] file=%s | true=%s | group=%s | "
        "stage_a_prob=%s | stage_a_route=%s | "
        "top1=%s | top1_conf=%s | top2=%s | top2_conf=%s | margin=%s | "
        "final_route=%s | eval=%s | reason=%s",
        index,
        total,
        row["image_path"],
        row["true_label"] or row["normalized_label"],
        row["group"],
        row["stage_a_fruit_probability"],
        row["stage_a_route"],
        row["stage_b_top1_label"],
        row["stage_b_top1_confidence"],
        row["stage_b_top2_label"],
        row["stage_b_top2_confidence"],
        row["stage_b_margin"],
        row["final_route"],
        row["evaluation_result"],
        row["reason"],
    )


def log_summary(
    logger: logging.Logger,
    summary: dict[str, Any],
    error_cases: list[dict[str, Any]],
    confused_pairs: list[dict[str, Any]],
) -> None:
    """Ghi phần tổng kết cuối log production."""

    logger.info("")
    logger.info("========== SUMMARY UPLOAD FLOW PRODUCTION ==========")
    logger.info("Tong anh: %s", summary["total_images"])
    logger.info("Supported: %s", summary["supported_count"])
    logger.info("Unsupported v1: %s", summary["unsupported_v1_count"])
    logger.info("Out of scope / other: %s", summary["out_of_scope_count"])
    logger.info("Auto accepted: %s", summary["auto_accepted_count"])
    logger.info("Manual review: %s", summary["manual_review_count"])
    logger.info("Manual review rate: %.4f", summary["manual_review_rate"])
    logger.info("Supported auto accuracy: %.4f", summary["supported_auto_accuracy"])
    logger.info("Supported auto coverage: %.4f", summary["supported_auto_coverage"])
    logger.info("Supported manual review count: %s", summary["supported_manual_review_count"])
    logger.info(
        "Unsupported correct manual review rate: %.4f",
        summary["unsupported_correct_manual_review_rate"],
    )
    logger.info(
        "Out-of-scope block/review rate: %.4f",
        summary["out_of_scope_block_or_review_rate"],
    )
    logger.info(
        "Stage A chan nham supported: %s",
        summary["stage_a_blocked_supported_wrong_count"],
    )
    logger.info(
        "Stage B doan sai supported: %s",
        summary["stage_b_wrong_supported_count"],
    )
    logger.info("Class sai nhieu nhat: %s", summary["class_sai_nhieu_nhat"])

    logger.info("")
    logger.info("========== TOP CONFUSED PAIRS ==========")
    if not confused_pairs:
        logger.info("Khong co confused pair supported nao.")
    for row in confused_pairs[:10]:
        logger.info(
            "%s -> %s | count=%s | examples=%s",
            row["true_label"],
            row["predicted_label"],
            row["count"],
            json.dumps(row["examples"], ensure_ascii=False),
        )

    logger.info("")
    logger.info("========== ERROR / WRONG CASES ==========")
    if not error_cases:
        logger.info("Khong co case sai hoac anh loi.")
    for row in error_cases:
        logger.info(
            "%s | true=%s | group=%s | pred=%s | final_route=%s | reason=%s",
            row["image_path"],
            row["true_label"] or row["normalized_label"],
            row["group"],
            row["stage_b_top1_label"],
            row["final_route"],
            row["reason"],
        )


def run_production_flow_test(config: ProductionFlowConfig) -> dict[str, Any]:
    """Chạy full upload flow production, lưu log/CSV/JSON đầy đủ."""

    stage_a_experiment = config.stage_a_experiment or find_latest_experiment(STAGE_A_PREFIX)
    stage_b_experiment = config.stage_b_experiment or find_latest_experiment(STAGE_B_PREFIX)

    config.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(config.logs_dir / "upload_flow_production.log")

    logger.info("Bat dau test full upload flow production.")
    logger.info("Uploads root: %s", config.uploads_root)
    logger.info("Stage A experiment: %s", stage_a_experiment)
    logger.info("Stage B experiment: %s", stage_b_experiment)
    logger.info("Stage A fruit threshold: %.6f", config.stage_a_fruit_threshold)
    logger.info("Stage A other threshold: %.6f", config.stage_a_other_threshold)
    logger.info("Stage B confidence threshold: %.6f", config.stage_b_confidence_threshold)
    logger.info("Stage B margin threshold: %.6f", config.stage_b_margin_threshold)

    stage_a_manifest = load_json(stage_a_experiment / "labels.json")
    stage_b_manifest = load_json(stage_b_experiment / "labels.json")
    class_names = load_class_names(stage_b_experiment)

    stage_a_model_path = stage_a_experiment / "model.keras"
    stage_b_model_path = stage_b_experiment / "model.keras"
    if not stage_a_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay Stage A model: {stage_a_model_path}")
    if not stage_b_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay Stage B model: {stage_b_model_path}")

    logger.info("Dang load Stage A model...")
    stage_a_model = tf.keras.models.load_model(str(stage_a_model_path), compile=False)
    logger.info("Dang load Stage B model...")
    stage_b_model = tf.keras.models.load_model(str(stage_b_model_path), compile=False)

    upload_items = list_upload_images(config.uploads_root, config.valid_extensions)
    logger.info("Tong anh upload tim thay: %s", len(upload_items))

    rows: list[dict[str, Any]] = []
    for index, (image_path, raw_label) in enumerate(upload_items, start=1):
        row = run_single_image(
            image_path=image_path,
            raw_label=raw_label,
            class_names=class_names,
            stage_a_model=stage_a_model,
            stage_b_model=stage_b_model,
            stage_a_manifest=stage_a_manifest,
            stage_b_manifest=stage_b_manifest,
            config=config,
        )
        rows.append(row)
        log_row(logger, index, len(upload_items), row)

    prediction_fields = [
        "image_path",
        "raw_label",
        "normalized_label",
        "true_label",
        "group",
        "stage_a_fruit_probability",
        "stage_a_route",
        "stage_b_top1_label",
        "stage_b_top1_confidence",
        "stage_b_top2_label",
        "stage_b_top2_confidence",
        "stage_b_margin",
        "final_route",
        "evaluation_result",
        "reason",
        "is_correct_flow",
        "is_wrong_flow",
        "stage_b_ran",
        "top_k",
        "warnings",
        "error",
    ]
    per_class_fields = [
        "label",
        "group",
        "total",
        "auto_accepted",
        "auto_correct",
        "auto_wrong",
        "manual_review",
        "stage_a_other",
        "error",
        "auto_accuracy",
    ]
    confused_pair_fields = ["true_label", "predicted_label", "count", "examples"]

    per_class_summary = build_per_class_summary(rows)
    error_cases = build_error_cases(rows)
    confused_pairs = build_confused_pairs(rows)
    summary = build_summary(
        rows=rows,
        confused_pairs=confused_pairs,
        config=config,
        stage_a_experiment=stage_a_experiment,
        stage_b_experiment=stage_b_experiment,
    )

    write_csv(config.logs_dir / "upload_flow_predictions.csv", rows, prediction_fields)
    write_csv(
        config.logs_dir / "upload_flow_per_class_summary.csv",
        per_class_summary,
        per_class_fields,
    )
    write_csv(
        config.logs_dir / "upload_flow_error_cases.csv",
        error_cases,
        prediction_fields,
    )
    write_csv(
        config.logs_dir / "upload_flow_confused_pairs.csv",
        confused_pairs,
        confused_pair_fields,
    )
    (config.logs_dir / "upload_flow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log_summary(logger, summary, error_cases, confused_pairs)
    return summary


def main() -> None:
    """Entry point."""

    config = build_config(parse_args())
    summary = run_production_flow_test(config)
    print("")
    print("Da luu ket qua:")
    print(f"- {config.logs_dir / 'upload_flow_production.log'}")
    print(f"- {config.logs_dir / 'upload_flow_predictions.csv'}")
    print(f"- {config.logs_dir / 'upload_flow_summary.json'}")
    print(f"Tong anh: {summary['total_images']}")
    print(f"Auto accepted: {summary['auto_accepted_count']}")
    print(f"Manual review: {summary['manual_review_count']}")
    print(f"Supported auto accuracy: {summary['supported_auto_accuracy']:.4f}")


if __name__ == "__main__":
    main()
