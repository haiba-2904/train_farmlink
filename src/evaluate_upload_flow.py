from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
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

    Trên máy macOS đang dùng, nếu chạy nhầm bằng Python base/Anaconda thì
    TensorFlow có thể bị segmentation fault ngay lúc import. Đoạn này giúp
    script test upload chạy giống các script train ổn định trong project.
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

    if os.environ.get("FARMLINK_UPLOAD_FLOW_REEXECED") == "1":
        return

    os.environ["FARMLINK_UPLOAD_FLOW_REEXECED"] = "1"
    print(
        f"Upload flow eval: dang chuyen interpreter tu {sys.executable} sang {venv_python}",
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


STAGE_A_EXPERIMENT_PREFIX = "stage_a_resnet50_"
STAGE_B_EXPERIMENT_PREFIX = "stage_b_supported_v1_resnet50_"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

# Các lớp đã loại khỏi production v1. Đây vẫn là nông sản, nhưng Stage B v1
# không có output class cho chúng, nên flow tốt nhất là đưa manual review.
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

# Các folder này đại diện cho ảnh ngoài phạm vi nông sản. Với nhóm này,
# route đúng nhất là Stage A chặn ở nhánh `stage_a_other`.
OUT_OF_SCOPE_LABELS = frozenset({"other", "hard", "unknown", "manual_review"})


@dataclass(frozen=True)
class UploadFlowEvalConfig:
    """Config cho test ảnh upload theo đúng flow Stage A -> Stage B."""

    uploads_root: Path = Path("test_uploads_labeled")
    stage_a_experiment: Path | None = None
    stage_b_experiment: Path | None = None
    output_dir: Path | None = None
    stage_a_threshold: float | None = None
    stage_b_threshold: float = 0.60
    top_k: int = 3
    strict_quality_check: bool = False
    clean_output: bool = False
    valid_extensions: tuple[str, ...] = VALID_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI.

    Ví dụ chạy nhanh:
    `python src/evaluate_upload_flow.py --uploads-root test_uploads_labeled`
    """

    parser = argparse.ArgumentParser(
        description="Test anh upload theo flow production: Stage A fruit/other -> Stage B supported v1."
    )
    parser.add_argument("--uploads-root", type=Path, default=Path("test_uploads_labeled"))
    parser.add_argument("--stage-a-experiment", type=Path, default=None)
    parser.add_argument("--stage-b-experiment", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage-a-threshold", type=float, default=None)
    parser.add_argument("--stage-b-threshold", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strict-quality-check", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UploadFlowEvalConfig:
    """Ghép CLI args thành object config rõ ràng."""

    return UploadFlowEvalConfig(
        uploads_root=args.uploads_root,
        stage_a_experiment=args.stage_a_experiment,
        stage_b_experiment=args.stage_b_experiment,
        output_dir=args.output_dir,
        stage_a_threshold=args.stage_a_threshold,
        stage_b_threshold=args.stage_b_threshold,
        top_k=args.top_k,
        strict_quality_check=args.strict_quality_check,
        clean_output=args.clean_output,
    )


def setup_logger(log_path: Path) -> logging.Logger:
    """Tạo logger ghi đầy đủ từng ảnh vào file log test."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("upload_flow_eval")
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
    """Tìm experiment mới nhất theo prefix timestamp trong thư mục experiments."""

    if not root.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc experiments: {root}")

    candidates = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"Khong tim thay experiment {prefix}* trong {root}")
    return candidates[-1]


def prepare_output_dir(config: UploadFlowEvalConfig, stage_b_experiment: Path) -> Path:
    """Tạo thư mục output riêng cho lần test upload.

    Nếu người dùng không truyền `--output-dir`, script tự tạo thư mục timestamp
    nằm trong experiment Stage B mới nhất để dễ truy vết model nào đã được test.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir or stage_b_experiment / f"upload_flow_eval_{timestamp}"

    if output_dir.exists():
        if not config.clean_output:
            raise FileExistsError(
                f"Output da ton tai: {output_dir}. Dung --clean-output neu muon ghi de."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def load_json(path: Path) -> dict[str, Any]:
    """Đọc JSON và báo lỗi rõ nếu file không tồn tại."""

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_a_threshold(
    stage_a_experiment: Path,
    manifest: dict[str, Any],
    override_threshold: float | None,
) -> float:
    """Lấy threshold Stage A.

    Ưu tiên:
    1. CLI `--stage-a-threshold`;
    2. `labels.json` field `fruit_threshold`;
    3. `threshold_analysis.json` field `selected.threshold`.
    """

    if override_threshold is not None:
        return float(override_threshold)

    if manifest.get("fruit_threshold") is not None:
        return float(manifest["fruit_threshold"])

    threshold_analysis_path = stage_a_experiment / "threshold_analysis.json"
    if threshold_analysis_path.exists():
        threshold_analysis = load_json(threshold_analysis_path)
        selected = threshold_analysis.get("selected", {})
        if selected.get("threshold") is not None:
            return float(selected["threshold"])

    raise ValueError(
        "Khong tim thay threshold Stage A. Hay truyen --stage-a-threshold."
    )


def validate_stage_manifests(
    stage_a_manifest: dict[str, Any],
    stage_b_manifest: dict[str, Any],
) -> list[str]:
    """Kiểm tra manifest của 2 model trước khi test upload."""

    warnings: list[str] = []

    stage_a_classes = stage_a_manifest.get("class_names")
    if stage_a_classes != ["other", "fruit"]:
        raise ValueError(
            "Stage A labels.json phai co class_names = ['other', 'fruit']; "
            f"nhan duoc {stage_a_classes}"
        )

    stage_b_classes = stage_b_manifest.get("class_names")
    if not isinstance(stage_b_classes, list) or not stage_b_classes:
        raise ValueError("Stage B labels.json khong co class_names hop le.")
    if "other" in stage_b_classes:
        raise ValueError("Stage B supported v1 khong duoc chua class other.")
    if any(class_name in stage_b_classes for class_name in UNSUPPORTED_CLASSES_V1):
        raise ValueError("Stage B supported v1 van chua unsupported class.")

    stage_a_size = stage_a_manifest.get("image_size", [320, 320])
    stage_b_size = stage_b_manifest.get("image_size", [320, 320])
    if stage_a_size != stage_b_size:
        warnings.append(
            f"Stage A image_size {stage_a_size} khac Stage B image_size {stage_b_size}."
        )
    return warnings


def normalize_label_token(value: str) -> str:
    """Chuẩn hóa tên folder upload thành snake_case không dấu.

    Hàm này xử lý các folder có dấu cách thừa như `caimito ` hoặc
    `pomegranate `, và tên gõ sai nhẹ như `suger_apple` sẽ được map tiếp
    trong `map_upload_label`.
    """

    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def map_upload_label(raw_label: str, class_names: list[str]) -> tuple[str | None, str, str]:
    """Map nhãn thật trong folder upload sang nhóm đánh giá.

    Return:
    - expected_label: class của Stage B nếu ảnh thuộc supported v1;
    - normalized_label: tên label đã chuẩn hóa;
    - truth_status:
      `supported`: có trong Stage B v1;
      `unsupported_v1`: là nông sản nhưng đã loại khỏi v1;
      `out_of_scope`: ảnh ngoài nông sản, Stage A nên chặn;
      `unknown_label`: folder lạ, vẫn ghi log nhưng không tính đúng/sai tự động.
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
    """Liệt kê ảnh upload theo cấu trúc `test_uploads_labeled/<true_label>/<image>`."""

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
    strict_quality_check: bool,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess ảnh upload giống ResNet50 training/inference.

    Flow xử lý một ảnh:
    1. mở ảnh bằng PIL qua `open_image_safely`, tự fix EXIF và RGB;
    2. kiểm tra chất lượng cơ bản để bắt ảnh lỗi/ảnh quá nhỏ;
    3. resize/pad về đúng `image_size`, giữ aspect ratio;
    4. áp dụng `tf.keras.applications.resnet.preprocess_input`;
    5. thêm batch dimension để đưa vào model.
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


def top_k_predictions(
    probabilities: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Format top-k prediction để lưu vào log/report."""

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


def predict_stage_a_fruit_probability(model: tf.keras.Model, image_tensor: np.ndarray) -> float:
    """Trả về xác suất ảnh là nông sản của Stage A.

    Stage A trong project dùng output sigmoid 1 node, nên giá trị model trả về
    chính là `fruit_probability`.
    """

    output = model.predict(image_tensor, verbose=0)
    return float(np.asarray(output).reshape(-1)[0])


def predict_stage_b(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], str, float]:
    """Predict Stage B và trả về top-k, nhãn tốt nhất, confidence tốt nhất."""

    probabilities = np.asarray(model.predict(image_tensor, verbose=0))[0]
    predictions = top_k_predictions(probabilities, class_names, top_k)
    best_prediction = predictions[0]
    return predictions, str(best_prediction["label"]), float(best_prediction["confidence"])


def classify_flow_result(
    truth_status: str,
    expected_label: str | None,
    route: str,
    predicted_label: str | None,
) -> tuple[str, bool | None, str]:
    """Quy đổi route production thành trạng thái đánh giá dễ đọc.

    `evaluation_status` được chia thành:
    - `correct`: hệ thống tự xử lý đúng;
    - `wrong`: hệ thống tự xử lý sai;
    - `manual_review`: model không chắc, chuyển người kiểm duyệt;
    - `error`: ảnh lỗi hoặc không đọc được;
    - `unknown_label`: nhãn thật không nằm trong taxonomy đánh giá.
    """

    if route == "error":
        return "error", None, "image_error"

    if truth_status == "unknown_label":
        if route == "low_confidence":
            return "manual_review", None, "unknown_label_manual_review"
        return "unknown_label", None, "unknown_true_label"

    if route == "low_confidence":
        return "manual_review", None, "low_confidence_manual_review"

    if route == "stage_a_other":
        if truth_status == "out_of_scope":
            return "correct", True, "stage_a_correctly_blocked_other"
        return "wrong", False, "stage_a_blocked_real_fruit"

    if route == "stage_b_supported":
        if truth_status == "supported" and predicted_label == expected_label:
            return "correct", True, "stage_b_correct_supported_class"
        if truth_status == "supported":
            return "wrong", False, "stage_b_wrong_supported_class"
        if truth_status == "out_of_scope":
            return "wrong", False, "stage_a_false_pass_other_then_stage_b_accepted"
        if truth_status == "unsupported_v1":
            return "wrong", False, "unsupported_v1_auto_accepted_as_supported"

    return "unknown_label", None, "unhandled_case"


def save_predictions_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Lưu toàn bộ kết quả từng ảnh dạng CSV để dễ mở bằng Excel."""

    fieldnames = [
        "image_path",
        "raw_label",
        "normalized_label",
        "truth_label",
        "truth_status",
        "stage_a_fruit_probability",
        "stage_a_threshold",
        "stage_b_threshold",
        "route",
        "evaluation_status",
        "correct",
        "predicted_label",
        "confidence",
        "reason",
        "top_k",
        "warnings",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["top_k"] = json.dumps(row.get("top_k", []), ensure_ascii=False)
            csv_row["warnings"] = json.dumps(row.get("warnings", []), ensure_ascii=False)
            writer.writerow(csv_row)


def save_predictions_jsonl(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Lưu toàn bộ kết quả từng ảnh dạng JSONL để máy đọc lại dễ hơn."""

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Lưu confusion matrix PNG cho các ảnh supported đã được Stage B nhận."""

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
    plt.title("Upload Flow Stage B Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_stage_b_supported_report(
    rows: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any] | None, list[list[int]] | None]:
    """Sinh classification report cho subset supported được Stage B tự nhận.

    Chỉ tính các ảnh:
    - nhãn thật thuộc supported v1;
    - không bị Stage A chặn;
    - không bị low confidence;
    - đã có predicted_label của Stage B.
    """

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    y_true: list[int] = []
    y_pred: list[int] = []
    for row in rows:
        if row["truth_status"] != "supported":
            continue
        if row["route"] != "stage_b_supported":
            continue
        truth_label = row["truth_label"]
        predicted_label = row["predicted_label"]
        if truth_label in class_to_index and predicted_label in class_to_index:
            y_true.append(class_to_index[truth_label])
            y_pred.append(class_to_index[predicted_label])

    if not y_true:
        return None, None

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
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    (output_dir / "stage_b_supported_classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    save_confusion_matrix_plot(
        matrix,
        class_names,
        output_dir / "stage_b_supported_confusion_matrix.png",
    )
    return report_dict, matrix.astype(int).tolist()


def write_summary_log(
    logger: logging.Logger,
    report: dict[str, Any],
    stage_a_blocked_wrong: list[dict[str, Any]],
    stage_b_wrong: list[dict[str, Any]],
    manual_review: list[dict[str, Any]],
) -> None:
    """Ghi phần tổng kết cuối file log test bằng tiếng Việt."""

    summary = report["summary"]
    logger.info("")
    logger.info("========== TONG QUAN TEST UPLOAD FLOW ==========")
    logger.info("Tong anh: %s", summary["total_images"])
    logger.info("Anh nhan dung: %s", summary["correct_count"])
    logger.info("Anh manual_review / low_confidence: %s", summary["manual_review_count"])
    logger.info("Anh sai: %s", summary["wrong_count"])
    logger.info("Anh error: %s", summary["error_count"])
    logger.info("Route stage_a_other: %s", summary["route_counts"].get("stage_a_other", 0))
    logger.info("Route stage_b_supported: %s", summary["route_counts"].get("stage_b_supported", 0))
    logger.info("Route low_confidence: %s", summary["route_counts"].get("low_confidence", 0))
    logger.info("Route error: %s", summary["route_counts"].get("error", 0))
    logger.info("Accuracy tren anh auto-resolved: %.4f", summary["auto_resolved_accuracy"])
    logger.info("Ti le dung tren tong anh: %.4f", summary["overall_correct_rate"])
    logger.info("Class sai nhieu nhat: %s", report["wrong_analysis"]["most_wrong_class"])

    logger.info("")
    logger.info("========== ANH STAGE A CHAN NHAM ==========")
    if not stage_a_blocked_wrong:
        logger.info("Khong co anh nao bi Stage A chan nham.")
    for item in stage_a_blocked_wrong:
        logger.info(
            "%s | true=%s | status=%s | fruit_prob=%.6f",
            item["image_path"],
            item["truth_label"] or item["normalized_label"],
            item["truth_status"],
            item["stage_a_fruit_probability"],
        )

    logger.info("")
    logger.info("========== ANH STAGE B NHAM CLASS ==========")
    if not stage_b_wrong:
        logger.info("Khong co anh supported nao bi Stage B nham class.")
    for item in stage_b_wrong:
        logger.info(
            "%s | true=%s | pred=%s | confidence=%.6f | top_k=%s",
            item["image_path"],
            item["truth_label"],
            item["predicted_label"],
            item["confidence"],
            json.dumps(item["top_k"], ensure_ascii=False),
        )

    logger.info("")
    logger.info("========== ANH LOW CONFIDENCE / MANUAL REVIEW ==========")
    if not manual_review:
        logger.info("Khong co anh low_confidence.")
    for item in manual_review:
        logger.info(
            "%s | true=%s | status=%s | pred=%s | confidence=%.6f | fruit_prob=%.6f",
            item["image_path"],
            item["truth_label"] or item["normalized_label"],
            item["truth_status"],
            item["predicted_label"],
            item["confidence"],
            item["stage_a_fruit_probability"],
        )


def run_upload_flow_eval(config: UploadFlowEvalConfig) -> tuple[Path, dict[str, Any]]:
    """Chạy full flow Stage A -> Stage B trên thư mục ảnh upload có nhãn."""

    stage_a_experiment = config.stage_a_experiment or find_latest_experiment(
        STAGE_A_EXPERIMENT_PREFIX
    )
    stage_b_experiment = config.stage_b_experiment or find_latest_experiment(
        STAGE_B_EXPERIMENT_PREFIX
    )
    output_dir = prepare_output_dir(config, stage_b_experiment)
    logger = setup_logger(output_dir / "upload_flow_test.log")

    stage_a_manifest = load_json(stage_a_experiment / "labels.json")
    stage_b_manifest = load_json(stage_b_experiment / "labels.json")
    manifest_warnings = validate_stage_manifests(stage_a_manifest, stage_b_manifest)
    class_names = [str(item) for item in stage_b_manifest["class_names"]]
    stage_a_threshold = load_stage_a_threshold(
        stage_a_experiment,
        stage_a_manifest,
        config.stage_a_threshold,
    )
    stage_b_threshold = float(config.stage_b_threshold)

    stage_a_model_path = stage_a_experiment / "model.keras"
    stage_b_model_path = stage_b_experiment / "model.keras"
    if not stage_a_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay Stage A model: {stage_a_model_path}")
    if not stage_b_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay Stage B model: {stage_b_model_path}")

    logger.info("Bat dau test upload theo full flow production.")
    logger.info("Uploads root: %s", config.uploads_root)
    logger.info("Stage A experiment: %s", stage_a_experiment)
    logger.info("Stage B experiment: %s", stage_b_experiment)
    logger.info("Stage A threshold fruit: %.6f", stage_a_threshold)
    logger.info("Stage B threshold supported confidence: %.6f", stage_b_threshold)
    for warning in manifest_warnings:
        logger.warning(warning)

    logger.info("Dang load Stage A model...")
    stage_a_model = tf.keras.models.load_model(str(stage_a_model_path), compile=False)
    logger.info("Dang load Stage B model...")
    stage_b_model = tf.keras.models.load_model(str(stage_b_model_path), compile=False)

    upload_items = list_upload_images(config.uploads_root, config.valid_extensions)
    logger.info("Tong anh upload tim thay: %s", len(upload_items))

    rows: list[dict[str, Any]] = []
    raw_label_counts: Counter[str] = Counter()
    truth_status_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    evaluation_status_counts: Counter[str] = Counter()
    wrong_by_true_label: Counter[str] = Counter()
    wrong_by_predicted_label: Counter[str] = Counter()
    stage_b_confused_pairs: Counter[str] = Counter()
    unsupported_auto_predictions: dict[str, Counter[str]] = defaultdict(Counter)

    for image_index, (image_path, raw_label) in enumerate(upload_items, start=1):
        raw_label_counts[raw_label.strip()] += 1
        expected_label, normalized_label, truth_status = map_upload_label(
            raw_label,
            class_names,
        )
        truth_status_counts[truth_status] += 1

        route = "error"
        evaluation_status = "error"
        correct: bool | None = None
        reason = "image_error"
        predicted_label: str | None = None
        confidence = 0.0
        stage_a_fruit_probability = 0.0
        top_k: list[dict[str, Any]] = []
        warnings: list[str] = []
        error_text = ""

        try:
            # Stage A preprocess trước vì mọi ảnh upload đều phải qua bước lọc
            # fruit-vs-other. Stage B dùng cùng ResNet preprocess và cùng size.
            image_tensor, warnings = prepare_image_tensor(
                image_path=image_path,
                manifest=stage_a_manifest,
                strict_quality_check=config.strict_quality_check,
            )
            stage_a_fruit_probability = predict_stage_a_fruit_probability(
                stage_a_model,
                image_tensor,
            )

            if stage_a_fruit_probability < stage_a_threshold:
                route = "stage_a_other"
            else:
                top_k, predicted_label, confidence = predict_stage_b(
                    stage_b_model,
                    image_tensor,
                    class_names,
                    config.top_k,
                )
                if confidence < stage_b_threshold:
                    route = "low_confidence"
                else:
                    route = "stage_b_supported"

            evaluation_status, correct, reason = classify_flow_result(
                truth_status=truth_status,
                expected_label=expected_label,
                route=route,
                predicted_label=predicted_label,
            )
        except Exception as error:  # noqa: BLE001
            error_text = str(error)
            route = "error"
            evaluation_status = "error"
            correct = None
            reason = "image_error"

        route_counts[route] += 1
        evaluation_status_counts[evaluation_status] += 1

        truth_label_for_count = expected_label or normalized_label
        if evaluation_status == "wrong":
            wrong_by_true_label[truth_label_for_count] += 1
            if predicted_label:
                wrong_by_predicted_label[predicted_label] += 1

        if (
            route == "stage_b_supported"
            and truth_status == "supported"
            and expected_label is not None
            and predicted_label is not None
            and predicted_label != expected_label
        ):
            stage_b_confused_pairs[f"{expected_label}->{predicted_label}"] += 1

        if (
            route == "stage_b_supported"
            and truth_status == "unsupported_v1"
            and predicted_label is not None
        ):
            unsupported_auto_predictions[normalized_label][predicted_label] += 1

        row = {
            "image_path": str(image_path),
            "raw_label": raw_label,
            "normalized_label": normalized_label,
            "truth_label": expected_label or "",
            "truth_status": truth_status,
            "stage_a_fruit_probability": float(stage_a_fruit_probability),
            "stage_a_threshold": float(stage_a_threshold),
            "stage_b_threshold": float(stage_b_threshold),
            "route": route,
            "evaluation_status": evaluation_status,
            "correct": correct if correct is not None else "",
            "predicted_label": predicted_label or "",
            "confidence": float(confidence),
            "reason": reason,
            "top_k": top_k,
            "warnings": warnings,
            "error": error_text,
        }
        rows.append(row)

        logger.info(
            "[%03d/%03d] route=%s | eval=%s | true=%s/%s | pred=%s | "
            "stage_a_prob=%.6f | conf=%.6f | file=%s | reason=%s%s",
            image_index,
            len(upload_items),
            route,
            evaluation_status,
            truth_status,
            expected_label or normalized_label,
            predicted_label or "",
            stage_a_fruit_probability,
            confidence,
            image_path,
            reason,
            f" | error={error_text}" if error_text else "",
        )

    stage_a_blocked_wrong = [
        row
        for row in rows
        if row["route"] == "stage_a_other" and row["evaluation_status"] == "wrong"
    ]
    stage_b_wrong = [
        row
        for row in rows
        if row["truth_status"] == "supported"
        and row["route"] == "stage_b_supported"
        and row["evaluation_status"] == "wrong"
    ]
    manual_review = [row for row in rows if row["route"] == "low_confidence"]
    errors = [row for row in rows if row["route"] == "error"]
    confident_wrong = [row for row in rows if row["evaluation_status"] == "wrong"]
    correct_count = int(evaluation_status_counts.get("correct", 0))
    wrong_count = int(evaluation_status_counts.get("wrong", 0))
    manual_review_count = int(evaluation_status_counts.get("manual_review", 0))
    error_count = int(evaluation_status_counts.get("error", 0))
    auto_resolved_total = correct_count + wrong_count
    total_images = len(rows)

    stage_b_report, stage_b_confusion = build_stage_b_supported_report(
        rows,
        class_names,
        output_dir,
    )

    unsupported_summary = {
        label: {
            "count": int(sum(counter.values())),
            "predicted_as": dict(counter.most_common()),
        }
        for label, counter in sorted(unsupported_auto_predictions.items())
    }

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "uploads_root": str(config.uploads_root),
        "output_dir": str(output_dir),
        "stage_a_experiment": str(stage_a_experiment),
        "stage_a_model_path": str(stage_a_model_path),
        "stage_a_threshold": stage_a_threshold,
        "stage_b_experiment": str(stage_b_experiment),
        "stage_b_model_path": str(stage_b_model_path),
        "stage_b_threshold": stage_b_threshold,
        "stage_b_class_count": len(class_names),
        "stage_b_class_names": class_names,
        "summary": {
            "total_images": total_images,
            "correct_count": correct_count,
            "manual_review_count": manual_review_count,
            "wrong_count": wrong_count,
            "error_count": error_count,
            "unknown_label_count": int(evaluation_status_counts.get("unknown_label", 0)),
            "auto_resolved_total": auto_resolved_total,
            "auto_resolved_accuracy": (
                float(correct_count / auto_resolved_total) if auto_resolved_total else 0.0
            ),
            "overall_correct_rate": (
                float(correct_count / total_images) if total_images else 0.0
            ),
            "manual_review_rate": (
                float(manual_review_count / total_images) if total_images else 0.0
            ),
            "wrong_rate": float(wrong_count / total_images) if total_images else 0.0,
            "route_counts": dict(route_counts),
            "truth_status_counts": dict(truth_status_counts),
            "evaluation_status_counts": dict(evaluation_status_counts),
            "raw_label_counts": dict(sorted(raw_label_counts.items())),
        },
        "wrong_analysis": {
            "most_wrong_class": (
                wrong_by_true_label.most_common(1)[0][0] if wrong_by_true_label else None
            ),
            "wrong_by_true_label": dict(wrong_by_true_label.most_common()),
            "wrong_by_predicted_label": dict(wrong_by_predicted_label.most_common()),
            "stage_b_confused_pairs": dict(stage_b_confused_pairs.most_common()),
        },
        "stage_a_blocked_wrong_count": len(stage_a_blocked_wrong),
        "stage_a_blocked_wrong_images": stage_a_blocked_wrong,
        "stage_b_wrong_count": len(stage_b_wrong),
        "stage_b_wrong_images": stage_b_wrong,
        "manual_review_count": len(manual_review),
        "manual_review_images": manual_review,
        "error_count": len(errors),
        "error_images": errors,
        "unsupported_v1_auto_predictions": unsupported_summary,
        "stage_b_supported_classification_report": stage_b_report,
        "stage_b_supported_confusion_matrix": stage_b_confusion,
        "note": (
            "Flow nay do nhu user upload that: Stage A quyet dinh fruit/other; "
            "neu fruit thi Stage B supported v1 predict; neu confidence Stage B "
            "thap hon threshold thi dua manual_review."
        ),
    }

    save_predictions_csv(output_dir / "upload_flow_predictions.csv", rows)
    save_predictions_jsonl(output_dir / "upload_flow_predictions.jsonl", rows)
    (output_dir / "upload_flow_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_log(
        logger,
        report,
        stage_a_blocked_wrong,
        stage_b_wrong,
        manual_review,
    )
    return output_dir, report


def main() -> None:
    """Entry point của script test upload flow."""

    args = parse_args()
    config = build_config(args)
    output_dir, report = run_upload_flow_eval(config)
    summary = report["summary"]
    print(f"Output: {output_dir}")
    print(f"Tong anh: {summary['total_images']}")
    print(f"Dung: {summary['correct_count']}")
    print(f"Manual review: {summary['manual_review_count']}")
    print(f"Sai: {summary['wrong_count']}")
    print(f"Error: {summary['error_count']}")
    print(f"Log: {output_dir / 'upload_flow_test.log'}")


if __name__ == "__main__":
    main()
