from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass(frozen=True)
class AnalyzeConfusionConfig:
    """Cấu hình trung tâm cho script phân tích lỗi từ confusion matrix."""

    confusion_matrix_path: Path = Path("logs/confusion_matrix.npy")
    y_true_path: Path | None = None
    y_pred_path: Path | None = None
    labels_path: Path = Path("models/mobilenetv2_farm.labels.json")
    class_names_path: Path | None = None
    error_analysis_path: Path = Path("logs/error_analysis.json")
    pair_summary_path: Path = Path("logs/confusion_pairs_summary.json")
    hard_classes_path: Path = Path("logs/hard_classes.txt")
    runtime_log_path: Path = Path("logs/analyze_confusion.log")
    weak_recall_threshold: float = 0.6
    weak_f1_threshold: float = 0.6
    top_k_confusions: int = 3
    top_pair_count: int = 10
    save_generated_confusion_matrix: bool = True


def parse_args() -> argparse.Namespace:
    """Nhận tham số CLI để script dùng được cho nhiều dataset/model khác nhau."""

    parser = argparse.ArgumentParser(
        description="Phan tich confusion matrix va tu dong tim lop yeu / cap nham lan."
    )
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path("logs/confusion_matrix.npy"),
        help="Duong dan toi file confusion_matrix.npy neu da co san.",
    )
    parser.add_argument(
        "--y-true",
        type=Path,
        default=None,
        help="File numpy chua ground-truth labels de tao confusion matrix.",
    )
    parser.add_argument(
        "--y-pred",
        type=Path,
        default=None,
        help="File numpy chua predicted labels de tao confusion matrix.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("models/mobilenetv2_farm.labels.json"),
        help="Label manifest JSON chua class_names.",
    )
    parser.add_argument(
        "--class-names-file",
        type=Path,
        default=None,
        help=(
            "File chua class_names. Ho tro .json, .txt, .npy. "
            "Neu bo trong, script se doc tu --labels."
        ),
    )
    parser.add_argument(
        "--error-analysis-out",
        type=Path,
        default=Path("logs/error_analysis.json"),
        help="File JSON dau ra cho ket qua phan tich loi.",
    )
    parser.add_argument(
        "--pair-summary-out",
        type=Path,
        default=Path("logs/confusion_pairs_summary.json"),
        help="File JSON gon chua top confusion pairs hai chieu va weak_classes.",
    )
    parser.add_argument(
        "--hard-classes-out",
        type=Path,
        default=Path("logs/hard_classes.txt"),
        help="File text dau ra chua danh sach lop yeu.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/analyze_confusion.log"),
        help="File log runtime cua script.",
    )
    parser.add_argument(
        "--weak-recall-threshold",
        type=float,
        default=0.6,
        help="Lop bi xem la yeu neu recall < nguong nay.",
    )
    parser.add_argument(
        "--weak-f1-threshold",
        type=float,
        default=0.6,
        help="Lop bi xem la yeu neu f1-score < nguong nay.",
    )
    parser.add_argument(
        "--top-k-confusions",
        type=int,
        default=3,
        help="So luong lop nham lan cao nhat can giu lai cho moi lop.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="So cap class nham lan hai chieu cao nhat can export.",
    )
    parser.add_argument(
        "--no-save-generated-confusion-matrix",
        action="store_true",
        help="Neu tao confusion matrix tu y_true/y_pred thi khong ghi ra file .npy.",
    )
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> AnalyzeConfusionConfig:
    """Chuyển tham số CLI thành config typed rõ ràng."""

    return AnalyzeConfusionConfig(
        confusion_matrix_path=args.confusion_matrix,
        y_true_path=args.y_true,
        y_pred_path=args.y_pred,
        labels_path=args.labels,
        class_names_path=args.class_names_file,
        error_analysis_path=args.error_analysis_out,
        pair_summary_path=args.pair_summary_out,
        hard_classes_path=args.hard_classes_out,
        runtime_log_path=args.log_file,
        weak_recall_threshold=args.weak_recall_threshold,
        weak_f1_threshold=args.weak_f1_threshold,
        top_k_confusions=args.top_k_confusions,
        top_pair_count=args.top_pairs,
        save_generated_confusion_matrix=not args.no_save_generated_confusion_matrix,
    )


def setup_logger(log_path: Path) -> logging.Logger:
    """Thiết lập logger để vừa ghi file vừa in console khi chạy local."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analyze_confusion")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def validate_config(config: AnalyzeConfusionConfig) -> None:
    """Kiểm tra config đầu vào để fail fast nếu thông số không hợp lệ."""

    if config.top_k_confusions <= 0:
        raise ValueError("top_k_confusions phai lon hon 0.")

    if config.top_pair_count <= 0:
        raise ValueError("top_pair_count phai lon hon 0.")

    if not 0.0 <= config.weak_recall_threshold <= 1.0:
        raise ValueError("weak_recall_threshold phai nam trong [0, 1].")

    if not 0.0 <= config.weak_f1_threshold <= 1.0:
        raise ValueError("weak_f1_threshold phai nam trong [0, 1].")

    using_confusion_matrix = config.confusion_matrix_path.exists()
    using_predictions = config.y_true_path is not None and config.y_pred_path is not None

    if not using_confusion_matrix and not using_predictions:
        raise FileNotFoundError(
            "Khong tim thay confusion_matrix.npy va cung khong co cap --y-true/--y-pred."
        )

    if (config.y_true_path is None) ^ (config.y_pred_path is None):
        raise ValueError("Can cung cap dong thoi ca --y-true va --y-pred neu muon tao ma tran.")


def _normalize_class_names(class_names: list[str]) -> list[str]:
    """Chuẩn hóa class_names và kiểm tra trùng/rỗng để tránh phân tích sai."""

    normalized_names = [str(class_name).strip() for class_name in class_names]
    if not normalized_names:
        raise ValueError("class_names dang rong.")

    if any(not class_name for class_name in normalized_names):
        raise ValueError("Phat hien class_name rong trong danh sach nhan.")

    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("class_names dang bi trung nhau.")

    return normalized_names


def _load_class_names_from_json_file(json_path: Path) -> list[str]:
    """Đọc class_names từ file JSON ở 2 dạng phổ biến:

    1. list[str]
    2. dict có key 'class_names'
    """

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return _normalize_class_names([str(item) for item in payload])

    if isinstance(payload, dict) and "class_names" in payload:
        return _normalize_class_names([str(item) for item in payload["class_names"]])

    raise ValueError(
        f"File JSON khong chua class_names hop le: {json_path}"
    )


def load_class_names(config: AnalyzeConfusionConfig) -> list[str]:
    """Nạp class_names từ file chỉ định hoặc từ label manifest mặc định."""

    source_path = config.class_names_path or config.labels_path
    if not source_path.exists():
        raise FileNotFoundError(f"Khong tim thay file class_names/labels: {source_path}")

    if source_path.suffix.lower() == ".json":
        return _load_class_names_from_json_file(source_path)

    if source_path.suffix.lower() == ".txt":
        with source_path.open("r", encoding="utf-8") as file:
            class_names = [line.strip() for line in file if line.strip()]
        return _normalize_class_names(class_names)

    if source_path.suffix.lower() == ".npy":
        class_names = np.load(source_path, allow_pickle=True).tolist()
        if not isinstance(class_names, list):
            raise ValueError(f"File .npy class_names khong dung dinh dang list: {source_path}")
        return _normalize_class_names([str(item) for item in class_names])

    raise ValueError(
        "Khong ho tro dinh dang file class_names. Ho tro: .json, .txt, .npy"
    )


def load_label_array(array_path: Path, array_name: str) -> np.ndarray:
    """Đọc mảng label từ file numpy và ép về vector 1 chiều số nguyên."""

    if not array_path.exists():
        raise FileNotFoundError(f"Khong tim thay file {array_name}: {array_path}")

    label_array = np.load(array_path, allow_pickle=False)
    label_array = np.asarray(label_array).reshape(-1)
    if label_array.size == 0:
        raise ValueError(f"Mang {array_name} dang rong: {array_path}")

    if not np.issubdtype(label_array.dtype, np.integer):
        # Trường hợp file được lưu kiểu float nhưng thực chất vẫn là chỉ số lớp.
        if np.any(label_array != np.floor(label_array)):
            raise ValueError(f"Mang {array_name} phai chua gia tri chi so lop nguyen.")
        label_array = label_array.astype(np.int32)
    else:
        label_array = label_array.astype(np.int32)

    return label_array


def load_or_generate_confusion_matrix(
    config: AnalyzeConfusionConfig,
    class_names: list[str],
    logger: logging.Logger,
) -> tuple[np.ndarray, str]:
    """Lấy confusion matrix từ file .npy hoặc tạo mới từ y_true/y_pred."""

    expected_num_classes = len(class_names)

    if config.confusion_matrix_path.exists():
        matrix = np.load(config.confusion_matrix_path, allow_pickle=False)
        logger.info("Da nap confusion matrix tu: %s", config.confusion_matrix_path)
        source_description = str(config.confusion_matrix_path)
    else:
        if config.y_true_path is None or config.y_pred_path is None:
            raise FileNotFoundError(
                "Khong tim thay confusion_matrix.npy va khong du thong tin y_true/y_pred."
            )

        y_true = load_label_array(config.y_true_path, "y_true")
        y_pred = load_label_array(config.y_pred_path, "y_pred")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("So luong phan tu cua y_true va y_pred khong khop nhau.")

        max_label = max(int(y_true.max()), int(y_pred.max()))
        if max_label >= expected_num_classes or int(y_true.min()) < 0 or int(y_pred.min()) < 0:
            raise ValueError(
                "Co label nam ngoai khoang class_names. Kiem tra lai thu tu nhan dau vao."
            )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(expected_num_classes),
        )
        logger.info(
            "Da tao confusion matrix tu y_true=%s va y_pred=%s",
            config.y_true_path,
            config.y_pred_path,
        )
        source_description = f"generated_from:{config.y_true_path},{config.y_pred_path}"

        if config.save_generated_confusion_matrix:
            config.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(config.confusion_matrix_path, matrix)
            logger.info("Da luu confusion matrix moi tai: %s", config.confusion_matrix_path)

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix phai la ma tran vuong 2 chieu.")

    if matrix.shape[0] != expected_num_classes:
        raise ValueError(
            "Kich thuoc confusion_matrix khong khop so luong class_names: "
            f"{matrix.shape[0]} != {expected_num_classes}"
        )

    if np.any(matrix < 0):
        raise ValueError("confusion_matrix khong duoc chua gia tri am.")

    return matrix, source_description


def safe_divide(numerator: float, denominator: float) -> float:
    """Chia an toàn để tránh chia cho 0 trong các lớp không có support."""

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_per_class_metrics(confusion: np.ndarray) -> dict[str, np.ndarray]:
    """Tính precision / recall / f1 / support trực tiếp từ confusion matrix."""

    true_positives = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)

    precision = np.array(
        [safe_divide(tp, predicted) for tp, predicted in zip(true_positives, predicted_count)],
        dtype=np.float64,
    )
    recall = np.array(
        [safe_divide(tp, true_count) for tp, true_count in zip(true_positives, support)],
        dtype=np.float64,
    )
    f1_score = np.array(
        [
            safe_divide(2.0 * p * r, p + r) if (p + r) > 0 else 0.0
            for p, r in zip(precision, recall)
        ],
        dtype=np.float64,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "support": support,
        "predicted_count": predicted_count,
        "true_positives": true_positives,
    }


def get_top_confusions_for_class(
    confusion: np.ndarray,
    class_index: int,
    class_names: list[str],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Lấy top-k lớp mà một lớp hiện tại bị nhầm nhiều nhất.

    Chúng ta xét theo hàng của confusion matrix:
    - hàng i: ground-truth là lớp i
    - cột j: model dự đoán thành lớp j
    """

    row = confusion[class_index].copy()
    row[class_index] = 0
    support = int(confusion[class_index].sum())

    confusion_candidates = [
        (target_index, int(count))
        for target_index, count in enumerate(row.tolist())
        if int(count) > 0
    ]
    confusion_candidates.sort(key=lambda item: item[1], reverse=True)
    top_candidates = confusion_candidates[:top_k]

    confused_with = [class_names[target_index] for target_index, _ in top_candidates]
    detailed_confusions = [
        {
            "class_name": class_names[target_index],
            "count": count,
            "rate_within_class": safe_divide(count, support),
        }
        for target_index, count in top_candidates
    ]
    return confused_with, detailed_confusions


def build_confusion_pair_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    top_pair_count: int = 10,
    weak_recall_threshold: float = 0.6,
) -> dict[str, Any]:
    """Tạo summary gọn từ y_true/y_pred theo đúng công thức pair hai chiều.

    Đây là hàm reusable cho notebook/script khác:
    1. Build confusion matrix kích thước num_classes x num_classes.
    2. Zero diagonal để bỏ dự đoán đúng.
    3. Với mỗi cặp i < j, tính confusion_count = cm[i][j] + cm[j][i].
    4. Sắp xếp giảm dần và lấy top 8-10 cặp tùy `top_pair_count`.
    5. Tính recall từng lớp = TP / (TP + FN).
    6. Chọn weak_classes theo rule recall < 0.6 mặc định.
    """

    y_true = np.asarray(y_true).reshape(-1).astype(np.int32)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int32)
    class_names = _normalize_class_names(class_names)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true/y_pred khong duoc rong.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("So luong phan tu cua y_true va y_pred khong khop nhau.")
    if top_pair_count <= 0:
        raise ValueError("top_pair_count phai lon hon 0.")
    if not 0.0 <= weak_recall_threshold <= 1.0:
        raise ValueError("weak_recall_threshold phai nam trong [0, 1].")

    num_classes = len(class_names)
    min_label = int(min(y_true.min(), y_pred.min()))
    max_label = int(max(y_true.max(), y_pred.max()))
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            "Co label nam ngoai khoang class_names. "
            f"min_label={min_label}, max_label={max_label}, num_classes={num_classes}"
        )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
    ).astype(np.int64)
    return build_confusion_pair_summary_from_matrix(
        confusion=confusion,
        class_names=class_names,
        top_pair_count=top_pair_count,
        weak_recall_threshold=weak_recall_threshold,
    )


def build_confusion_pair_summary_from_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    top_pair_count: int = 10,
    weak_recall_threshold: float = 0.6,
) -> dict[str, Any]:
    """Tạo summary gọn từ confusion matrix đã có sẵn.

    Output có trường `confusion_pairs` dạng list[list[str]] đúng format yêu cầu,
    đồng thời có `confusion_pair_details` để xem count và chiều nhầm cụ thể.
    """

    class_names = _normalize_class_names(class_names)
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix phai la ma tran vuong.")
    if confusion.shape[0] != len(class_names):
        raise ValueError(
            "Kich thuoc confusion matrix khong khop class_names: "
            f"{confusion.shape[0]} != {len(class_names)}"
        )
    if np.any(confusion < 0):
        raise ValueError("confusion matrix khong duoc chua gia tri am.")
    if top_pair_count <= 0:
        raise ValueError("top_pair_count phai lon hon 0.")
    if not 0.0 <= weak_recall_threshold <= 1.0:
        raise ValueError("weak_recall_threshold phai nam trong [0, 1].")

    confusion_without_diagonal = confusion.copy()
    np.fill_diagonal(confusion_without_diagonal, 0)

    pair_candidates: list[dict[str, Any]] = []
    num_classes = len(class_names)
    for source_index in range(num_classes):
        for target_index in range(source_index + 1, num_classes):
            source_to_target = int(confusion_without_diagonal[source_index, target_index])
            target_to_source = int(confusion_without_diagonal[target_index, source_index])
            confusion_count = source_to_target + target_to_source
            if confusion_count <= 0:
                continue

            pair_candidates.append(
                {
                    "pair": [class_names[source_index], class_names[target_index]],
                    "confusion_count": confusion_count,
                    "directional_counts": {
                        f"{class_names[source_index]}->{class_names[target_index]}": source_to_target,
                        f"{class_names[target_index]}->{class_names[source_index]}": target_to_source,
                    },
                }
            )

    pair_candidates.sort(
        key=lambda item: (
            item["confusion_count"],
            max(item["directional_counts"].values()),
            item["pair"][0],
            item["pair"][1],
        ),
        reverse=True,
    )
    top_pairs = pair_candidates[:top_pair_count]

    true_positives = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    recall_values = np.array(
        [safe_divide(tp, true_count) for tp, true_count in zip(true_positives, support)],
        dtype=np.float64,
    )
    per_class_recall = {
        class_name: float(recall_values[class_index])
        for class_index, class_name in enumerate(class_names)
    }
    weak_classes = [
        class_name
        for class_name, recall in per_class_recall.items()
        if recall < weak_recall_threshold
    ]

    return {
        "confusion_pairs": [item["pair"] for item in top_pairs],
        "weak_classes": weak_classes,
        "per_class_recall": per_class_recall,
        "confusion_pair_details": top_pairs,
        "rule": {
            "pair_score": "cm[i][j] + cm[j][i]",
            "diagonal_zeroed": True,
            "weak_class": f"recall < {weak_recall_threshold}",
            "top_pair_count": top_pair_count,
        },
    }


def build_error_analysis(
    confusion: np.ndarray,
    class_names: list[str],
    config: AnalyzeConfusionConfig,
    confusion_source: str,
) -> dict[str, Any]:
    """Tạo payload phân tích lỗi hoàn chỉnh để export ra JSON."""

    metric_arrays = compute_per_class_metrics(confusion)
    weak_classes: list[str] = []
    per_class_analysis: dict[str, Any] = {}
    confusion_pairs: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(class_names):
        precision = float(metric_arrays["precision"][class_index])
        recall = float(metric_arrays["recall"][class_index])
        f1_score = float(metric_arrays["f1_score"][class_index])
        support = int(metric_arrays["support"][class_index])
        predicted_count = int(metric_arrays["predicted_count"][class_index])
        tp = int(metric_arrays["true_positives"][class_index])

        confused_with, detailed_confusions = get_top_confusions_for_class(
            confusion=confusion,
            class_index=class_index,
            class_names=class_names,
            top_k=config.top_k_confusions,
        )

        is_weak_class = (
            recall < config.weak_recall_threshold or f1_score < config.weak_f1_threshold
        )
        if is_weak_class:
            weak_classes.append(class_name)

        per_class_analysis[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "predicted_count": predicted_count,
            "true_positives": tp,
            "weak_class": is_weak_class,
            "confused_with": confused_with,
            "top_confusions": detailed_confusions,
        }

        for confusion_item in detailed_confusions:
            confusion_pairs.append(
                {
                    "source_class": class_name,
                    "target_class": confusion_item["class_name"],
                    "count": int(confusion_item["count"]),
                    "rate_within_source": float(confusion_item["rate_within_class"]),
                }
            )

    confusion_pairs.sort(
        key=lambda item: (item["count"], item["rate_within_source"]),
        reverse=True,
    )

    total_samples = int(confusion.sum())
    overall_accuracy = safe_divide(float(np.trace(confusion)), float(total_samples))

    return {
        "generated_at": datetime.now().isoformat(),
        "confusion_source": confusion_source,
        "num_classes": len(class_names),
        "total_samples": total_samples,
        "overall_accuracy": overall_accuracy,
        "weak_class_rule": {
            "recall_lt": config.weak_recall_threshold,
            "f1_lt": config.weak_f1_threshold,
        },
        "weak_classes": weak_classes,
        "per_class": per_class_analysis,
        "confusion_pairs": confusion_pairs,
    }


def save_pair_summary_json(payload: dict[str, Any], output_path: Path) -> None:
    """Lưu JSON gọn gồm top confusion pairs, weak_classes và per-class recall."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_error_analysis_json(payload: dict[str, Any], output_path: Path) -> None:
    """Lưu payload JSON phục vụ downstream dashboard hoặc inspect thủ công."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_hard_classes_text(payload: dict[str, Any], output_path: Path) -> None:
    """Lưu danh sách lớp yếu ra text để đọc nhanh hoặc dùng trong vòng lặp cải thiện dữ liệu."""

    weak_classes = payload.get("weak_classes", [])
    per_class = payload.get("per_class", {})
    rule = payload.get("weak_class_rule", {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("Hard Classes Analysis\n")
        file.write("=====================\n")
        file.write(
            f"Rule: recall < {rule.get('recall_lt', 0.5)} OR "
            f"f1 < {rule.get('f1_lt', 0.6)}\n\n"
        )

        if not weak_classes:
            file.write("Khong tim thay lop yeu theo nguong hien tai.\n")
            return

        for class_name in weak_classes:
            class_info = per_class[class_name]
            file.write(
                f"{class_name} | precision={class_info['precision']:.4f} | "
                f"recall={class_info['recall']:.4f} | f1={class_info['f1_score']:.4f} | "
                f"support={class_info['support']}\n"
            )
            top_confusions = class_info.get("top_confusions", [])
            if top_confusions:
                formatted_confusions = ", ".join(
                    [
                        (
                            f"{item['class_name']} "
                            f"(count={item['count']}, rate={item['rate_within_class']:.4f})"
                        )
                        for item in top_confusions
                    ]
                )
                file.write(f"  confused_with: {formatted_confusions}\n")
            else:
                file.write("  confused_with: none\n")


def main() -> None:
    """Entry point chính của script phân tích confusion matrix."""

    args = parse_args()
    config = build_runtime_config(args)
    validate_config(config)
    logger = setup_logger(config.runtime_log_path)

    logger.info("Bat dau phan tich confusion matrix voi config: %s", config)
    class_names = load_class_names(config)
    logger.info("Da nap %d class_names.", len(class_names))

    confusion, confusion_source = load_or_generate_confusion_matrix(
        config=config,
        class_names=class_names,
        logger=logger,
    )
    logger.info("Confusion matrix shape: %s", confusion.shape)

    payload = build_error_analysis(
        confusion=confusion,
        class_names=class_names,
        config=config,
        confusion_source=confusion_source,
    )
    pair_summary = build_confusion_pair_summary_from_matrix(
        confusion=confusion,
        class_names=class_names,
        top_pair_count=config.top_pair_count,
        weak_recall_threshold=config.weak_recall_threshold,
    )
    save_error_analysis_json(payload, config.error_analysis_path)
    save_pair_summary_json(pair_summary, config.pair_summary_path)
    save_hard_classes_text(payload, config.hard_classes_path)

    logger.info("So luong lop yeu: %d", len(payload["weak_classes"]))
    logger.info("Da luu error_analysis.json tai: %s", config.error_analysis_path)
    logger.info("Da luu confusion_pairs_summary.json tai: %s", config.pair_summary_path)
    logger.info("Da luu hard_classes.txt tai: %s", config.hard_classes_path)


if __name__ == "__main__":
    main()
