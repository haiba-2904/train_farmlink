from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


STAGE_B_SUPPORTED_PREFIX = "stage_b_supported_v1_resnet50_"
MOBILENET_HISTORY_NAME = "train_history.json"


@dataclass(frozen=True)
class ModelRun:
    """Thông tin một lần train dùng để vẽ biểu đồ và so sánh."""

    display_name: str
    model_type: str
    run_dir: Path
    history_path: Path
    test_results_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path | None
    history: dict[str, Any]
    metrics: dict[str, float]
    notes: str


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI."""

    parser = argparse.ArgumentParser(
        description="Build training result charts and comparison charts for current ResNet50 and best old MobileNetV2."
    )
    parser.add_argument(
        "--current-experiment",
        type=Path,
        default=None,
        help="Experiment ResNet50 currently used. Default: latest stage_b_supported_v1_resnet50_*.",
    )
    parser.add_argument(
        "--mobilenet-run",
        type=Path,
        default=None,
        help="Folder log MobileNetV2 old run. Default: auto-select best macro F1 from archive logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder output report/charts. Default: reports/model_training_comparison_<timestamp>.",
    )
    return parser.parse_args()


def find_latest_current_experiment(root: Path = Path("experiments")) -> Path:
    """Tìm experiment Stage B supported v1 mới nhất đang dùng."""

    candidates = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(STAGE_B_SUPPORTED_PREFIX)
        ],
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Khong tim thay experiment {STAGE_B_SUPPORTED_PREFIX}* trong {root}"
        )
    return candidates[-1]


def read_json(path: Path) -> dict[str, Any]:
    """Đọc JSON và báo lỗi rõ ràng."""

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_key_value_metrics(path: Path) -> dict[str, float]:
    """Parse test_results.txt dạng `key: value`."""

    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def parse_classification_report(path: Path) -> dict[str, float]:
    """Parse các dòng accuracy/macro avg/weighted avg trong classification_report.txt."""

    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("accuracy"):
            parts = line.split()
            if len(parts) >= 2:
                metrics["test_accuracy"] = float(parts[1])
            continue

        if line.startswith("macro avg"):
            parts = line.split()
            if len(parts) >= 5:
                metrics["macro_precision"] = float(parts[2])
                metrics["macro_recall"] = float(parts[3])
                metrics["macro_f1"] = float(parts[4])
            continue

        if line.startswith("weighted avg"):
            parts = line.split()
            if len(parts) >= 5:
                metrics["weighted_precision"] = float(parts[2])
                metrics["weighted_recall"] = float(parts[3])
                metrics["weighted_f1"] = float(parts[4])
            continue

    return metrics


def normalize_metrics(
    test_results_path: Path,
    classification_report_path: Path,
) -> dict[str, float]:
    """Gộp metric từ test_results.txt và classification_report.txt.

    Một số run cũ chỉ lưu accuracy trong test_results.txt nhưng macro F1 nằm
    trong classification_report.txt, nên cần đọc cả hai nguồn.
    """

    report_metrics = parse_classification_report(classification_report_path)
    test_metrics = parse_key_value_metrics(test_results_path)
    metrics = dict(report_metrics)
    metrics.update(test_metrics)

    if "test_accuracy_reference_only" in metrics:
        metrics["test_accuracy"] = metrics["test_accuracy_reference_only"]
    if "accuracy" in metrics and "test_accuracy" not in metrics:
        metrics["test_accuracy"] = metrics["accuracy"]
    return metrics


def is_mobilenet_history(path: Path) -> bool:
    """Kiểm tra một history cũ có thuộc MobileNetV2 hay không."""

    try:
        history = read_json(path)
    except Exception:  # noqa: BLE001
        return False

    config = history.get("config", {})
    final_model_path = str(config.get("final_model_path", "")).lower()
    label_manifest_path = str(config.get("label_manifest_path", "")).lower()
    input_shape = config.get("input_shape")
    return (
        "mobilenet" in final_model_path
        or "mobilenet" in label_manifest_path
        or input_shape == [224, 224, 3]
    )


def find_best_mobilenet_run(
    archive_root: Path = Path("archive_unused"),
) -> Path:
    """Tự chọn MobileNetV2 tốt nhất theo macro F1, fallback theo accuracy."""

    candidates: list[tuple[float, Path, dict[str, float]]] = []
    for history_path in sorted(archive_root.rglob(MOBILENET_HISTORY_NAME)):
        if not is_mobilenet_history(history_path):
            continue
        run_dir = history_path.parent
        metrics = normalize_metrics(
            test_results_path=run_dir / "test_results.txt",
            classification_report_path=run_dir / "classification_report.txt",
        )
        score = metrics.get("macro_f1", metrics.get("test_accuracy", -1.0))
        candidates.append((float(score), run_dir, metrics))

    if not candidates:
        raise FileNotFoundError(
            "Khong tim thay run MobileNetV2 cu co train_history.json trong archive_unused."
        )

    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def load_current_resnet_run(run_dir: Path) -> ModelRun:
    """Load experiment ResNet50 supported v1 đang dùng."""

    history_path = run_dir / "history.json"
    test_results_path = run_dir / "test_results.txt"
    classification_report_path = run_dir / "classification_report.txt"
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    return ModelRun(
        display_name="ResNet50 supported v1",
        model_type="resnet50",
        run_dir=run_dir,
        history_path=history_path,
        test_results_path=test_results_path,
        classification_report_path=classification_report_path,
        confusion_matrix_path=confusion_matrix_path if confusion_matrix_path.exists() else None,
        history=read_json(history_path),
        metrics=normalize_metrics(test_results_path, classification_report_path),
        notes="Model hien tai dang dung cho Stage B production v1, 32 supported classes, khong co class other.",
    )


def load_mobilenet_run(run_dir: Path) -> ModelRun:
    """Load run MobileNetV2 cũ."""

    history_path = run_dir / "train_history.json"
    test_results_path = run_dir / "test_results.txt"
    classification_report_path = run_dir / "classification_report.txt"
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    return ModelRun(
        display_name="MobileNetV2 best old run",
        model_type="mobilenetv2",
        run_dir=run_dir,
        history_path=history_path,
        test_results_path=test_results_path,
        classification_report_path=classification_report_path,
        confusion_matrix_path=confusion_matrix_path if confusion_matrix_path.exists() else None,
        history=read_json(history_path),
        metrics=normalize_metrics(test_results_path, classification_report_path),
        notes="Run MobileNetV2 cu tot nhat tim thay trong archive, dataset/class taxonomy khac pipeline supported v1 hien tai.",
    )


def stage_names(history: dict[str, Any]) -> list[str]:
    """Lấy thứ tự stage train trong history."""

    preferred_order = ["stage1", "stage2", "hard_mining"]
    names = [name for name in preferred_order if name in history]
    for name in history:
        if name != "config" and name not in names and isinstance(history[name], dict):
            names.append(name)
    return names


def flatten_history(history: dict[str, Any]) -> dict[str, list[Any]]:
    """Ghép stage1/stage2/... thành chuỗi epoch liên tục để vẽ line chart."""

    flat: dict[str, list[Any]] = {
        "epoch": [],
        "stage": [],
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }
    epoch_index = 1
    for stage_name in stage_names(history):
        stage_history = history.get(stage_name, {})
        losses = stage_history.get("loss", [])
        for idx in range(len(losses)):
            flat["epoch"].append(epoch_index)
            flat["stage"].append(stage_name)
            for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
                values = stage_history.get(key, [])
                flat[key].append(values[idx] if idx < len(values) else None)
            epoch_index += 1
    return flat


def stage_boundaries(history: dict[str, Any]) -> list[tuple[int, str]]:
    """Tính vị trí ranh giới giữa các stage để vẽ đường dọc."""

    boundaries: list[tuple[int, str]] = []
    total_epochs = 0
    names = stage_names(history)
    for stage_name in names[:-1]:
        stage_history = history.get(stage_name, {})
        total_epochs += len(stage_history.get("loss", []))
        boundaries.append((total_epochs, stage_name))
    return boundaries


def plot_training_curves(run: ModelRun, output_path: Path) -> None:
    """Vẽ accuracy/loss train-val theo epoch cho một model."""

    flat = flatten_history(run.history)
    if not flat["epoch"]:
        raise ValueError(f"History rong, khong the ve chart: {run.history_path}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    fig.suptitle(f"Training Curves - {run.display_name}", fontsize=15, fontweight="bold")

    axes[0].plot(flat["epoch"], flat["accuracy"], label="train accuracy", linewidth=2)
    axes[0].plot(flat["epoch"], flat["val_accuracy"], label="val accuracy", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(flat["epoch"], flat["loss"], label="train loss", linewidth=2)
    axes[1].plot(flat["epoch"], flat["val_loss"], label="val loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for boundary_epoch, stage_name in stage_boundaries(run.history):
        for axis in axes:
            axis.axvline(boundary_epoch + 0.5, color="gray", linestyle="--", alpha=0.45)
            axis.text(
                boundary_epoch + 0.65,
                axis.get_ylim()[1] * 0.92,
                f"after {stage_name}",
                fontsize=8,
                color="gray",
            )

    best_val_acc = max(value for value in flat["val_accuracy"] if value is not None)
    min_val_loss = min(value for value in flat["val_loss"] if value is not None)
    fig.text(
        0.5,
        0.01,
        f"Best val accuracy: {best_val_acc:.4f} | Min val loss: {min_val_loss:.4f}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_overlay(runs: list[ModelRun], output_path: Path) -> None:
    """Vẽ overlay val_accuracy và val_loss của hai model."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Validation Curve Comparison", fontsize=15, fontweight="bold")

    for run in runs:
        flat = flatten_history(run.history)
        axes[0].plot(
            flat["epoch"],
            flat["val_accuracy"],
            label=f"{run.display_name}",
            linewidth=2.4,
        )
        axes[1].plot(
            flat["epoch"],
            flat["val_loss"],
            label=f"{run.display_name}",
            linewidth=2.4,
        )

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(runs: list[ModelRun], output_path: Path) -> None:
    """Vẽ bar chart so sánh metric test chính."""

    metric_keys = [
        ("test_accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    x_positions = list(range(len(metric_keys)))
    width = 0.36

    fig, axis = plt.subplots(figsize=(13.5, 6.2))
    for run_index, run in enumerate(runs):
        values = [run.metrics.get(key, 0.0) for key, _ in metric_keys]
        offsets = [position + (run_index - 0.5) * width for position in x_positions]
        bars = axis.bar(offsets, values, width=width, label=run.display_name)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_title("Test Metrics Comparison", fontsize=15, fontweight="bold")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([label for _, label in metric_keys], rotation=0)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_validation_summary(runs: list[ModelRun], output_path: Path) -> None:
    """Vẽ bar chart best val accuracy và min val loss."""

    labels = [run.display_name for run in runs]
    best_val_acc = []
    min_val_loss = []
    for run in runs:
        flat = flatten_history(run.history)
        best_val_acc.append(max(value for value in flat["val_accuracy"] if value is not None))
        min_val_loss.append(min(value for value in flat["val_loss"] if value is not None))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bars_acc = axes[0].bar(labels, best_val_acc, color=["#3b82f6", "#10b981"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Best Validation Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars_acc, best_val_acc):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
        )

    bars_loss = axes[1].bar(labels, min_val_loss, color=["#6366f1", "#f97316"])
    axes[1].set_title("Minimum Validation Loss")
    axes[1].set_ylabel("Loss")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars_loss, min_val_loss):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def copy_confusion_matrix(run: ModelRun, output_dir: Path) -> Path | None:
    """Copy confusion matrix gốc nếu run có lưu ảnh."""

    if run.confusion_matrix_path is None:
        return None
    target = output_dir / f"confusion_matrix_{run.model_type}.png"
    shutil.copy2(run.confusion_matrix_path, target)
    return target


def metric_summary_rows(runs: list[ModelRun]) -> list[dict[str, Any]]:
    """Tạo rows cho metrics_summary.csv."""

    rows = []
    for run in runs:
        flat = flatten_history(run.history)
        rows.append(
            {
                "model": run.display_name,
                "model_type": run.model_type,
                "run_dir": str(run.run_dir),
                "test_accuracy": run.metrics.get("test_accuracy", ""),
                "macro_precision": run.metrics.get("macro_precision", ""),
                "macro_recall": run.metrics.get("macro_recall", ""),
                "macro_f1": run.metrics.get("macro_f1", ""),
                "weighted_f1": run.metrics.get("weighted_f1", ""),
                "best_val_accuracy": max(
                    value for value in flat["val_accuracy"] if value is not None
                ),
                "min_val_loss": min(value for value in flat["val_loss"] if value is not None),
                "total_epochs": len(flat["epoch"]),
                "notes": run.notes,
            }
        )
    return rows


def write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Ghi metric summary CSV."""

    fieldnames = [
        "model",
        "model_type",
        "run_dir",
        "test_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "best_val_accuracy",
        "min_val_loss",
        "total_epochs",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    runs: list[ModelRun],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Ghi report markdown ngắn để đưa vào báo cáo tốt nghiệp."""

    current, mobilenet = runs
    current_row, mobilenet_row = rows
    macro_f1_gain = current_row["macro_f1"] - mobilenet_row["macro_f1"]
    accuracy_gain = current_row["test_accuracy"] - mobilenet_row["test_accuracy"]
    report = f"""# So sánh kết quả huấn luyện model

Generated at: {datetime.now().isoformat(timespec="seconds")}

## Model được so sánh

| Model | Run | Ghi chú |
|---|---|---|
| {current.display_name} | `{current.run_dir}` | {current.notes} |
| {mobilenet.display_name} | `{mobilenet.run_dir}` | {mobilenet.notes} |

## Metric chính trên test set

| Metric | ResNet50 supported v1 | MobileNetV2 best old run | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | {current_row["test_accuracy"]:.4f} | {mobilenet_row["test_accuracy"]:.4f} | {accuracy_gain:+.4f} |
| Macro Precision | {current_row["macro_precision"]:.4f} | {mobilenet_row["macro_precision"]:.4f} | {current_row["macro_precision"] - mobilenet_row["macro_precision"]:+.4f} |
| Macro Recall | {current_row["macro_recall"]:.4f} | {mobilenet_row["macro_recall"]:.4f} | {current_row["macro_recall"] - mobilenet_row["macro_recall"]:+.4f} |
| Macro F1 | {current_row["macro_f1"]:.4f} | {mobilenet_row["macro_f1"]:.4f} | {macro_f1_gain:+.4f} |
| Weighted F1 | {current_row["weighted_f1"]:.4f} | {mobilenet_row["weighted_f1"]:.4f} | {current_row["weighted_f1"] - mobilenet_row["weighted_f1"]:+.4f} |

## Kết luận nhanh

- ResNet50 supported v1 có Macro F1 cao hơn MobileNetV2 cũ khoảng `{macro_f1_gain:+.4f}`.
- ResNet50 supported v1 có Accuracy cao hơn MobileNetV2 cũ khoảng `{accuracy_gain:+.4f}`.
- Lưu ý: hai model không hoàn toàn cùng taxonomy. MobileNetV2 cũ train trên 45 class gồm `other` và taxonomy cũ; ResNet50 supported v1 train trên 32 supported classes, không gồm `other` và đã loại unsupported classes.

## File biểu đồ

- `training_curves_resnet50_supported_v1.png`
- `training_curves_mobilenetv2_best_old_run.png`
- `comparison_test_metrics.png`
- `comparison_validation_curves.png`
- `comparison_best_validation_summary.png`
- `confusion_matrix_resnet50.png`
- `confusion_matrix_mobilenetv2.png`
"""
    (output_dir / "model_comparison_report.md").write_text(report, encoding="utf-8")


def build_output_dir(output_dir: Path | None) -> Path:
    """Tạo thư mục output."""

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / f"model_training_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    """Entry point."""

    args = parse_args()
    output_dir = build_output_dir(args.output_dir)

    current_experiment = args.current_experiment or find_latest_current_experiment()
    mobilenet_run_dir = args.mobilenet_run or find_best_mobilenet_run()

    current_run = load_current_resnet_run(current_experiment)
    mobilenet_run = load_mobilenet_run(mobilenet_run_dir)
    runs = [current_run, mobilenet_run]

    plot_training_curves(
        current_run,
        output_dir / "training_curves_resnet50_supported_v1.png",
    )
    plot_training_curves(
        mobilenet_run,
        output_dir / "training_curves_mobilenetv2_best_old_run.png",
    )
    plot_metric_comparison(runs, output_dir / "comparison_test_metrics.png")
    plot_validation_overlay(runs, output_dir / "comparison_validation_curves.png")
    plot_best_validation_summary(runs, output_dir / "comparison_best_validation_summary.png")
    copy_confusion_matrix(current_run, output_dir)
    copy_confusion_matrix(mobilenet_run, output_dir)

    rows = metric_summary_rows(runs)
    write_metrics_csv(rows, output_dir / "metrics_summary.csv")
    write_report(runs, rows, output_dir)

    print(f"Output: {output_dir}")
    for row in rows:
        print(
            f"{row['model']}: accuracy={row['test_accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, best_val_acc={row['best_val_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
