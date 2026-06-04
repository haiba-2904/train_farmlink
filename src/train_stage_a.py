from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

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
    from src.dataloader import extract_clean_label, load_binary_datasets
    from src.model.binary import (
        build_resnet50_binary_classifier,
        compile_binary_model,
        set_fine_tuning_binary_resnet,
    )
except ImportError:  # pragma: no cover
    from dataloader import extract_clean_label, load_binary_datasets
    from model.binary import (
        build_resnet50_binary_classifier,
        compile_binary_model,
        set_fine_tuning_binary_resnet,
    )


@dataclass(frozen=True)
class StageATrainingConfig:
    """Cấu hình train Stage A: binary gate `other` vs `fruit`.

    Stage A là lớp bảo vệ đầu vào cho website: nếu ảnh upload không phải nông
    sản, router trả về `other` ngay và không đẩy ảnh đó vào Stage B 43 classes.
    """

    dataset_root: Path = Path("dataset")
    experiment_root: Path = Path("experiments")
    model_type: str = "resnet50"
    image_size: tuple[int, int] = (320, 320)
    input_shape: tuple[int, int, int] = (320, 320, 3)
    batch_size: int = 32
    seed: int = 42
    stage1_epochs: int = 15
    stage2_epochs: int = 15
    stage1_learning_rate: float = 3e-4
    stage2_learning_rate: float = 1e-5
    fine_tune_last_layers: int = 50
    dropout_rate: float = 0.5
    head_units: int = 256
    early_stopping_patience: int = 5
    shuffle_buffer_size: int = 1000
    threshold_start: float = 0.30
    threshold_end: float = 0.80
    threshold_step: float = 0.01
    min_fruit_recall: float = 0.90
    min_other_recall: float = 0.85
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Log learning rate mỗi epoch để debug CosineDecay/optimizer dễ hơn."""

    def __init__(self, logger: logging.Logger, stage_name: str) -> None:
        super().__init__()
        self.logger = logger
        self.stage_name = stage_name

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        optimizer = self.model.optimizer
        learning_rate = optimizer.learning_rate
        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = float(learning_rate(optimizer.iterations).numpy())
        else:
            current_lr = float(tf.keras.backend.get_value(learning_rate))
        logs = logs or {}
        self.logger.info(
            "[%s] epoch=%d | loss=%.6f | accuracy=%.6f | val_loss=%.6f | val_accuracy=%.6f | lr=%.8f",
            self.stage_name,
            epoch + 1,
            float(logs.get("loss", float("nan"))),
            float(logs.get("accuracy", float("nan"))),
            float(logs.get("val_loss", float("nan"))),
            float(logs.get("val_accuracy", float("nan"))),
            current_lr,
        )


def parse_args() -> argparse.Namespace:
    """Đọc cấu hình train từ command line.

    Mặc định script dùng đúng dataset chuẩn:
    - `dataset/train`
    - `dataset/val`
    - `dataset/test`

    Không có tham số chọn model khác vì Stage A chính thức chỉ dùng ResNet50.
    """

    parser = argparse.ArgumentParser(description="Train rieng Stage A: fruit vs other.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--image-size", type=int, default=320, choices=(320, 384))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=15)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-last-layers", type=int, default=50)
    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-end", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-fruit-recall", type=float, default=0.90)
    parser.add_argument("--min-other-recall", type=float, default=0.85)
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_a_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def create_experiment_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = root / f"stage_a_resnet50_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def set_reproducibility(seed: int, logger: logging.Logger) -> None:
    """Cố định seed và bật deterministic ops khi TensorFlow hỗ trợ."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        logger.info("Da bat TensorFlow deterministic ops.")
    except Exception as error:  # pragma: no cover
        logger.warning("Khong bat duoc deterministic ops: %s", error)


def dataset_cardinality(dataset: tf.data.Dataset, name: str) -> int:
    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality <= 0:
        raise ValueError(f"Dataset {name} rong hoac cardinality khong hop le.")
    return cardinality


def count_binary_samples(train_dir: Path, valid_extensions: tuple[str, ...]) -> dict[int, int]:
    """Đếm binary label vật lý trong train để tính class_weight đúng."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    counts = {0: 0, 1: 0}
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        clean_label = extract_clean_label(class_dir.name)
        binary_label = 0 if clean_label == "other" else 1
        counts[binary_label] += sum(
            1
            for image_path in class_dir.iterdir()
            if image_path.is_file()
            and not image_path.name.startswith(".")
            and image_path.suffix.lower() in valid_extension_set
        )
    if counts[0] <= 0 or counts[1] <= 0:
        raise ValueError(f"Stage A can du ca 2 lop other/fruit, counts={counts}")
    return counts


def compute_binary_class_weight(counts: dict[int, int]) -> dict[int, float]:
    """Bù mất cân bằng vì `fruit` là tổng của nhiều class, thường nhiều hơn `other`."""

    total = counts[0] + counts[1]
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }


def build_callbacks(
    checkpoint_path: Path,
    patience: int,
    logger: logging.Logger,
    stage_name: str,
) -> list[tf.keras.callbacks.Callback]:
    """Stage A chỉ dùng callback tương thích với CosineDecay, không ReduceLROnPlateau."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        LearningRateLogger(logger=logger, stage_name=stage_name),
    ]


def cosine_decay(
    initial_learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
) -> tf.keras.optimizers.schedules.CosineDecay:
    """CosineDecay chỉ dùng ở Stage 2 để fine-tune ổn định với LR thấp."""

    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=max(1, steps_per_epoch * epochs),
    )


def history_to_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {
        metric_name: [float(value) for value in metric_values]
        for metric_name, metric_values in history.history.items()
    }


def collect_binary_probabilities(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Thu y_true và fruit_probability từ dataset đã preprocess đúng contract."""

    probabilities = model.predict(dataset, verbose=0).reshape(-1).astype(np.float32)
    y_true_batches: list[np.ndarray] = []
    for _, labels in dataset:
        y_true_batches.append(labels.numpy().reshape(-1).astype(np.int32))
    return np.concatenate(y_true_batches), probabilities


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    fruit_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Tính đầy đủ metric nhị phân tại một threshold.

    Quy ước:
    - label 0 = `other`
    - label 1 = `fruit`
    - nếu `fruit_probability >= threshold` thì dự đoán là fruit
    - nếu thấp hơn threshold thì dự đoán là other

    Confusion matrix với labels=[0, 1]:
    - TN: ảnh other dự đoán đúng là other
    - FP: ảnh other bị nhầm thành fruit
    - FN: ảnh fruit bị nhầm thành other
    - TP: ảnh fruit dự đoán đúng là fruit
    """

    y_pred = (fruit_probabilities >= threshold).astype(np.int32)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) else 0.0
    other_precision = tn / (tn + fn) if (tn + fn) else 0.0
    other_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fruit_precision = tp / (tp + fp) if (tp + fp) else 0.0
    fruit_recall = tp / (tp + fn) if (tp + fn) else 0.0
    other_f1 = (
        2.0 * other_precision * other_recall / (other_precision + other_recall)
        if (other_precision + other_recall)
        else 0.0
    )
    fruit_f1 = (
        2.0 * fruit_precision * fruit_recall / (fruit_precision + fruit_recall)
        if (fruit_precision + fruit_recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "other_precision": float(other_precision),
        "other_recall": float(other_recall),
        "other_f1": float(other_f1),
        "fruit_precision": float(fruit_precision),
        "fruit_recall": float(fruit_recall),
        "fruit_f1": float(fruit_f1),
        "accuracy": float(accuracy),
        "macro_f1": float((other_f1 + fruit_f1) / 2.0),
        "balanced_accuracy": float((other_recall + fruit_recall) / 2.0),
        "min_recall": float(min(other_recall, fruit_recall)),
        "other_misclassified_as_fruit": int(fp),
        "fruit_misclassified_as_other": int(fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_fruit_threshold(
    y_true: np.ndarray,
    fruit_probabilities: np.ndarray,
    config: StageATrainingConfig,
) -> tuple[float, dict[str, Any]]:
    """Tune threshold trên validation set, không dùng mặc định 0.5.

    Script thử các threshold từ 0.3 đến 0.8 theo `threshold_step`.

    Cách chọn:
    1. Ưu tiên threshold đạt cả `min_fruit_recall` và `min_other_recall`.
    2. Trong nhóm đạt điều kiện, chọn threshold có macro F1 cao nhất.
    3. Nếu không threshold nào đạt cả hai recall tối thiểu, fallback sang điểm
       cân bằng nhất theo macro F1, min recall và balanced accuracy.

    Lý do:
    - Fruit recall thấp làm bỏ sót nông sản thật, rất nguy hiểm cho Stage B.
    - Other recall thấp làm ảnh ngoài bị đẩy nhầm vào Stage B.
    - Macro F1 tránh chọn threshold chỉ tốt cho một phía.
    """

    if config.threshold_step <= 0:
        raise ValueError("threshold_step phai lon hon 0.")
    if config.threshold_start >= config.threshold_end:
        raise ValueError("threshold_start phai nho hon threshold_end.")

    thresholds = np.arange(
        config.threshold_start,
        config.threshold_end + config.threshold_step / 2.0,
        config.threshold_step,
    )
    threshold_table = [
        binary_metrics_at_threshold(y_true, fruit_probabilities, threshold)
        for threshold in thresholds
    ]
    feasible = [
        row
        for row in threshold_table
        if row["fruit_recall"] >= config.min_fruit_recall
        and row["other_recall"] >= config.min_other_recall
    ]
    search_space = feasible if feasible else threshold_table
    selected = max(
        search_space,
        key=lambda row: (
            row["macro_f1"],
            row["min_recall"],
            row["balanced_accuracy"],
            row["fruit_recall"],
            row["other_recall"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    top_candidates = sorted(
        threshold_table,
        key=lambda row: (row["macro_f1"], row["balanced_accuracy"]),
        reverse=True,
    )[:20]
    return selected["threshold"], {
        "mode": "auto",
        "selection_rule": (
            "maximize macro_f1 among thresholds meeting min_fruit_recall and min_other_recall"
            if feasible
            else "fallback maximize macro_f1 because both recall targets were not reachable"
        ),
        "threshold_range": {
            "start": config.threshold_start,
            "end": config.threshold_end,
            "step": config.threshold_step,
        },
        "min_fruit_recall": config.min_fruit_recall,
        "min_other_recall": config.min_other_recall,
        "selected": selected,
        "all_candidates": threshold_table,
        "top_candidates": top_candidates,
    }


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None or sns is None:
        logger.warning("Khong co matplotlib/seaborn, bo qua confusion_matrix.png")
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Stage A Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_test_results(
    output_path: Path,
    selected_threshold: float,
    metrics: dict[str, Any],
    matrix: np.ndarray,
) -> None:
    """Lưu kết quả test dạng text để đọc nhanh khi báo cáo.

    Các dòng quan trọng nhất:
    - `macro_f1`: metric chính vì Stage A có 2 nhóm lệch số lượng.
    - `fruit_recall`: khả năng không bỏ sót nông sản thật.
    - `other_recall`: khả năng chặn ảnh ngoài.
    - `fruit_misclassified_as_other`: số ảnh nông sản bị chặn nhầm.
    - `other_misclassified_as_fruit`: số ảnh ngoài bị lọt vào Stage B.
    """

    tn, fp, fn, tp = matrix.ravel()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"selected_fruit_threshold: {selected_threshold:.4f}\n")
        file.write(f"accuracy: {float(metrics['accuracy']):.6f}\n")
        file.write(f"macro_f1: {float(metrics['macro_f1']):.6f}\n")
        file.write(f"balanced_accuracy: {float(metrics['balanced_accuracy']):.6f}\n")
        file.write(f"other_precision: {float(metrics['other_precision']):.6f}\n")
        file.write(f"other_recall: {float(metrics['other_recall']):.6f}\n")
        file.write(f"other_f1: {float(metrics['other_f1']):.6f}\n")
        file.write(f"fruit_precision: {float(metrics['fruit_precision']):.6f}\n")
        file.write(f"fruit_recall: {float(metrics['fruit_recall']):.6f}\n")
        file.write(f"fruit_f1: {float(metrics['fruit_f1']):.6f}\n")
        file.write(f"true_other_pred_other: {int(tn)}\n")
        file.write(f"other_misclassified_as_fruit: {int(fp)}\n")
        file.write(f"fruit_misclassified_as_other: {int(fn)}\n")
        file.write(f"true_fruit_pred_fruit: {int(tp)}\n")


def run_training(config: StageATrainingConfig) -> Path:
    experiment_dir = create_experiment_dir(config.experiment_root)
    logger = setup_logger(experiment_dir / "train.log")
    set_reproducibility(config.seed, logger)
    save_json(experiment_dir / "config.json", asdict(config))
    logger.info("Bat dau train Stage A | config=%s", make_json_safe(asdict(config)))

    train_ds, val_ds, test_ds, class_names = load_binary_datasets(
        dataset_root=config.dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        model_type=config.model_type,
        shuffle_buffer_size=config.shuffle_buffer_size,
        data_cleaning_log_path=experiment_dir / "data_cleaning.log",
    )
    if class_names != ["other", "fruit"]:
        raise ValueError(f"Stage A class_names khong dung: {class_names}")

    train_steps = dataset_cardinality(train_ds, "stage_a_train")
    val_steps = dataset_cardinality(val_ds, "stage_a_val")
    test_steps = dataset_cardinality(test_ds, "stage_a_test")
    sample_counts = count_binary_samples(config.dataset_root / "train", config.valid_extensions)
    class_weight = compute_binary_class_weight(sample_counts)
    logger.info(
        "Dataset Stage A | train_steps=%d | val_steps=%d | test_steps=%d | sample_counts=%s | class_weight=%s",
        train_steps,
        val_steps,
        test_steps,
        sample_counts,
        class_weight,
    )

    model, base_model = build_resnet50_binary_classifier(
        input_shape=config.input_shape,
        dropout_rate=config.dropout_rate,
        head_units=config.head_units,
    )
    model_summary_lines: list[str] = []
    model.summary(print_fn=model_summary_lines.append)
    for summary_line in model_summary_lines:
        logger.info("%s", summary_line)

    # Stage 1: backbone frozen, chỉ train head để học boundary other/fruit nhanh và ổn định.
    base_model.trainable = False
    compile_binary_model(model, learning_rate=config.stage1_learning_rate)
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            checkpoint_path=experiment_dir / "checkpoints" / "stage1_best.keras",
            patience=config.early_stopping_patience,
            logger=logger,
            stage_name="stage1",
        ),
        verbose=1,
    )

    # Stage 2: fine-tune phần cuối ResNet50 với LR thấp; BatchNorm luôn frozen
    # trong `set_fine_tuning_binary_resnet` để tránh lệch moving statistics.
    set_fine_tuning_binary_resnet(
        base_model=base_model,
        fine_tune_last_layers=config.fine_tune_last_layers,
    )
    compile_binary_model(
        model,
        learning_rate=cosine_decay(
            initial_learning_rate=config.stage2_learning_rate,
            steps_per_epoch=train_steps,
            epochs=config.stage2_epochs,
        ),
    )
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            checkpoint_path=experiment_dir / "checkpoints" / "stage2_best.keras",
            patience=config.early_stopping_patience,
            logger=logger,
            stage_name="stage2",
        ),
        verbose=1,
    )

    save_json(
        experiment_dir / "history.json",
        {
            "stage1": history_to_dict(history_stage1),
            "stage2": history_to_dict(history_stage2),
        },
    )

    val_true, val_probabilities = collect_binary_probabilities(model, val_ds)
    selected_threshold, threshold_analysis = select_fruit_threshold(
        y_true=val_true,
        fruit_probabilities=val_probabilities,
        config=config,
    )
    save_json(experiment_dir / "threshold_analysis.json", threshold_analysis)
    logger.info(
        "Selected fruit_threshold=%.4f | detail=%s",
        selected_threshold,
        threshold_analysis["selected"],
    )

    test_true, test_probabilities = collect_binary_probabilities(model, test_ds)
    test_pred = (test_probabilities >= selected_threshold).astype(np.int32)
    test_metrics = binary_metrics_at_threshold(
        y_true=test_true,
        fruit_probabilities=test_probabilities,
        threshold=selected_threshold,
    )
    report = classification_report(
        test_true,
        test_pred,
        labels=[0, 1],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(test_true, test_pred, labels=[0, 1])
    (experiment_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_test_results(
        output_path=experiment_dir / "test_results.txt",
        selected_threshold=selected_threshold,
        metrics=test_metrics,
        matrix=matrix,
    )
    np.save(experiment_dir / "confusion_matrix.npy", matrix)
    np.savez_compressed(
        experiment_dir / "predictions_test.npz",
        y_true=test_true,
        fruit_probability=test_probabilities,
        y_pred=test_pred,
        fruit_threshold=np.array([selected_threshold], dtype=np.float32),
    )
    save_confusion_matrix_plot(
        matrix=matrix,
        class_names=class_names,
        output_path=experiment_dir / "confusion_matrix.png",
        logger=logger,
    )
    logger.info("Stage A classification report:\n%s", report)
    logger.info(
        (
            "Stage A test summary | threshold=%.4f | accuracy=%.6f | macro_f1=%.6f | "
            "fruit_recall=%.6f | other_recall=%.6f | "
            "fruit_misclassified_as_other=%d | other_misclassified_as_fruit=%d"
        ),
        selected_threshold,
        float(test_metrics["accuracy"]),
        float(test_metrics["macro_f1"]),
        float(test_metrics["fruit_recall"]),
        float(test_metrics["other_recall"]),
        int(test_metrics["fruit_misclassified_as_other"]),
        int(test_metrics["other_misclassified_as_fruit"]),
    )

    model_path = experiment_dir / "model.keras"
    labels_path = experiment_dir / "labels.json"
    model.save(model_path)
    save_json(
        labels_path,
        {
            "stage": "A",
            "task": "fruit_vs_other",
            "model_type": config.model_type,
            "class_names": class_names,
            "other_class_index": 0,
            "fruit_class_index": 1,
            "fruit_threshold": selected_threshold,
            "min_fruit_recall": config.min_fruit_recall,
            "min_other_recall": config.min_other_recall,
            "image_size": list(config.image_size),
            "input_shape": list(config.input_shape),
            "preprocess": {
                "normalization": "tf.keras.applications.resnet.preprocess_input",
                "normalization_location": "dataloader_and_router",
                "preserve_aspect_ratio": True,
                "padding": "black",
            },
        },
    )
    logger.info("Da luu Stage A model: %s", model_path)
    logger.info("Da luu Stage A labels manifest: %s", labels_path)
    return experiment_dir


def main() -> None:
    args = parse_args()
    image_size = (args.image_size, args.image_size)
    config = StageATrainingConfig(
        dataset_root=args.dataset_root,
        experiment_root=args.experiment_root,
        image_size=image_size,
        input_shape=(args.image_size, args.image_size, 3),
        batch_size=args.batch_size,
        seed=args.seed,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_learning_rate=args.stage2_learning_rate,
        fine_tune_last_layers=args.fine_tune_last_layers,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        min_fruit_recall=args.min_fruit_recall,
        min_other_recall=args.min_other_recall,
    )
    run_training(config)


if __name__ == "__main__":
    main()
