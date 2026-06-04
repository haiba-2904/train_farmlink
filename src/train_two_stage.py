from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    from src.analyze_confusion import build_confusion_pair_summary
    from src.dataloader import extract_clean_label, load_binary_datasets, load_fruit_datasets
    from src.model.binary import (
        build_resnet50_binary_classifier,
        compile_binary_model,
        set_fine_tuning_binary_resnet,
    )
    from src.model.mobilenetv2 import compile_model
    from src.model.model_factory import build_model
    from src.model.resnet50 import set_fine_tuning_resnet
except ImportError:  # pragma: no cover
    from analyze_confusion import build_confusion_pair_summary
    from dataloader import extract_clean_label, load_binary_datasets, load_fruit_datasets
    from model.binary import (
        build_resnet50_binary_classifier,
        compile_binary_model,
        set_fine_tuning_binary_resnet,
    )
    from model.mobilenetv2 import compile_model
    from model.model_factory import build_model
    from model.resnet50 import set_fine_tuning_resnet


@dataclass(frozen=True)
class TwoStageTrainingConfig:
    """Cấu hình train cho hệ thống 2-stage ResNet50."""

    dataset_root: Path = Path("dataset")
    stage_b_dataset_root: Path = Path("dataset_fruit_only")
    experiment_root: Path = Path("experiments")
    model_type: str = "resnet50"
    image_size: tuple[int, int] = (320, 320)
    input_shape: tuple[int, int, int] = (320, 320, 3)
    batch_size: int = 32
    seed: int = 42
    stage1_epochs: int = 15
    stage2_epochs: int = 25
    stage1_learning_rate: float = 3e-4
    stage2_learning_rate: float = 1e-5
    fine_tune_last_layers: int = 50
    early_stopping_patience: int = 5
    dropout_rate_stage_a: float = 0.5
    dropout_rate_stage_b: float = 0.5
    stage_a_head_units: int = 256
    stage_b_head_units: int = 512
    stage_b_expected_classes: int = 43
    stage_b_focal_alpha: float = 0.25
    stage_b_focal_gamma: float = 2.0
    stage_b_label_smoothing: float = 0.0
    shuffle_buffer_size: int = 1000
    fruit_threshold: float = 0.5
    stage_b_confidence_threshold: float = 0.6
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def parse_args() -> argparse.Namespace:
    """CLI để train hệ thống 2-stage trên macOS/local."""

    parser = argparse.ArgumentParser(description="Train two-stage fruit classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--stage-b-dataset-root",
        type=Path,
        default=Path("dataset_fruit_only"),
        help="Dataset fruit-only cho Stage B, khong chua class other.",
    )
    parser.add_argument(
        "--stage",
        choices=("full", "stage_a", "stage_b"),
        default="full",
        help="Chon full two-stage, chi Stage A, hoac chi Stage B.",
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--image-size", type=int, default=320, choices=(320, 384))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    """Logger vừa ghi file vừa in console cho từng experiment."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("two_stage_training")
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


def create_experiment_dir(root: Path) -> Path:
    """Tạo folder experiment riêng để không ghi đè model cũ."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"two_stage_{timestamp}_resnet50"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def make_json_safe(value: Any) -> Any:
    """Chuyển dataclass/path/numpy value sang JSON-safe."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_callbacks(checkpoint_path: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    """Callbacks tối giản cho baseline: EarlyStopping + ModelCheckpoint."""

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
    ]


def cosine_decay(initial_learning_rate: float, steps_per_epoch: int, epochs: int) -> tf.keras.optimizers.schedules.CosineDecay:
    """CosineDecay cho Stage 2 fine-tuning."""

    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=max(1, steps_per_epoch * epochs),
    )


def dataset_cardinality(dataset: tf.data.Dataset, name: str) -> int:
    """Lấy số batch để log và tạo LR schedule."""

    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality <= 0:
        raise ValueError(f"Dataset {name} dang rong hoac cardinality khong hop le.")
    return cardinality


def count_binary_samples(train_dir: Path, valid_extensions: tuple[str, ...]) -> dict[int, int]:
    """Đếm sample cho Stage A để tính class_weight binary."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    counts = {0: 0, 1: 0}
    for class_dir in sorted([path for path in train_dir.iterdir() if path.is_dir()]):
        clean_label = extract_clean_label(class_dir.name)
        binary_label = 0 if clean_label == "other" else 1
        counts[binary_label] += sum(
            1
            for file_path in class_dir.iterdir()
            if file_path.is_file()
            and not file_path.name.startswith(".")
            and file_path.suffix.lower() in valid_extension_set
        )
    if counts[0] <= 0 or counts[1] <= 0:
        raise ValueError(f"Binary dataset khong du ca 2 lop: {counts}")
    return counts


def compute_binary_class_weight(counts: dict[int, int]) -> dict[int, float]:
    """Tính class_weight cho Stage A vì `other` thường ít hơn fruit rất nhiều."""

    total = counts[0] + counts[1]
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }


def collect_binary_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Thu y_true/y_pred cho Stage A."""

    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    y_pred = (probabilities >= threshold).astype(np.int32)
    y_true_batches = []
    for _, labels in dataset:
        y_true_batches.append(labels.numpy().reshape(-1).astype(np.int32))
    return np.concatenate(y_true_batches), y_pred


def collect_multiclass_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Thu y_true/y_pred cho Stage B one-hot labels."""

    probabilities = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
    y_true_batches = []
    for _, labels in dataset:
        label_array = labels.numpy()
        if label_array.ndim == 2:
            label_array = np.argmax(label_array, axis=1)
        y_true_batches.append(label_array.astype(np.int32))
    return np.concatenate(y_true_batches), y_pred


def save_manifest(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """Lưu manifest để inference router dùng đúng label/preprocess contract."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, ensure_ascii=False, indent=2)


def train_stage_a(
    config: TwoStageTrainingConfig,
    experiment_dir: Path,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    """Train binary classifier `other` vs `fruit`."""

    train_ds, val_ds, test_ds, class_names = load_binary_datasets(
        dataset_root=config.dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        model_type=config.model_type,
        shuffle_buffer_size=config.shuffle_buffer_size,
        data_cleaning_log_path=experiment_dir / "stage_a_data_cleaning.log",
    )
    train_steps = dataset_cardinality(train_ds, "stage_a_train")
    logger.info("Stage A class_names=%s | train_steps=%d", class_names, train_steps)

    model, base_model = build_resnet50_binary_classifier(
        input_shape=config.input_shape,
        dropout_rate=config.dropout_rate_stage_a,
        head_units=config.stage_a_head_units,
    )
    class_weight = compute_binary_class_weight(
        count_binary_samples(config.dataset_root / "train", config.valid_extensions)
    )
    logger.info("Stage A class_weight=%s", class_weight)

    compile_binary_model(model, learning_rate=config.stage1_learning_rate)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            experiment_dir / "stage_a" / "checkpoints" / "stage1_best.keras",
            config.early_stopping_patience,
        ),
        verbose=1,
    )

    set_fine_tuning_binary_resnet(base_model, config.fine_tune_last_layers)
    compile_binary_model(
        model,
        learning_rate=cosine_decay(
            config.stage2_learning_rate,
            train_steps,
            config.stage2_epochs,
        ),
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            experiment_dir / "stage_a" / "checkpoints" / "stage2_best.keras",
            config.early_stopping_patience,
        ),
        verbose=1,
    )

    y_true, y_pred = collect_binary_predictions(model, test_ds, config.fruit_threshold)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (experiment_dir / "stage_a").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "stage_a" / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    logger.info("Stage A classification report:\n%s", report)

    model_path = experiment_dir / "stage_a" / "model.keras"
    labels_path = experiment_dir / "stage_a" / "labels.json"
    model.save(model_path)
    save_manifest(
        labels_path,
        {
            "stage": "A",
            "task": "fruit_vs_other",
            "model_type": config.model_type,
            "class_names": class_names,
            "fruit_class_index": 1,
            "other_class_index": 0,
            "fruit_threshold": config.fruit_threshold,
            "image_size": list(config.image_size),
            "input_shape": list(config.input_shape),
            "preprocess": {
                "normalization": "tf.keras.applications.resnet.preprocess_input",
                "normalization_location": "dataloader",
                "preserve_aspect_ratio": True,
                "padding": "black",
            },
        },
    )
    return model_path, labels_path


def train_stage_b(
    config: TwoStageTrainingConfig,
    experiment_dir: Path,
    logger: logging.Logger,
) -> tuple[Path, Path, list[str]]:
    """Train fruit-only multi-class classifier."""

    train_ds, val_ds, test_ds, fruit_class_names = load_fruit_datasets(
        dataset_root=config.stage_b_dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        one_hot_labels=True,
        model_type=config.model_type,
        shuffle_buffer_size=config.shuffle_buffer_size,
        data_cleaning_log_path=experiment_dir / "stage_b_data_cleaning.log",
    )
    if "other" in fruit_class_names:
        raise ValueError("Stage B khong duoc chua class 'other'.")
    if len(fruit_class_names) != config.stage_b_expected_classes:
        raise ValueError(
            "Stage B sai so class fruit-only: "
            f"expected={config.stage_b_expected_classes}, actual={len(fruit_class_names)}"
        )

    train_steps = dataset_cardinality(train_ds, "stage_b_train")
    logger.info(
        "Stage B dataset_root=%s | fruit_classes=%d | train_steps=%d | "
        "class_weight=disabled | hard_mining=disabled",
        config.stage_b_dataset_root,
        len(fruit_class_names),
        train_steps,
    )

    model, base_model = build_model(
        model_type=config.model_type,
        num_classes=len(fruit_class_names),
        input_shape=config.input_shape,
        dropout_rate=config.dropout_rate_stage_b,
        head_units=config.stage_b_head_units,
    )

    # Stage B dùng focal loss với one-hot labels để tập trung vào sample khó,
    # nhưng không dùng class_weight/hard mining trong baseline này để đo sạch
    # tác động của kiến trúc 2-stage fruit-only.
    compile_model(
        model,
        learning_rate=config.stage1_learning_rate,
        alpha=config.stage_b_focal_alpha,
        gamma=config.stage_b_focal_gamma,
        label_smoothing=config.stage_b_label_smoothing,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        callbacks=build_callbacks(
            experiment_dir / "stage_b" / "checkpoints" / "stage1_best.keras",
            config.early_stopping_patience,
        ),
        verbose=1,
    )

    set_fine_tuning_resnet(base_model, config.fine_tune_last_layers)
    compile_model(
        model,
        learning_rate=cosine_decay(
            config.stage2_learning_rate,
            train_steps,
            config.stage2_epochs,
        ),
        alpha=config.stage_b_focal_alpha,
        gamma=config.stage_b_focal_gamma,
        label_smoothing=config.stage_b_label_smoothing,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        callbacks=build_callbacks(
            experiment_dir / "stage_b" / "checkpoints" / "stage2_best.keras",
            config.early_stopping_patience,
        ),
        verbose=1,
    )

    y_true, y_pred = collect_multiclass_predictions(model, test_ds)
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(fruit_class_names)),
        target_names=fruit_class_names,
        digits=4,
        zero_division=0,
    )
    stage_b_dir = experiment_dir / "stage_b"
    stage_b_dir.mkdir(parents=True, exist_ok=True)
    (stage_b_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # Lưu evaluation arrays để các vòng cải thiện tiếp theo có thể phân tích
    # confusion pair chính xác, không cần đọc lại từ ảnh confusion_matrix.png.
    np.save(stage_b_dir / "y_true.npy", y_true.astype(np.int32))
    np.save(stage_b_dir / "y_pred.npy", y_pred.astype(np.int32))
    np.save(
        stage_b_dir / "confusion_matrix.npy",
        confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(fruit_class_names)),
        ).astype(np.int64),
    )

    pair_summary = build_confusion_pair_summary(
        y_true=y_true,
        y_pred=y_pred,
        class_names=fruit_class_names,
        top_pair_count=10,
        weak_recall_threshold=0.6,
    )
    (stage_b_dir / "confusion_pairs_summary.json").write_text(
        json.dumps(pair_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Stage B classification report:\n%s", report)

    model_path = stage_b_dir / "model.keras"
    labels_path = stage_b_dir / "labels.json"
    model.save(model_path)
    save_manifest(
        labels_path,
        {
            "stage": "B",
            "task": "fruit_multiclass",
            "model_type": config.model_type,
            "class_names": fruit_class_names,
            "image_size": list(config.image_size),
            "input_shape": list(config.input_shape),
            "confidence_threshold": config.stage_b_confidence_threshold,
            "preprocess": {
                "normalization": "tf.keras.applications.resnet.preprocess_input",
                "normalization_location": "dataloader",
                "preserve_aspect_ratio": True,
                "padding": "black",
            },
        },
    )
    return model_path, labels_path, fruit_class_names


def main() -> None:
    """Train đầy đủ hệ thống 2-stage."""

    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    config = TwoStageTrainingConfig(
        dataset_root=args.dataset_root,
        stage_b_dataset_root=args.stage_b_dataset_root,
        experiment_root=args.experiment_root,
        image_size=(args.image_size, args.image_size),
        input_shape=(args.image_size, args.image_size, 3),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    experiment_dir = create_experiment_dir(config.experiment_root)
    logger = setup_logger(experiment_dir / "train_two_stage.log")
    save_manifest(
        experiment_dir / "config.json",
        {
            **asdict(config),
            "run_stage": args.stage,
        },
    )
    logger.info(
        "Bat dau train two-stage system | stage=%s | config=%s",
        args.stage,
        asdict(config),
    )

    stage_a_model_path: Path | None = None
    stage_a_labels_path: Path | None = None
    stage_b_model_path: Path | None = None
    stage_b_labels_path: Path | None = None
    fruit_class_names: list[str] = []

    if args.stage in ("full", "stage_a"):
        stage_a_model_path, stage_a_labels_path = train_stage_a(
            config,
            experiment_dir,
            logger,
        )

    if args.stage in ("full", "stage_b"):
        stage_b_model_path, stage_b_labels_path, fruit_class_names = train_stage_b(
            config,
            experiment_dir,
            logger,
        )

    if args.stage == "full":
        # Stage B chỉ output fruit classes; system-level output vẫn là 44 class vì
        # Stage A route trực tiếp `other`, còn Stage B xử lý các lớp còn lại.
        save_manifest(
            experiment_dir / "router_manifest.json",
            {
                "system": "two_stage_fruit_classifier",
                "routing_logic": "if stageA fruit_probability < fruit_threshold: return other; else return stageB(image)",
                "stage_a_model_path": stage_a_model_path,
                "stage_a_label_manifest_path": stage_a_labels_path,
                "stage_b_model_path": stage_b_model_path,
                "stage_b_label_manifest_path": stage_b_labels_path,
                "system_class_names": ["other", *fruit_class_names],
                "fruit_threshold": config.fruit_threshold,
                "stage_b_confidence_threshold": config.stage_b_confidence_threshold,
            },
        )
    elif args.stage == "stage_b":
        save_manifest(
            experiment_dir / "stage_b_manifest.json",
            {
                "system": "stage_b_fruit_only_classifier",
                "model_path": stage_b_model_path,
                "label_manifest_path": stage_b_labels_path,
                "class_names": fruit_class_names,
                "confidence_threshold": config.stage_b_confidence_threshold,
            },
        )

    logger.info("Da luu two-stage artifacts tai: %s", experiment_dir)


if __name__ == "__main__":
    main()
