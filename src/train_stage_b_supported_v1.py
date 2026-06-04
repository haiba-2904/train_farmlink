from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_config")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _reexec_with_project_venv_if_needed() -> None:
    """Tự chuyển sang `.venv/bin/python` trước khi import TensorFlow.

    Trên máy hiện tại, `python` có thể trỏ tới Anaconda base. TensorFlow/macOS
    Metal dễ bị segmentation fault nếu chạy sai interpreter. Hàm này chạy trước
    mọi import nặng để đảm bảo lệnh:

        python src/train_stage_b_supported_v1.py --dataset-root dataset_fruit_supported_v1

    vẫn dùng đúng Python trong `.venv`.
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

    if os.environ.get("FARMLINK_STAGE_B_SUPPORTED_V1_REEXECED") == "1":
        return

    os.environ["FARMLINK_STAGE_B_SUPPORTED_V1_REEXECED"] = "1"
    print(
        f"Stage B supported v1: dang chuyen interpreter tu {sys.executable} sang {venv_python}",
        file=sys.stderr,
        flush=True,
    )
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv_if_needed()

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.dataloader import load_fruit_datasets
    from src.model.resnet50 import build_resnet50_classifier, set_fine_tuning_resnet
    from src.train_stage_b import (
        build_callbacks,
        build_confused_pairs_payload,
        build_weak_classes_payload,
        collect_predictions,
        compile_stage_b_model,
        dataset_cardinality,
        get_best_val_loss,
        history_to_dict,
        load_best_checkpoint,
        make_json_safe,
        run_pretrain_sanity_check,
        save_confusion_matrix_plot,
        save_json,
        save_test_results,
        set_reproducibility,
        setup_logger,
    )
except ImportError:  # pragma: no cover
    from dataloader import load_fruit_datasets
    from model.resnet50 import build_resnet50_classifier, set_fine_tuning_resnet
    from train_stage_b import (
        build_callbacks,
        build_confused_pairs_payload,
        build_weak_classes_payload,
        collect_predictions,
        compile_stage_b_model,
        dataset_cardinality,
        get_best_val_loss,
        history_to_dict,
        load_best_checkpoint,
        make_json_safe,
        run_pretrain_sanity_check,
        save_confusion_matrix_plot,
        save_json,
        save_test_results,
        set_reproducibility,
        setup_logger,
    )


OTHER_CLASS_NAME = "other"
UNSUPPORTED_CLASSES_V1: tuple[str, ...] = (
    "bell_pepper",
    "coffee",
    "lime",
    "longan",
    "mango",
    "gourd",
    "canistel",
    "burmese_grape",
)


@dataclass(frozen=True)
class StageBSupportedV1Config:
    """Config trung tâm cho Stage B production v1.

    Production v1 chỉ train các class đã được chọn là đủ tin cậy trong
    `dataset_fruit_supported_v1`. Script không train Stage A, không sửa dataset
    gốc và không tự thêm lại class unsupported.
    """

    dataset_root: Path = Path("dataset_fruit_supported_v1")
    experiment_root: Path = Path("experiments")
    model_type: str = "resnet50"
    image_size: tuple[int, int] = (320, 320)
    input_shape: tuple[int, int, int] = (320, 320, 3)
    batch_size: int = 16
    seed: int = 42
    stage1_epochs: int = 12
    stage2_epochs: int = 20
    stage1_learning_rate: float = 3e-4
    stage2_learning_rate: float = 1e-5
    fine_tune_last_layers: int = 50
    dropout_rate: float = 0.5
    head_units: int = 512
    early_stopping_patience: int = 5
    shuffle_buffer_size: int = 1000
    resnet50_rotation_factor: float = 0.05
    resnet50_zoom_factor: float = 0.08
    sanity_loss_ceiling: float = 10.0
    weak_recall_threshold: float = 0.60
    use_class_weight: bool = False
    class_weight_clip_min: float = 0.8
    class_weight_clip_max: float = 1.2
    unsupported_classes: tuple[str, ...] = UNSUPPORTED_CLASSES_V1
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho production v1."""

    parser = argparse.ArgumentParser(
        description="Train Stage B production v1 ResNet50 supported classes."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_fruit_supported_v1"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-5)
    parser.add_argument("--fine-tune-last-layers", type=int, default=50)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--head-units", type=int, default=512)
    parser.add_argument("--resnet50-rotation-factor", type=float, default=0.05)
    parser.add_argument("--resnet50-zoom-factor", type=float, default=0.08)
    parser.add_argument("--sanity-loss-ceiling", type=float, default=10.0)
    parser.add_argument("--weak-recall-threshold", type=float, default=0.60)
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="Bat class_weight clipped [0.8, 1.2]. Mac dinh tat.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StageBSupportedV1Config:
    """Ghép CLI args thành config bất biến."""

    image_size = (args.image_size, args.image_size)
    return StageBSupportedV1Config(
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
        dropout_rate=args.dropout_rate,
        head_units=args.head_units,
        resnet50_rotation_factor=args.resnet50_rotation_factor,
        resnet50_zoom_factor=args.resnet50_zoom_factor,
        sanity_loss_ceiling=args.sanity_loss_ceiling,
        weak_recall_threshold=args.weak_recall_threshold,
        use_class_weight=args.use_class_weight,
    )


def create_experiment_dir(root: Path) -> Path:
    """Tạo experiment dir đúng chuẩn production v1."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = root / f"stage_b_supported_v1_resnet50_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def list_split_class_names(split_dir: Path) -> list[str]:
    """Lấy class folder của một split theo thứ tự ổn định."""

    if not split_dir.exists():
        raise FileNotFoundError(f"Khong tim thay split dir: {split_dir}")
    class_names = sorted(
        [path.name for path in split_dir.iterdir() if path.is_dir()],
        key=str.lower,
    )
    if not class_names:
        raise ValueError(f"Split dir khong co class folder: {split_dir}")
    return class_names


def read_supported_class_names(dataset_root: Path) -> list[str]:
    """Đọc class order từ `supported_classes_v1.txt`.

    File này được tạo bởi `build_supported_v1_dataset.py` và là nguồn class order
    chính thức cho train/evaluate/inference production v1. Nếu thiếu file này,
    script dừng lại để tránh train nhầm dataset không phải supported v1.
    """

    class_names_path = dataset_root / "supported_classes_v1.txt"
    if not class_names_path.exists():
        raise FileNotFoundError(
            "Khong tim thay supported_classes_v1.txt. "
            f"Hay build dataset_fruit_supported_v1 truoc: {class_names_path}"
        )

    class_names = [
        line.strip()
        for line in class_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"supported_classes_v1.txt rong: {class_names_path}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"supported_classes_v1.txt co class bi trung: {class_names_path}")
    return class_names


def count_images_in_class_dir(
    class_dir: Path,
    valid_extensions: tuple[str, ...],
) -> int:
    """Đếm ảnh hợp lệ trong một class folder."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    return sum(
        1
        for file_path in class_dir.iterdir()
        if file_path.is_file()
        and not file_path.name.startswith(".")
        and file_path.suffix.lower() in valid_extension_set
    )


def validate_supported_v1_dataset(
    config: StageBSupportedV1Config,
    logger: logging.Logger,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Validate dataset production v1 trước khi train.

    Điều kiện bắt buộc:
    - train/val/test có cùng class list.
    - không có `other`.
    - không có class unsupported.
    - không có class rỗng, split rỗng.
    - không hardcode số class; số class lấy từ `supported_classes_v1.txt`.
    """

    class_names = read_supported_class_names(config.dataset_root)
    class_name_set = set(class_names)
    unsupported_leak = sorted(class_name_set & set(config.unsupported_classes))
    if unsupported_leak:
        raise ValueError(f"supported_classes_v1.txt van con unsupported class: {unsupported_leak}")
    if OTHER_CLASS_NAME in class_name_set:
        raise ValueError("Stage B supported v1 khong duoc co class other.")

    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = config.dataset_root / split_name
        current_class_names = list_split_class_names(split_dir)
        if current_class_names != class_names:
            raise ValueError(
                "Class list split khong khop supported_classes_v1.txt. "
                f"split={split_name}, expected={class_names}, actual={current_class_names}"
            )

        split_counts[split_name] = {}
        for class_name in current_class_names:
            if class_name in config.unsupported_classes:
                raise ValueError(f"Output supported v1 van co unsupported class: {class_name}")
            if class_name == OTHER_CLASS_NAME:
                raise ValueError("Output supported v1 van co class other.")

            image_count = count_images_in_class_dir(
                class_dir=split_dir / class_name,
                valid_extensions=config.valid_extensions,
            )
            if image_count <= 0:
                raise ValueError(f"Class rong: split={split_name}, class={class_name}")
            split_counts[split_name][class_name] = image_count

    logger.info(
        "Dataset supported v1 hop le | dataset_root=%s | num_classes=%d | class_names=%s",
        config.dataset_root,
        len(class_names),
        class_names,
    )
    logger.info("Supported v1 split_counts=%s", split_counts)
    return class_names, split_counts


def iter_image_files_in_class(
    class_dir: Path,
    valid_extensions: tuple[str, ...],
) -> list[Path]:
    """Liệt kê ảnh trong class folder để tính class_weight optional."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in valid_extension_set
        ],
        key=lambda path: path.name.lower(),
    )


def build_optional_class_weight(
    config: StageBSupportedV1Config,
    class_names: list[str],
    logger: logging.Logger,
) -> dict[int, float] | None:
    """Tính class_weight optional cho production v1.

    Mặc định production v1 không dùng class_weight để tránh model bị kéo quá
    mạnh về class ít mẫu và over-predict class đó. Khi bật `--use-class-weight`,
    weight vẫn bị clip trong [0.8, 1.2] để chỉ điều chỉnh nhẹ.
    """

    if not config.use_class_weight:
        logger.info("Class weight disabled | class_weight=None")
        return None

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    label_values: list[int] = []
    per_class_counts = {class_name: 0 for class_name in class_names}

    for class_name in class_names:
        class_dir = config.dataset_root / "train" / class_name
        image_files = iter_image_files_in_class(class_dir, config.valid_extensions)
        if not image_files:
            raise ValueError(f"Khong the tinh class_weight vi class train rong: {class_name}")
        class_index = class_to_index[class_name]
        per_class_counts[class_name] = len(image_files)
        label_values.extend([class_index] * len(image_files))

    classes = np.arange(len(class_names), dtype=np.int32)
    y_train = np.asarray(label_values, dtype=np.int32)
    raw_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    clipped_weight_values = np.clip(
        raw_weight_values,
        config.class_weight_clip_min,
        config.class_weight_clip_max,
    )
    class_weight = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, clipped_weight_values)
    }

    logger.info("Class weight enabled | per_class_counts=%s", per_class_counts)
    logger.info(
        "Class weight raw_by_name=%s",
        {
            class_name: float(raw_weight_values[index])
            for index, class_name in enumerate(class_names)
        },
    )
    logger.info(
        "Class weight clipped_by_name=%s",
        {
            class_name: class_weight[index]
            for index, class_name in enumerate(class_names)
        },
    )
    return class_weight


def save_class_weight_manifest(
    output_path: Path,
    class_names: list[str],
    class_weight: dict[int, float] | None,
    config: StageBSupportedV1Config,
) -> None:
    """Lưu class_weight policy để experiment audit được."""

    save_json(
        output_path,
        {
            "use_class_weight": config.use_class_weight,
            "clip_min": config.class_weight_clip_min,
            "clip_max": config.class_weight_clip_max,
            "class_weight_by_index": (
                {str(index): weight for index, weight in class_weight.items()}
                if class_weight is not None
                else None
            ),
            "class_weight_by_name": (
                {
                    class_name: class_weight[index]
                    for index, class_name in enumerate(class_names)
                }
                if class_weight is not None
                else None
            ),
        },
    )


def save_supported_label_manifest(
    output_path: Path,
    class_names: list[str],
    config: StageBSupportedV1Config,
) -> None:
    """Lưu labels.json cho inference production v1."""

    save_json(
        output_path,
        {
            "stage": "B",
            "variant": "supported_v1",
            "task": "fruit_multiclass_supported_v1",
            "model_name": "stage_b_supported_v1_resnet50_classifier",
            "model_type": config.model_type,
            "class_names": class_names,
            "class_to_index": {
                class_name: index for index, class_name in enumerate(class_names)
            },
            "index_to_class": {
                str(index): class_name for index, class_name in enumerate(class_names)
            },
            "num_classes": len(class_names),
            "unsupported_classes_v1": list(config.unsupported_classes),
            "image_size": list(config.image_size),
            "input_shape": list(config.input_shape),
            "preprocess": {
                "fix_exif_orientation": True,
                "convert_to_rgb": True,
                "preserve_aspect_ratio": True,
                "padding": "black",
                "normalization": "tf.keras.applications.resnet.preprocess_input",
                "normalization_location": "dataloader",
            },
            "serving": {
                "output_activation": "softmax",
                "note": (
                    "Model nay chi phan loai supported_classes_v1. Unsupported "
                    "classes can route sang manual_review hoac khong dua vao Stage B v1."
                ),
            },
        },
    )


def run_training(config: StageBSupportedV1Config) -> Path:
    """Điều phối train/evaluate/export Stage B production v1."""

    experiment_dir = create_experiment_dir(config.experiment_root)
    logger = setup_logger(experiment_dir / "train.log")
    set_reproducibility(config.seed, logger)
    save_json(experiment_dir / "config.json", asdict(config))
    logger.info("Bat dau train Stage B supported v1 | config=%s", make_json_safe(asdict(config)))

    expected_class_names, _ = validate_supported_v1_dataset(config, logger)

    train_ds, val_ds, test_ds, class_names = load_fruit_datasets(
        dataset_root=config.dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        one_hot_labels=False,
        cache_enabled=False,
        cache_root=None,
        model_type=config.model_type,
        shuffle_buffer_size=config.shuffle_buffer_size,
        resnet50_rotation_factor=config.resnet50_rotation_factor,
        resnet50_zoom_factor=config.resnet50_zoom_factor,
        data_cleaning_log_path=experiment_dir / "data_cleaning.log",
    )
    if class_names != expected_class_names:
        raise ValueError(
            "Class order cua dataloader khong khop supported_classes_v1.txt. "
            f"expected={expected_class_names}, actual={class_names}"
        )
    if OTHER_CLASS_NAME in class_names:
        raise ValueError("Dataloader Stage B supported v1 khong duoc co other.")
    leaked_unsupported = sorted(set(class_names) & set(config.unsupported_classes))
    if leaked_unsupported:
        raise ValueError(f"Dataloader Stage B supported v1 van co unsupported: {leaked_unsupported}")

    train_steps = dataset_cardinality(train_ds, "stage_b_supported_v1_train")
    val_steps = dataset_cardinality(val_ds, "stage_b_supported_v1_val")
    test_steps = dataset_cardinality(test_ds, "stage_b_supported_v1_test")
    logger.info(
        "Stage B supported v1 dataset | classes=%d | train_steps=%d | val_steps=%d | test_steps=%d",
        len(class_names),
        train_steps,
        val_steps,
        test_steps,
    )
    logger.info("Stage B supported v1 class_names=%s", class_names)
    logger.info(
        "Dataloader contract | integer labels | one_hot_labels=False | loss=sparse_ce | cache_enabled=False"
    )

    class_weight = build_optional_class_weight(config, class_names, logger)
    save_class_weight_manifest(
        output_path=experiment_dir / "class_weight.json",
        class_names=class_names,
        class_weight=class_weight,
        config=config,
    )

    model, base_model = build_resnet50_classifier(
        num_classes=len(class_names),
        input_shape=config.input_shape,
        dropout_rate=config.dropout_rate,
        head_units=config.head_units,
    )
    model_summary_lines: list[str] = []
    model.summary(print_fn=model_summary_lines.append)
    for summary_line in model_summary_lines:
        logger.info("%s", summary_line)

    stage1_checkpoint = experiment_dir / "checkpoints" / "stage1_best.keras"
    stage2_checkpoint = experiment_dir / "checkpoints" / "stage2_best.keras"

    # Stage 1: freeze ResNet50 backbone, chỉ train classifier head.
    base_model.trainable = False
    compile_stage_b_model(model, learning_rate=config.stage1_learning_rate)
    run_pretrain_sanity_check(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=len(class_names),
        loss_ceiling=config.sanity_loss_ceiling,
        logger=logger,
    )
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            checkpoint_path=stage1_checkpoint,
            logger=logger,
            stage_name="stage1",
            patience=config.early_stopping_patience,
        ),
        verbose=1,
    )
    stage1_best_val_loss = get_best_val_loss(history_stage1)
    logger.info("Stage 1 hoan tat | best_val_loss=%.6f", stage1_best_val_loss)

    # Stage 2: fine-tune các layer cuối của ResNet50.
    # Helper `set_fine_tuning_resnet` luôn freeze BatchNorm để tránh batch nhỏ
    # làm lệch moving mean/variance pretrained.
    set_fine_tuning_resnet(
        base_model=base_model,
        fine_tune_last_layers=config.fine_tune_last_layers,
    )
    compile_stage_b_model(model, learning_rate=config.stage2_learning_rate)
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            checkpoint_path=stage2_checkpoint,
            logger=logger,
            stage_name="stage2",
            patience=config.early_stopping_patience,
        ),
        verbose=1,
    )
    stage2_best_val_loss = get_best_val_loss(history_stage2)
    logger.info("Stage 2 hoan tat | best_val_loss=%.6f", stage2_best_val_loss)

    save_json(
        experiment_dir / "history.json",
        {
            "stage1": history_to_dict(history_stage1),
            "stage2": history_to_dict(history_stage2),
        },
    )

    best_model, selected_stage = load_best_checkpoint(
        stage1_path=stage1_checkpoint,
        stage1_val_loss=stage1_best_val_loss,
        stage2_path=stage2_checkpoint,
        stage2_val_loss=stage2_best_val_loss,
        logger=logger,
    )
    compile_stage_b_model(best_model, learning_rate=config.stage2_learning_rate)

    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    y_true, y_pred, probabilities = collect_predictions(
        model=best_model,
        dataset=test_ds,
        num_classes=len(class_names),
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    (experiment_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_test_results(
        output_path=experiment_dir / "test_results.txt",
        selected_stage=selected_stage,
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
        report_dict=report_dict,
    )
    save_confusion_matrix_plot(
        matrix=matrix,
        class_names=class_names,
        output_path=experiment_dir / "confusion_matrix.png",
        logger=logger,
    )
    save_json(
        experiment_dir / "weak_classes.json",
        build_weak_classes_payload(
            report_dict=report_dict,
            class_names=class_names,
            weak_recall_threshold=config.weak_recall_threshold,
        ),
    )
    save_json(
        experiment_dir / "confused_pairs.json",
        build_confused_pairs_payload(matrix=matrix, class_names=class_names, top_k=3),
    )

    np.save(experiment_dir / "y_true.npy", y_true.astype(np.int32))
    np.save(experiment_dir / "y_pred.npy", y_pred.astype(np.int32))
    np.savez_compressed(
        experiment_dir / "predictions_test.npz",
        y_true=y_true.astype(np.int32),
        y_pred=y_pred.astype(np.int32),
        probabilities=probabilities.astype(np.float32),
    )

    model_path = experiment_dir / "model.keras"
    labels_path = experiment_dir / "labels.json"
    best_model.save(model_path)
    save_supported_label_manifest(labels_path, class_names, config)

    macro_avg = report_dict.get("macro avg", {})
    logger.info("Stage B supported v1 classification report:\n%s", report_text)
    logger.info(
        (
            "Stage B supported v1 test summary | selected_stage=%s | "
            "test_accuracy_reference_only=%.6f | macro_recall=%.6f | macro_f1=%.6f"
        ),
        selected_stage,
        float(test_accuracy),
        float(macro_avg.get("recall", 0.0)),
        float(macro_avg.get("f1-score", 0.0)),
    )
    logger.info("Da luu model: %s", model_path)
    logger.info("Da luu labels: %s", labels_path)
    logger.info("Hoan tat Stage B supported v1 experiment: %s", experiment_dir)
    return experiment_dir


def main() -> None:
    """Entry point: `python src/train_stage_b_supported_v1.py --dataset-root dataset_fruit_supported_v1`."""

    args = parse_args()
    config = build_config(args)
    run_training(config)


if __name__ == "__main__":
    main()
