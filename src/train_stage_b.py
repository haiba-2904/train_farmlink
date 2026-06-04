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

    Lý do:
    - Trên máy hiện tại, lệnh `python` đang trỏ tới Anaconda base Python 3.13.
    - TensorFlow/macOS Metal rất dễ crash cứng ở bước import nếu sai version
      Python hoặc sai binary wheel. Crash kiểu này là segmentation fault nên
      Python không kịp raise exception.
    - Project đã có `.venv/bin/python` dùng Python 3.12 và import TensorFlow ổn.

    Hàm này chạy ở đầu file, trước `import tensorflow as tf`. Nhờ vậy người dùng
    vẫn có thể gõ `python src/train_stage_b.py`, script sẽ tự exec lại bằng venv
    đúng của project. Nếu muốn tắt cơ chế này để debug môi trường khác, set:

        FARMLINK_DISABLE_VENV_REEXEC=1
    """

    if os.environ.get("FARMLINK_DISABLE_VENV_REEXEC") == "1":
        return

    project_root = Path(__file__).resolve().parents[1]
    venv_root = project_root / ".venv"
    venv_python = project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable)
    if Path(sys.prefix).resolve() == venv_root.resolve():
        return

    if os.environ.get("FARMLINK_STAGE_B_REEXECED") == "1":
        return

    os.environ["FARMLINK_STAGE_B_REEXECED"] = "1"
    print(
        f"Stage B: dang chuyen interpreter tu {current_python} sang {venv_python}",
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None
    sns = None

try:
    from src.dataloader import load_fruit_datasets
    from src.model.resnet50 import build_resnet50_classifier, set_fine_tuning_resnet
except ImportError:  # pragma: no cover
    from dataloader import load_fruit_datasets
    from model.resnet50 import build_resnet50_classifier, set_fine_tuning_resnet


OLD_MERGED_STAGE_B_FOLDERS = frozenset(
    {
        "black_mulberry",
        "red_mulberry",
        "mullberry",
        "cempedak",
        "jackfruit",
        "bitter_gourd",
        "ridged_gourd",
    }
)
REQUIRED_MERGED_STAGE_B_FOLDERS = frozenset(
    {"mulberry", "jackfruit_cempedak", "gourd"}
)
OTHER_CLASS_NAME = "other"


@dataclass(frozen=True)
class StageBTrainingConfig:
    """Cấu hình trung tâm cho Stage B.

    Stage B chỉ phân loại các loại nông sản cụ thể. Class `other` đã được xử lý
    bởi Stage A, nên Stage B tuyệt đối không được nhìn thấy dữ liệu `other`.

    dataset_root:
        Folder `dataset_fruit_only` có đủ `train`, `val`, `test`.
    experiment_root:
        Folder chứa kết quả train/evaluate của từng lần chạy.
    weak_recall_threshold:
        Ngưỡng recall để đánh dấu class yếu. Nếu recall < 0.6 thì class đó cần
        kiểm tra lại data hoặc bổ sung dữ liệu.
    """

    dataset_root: Path = Path("dataset_fruit_only")
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
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


class EpochMetricsLogger(tf.keras.callbacks.Callback):
    """Callback log metric cuối mỗi epoch vào `train.log`.

    Không dùng print rải rác vì log file là nguồn audit chính của experiment.
    Log gồm loss/accuracy/val_loss/val_accuracy và learning rate để biết quá
    trình fine-tune có ổn định hay không.
    """

    def __init__(self, logger: logging.Logger, stage_name: str) -> None:
        super().__init__()
        self.logger = logger
        self.stage_name = stage_name

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs or {}
        learning_rate = self._get_learning_rate()
        self.logger.info(
            (
                "[%s] epoch=%d | loss=%.6f | accuracy=%.6f | "
                "val_loss=%.6f | val_accuracy=%.6f | lr=%.8f"
            ),
            self.stage_name,
            epoch + 1,
            float(logs.get("loss", float("nan"))),
            float(logs.get("accuracy", float("nan"))),
            float(logs.get("val_loss", float("nan"))),
            float(logs.get("val_accuracy", float("nan"))),
            learning_rate,
        )

    def _get_learning_rate(self) -> float:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return 0.0

        learning_rate = getattr(optimizer, "learning_rate", None)
        if learning_rate is None:
            return 0.0

        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            learning_rate = learning_rate(optimizer.iterations)

        try:
            return float(tf.keras.backend.get_value(learning_rate))
        except Exception:  # noqa: BLE001
            return 0.0


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI.

    Script có default path rõ ràng nhưng không hardcode trong logic train. Nếu
    cần chạy trên dataset khác, chỉ cần truyền `--dataset-root` hoặc
    `--experiment-root`.
    """

    parser = argparse.ArgumentParser(
        description="Train Stage B ResNet50 fruit-only multi-class classifier."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_fruit_only"))
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
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StageBTrainingConfig:
    """Ghép CLI args thành dataclass config bất biến."""

    image_size = (args.image_size, args.image_size)
    return StageBTrainingConfig(
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
    )


def setup_logger(log_path: Path) -> logging.Logger:
    """Tạo logger ghi đồng thời ra terminal và file `train.log`."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_b_training")
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
    """Chuyển Path/numpy/dataclass value thành JSON-safe."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Lưu JSON UTF-8 có indent để dễ đọc trong báo cáo."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def create_experiment_dir(root: Path) -> Path:
    """Tạo thư mục experiment Stage B theo timestamp."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = root / f"stage_b_resnet50_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def set_reproducibility(seed: int, logger: logging.Logger) -> None:
    """Cố định seed để giảm dao động giữa các lần train."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        logger.info("Da bat TensorFlow deterministic ops.")
    except Exception as error:  # pragma: no cover
        logger.warning("Khong bat duoc deterministic ops: %s", error)


def list_split_class_names(split_dir: Path) -> list[str]:
    """Lấy danh sách class folder trong một split theo thứ tự ổn định."""

    if not split_dir.exists():
        raise FileNotFoundError(f"Khong tim thay split_dir: {split_dir}")
    class_names = sorted(
        [path.name for path in split_dir.iterdir() if path.is_dir()],
        key=lambda value: value.lower(),
    )
    if not class_names:
        raise ValueError(f"Split_dir khong co class folder: {split_dir}")
    return class_names


def read_stage_b_class_names(dataset_root: Path) -> list[str]:
    """Đọc class order chính thức của Stage B từ `class_names.txt`.

    Stage B baseline cần dễ so sánh giữa các lần train, nên thứ tự nhãn phải có
    một nguồn duy nhất. File `dataset_fruit_only/class_names.txt` được tạo ở
    bước build dataset và được dùng làm class order chính thức cho train,
    evaluate và inference.
    """

    class_names_path = dataset_root / "class_names.txt"
    if not class_names_path.exists():
        raise FileNotFoundError(
            "Stage B can file class_names.txt de co class order on dinh: "
            f"{class_names_path}"
        )

    class_names = [
        line.strip()
        for line in class_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"class_names.txt rong: {class_names_path}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class_names.txt co class bi trung: {class_names_path}")
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


def validate_stage_b_taxonomy(class_names: list[str]) -> None:
    """Validate taxonomy Stage B sau khi đã gộp class.

    Không được còn folder cũ đã gộp vì sẽ làm model học lại nhãn sai. Đồng thời
    phải có các class mới để chứng minh taxonomy đã được rebuild đúng.
    """

    class_name_set = set(class_names)
    if OTHER_CLASS_NAME in class_name_set:
        raise ValueError("Stage B khong duoc chua class 'other'.")

    old_folders = sorted(class_name_set & OLD_MERGED_STAGE_B_FOLDERS)
    if old_folders:
        raise ValueError(
            "Stage B van con folder cu da merge, can rebuild dataset_fruit_only: "
            f"{old_folders}"
        )

    missing_required = sorted(REQUIRED_MERGED_STAGE_B_FOLDERS - class_name_set)
    if missing_required:
        raise ValueError(
            "Stage B thieu class moi sau taxonomy merge: "
            f"{missing_required}"
        )


def validate_fruit_only_dataset(
    config: StageBTrainingConfig,
    logger: logging.Logger,
) -> list[str]:
    """Validate dataset Stage B trước khi train.

    Điều kiện bắt buộc:
    - Có đủ `train`, `val`, `test`.
    - Không chứa class `other`.
    - Không còn folder cũ đã gộp taxonomy.
    - Có đủ folder mới sau merge taxonomy.
    - `train/val/test` có cùng class list và khớp `class_names.txt`.
    - Không có split/class rỗng.

    Hàm trả về `class_names` để tất cả bước sau dùng cùng class order chính thức,
    không hardcode số class.
    """

    class_names = read_stage_b_class_names(config.dataset_root)
    validate_stage_b_taxonomy(class_names)

    split_class_names: dict[str, list[str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = config.dataset_root / split_name
        current_class_names = list_split_class_names(split_dir)
        split_class_names[split_name] = current_class_names

        if current_class_names != class_names:
            raise ValueError(
                "Class list cua split khong khop class_names.txt. "
                f"split={split_name}, expected={class_names}, actual={current_class_names}"
            )

        split_counts[split_name] = {}
        for class_name in current_class_names:
            image_count = count_images_in_class_dir(
                class_dir=split_dir / class_name,
                valid_extensions=config.valid_extensions,
            )
            if image_count <= 0:
                raise ValueError(
                    f"Stage B co class rong: split={split_name}, class={class_name}"
                )
            split_counts[split_name][class_name] = image_count

    logger.info(
        "Dataset Stage B hop le | dataset_root=%s | num_classes=%d | class_names=%s",
        config.dataset_root,
        len(class_names),
        class_names,
    )
    logger.info("Stage B split_counts=%s", split_counts)
    return class_names


def dataset_cardinality(dataset: tf.data.Dataset, dataset_name: str) -> int:
    """Lấy số batch của tf.data.Dataset và fail nếu dataset rỗng."""

    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality <= 0:
        raise ValueError(f"Dataset '{dataset_name}' rong hoac cardinality khong hop le.")
    return cardinality


def iter_image_files_in_class(
    class_dir: Path,
    valid_extensions: tuple[str, ...],
) -> list[Path]:
    """Liệt kê ảnh hợp lệ trong một class folder.

    Hàm này chỉ dùng để tính class_weight từ split train. Không load ảnh vào RAM,
    chỉ đếm file theo extension để biết class nào ít mẫu hơn class khác.
    """

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    return sorted(
        [
            file_path
            for file_path in class_dir.iterdir()
            if file_path.is_file()
            and not file_path.name.startswith(".")
            and file_path.suffix.lower() in valid_extension_set
        ],
        key=lambda path: path.name.lower(),
    )


def compute_stage_b_class_weight(
    train_dir: Path,
    class_names: list[str],
    valid_extensions: tuple[str, ...],
    logger: logging.Logger,
) -> dict[int, float]:
    """Tính class_weight cho Stage B từ `dataset_fruit_only/train`.

    `class_names` là thứ tự nhãn chính thức đọc từ `class_names.txt`. Baseline
    mới không boost thủ công class yếu; chỉ dùng công thức balanced của sklearn
    để bù class imbalance ở mức nhẹ và dễ so sánh.
    """

    if not train_dir.exists():
        raise FileNotFoundError(f"Khong tim thay train_dir de tinh class_weight: {train_dir}")

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    label_values: list[int] = []
    per_class_counts = {class_name: 0 for class_name in class_names}

    for class_dir in sorted(
        [path for path in train_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    ):
        class_name = class_dir.name
        if class_name == OTHER_CLASS_NAME:
            raise ValueError(
                "Stage B khong duoc co class 'other' khi tinh class_weight. "
                f"Folder vi pham: {class_dir}"
            )
        if class_name not in class_to_index:
            raise ValueError(
                "Folder train khong khop class_names.txt cua Stage B. "
                f"folder={class_dir.name}"
            )

        image_files = iter_image_files_in_class(
            class_dir=class_dir,
            valid_extensions=valid_extensions,
        )
        class_index = class_to_index[class_name]
        per_class_counts[class_name] += len(image_files)
        label_values.extend([class_index] * len(image_files))

    missing_classes = [
        class_name
        for class_name, image_count in per_class_counts.items()
        if image_count <= 0
    ]
    if missing_classes:
        raise ValueError(
            "Khong the tinh class_weight vi co class train rong: "
            f"{missing_classes}"
        )

    y_train = np.asarray(label_values, dtype=np.int32)
    classes = np.arange(len(class_names), dtype=np.int32)
    weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weight_values)
    }

    logger.info("Stage B train per_class_counts=%s", per_class_counts)
    logger.info(
        "Stage B class_weight_by_index=%s",
        {str(index): weight for index, weight in class_weight.items()},
    )
    logger.info(
        "Stage B class_weight_by_name=%s",
        {
            class_name: class_weight[index]
            for index, class_name in enumerate(class_names)
        },
    )
    return class_weight


def validate_integer_label_batch(
    labels: tf.Tensor,
    dataset_name: str,
    num_classes: int,
) -> tuple[np.ndarray, int, int]:
    """Kiểm tra một batch label phải là integer và nằm trong range hợp lệ."""

    label_array = labels.numpy()
    if label_array.ndim != 1:
        raise ValueError(
            "Stage B dataloader phai tra label integer shape=(batch,), "
            f"nhung {dataset_name} dang co label_shape={label_array.shape}. "
            "Hay kiem tra one_hot_labels phai bang False."
        )

    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError(
            "Stage B label phai la integer dtype. "
            f"{dataset_name} dang co dtype={label_array.dtype}."
        )

    label_min = int(label_array.min())
    label_max = int(label_array.max())
    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            "Stage B label nam ngoai range [0, num_classes - 1]. "
            f"{dataset_name}: label_min={label_min}, label_max={label_max}, "
            f"num_classes={num_classes}"
        )

    return label_array.astype(np.int32), label_min, label_max


def log_dataset_batch_sanity(
    dataset: tf.data.Dataset,
    dataset_name: str,
    num_classes: int,
    logger: logging.Logger,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Lấy một batch từ dataset và log shape/range label trước khi train."""

    images, labels = next(iter(dataset))
    _, label_min, label_max = validate_integer_label_batch(
        labels=labels,
        dataset_name=dataset_name,
        num_classes=num_classes,
    )
    logger.info(
        (
            "Sanity %s | image_shape=%s | label_shape=%s | "
            "label_min=%d | label_max=%d | num_classes=%d"
        ),
        dataset_name,
        tuple(images.shape.as_list()),
        tuple(labels.shape.as_list()),
        label_min,
        label_max,
        num_classes,
    )
    return images, labels


def run_pretrain_sanity_check(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    num_classes: int,
    loss_ceiling: float,
    logger: logging.Logger,
) -> None:
    """Sanity check label/loss/model output trước khi gọi `model.fit()`.

    Nếu loss thử trên một batch quá cao, khả năng lớn là label mode, loss hoặc
    output softmax đang lệch nhau. Fail sớm giúp tránh train nhiều epoch vô ích.
    """

    train_images, train_labels = log_dataset_batch_sanity(
        dataset=train_ds,
        dataset_name="train",
        num_classes=num_classes,
        logger=logger,
    )
    log_dataset_batch_sanity(
        dataset=val_ds,
        dataset_name="val",
        num_classes=num_classes,
        logger=logger,
    )

    model_outputs = model(train_images, training=False)
    output_shape = tuple(model_outputs.shape.as_list())
    if model_outputs.shape.rank != 2 or model_outputs.shape[-1] != num_classes:
        raise ValueError(
            "Model output Stage B phai co shape=(batch, num_classes). "
            f"Nhan duoc output_shape={output_shape}, num_classes={num_classes}"
        )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    trial_loss = float(loss_fn(train_labels, model_outputs).numpy())
    logger.info(
        "Sanity model | output_shape=%s | trial_sparse_ce_loss=%.6f",
        output_shape,
        trial_loss,
    )
    if not np.isfinite(trial_loss):
        raise ValueError(f"Sanity loss khong finite: {trial_loss}")
    if trial_loss > loss_ceiling:
        raise ValueError(
            "Sanity loss qua cao, co the label/loss/model output dang sai. "
            f"trial_loss={trial_loss:.6f}, loss_ceiling={loss_ceiling:.6f}"
        )


def compile_stage_b_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    """Compile model Stage B.

    Loss dùng `SparseCategoricalCrossentropy` vì Stage B trả label integer
    dạng 0..num_classes-1, không dùng one-hot. Đây là cách ít rủi ro hơn cho
    multi-class khi muốn kiểm soát rõ label/loss/model output.
    Accuracy vẫn được log để tham khảo, nhưng metric chính khi kết luận là
    macro F1, macro recall và per-class recall/F1 sau evaluation.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def build_callbacks(
    checkpoint_path: Path,
    logger: logging.Logger,
    stage_name: str,
    patience: int,
) -> list[tf.keras.callbacks.Callback]:
    """Callback chuẩn cho từng stage train."""

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
        EpochMetricsLogger(logger=logger, stage_name=stage_name),
    ]


def history_to_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    """Chuyển Keras History sang dict JSON-safe."""

    return {
        metric_name: [float(value) for value in metric_values]
        for metric_name, metric_values in history.history.items()
    }


def get_best_val_loss(history: tf.keras.callbacks.History) -> float:
    """Lấy val_loss tốt nhất để chọn checkpoint cuối cùng."""

    val_losses = history.history.get("val_loss", [])
    if not val_losses:
        raise ValueError("Khong tim thay val_loss trong history.")
    return float(min(val_losses))


def load_best_checkpoint(
    stage1_path: Path,
    stage1_val_loss: float,
    stage2_path: Path,
    stage2_val_loss: float,
    logger: logging.Logger,
) -> tuple[tf.keras.Model, str]:
    """Chọn checkpoint tốt nhất giữa Stage 1 và Stage 2 theo val_loss."""

    if stage2_val_loss <= stage1_val_loss:
        logger.info(
            "Chon checkpoint Stage 2 | stage2_val_loss=%.6f <= stage1_val_loss=%.6f",
            stage2_val_loss,
            stage1_val_loss,
        )
        return tf.keras.models.load_model(str(stage2_path), compile=False), "stage2"

    logger.info(
        "Chon checkpoint Stage 1 | stage1_val_loss=%.6f < stage2_val_loss=%.6f",
        stage1_val_loss,
        stage2_val_loss,
    )
    return tf.keras.models.load_model(str(stage1_path), compile=False), "stage1"


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thu y_true, y_pred và softmax probabilities từ test dataset."""

    probabilities = model.predict(dataset, verbose=0)
    if probabilities.ndim != 2:
        raise ValueError("Output model phai la ma tran softmax shape=(N, num_classes).")
    if probabilities.shape[1] != num_classes:
        raise ValueError(
            "So class output model khong khop Stage B. "
            f"output_classes={probabilities.shape[1]}, num_classes={num_classes}"
        )

    y_pred = np.argmax(probabilities, axis=1).astype(np.int32)
    y_true_batches: list[np.ndarray] = []

    for _, labels in dataset:
        label_array, _, _ = validate_integer_label_batch(
            labels=labels,
            dataset_name="test",
            num_classes=num_classes,
        )
        y_true_batches.append(label_array.astype(np.int32))

    y_true = np.concatenate(y_true_batches, axis=0)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("So luong y_true va y_pred khong khop.")

    return y_true, y_pred, probabilities


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Lưu confusion matrix dạng ảnh PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plt is None or sns is None:
        logger.warning("Khong co matplotlib/seaborn, bo qua confusion_matrix.png")
        return

    figure_width = max(16, len(class_names) * 0.45)
    figure_height = max(14, len(class_names) * 0.40)
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
    plt.title("Stage B Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_weak_classes_payload(
    report_dict: dict[str, Any],
    class_names: list[str],
    weak_recall_threshold: float,
) -> dict[str, Any]:
    """Tìm class yếu theo recall.

    Class có recall < threshold nghĩa là ảnh thật của class đó thường bị model
    bỏ sót hoặc dự đoán sang class khác. Đây là danh sách cần soi lại data trước
    khi tăng kỹ thuật phức tạp.
    """

    weak_classes: list[dict[str, Any]] = []
    for class_name in class_names:
        class_report = report_dict.get(class_name, {})
        recall = float(class_report.get("recall", 0.0))
        precision = float(class_report.get("precision", 0.0))
        f1_score = float(class_report.get("f1-score", 0.0))
        support = int(class_report.get("support", 0))
        if recall < weak_recall_threshold:
            weak_classes.append(
                {
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "support": support,
                    "note": "Recall thap, can kiem tra lai data class nay.",
                }
            )

    weak_classes.sort(key=lambda item: (item["recall"], item["f1_score"], item["class_name"]))
    return {
        "weak_recall_threshold": weak_recall_threshold,
        "weak_class_count": len(weak_classes),
        "weak_classes": weak_classes,
    }


def build_confused_pairs_payload(
    matrix: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Lấy top class bị nhầm nhiều nhất cho từng class thật.

    Với mỗi hàng của confusion matrix:
    - hàng = class thật
    - cột = class model dự đoán
    - bỏ ô diagonal vì đó là dự đoán đúng
    - lấy top 3 cột có số nhầm cao nhất
    """

    payload: dict[str, list[dict[str, Any]]] = {}
    for true_index, class_name in enumerate(class_names):
        row = matrix[true_index].astype(np.int64).copy()
        total_true = int(row.sum())
        row[true_index] = 0

        confused_indices = [
            int(index)
            for index in np.argsort(row)[::-1]
            if int(row[index]) > 0
        ][:top_k]

        payload[class_name] = [
            {
                "predicted_class": class_names[pred_index],
                "count": int(row[pred_index]),
                "rate_in_true_class": (
                    float(row[pred_index] / total_true) if total_true > 0 else 0.0
                ),
            }
            for pred_index in confused_indices
        ]
    return payload


def save_test_results(
    output_path: Path,
    selected_stage: str,
    test_loss: float,
    test_accuracy: float,
    report_dict: dict[str, Any],
) -> None:
    """Lưu metric tổng hợp dạng text.

    Accuracy được lưu để tham khảo, nhưng macro F1 và macro recall mới là metric
    chính cho Stage B vì dataset có nhiều class và chất lượng từng class quan
    trọng hơn điểm đúng tổng thể.
    """

    macro_avg = report_dict.get("macro avg", {})
    weighted_avg = report_dict.get("weighted avg", {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"selected_stage: {selected_stage}\n")
        file.write(f"test_loss: {test_loss:.6f}\n")
        file.write(f"test_accuracy_reference_only: {test_accuracy:.6f}\n")
        file.write(f"macro_precision: {float(macro_avg.get('precision', 0.0)):.6f}\n")
        file.write(f"macro_recall: {float(macro_avg.get('recall', 0.0)):.6f}\n")
        file.write(f"macro_f1: {float(macro_avg.get('f1-score', 0.0)):.6f}\n")
        file.write(f"weighted_precision: {float(weighted_avg.get('precision', 0.0)):.6f}\n")
        file.write(f"weighted_recall: {float(weighted_avg.get('recall', 0.0)):.6f}\n")
        file.write(f"weighted_f1: {float(weighted_avg.get('f1-score', 0.0)):.6f}\n")


def save_label_manifest(
    output_path: Path,
    class_names: list[str],
    config: StageBTrainingConfig,
) -> None:
    """Lưu manifest nhãn cho inference Stage B."""

    save_json(
        output_path,
        {
            "stage": "B",
            "task": "fruit_multiclass",
            "model_name": "stage_b_resnet50_fruit_classifier",
            "model_type": config.model_type,
            "class_names": class_names,
            "class_to_index": {
                class_name: index for index, class_name in enumerate(class_names)
            },
            "index_to_class": {
                str(index): class_name for index, class_name in enumerate(class_names)
            },
            "num_classes": len(class_names),
            "image_size": list(config.image_size),
            "input_shape": list(config.input_shape),
            "preprocess": {
                "fix_exif_orientation": True,
                "convert_to_rgb": True,
                "preserve_aspect_ratio": True,
                "padding": "black",
                "normalization": "tf.keras.applications.resnet.preprocess_input",
                "normalization_location": "dataloader_and_router",
            },
            "serving": {
                "output_activation": "softmax",
                "note": "Stage B chi nhan anh da qua Stage A va duoc xac dinh la nong san.",
            },
        },
    )


def run_training(config: StageBTrainingConfig) -> Path:
    """Điều phối toàn bộ Stage B: load data -> train -> evaluate -> export."""

    experiment_dir = create_experiment_dir(config.experiment_root)
    logger = setup_logger(experiment_dir / "train.log")
    set_reproducibility(config.seed, logger)
    save_json(experiment_dir / "config.json", asdict(config))
    logger.info("Bat dau train Stage B | config=%s", make_json_safe(asdict(config)))

    expected_class_names = validate_fruit_only_dataset(config, logger)

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
            "Class order cua dataloader khong khop class_names.txt. "
            f"expected={expected_class_names}, actual={class_names}"
        )
    if OTHER_CLASS_NAME in class_names:
        raise ValueError("Stage B class_names khong duoc chua 'other'.")
    validate_stage_b_taxonomy(class_names)

    train_steps = dataset_cardinality(train_ds, "stage_b_train")
    val_steps = dataset_cardinality(val_ds, "stage_b_val")
    test_steps = dataset_cardinality(test_ds, "stage_b_test")
    logger.info(
        "Stage B dataset | classes=%d | train_steps=%d | val_steps=%d | test_steps=%d",
        len(class_names),
        train_steps,
        val_steps,
        test_steps,
    )
    logger.info("Stage B class_names=%s", class_names)
    logger.info(
        "Stage B dataloader contract | one_hot_labels=False | loss=sparse_ce | cache_enabled=False"
    )

    class_weight = compute_stage_b_class_weight(
        train_dir=config.dataset_root / "train",
        class_names=class_names,
        valid_extensions=config.valid_extensions,
        logger=logger,
    )
    save_json(
        experiment_dir / "class_weight.json",
        {
            "class_weight_by_index": {
                str(index): weight for index, weight in class_weight.items()
            },
            "class_weight_by_name": {
                class_name: class_weight[index]
                for index, class_name in enumerate(class_names)
            },
        },
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

    # Stage 1: freeze backbone, chỉ train classifier head.
    # Mục tiêu là để head học mapping từ feature ImageNet sang num_classes hiện tại.
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

    # Stage 2: mở các layer cuối của ResNet50 để fine-tune nhẹ theo domain nông sản.
    # BatchNorm vẫn bị freeze trong helper để tránh làm lệch thống kê pretrained.
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

    (experiment_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
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
        build_confused_pairs_payload(
            matrix=matrix,
            class_names=class_names,
            top_k=3,
        ),
    )

    np.save(experiment_dir / "y_true.npy", y_true.astype(np.int32))
    np.save(experiment_dir / "y_pred.npy", y_pred.astype(np.int32))
    np.save(experiment_dir / "confusion_matrix.npy", matrix.astype(np.int64))
    np.savez_compressed(
        experiment_dir / "predictions_test.npz",
        y_true=y_true.astype(np.int32),
        y_pred=y_pred.astype(np.int32),
        probabilities=probabilities.astype(np.float32),
    )

    model_path = experiment_dir / "model.keras"
    labels_path = experiment_dir / "labels.json"
    best_model.save(model_path)
    save_label_manifest(
        output_path=labels_path,
        class_names=class_names,
        config=config,
    )

    macro_avg = report_dict.get("macro avg", {})
    logger.info("Stage B classification report:\n%s", report_text)
    logger.info(
        (
            "Stage B test summary | selected_stage=%s | "
            "test_accuracy_reference_only=%.6f | macro_recall=%.6f | macro_f1=%.6f"
        ),
        selected_stage,
        float(test_accuracy),
        float(macro_avg.get("recall", 0.0)),
        float(macro_avg.get("f1-score", 0.0)),
    )
    logger.info("Da luu model: %s", model_path)
    logger.info("Da luu labels: %s", labels_path)
    logger.info("Hoan tat Stage B experiment: %s", experiment_dir)
    return experiment_dir


def main() -> None:
    """Entry point: `.venv/bin/python src/train_stage_b.py`."""

    args = parse_args()
    config = build_config(args)
    run_training(config)


if __name__ == "__main__":
    main()
