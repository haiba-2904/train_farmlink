from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from config import DEFAULT_CLASS_METADATA
    from utils import (
        HashRegistry,
        compute_blur_score,
        compute_perceptual_hash,
        open_image_safely,
        resize_with_padding,
        validate_image_quality,
    )
except ImportError:  # pragma: no cover
    from .config import DEFAULT_CLASS_METADATA
    from .utils import (
        HashRegistry,
        compute_blur_score,
        compute_perceptual_hash,
        open_image_safely,
        resize_with_padding,
        validate_image_quality,
    )


AUTOTUNE = tf.data.AUTOTUNE
UNKNOWN_CARDINALITY = tf.data.experimental.UNKNOWN_CARDINALITY
INFINITE_CARDINALITY = tf.data.experimental.INFINITE_CARDINALITY
SUPPORTED_MODEL_TYPES = ("mobilenetv2", "resnet50")
MODEL_DEFAULT_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "mobilenetv2": (224, 224),
    "resnet50": (320, 320),
}


def _normalize_label_token(value: str) -> str:
    """Chuẩn hóa token label từ tên folder hoặc alias trong config."""

    normalized_name = value.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name)
    normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")
    if normalized_name.endswith("_c"):
        normalized_name = normalized_name[:-2]
    return normalized_name


KNOWN_LABEL_ALIASES = {
    _normalize_label_token(raw_key): metadata.english_name
    for raw_key, metadata in DEFAULT_CLASS_METADATA.items()
}
KNOWN_LABEL_ALIASES.update(
    {
        _normalize_label_token(metadata.english_name): metadata.english_name
        for metadata in DEFAULT_CLASS_METADATA.values()
    }
)

# Taxonomy mới của project đã gộp một số class cho Stage B. Các label output
# mới này phải được nhận diện bằng exact match trước khi chạy logic match prefix
# theo metadata cũ. Nếu không, `jackfruit_cempedak` sẽ bị match nhầm thành
# `jackfruit` vì metadata cũ vẫn có alias `jackfruit`.
TAXONOMY_OUTPUT_LABEL_ALIASES = {
    _normalize_label_token("mulberry"): "mulberry",
    _normalize_label_token("jackfruit_cempedak"): "jackfruit_cempedak",
    _normalize_label_token("gourd"): "gourd",
}
KNOWN_LABEL_ALIASES.update(TAXONOMY_OUTPUT_LABEL_ALIASES)

KNOWN_ENGLISH_LABELS = tuple(
    sorted(
        {metadata.english_name for metadata in DEFAULT_CLASS_METADATA.values()},
        key=len,
        reverse=True,
    )
)
KNOWN_LABEL_ALIAS_KEYS = tuple(sorted(KNOWN_LABEL_ALIASES.keys(), key=len, reverse=True))


@dataclass(frozen=True)
class DataLoaderConfig:
    """Cấu hình trung tâm cho pipeline nạp dữ liệu ảnh."""

    dataset_root: Path = Path("dataset")
    model_type: str = "mobilenetv2"
    image_size: tuple[int, int] = MODEL_DEFAULT_IMAGE_SIZES["mobilenetv2"]
    batch_size: int = 32
    seed: int = 42
    label_mode: str = "int"
    one_hot_labels: bool = False
    color_mode: str = "rgb"
    train_dir_name: str = "train"
    val_dir_name: str = "val"
    test_dir_name: str = "test"
    cache_enabled: bool = True
    cache_root: Path | None = None
    shuffle_buffer_size: int = 1000
    resnet50_rotation_factor: float = 0.1
    resnet50_zoom_factor: float = 0.1
    drop_remainder: bool = False
    reshuffle_each_iteration: bool = True
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    weak_class_names: tuple[str, ...] = ()
    weak_classes_path: Path | None = None
    strict_validation: bool = False
    validate_split_overlap: bool = True
    min_image_side: int = 100
    max_image_side: int = 5000
    max_aspect_ratio: float = 4.0
    blur_threshold: float = 100.0
    duplicate_hash_tolerance: int = 0
    data_cleaning_log_path: Path = Path("logs/data_cleaning.log")


@dataclass(frozen=True)
class CleanedSplitData:
    """Danh sách file đã qua kiểm tra chất lượng cho một split."""

    split_name: str
    file_paths: tuple[Path, ...]
    labels: tuple[int, ...]
    raw_counts: dict[str, int]
    accepted_counts: dict[str, int]
    rejected_counts: dict[str, int]
    duplicate_counts: dict[str, int]
    content_hashes: set[str]
    latest_mtime_ns: int

    @property
    def total_raw(self) -> int:
        return sum(self.raw_counts.values())

    @property
    def total_accepted(self) -> int:
        return len(self.file_paths)

    @property
    def total_rejected(self) -> int:
        return sum(self.rejected_counts.values())

    @property
    def total_duplicates(self) -> int:
        return sum(self.duplicate_counts.values())


def _validate_config(config: DataLoaderConfig) -> None:
    """Kiểm tra cấu hình đầu vào để fail fast nếu có tham số bất thường."""

    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_type khong duoc ho tro: {config.model_type}. "
            f"Ho tro: {SUPPORTED_MODEL_TYPES}"
        )

    if config.batch_size <= 0:
        raise ValueError("batch_size phai lon hon 0.")

    if config.image_size[0] <= 0 or config.image_size[1] <= 0:
        raise ValueError("image_size phai lon hon 0 cho ca hai chieu.")

    if config.shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size phai lon hon 0.")

    if not 0.0 <= config.resnet50_rotation_factor < 1.0:
        raise ValueError("resnet50_rotation_factor phai nam trong khoang [0.0, 1.0).")

    if not 0.0 <= config.resnet50_zoom_factor < 1.0:
        raise ValueError("resnet50_zoom_factor phai nam trong khoang [0.0, 1.0).")

    if config.label_mode != "int":
        raise ValueError("Pipeline nay chi ho tro label_mode='int'.")

    if config.color_mode.lower() != "rgb":
        raise ValueError("Pipeline nay chi ho tro color_mode='rgb'.")

    if config.weak_classes_path is not None and not isinstance(config.weak_classes_path, Path):
        raise ValueError("weak_classes_path phai la pathlib.Path hoac None.")

    if config.min_image_side <= 0:
        raise ValueError("min_image_side phai lon hon 0.")

    if config.max_image_side <= config.min_image_side:
        raise ValueError("max_image_side phai lon hon min_image_side.")

    if config.max_aspect_ratio <= 1:
        raise ValueError("max_aspect_ratio phai lon hon 1.")

    if config.blur_threshold < 0:
        raise ValueError("blur_threshold khong duoc am.")

    if config.duplicate_hash_tolerance < 0:
        raise ValueError("duplicate_hash_tolerance khong duoc am.")


def _normalize_model_type(model_type: str) -> str:
    """Chuẩn hóa model_type để switch pipeline ổn định."""

    normalized_model_type = model_type.strip().lower()
    if normalized_model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_type khong duoc ho tro: {model_type}. Ho tro: {SUPPORTED_MODEL_TYPES}"
        )
    return normalized_model_type


def get_default_image_size_for_model(model_type: str) -> tuple[int, int]:
    """Trả về image_size mặc định theo từng backbone."""

    return MODEL_DEFAULT_IMAGE_SIZES[_normalize_model_type(model_type)]


def _is_strict_validation_enabled(config: DataLoaderConfig) -> bool:
    """ResNet50 luôn dùng strict loading; MobileNetV2 giữ tương thích mặc định."""

    return config.strict_validation or config.model_type == "resnet50"


def _setup_data_cleaning_logger(log_path: Path) -> logging.Logger:
    """Tạo logger riêng cho quá trình lọc dữ liệu trước khi train."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_cleaning")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _get_split_dir(config: DataLoaderConfig, split_name: str) -> Path:
    """Trả về đường dẫn thư mục của từng split."""

    return config.dataset_root / split_name


def _list_class_directories(split_dir: Path) -> list[str]:
    """Lấy danh sách class folder theo thứ tự ổn định."""

    if not split_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc split: {split_dir}")

    if not split_dir.is_dir():
        raise NotADirectoryError(f"Duong dan split khong phai thu muc: {split_dir}")

    class_names = sorted(
        [path.name for path in split_dir.iterdir() if path.is_dir()],
        key=lambda value: value.lower(),
    )
    if not class_names:
        raise ValueError(f"Khong tim thay class folder nao trong: {split_dir}")

    return class_names


def _normalize_folder_label(folder_name: str) -> str:
    """Chuẩn hóa tên folder về token label ổn định cho cả raw và metadata folder."""

    return _normalize_label_token(folder_name)


def extract_clean_label(folder_name: str) -> str:
    """Trích xuất clean label từ folder raw hoặc folder metadata một cách an toàn.

    Lưu ý:
    - Pipeline mới từ prepare_dataset.py giữ folder raw như 'ambarella',
      'bell_pepper', nên nếu folder không có metadata thì dùng chính tên đó.
    - Pipeline cũ dùng '<english>_<vietnamese>_<count>', ví dụ
      'banana_chuoi_726'. Với format này, hàm ưu tiên match prefix tiếng Anh
      dài nhất theo config để không làm hỏng lớp nhiều token như bell_pepper,
      black_mulberry, passion_fruit...
    - Hàm này ưu tiên match prefix dài nhất theo danh sách class chuẩn trong config.
    """

    normalized_name = _normalize_folder_label(folder_name)
    if not normalized_name:
        raise ValueError("folder_name khong duoc rong.")

    if normalized_name in KNOWN_LABEL_ALIASES:
        return KNOWN_LABEL_ALIASES[normalized_name]

    name_parts = normalized_name.split("_")

    # Bỏ token count ở cuối nếu đó là số lượng ảnh.
    stem_name = "_".join(name_parts[:-1]) if name_parts[-1].isdigit() else normalized_name
    for alias_key in KNOWN_LABEL_ALIAS_KEYS:
        if stem_name == alias_key or stem_name.startswith(f"{alias_key}_"):
            return KNOWN_LABEL_ALIASES[alias_key]

    # Fallback cho dataset tùy biến không có trong config:
    # - Folder raw không có hậu tố số lượng: dùng nguyên tên đã chuẩn hóa.
    # - Folder metadata lạ có hậu tố số lượng: giữ contract cũ bằng token đầu tiên.
    if not name_parts[-1].isdigit():
        return normalized_name
    return name_parts[0]


def _build_clean_class_names(raw_class_names: list[str]) -> list[str]:
    """Sinh class_names sạch và đảm bảo không có xung đột nhãn."""

    clean_class_names = [extract_clean_label(class_name) for class_name in raw_class_names]
    if len(set(clean_class_names)) != len(clean_class_names):
        raise ValueError(
            "Phat hien nhieu folder goc map ve cung mot clean label. "
            "Vui long kiem tra ten class trong dataset."
        )
    return clean_class_names


def _is_fruit_only_dataset_root(dataset_root: Path) -> bool:
    """Nhận diện dataset vật lý dành riêng cho Stage B.

    Dataloader vẫn giữ mặc định ``dataset/`` cho pipeline cũ. Khi caller truyền
    ``dataset_fruit_only/``, ta bật thêm các kiểm tra an toàn để chắc chắn Stage B
    không học nhãn ``other`` nữa.
    """

    return dataset_root.name == "dataset_fruit_only"


def _validate_fruit_only_class_names(
    raw_class_names: list[str],
    clean_class_names: list[str],
    dataset_root: Path,
) -> None:
    """Validate dataset Stage B không chứa class ``other``.

    Không hardcode số class ở đây: số lớp được suy ra động từ folder. Điều kiện
    quan trọng cho Stage B là không còn ``other`` và mỗi raw folder map đúng một
    clean label duy nhất.
    """

    other_like_raw_classes = [
        raw_class_name
        for raw_class_name, clean_class_name in zip(raw_class_names, clean_class_names)
        if clean_class_name == "other"
    ]
    if other_like_raw_classes:
        raise ValueError(
            "dataset_fruit_only khong duoc chua class 'other'. "
            f"dataset_root={dataset_root}, folders={other_like_raw_classes}"
        )

    if not clean_class_names:
        raise ValueError(f"dataset_fruit_only khong co class nao: {dataset_root}")


def _validate_split_class_alignment(
    train_classes: list[str],
    val_classes: list[str],
    test_classes: list[str],
) -> None:
    """Đảm bảo train/val/test có cùng danh sách lớp và cùng thứ tự."""

    if train_classes != val_classes:
        raise ValueError(
            "Danh sach class cua train va val khong khop nhau. "
            "Khong the tao dataloader an toan."
        )

    if train_classes != test_classes:
        raise ValueError(
            "Danh sach class cua train va test khong khop nhau. "
            "Khong the tao dataloader an toan."
        )


def _count_images_in_class_dir(
    class_dir: Path,
    valid_extensions: tuple[str, ...],
) -> tuple[int, int]:
    """Đếm số ảnh hợp lệ và lấy mốc thời gian mới nhất trong class."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    image_count = 0
    latest_mtime_ns = 0

    for file_path in class_dir.iterdir():
        if (
            file_path.is_file()
            and not file_path.name.startswith(".")
            and file_path.suffix.lower() in valid_extension_set
        ):
            image_count += 1
            latest_mtime_ns = max(latest_mtime_ns, file_path.stat().st_mtime_ns)

    return image_count, latest_mtime_ns


def _scan_split_metadata(
    split_dir: Path,
    class_names: list[str],
    valid_extensions: tuple[str, ...],
) -> tuple[dict[str, int], int, int]:
    """Quét số lượng ảnh từng lớp, tổng ảnh và mtime mới nhất của split."""

    per_class_counts: dict[str, int] = {}
    total_images = 0
    latest_mtime_ns = 0

    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Khong tim thay class dir: {class_dir}")

        image_count, class_latest_mtime_ns = _count_images_in_class_dir(
            class_dir=class_dir,
            valid_extensions=valid_extensions,
        )
        if image_count <= 0:
            raise ValueError(
                f"Class '{class_name}' trong split '{split_dir.name}' dang rong."
            )

        per_class_counts[class_name] = image_count
        total_images += image_count
        latest_mtime_ns = max(
            latest_mtime_ns,
            class_dir.stat().st_mtime_ns,
            class_latest_mtime_ns,
        )

    if total_images <= 0:
        raise ValueError(f"Split '{split_dir.name}' khong co anh hop le.")

    return per_class_counts, total_images, latest_mtime_ns


def _iter_image_files(
    class_dir: Path,
    valid_extensions: tuple[str, ...],
) -> list[Path]:
    """Liệt kê file ảnh trong một class theo thứ tự ổn định."""

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


def _compute_file_sha256(file_path: Path) -> str:
    """Tính hash nội dung file để phát hiện overlap thật giữa train/val/test."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_rejected_image(
    logger: logging.Logger,
    split_name: str,
    class_name: str,
    image_path: Path,
    reason: str,
    value: object | None = None,
) -> None:
    """Ghi log ảnh bị loại theo format dễ grep khi audit dataset."""

    logger.warning(
        "[%s/%s] rejected %s | reason=%s | value=%s",
        split_name,
        class_name,
        image_path.name,
        reason,
        "N/A" if value is None else value,
    )


def _validate_and_collect_split_files(
    split_name: str,
    raw_class_names: list[str],
    config: DataLoaderConfig,
    logger: logging.Logger,
) -> CleanedSplitData:
    """Quét, lọc chất lượng, loại duplicate và trả về file hợp lệ cho một split.

    ResNet50 sử dụng luồng này để đảm bảo ảnh vào training đã được:
    sửa EXIF orientation, chuyển RGB, kiểm tra kích thước/tỉ lệ/blur và loại
    duplicate theo perceptual hash trong phạm vi từng class.
    """

    split_dir = _get_split_dir(config, split_name)
    file_paths: list[Path] = []
    labels: list[int] = []
    raw_counts: dict[str, int] = {}
    accepted_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    content_hashes: set[str] = set()
    latest_mtime_ns = 0

    for class_index, class_name in enumerate(raw_class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Khong tim thay class dir: {class_dir}")

        image_files = _iter_image_files(class_dir, config.valid_extensions)
        raw_counts[class_name] = len(image_files)
        accepted_counts[class_name] = 0
        rejected_counts[class_name] = 0
        duplicate_counts[class_name] = 0
        hash_registry = HashRegistry(tolerance=config.duplicate_hash_tolerance)

        for image_path in image_files:
            latest_mtime_ns = max(latest_mtime_ns, image_path.stat().st_mtime_ns)

            try:
                image = open_image_safely(image_path)
            except Exception as exc:  # noqa: BLE001
                rejected_counts[class_name] += 1
                _log_rejected_image(
                    logger=logger,
                    split_name=split_name,
                    class_name=class_name,
                    image_path=image_path,
                    reason="corrupted_image",
                    value=str(exc),
                )
                continue

            is_valid_quality, quality_reason = validate_image_quality(
                image=image,
                min_image_side=config.min_image_side,
                max_image_side=config.max_image_side,
                max_aspect_ratio=config.max_aspect_ratio,
            )
            if not is_valid_quality:
                rejected_counts[class_name] += 1
                _log_rejected_image(
                    logger=logger,
                    split_name=split_name,
                    class_name=class_name,
                    image_path=image_path,
                    reason=quality_reason or "invalid_image",
                    value=f"size={image.size}",
                )
                continue

            blur_score = compute_blur_score(image)
            if blur_score < config.blur_threshold:
                rejected_counts[class_name] += 1
                _log_rejected_image(
                    logger=logger,
                    split_name=split_name,
                    class_name=class_name,
                    image_path=image_path,
                    reason="blurry_image",
                    value=f"laplacian_variance={blur_score:.4f}",
                )
                continue

            standardized_image = resize_with_padding(
                image=image,
                target_size=(config.image_size[1], config.image_size[0]),
                background_color=(0, 0, 0),
            )
            perceptual_hash = compute_perceptual_hash(standardized_image)
            if hash_registry.contains(perceptual_hash):
                rejected_counts[class_name] += 1
                duplicate_counts[class_name] += 1
                _log_rejected_image(
                    logger=logger,
                    split_name=split_name,
                    class_name=class_name,
                    image_path=image_path,
                    reason="duplicate_image",
                    value=f"phash={perceptual_hash}",
                )
                continue

            hash_registry.add(perceptual_hash)
            file_paths.append(image_path)
            labels.append(class_index)
            accepted_counts[class_name] += 1
            content_hashes.add(_compute_file_sha256(image_path))

        if accepted_counts[class_name] <= 0:
            raise ValueError(
                f"Class '{class_name}' trong split '{split_name}' khong con anh hop le "
                "sau strict cleaning."
            )

        if accepted_counts[class_name] < 10:
            logger.warning(
                "[%s/%s] class co qua it mau sau cleaning | final_count=%d < 10",
                split_name,
                class_name,
                accepted_counts[class_name],
            )

        logger.info(
            (
                "[%s/%s] summary | raw=%d | accepted=%d | rejected=%d | "
                "duplicates_removed=%d"
            ),
            split_name,
            class_name,
            raw_counts[class_name],
            accepted_counts[class_name],
            rejected_counts[class_name],
            duplicate_counts[class_name],
        )

    cleaned_split = CleanedSplitData(
        split_name=split_name,
        file_paths=tuple(file_paths),
        labels=tuple(labels),
        raw_counts=raw_counts,
        accepted_counts=accepted_counts,
        rejected_counts=rejected_counts,
        duplicate_counts=duplicate_counts,
        content_hashes=content_hashes,
        latest_mtime_ns=latest_mtime_ns,
    )

    logger.info(
        (
            "[%s] final summary | raw_total=%d | final_total=%d | "
            "rejected_total=%d | duplicates_removed_total=%d"
        ),
        split_name,
        cleaned_split.total_raw,
        cleaned_split.total_accepted,
        cleaned_split.total_rejected,
        cleaned_split.total_duplicates,
    )
    return cleaned_split


def _validate_no_split_overlap(cleaned_splits: list[CleanedSplitData]) -> None:
    """Fail fast nếu cùng nội dung file xuất hiện trong nhiều split."""

    seen_hash_to_split: dict[str, str] = {}
    overlaps: list[str] = []

    for split_data in cleaned_splits:
        for content_hash in split_data.content_hashes:
            previous_split = seen_hash_to_split.get(content_hash)
            if previous_split is not None and previous_split != split_data.split_name:
                overlaps.append(f"{previous_split}<->{split_data.split_name}:{content_hash[:12]}")
                continue
            seen_hash_to_split[content_hash] = split_data.split_name

    if overlaps:
        preview = ", ".join(overlaps[:20])
        raise ValueError(
            "Phat hien data leakage: cung mot file xuat hien trong nhieu split. "
            f"Vi du: {preview}"
        )


def _load_image_with_pil_numpy(image_path_tensor: tf.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    """Load ảnh bằng PIL utilities để giữ EXIF/RGB/aspect-ratio/padding nhất quán."""

    image_path_value = image_path_tensor.numpy()
    image_path = Path(image_path_value.decode("utf-8"))
    image = open_image_safely(image_path)
    standardized_image = resize_with_padding(
        image=image,
        target_size=(image_size[1], image_size[0]),
        background_color=(0, 0, 0),
    )
    return np.asarray(standardized_image, dtype=np.float32)


def _apply_dataset_options(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Bật deterministic mode để pipeline ổn định hơn giữa các lần chạy."""

    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)


def _create_base_dataset(
    directory: Path,
    raw_class_names: list[str],
    config: DataLoaderConfig,
) -> tf.data.Dataset:
    """Tạo dataset unbatched để dễ áp dụng preprocessing và augmentation đúng thứ tự."""

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode=config.label_mode,
        class_names=raw_class_names,
        color_mode=config.color_mode,
        batch_size=None,
        image_size=config.image_size,
        shuffle=False,
        seed=config.seed,
        interpolation="bilinear",
    )
    return _apply_dataset_options(dataset)


def _load_standardized_image_example(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load ảnh strict bằng PIL trong tf.data, output float32 RGB đã pad đúng size."""

    image = tf.py_function(
        func=lambda path: _load_image_with_pil_numpy(path, image_size),
        inp=[image_path],
        Tout=tf.float32,
    )
    image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
    label = tf.cast(label, tf.int32)
    label = tf.ensure_shape(label, ())
    return image, label


def _create_strict_base_dataset(
    cleaned_split_data: CleanedSplitData,
    config: DataLoaderConfig,
) -> tf.data.Dataset:
    """Tạo dataset path/label để parse ảnh nằm trong pipeline chính.

    Với train split, việc giữ path ở bước này giúp pipeline có thể shuffle trước
    rồi mới load/resize ảnh bằng PIL, giảm lãng phí CPU và đúng thứ tự production.
    """

    path_values = [str(file_path) for file_path in cleaned_split_data.file_paths]
    label_values = list(cleaned_split_data.labels)

    dataset = tf.data.Dataset.from_tensor_slices((path_values, label_values))
    return _apply_dataset_options(dataset)


def _ensure_example_shapes(
    image: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Đặt static shape rõ ràng để model downstream nhận tensor ổn định."""

    image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
    label = tf.cast(label, tf.int32)
    label = tf.ensure_shape(label, ())
    return image, label


def _encode_label_for_output(
    image: tf.Tensor,
    label: tf.Tensor,
    num_classes: int,
    one_hot_labels: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Chuẩn hóa format label ở cuối pipeline để khớp loss đang dùng.

    Mặc định pipeline trả label integer. Nếu caller cũ vẫn cần one-hot, việc
    one-hot hóa được đặt sau bước augment/preprocess để các hàm augment
    class-aware vẫn thao tác trên label integer gọn nhẹ.
    """

    if num_classes <= 0:
        raise ValueError("num_classes phai lon hon 0 de encode label.")

    label = tf.cast(label, tf.int32)
    tf.debugging.assert_greater_equal(
        label,
        tf.constant(0, dtype=tf.int32),
        message="Label index phai >= 0.",
    )
    tf.debugging.assert_less(
        label,
        tf.constant(num_classes, dtype=tf.int32),
        message="Label index phai nam trong range [0, num_classes - 1].",
    )

    if one_hot_labels:
        label = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        label = tf.ensure_shape(label, (num_classes,))
        return image, label

    label = tf.ensure_shape(label, ())
    return image, label


def _remap_label_index(
    image: tf.Tensor,
    label: tf.Tensor,
    label_index_mapping: tuple[int, ...],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map nhãn gốc sang nhãn output mới.

    Dùng cho Stage A binary classifier: nhiều folder trái cây cùng map về label
    `fruit`, riêng folder `other` map về label `other`. Việc map sau bước load
    giúp giữ nguyên cấu trúc folder trên đĩa, không phải tạo dataset vật lý mới.
    """

    mapping_tensor = tf.constant(label_index_mapping, dtype=tf.int32)
    label = tf.gather(mapping_tensor, tf.cast(label, tf.int32))
    label = tf.ensure_shape(label, ())
    return image, label


def _preprocess_for_model(
    image: tf.Tensor,
    label: tf.Tensor,
    config: DataLoaderConfig,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Áp dụng preprocess_input đúng theo backbone đang train."""

    image, label = _ensure_example_shapes(image, label, config.image_size)
    image = tf.cast(image, tf.float32)

    if config.model_type == "mobilenetv2":
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif config.model_type == "resnet50":
        # ResNet50 baseline preprocess ở dataloader, sau augmentation và trước
        # batch. Như vậy train/val/test đều có cùng input contract và model
        # không bị double-preprocess khi load lại để inference.
        image = tf.keras.applications.resnet.preprocess_input(image)
    else:  # pragma: no cover - config đã validate trước đó.
        raise ValueError(f"model_type khong duoc ho tro: {config.model_type}")

    return image, label


def _build_train_augmentation(seed: int) -> tf.keras.Sequential:
    """Tạo augmentation thường cho các lớp không nằm trong danh sách lớp yếu."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
        ],
        name="train_augmentation_normal",
    )


def _build_weak_class_augmentation(seed: int) -> tf.keras.Sequential:
    """Tạo augmentation nhẹ cho lớp yếu nhưng không làm lệch phân phối ảnh."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
        ],
        name="train_augmentation_weak",
    )


def _build_resnet50_train_augmentation(
    seed: int,
    rotation_factor: float = 0.1,
    zoom_factor: float = 0.1,
) -> tf.keras.Sequential:
    """Tạo augmentation nhẹ cho ResNet50, chỉ dùng trên train split.

    Augmentation phải chạy trên ảnh RGB float32 ở range 0..255, trước khi gọi
    `tf.keras.applications.resnet.preprocess_input`. Stage B có thể truyền
    rotation/zoom nhỏ hơn để tránh làm mất đặc trưng hình dáng của từng loại
    nông sản trong bài toán fine-grained classification.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(rotation_factor, seed=seed),
            tf.keras.layers.RandomZoom(zoom_factor, seed=seed),
        ],
        name="train_augmentation_resnet50",
    )


def _augment_train_example(
    image: tf.Tensor,
    label: tf.Tensor,
    augmentation_layer: tf.keras.Sequential,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Augment từng ảnh train theo đúng flow của pipeline hiện tại."""

    image, label = _ensure_example_shapes(image, label, image_size)
    image = augmentation_layer(image, training=True)
    image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
    return image, label


def _load_weak_classes_from_error_analysis(error_analysis_path: Path) -> list[str]:
    """Đọc danh sách weak classes từ file error_analysis.json.

    Script phân tích lỗi hiện tại export theo key:
    {
      "weak_classes": [...]
    }
    Hàm này cũng hỗ trợ fallback an toàn nếu người dùng cung cấp trực tiếp một
    JSON list thay vì object.
    """

    if not error_analysis_path.exists():
        raise FileNotFoundError(f"Khong tim thay error_analysis.json: {error_analysis_path}")

    with error_analysis_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        weak_classes = payload.get("weak_classes", [])
    elif isinstance(payload, list):
        weak_classes = payload
    else:
        raise ValueError(
            "File error_analysis khong dung dinh dang hop le de trich weak_classes."
        )

    return [
        str(class_name).strip().lower()
        for class_name in weak_classes
        if str(class_name).strip()
    ]


def _resolve_weak_class_names(
    requested_weak_classes: list[str],
    class_names: list[str],
) -> list[str]:
    """Map weak_classes đầu vào về đúng class_names thực tế của dataset.

    Ưu tiên exact match. Nếu không có exact match, cho phép match prefix duy nhất
    để hỗ trợ các trường hợp người dùng truyền alias ngắn hơn.
    """

    resolved_class_names: list[str] = []
    for requested_name in requested_weak_classes:
        normalized_name = requested_name.strip().lower()
        if not normalized_name:
            continue

        if normalized_name in class_names:
            resolved_class_names.append(normalized_name)
            continue

        prefix_matches = [
            class_name
            for class_name in class_names
            if class_name.startswith(f"{normalized_name}_")
        ]
        if len(prefix_matches) == 1:
            resolved_class_names.append(prefix_matches[0])
            continue

        if len(prefix_matches) > 1:
            raise ValueError(
                "Nhieu class_names cung khop prefix voi weak class duoc yeu cau: "
                f"{normalized_name} -> {prefix_matches}"
            )

    return sorted(set(resolved_class_names))


def _collect_effective_weak_class_names(
    config: DataLoaderConfig,
    class_names: list[str],
) -> list[str]:
    """Gộp weak_classes từ config trực tiếp và từ error_analysis.json nếu có."""

    requested_weak_classes = [
        class_name.strip().lower()
        for class_name in config.weak_class_names
        if class_name.strip()
    ]

    if config.weak_classes_path is not None:
        requested_weak_classes.extend(
            _load_weak_classes_from_error_analysis(config.weak_classes_path)
        )

    return _resolve_weak_class_names(requested_weak_classes, class_names)


def _build_weak_label_mask(
    class_names: list[str],
    weak_class_names: list[str],
) -> tf.Tensor:
    """Tạo mask bool theo chỉ số lớp để kiểm tra label yếu thật nhanh trong tf.data."""

    weak_class_name_set = set(weak_class_names)
    weak_label_mask = [
        class_name in weak_class_name_set for class_name in class_names
    ]
    return tf.constant(weak_label_mask, dtype=tf.bool)


def _augment_train_example_class_aware(
    image: tf.Tensor,
    label: tf.Tensor,
    normal_augmentation_layer: tf.keras.Sequential,
    weak_augmentation_layer: tf.keras.Sequential,
    weak_label_mask: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    """Áp dụng augmentation theo lớp: lớp yếu mạnh hơn, lớp thường nhẹ hơn."""

    image, label = _ensure_example_shapes(image, label, image_size)
    is_weak_class = tf.gather(weak_label_mask, label)

    image = tf.cond(
        is_weak_class,
        lambda: weak_augmentation_layer(image, training=True),
        lambda: normal_augmentation_layer(image, training=True),
    )
    image = tf.ensure_shape(image, (image_size[0], image_size[1], 3))
    return image, label


def _get_dataset_cardinality(dataset: tf.data.Dataset, dataset_name: str) -> int:
    """Lấy cardinality và fail fast nếu dataset rỗng/unknown/infinite."""

    cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
    if cardinality == UNKNOWN_CARDINALITY:
        raise ValueError(f"Dataset '{dataset_name}' co cardinality UNKNOWN.")
    if cardinality == INFINITE_CARDINALITY:
        raise ValueError(f"Dataset '{dataset_name}' co cardinality INFINITE.")
    if cardinality <= 0:
        raise ValueError(f"Dataset '{dataset_name}' dang rong.")
    return cardinality


def _validate_example_count(
    dataset: tf.data.Dataset,
    expected_examples: int,
    split_name: str,
) -> None:
    """Đối chiếu số phần tử unbatched của tf.data với số file thực tế trên đĩa."""

    cardinality = _get_dataset_cardinality(dataset, split_name)
    if cardinality != expected_examples:
        raise ValueError(
            f"Cardinality cua split '{split_name}' = {cardinality}, "
            f"khong khop so file tren dia = {expected_examples}."
        )


def _compute_expected_batch_count(
    total_examples: int,
    batch_size: int,
    drop_remainder: bool,
) -> int:
    """Tính số batch kỳ vọng sau khi batch."""

    if drop_remainder:
        expected_batches = total_examples // batch_size
    else:
        expected_batches = (total_examples + batch_size - 1) // batch_size

    if expected_batches <= 0:
        raise ValueError("So batch ky vong phai lon hon 0.")
    return expected_batches


def _validate_batch_count(
    dataset: tf.data.Dataset,
    expected_batches: int,
    split_name: str,
) -> None:
    """Kiểm tra cardinality của dataset sau bước batch."""

    cardinality = _get_dataset_cardinality(dataset, split_name)
    if cardinality != expected_batches:
        raise ValueError(
            f"So batch cua split '{split_name}' = {cardinality}, "
            f"khong khop gia tri ky vong = {expected_batches}."
        )


def _build_split_dataset(
    split_name: str,
    raw_class_names: list[str],
    clean_class_names: list[str],
    config: DataLoaderConfig,
    training: bool,
    cleaned_split_data: CleanedSplitData | None = None,
    label_index_mapping: tuple[int, ...] | None = None,
) -> tf.data.Dataset:
    """Xây dựng pipeline cho một split theo đúng thứ tự production-safe."""

    split_dir = _get_split_dir(config, split_name)
    strict_mode = cleaned_split_data is not None

    if strict_mode:
        total_images = cleaned_split_data.total_accepted
        latest_mtime_ns = cleaned_split_data.latest_mtime_ns
    else:
        _, total_images, latest_mtime_ns = _scan_split_metadata(
            split_dir=split_dir,
            class_names=raw_class_names,
            valid_extensions=config.valid_extensions,
        )
    del latest_mtime_ns

    # MobileNetV2 mặc định giữ loader cũ để backward compatible.
    # ResNet50 strict dùng PIL loader để sửa EXIF, convert RGB và pad không méo ảnh.
    if strict_mode:
        dataset = _create_strict_base_dataset(
            cleaned_split_data=cleaned_split_data,
            config=config,
        )
    else:
        dataset = _create_base_dataset(
            directory=split_dir,
            raw_class_names=raw_class_names,
            config=config,
        )

    _validate_example_count(
        dataset=dataset,
        expected_examples=total_images,
        split_name=split_name,
    )

    # Production order:
    # - ResNet50 train: shuffle -> parse/load -> augment -> preprocess_input -> batch -> prefetch.
    # - ResNet50 val/test: parse/load -> preprocess_input -> batch -> prefetch.
    # - MobileNetV2 giữ thứ tự cũ để không làm thay đổi baseline đang có.
    # Không dùng disk cache trong train để tránh warning partial cache và giảm bottleneck I/O.
    if training:
        dataset = dataset.shuffle(
            buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            reshuffle_each_iteration=config.reshuffle_each_iteration,
        )

    if strict_mode:
        dataset = dataset.map(
            lambda image_path, label: _load_standardized_image_example(
                image_path=image_path,
                label=label,
                image_size=config.image_size,
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    if label_index_mapping is not None:
        dataset = dataset.map(
            lambda image, label: _remap_label_index(
                image=image,
                label=label,
                label_index_mapping=label_index_mapping,
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    if training:
        if config.model_type == "resnet50":
            resnet50_augmentation_layer = _build_resnet50_train_augmentation(
                seed=config.seed,
                rotation_factor=config.resnet50_rotation_factor,
                zoom_factor=config.resnet50_zoom_factor,
            )
            dataset = dataset.map(
                lambda image, label: _augment_train_example(
                    image=image,
                    label=label,
                    augmentation_layer=resnet50_augmentation_layer,
                    image_size=config.image_size,
                ),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        else:
            dataset = dataset.map(
                lambda image, label: _preprocess_for_model(image, label, config),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
            normal_augmentation_layer = _build_train_augmentation(config.seed)
            effective_weak_class_names = _collect_effective_weak_class_names(
                config=config,
                class_names=clean_class_names,
            )

            if effective_weak_class_names:
                weak_augmentation_layer = _build_weak_class_augmentation(config.seed)
                weak_label_mask = _build_weak_label_mask(
                    class_names=clean_class_names,
                    weak_class_names=effective_weak_class_names,
                )
                dataset = dataset.map(
                    lambda image, label: _augment_train_example_class_aware(
                        image=image,
                        label=label,
                        normal_augmentation_layer=normal_augmentation_layer,
                        weak_augmentation_layer=weak_augmentation_layer,
                        weak_label_mask=weak_label_mask,
                        image_size=config.image_size,
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=True,
                )
            else:
                dataset = dataset.map(
                    lambda image, label: _augment_train_example(
                        image=image,
                        label=label,
                        augmentation_layer=normal_augmentation_layer,
                        image_size=config.image_size,
                    ),
                    num_parallel_calls=AUTOTUNE,
                    deterministic=True,
                )

    if config.model_type == "resnet50" or not training:
        dataset = dataset.map(
            lambda image, label: _preprocess_for_model(image, label, config),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    dataset = dataset.map(
        lambda image, label: _encode_label_for_output(
            image=image,
            label=label,
            num_classes=len(clean_class_names),
            one_hot_labels=config.one_hot_labels,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        config.batch_size,
        drop_remainder=config.drop_remainder,
    )
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = _apply_dataset_options(dataset)

    expected_batches = _compute_expected_batch_count(
        total_examples=total_images,
        batch_size=config.batch_size,
        drop_remainder=config.drop_remainder,
    )
    _validate_batch_count(
        dataset=dataset,
        expected_batches=expected_batches,
        split_name=split_name,
    )
    return dataset


def _build_loader_config(
    dataset_root: str | Path,
    image_size: tuple[int, int] | None,
    batch_size: int,
    seed: int,
    one_hot_labels: bool,
    cache_enabled: bool,
    cache_root: str | Path | None,
    shuffle_buffer_size: int,
    resnet50_rotation_factor: float,
    resnet50_zoom_factor: float,
    drop_remainder: bool,
    weak_classes: list[str] | tuple[str, ...] | None,
    weak_classes_path: str | Path | None,
    train_dir_name: str,
    val_dir_name: str,
    test_dir_name: str,
    model_type: str,
    strict_validation: bool | None,
    validate_split_overlap: bool,
    data_cleaning_log_path: str | Path,
) -> DataLoaderConfig:
    """Tạo DataLoaderConfig dùng chung cho các biến thể loader."""

    normalized_model_type = _normalize_model_type(model_type)
    return DataLoaderConfig(
        dataset_root=Path(dataset_root),
        model_type=normalized_model_type,
        image_size=image_size or get_default_image_size_for_model(normalized_model_type),
        batch_size=batch_size,
        seed=seed,
        one_hot_labels=one_hot_labels,
        train_dir_name=train_dir_name,
        val_dir_name=val_dir_name,
        test_dir_name=test_dir_name,
        cache_enabled=cache_enabled,
        cache_root=Path(cache_root) if cache_root is not None else None,
        shuffle_buffer_size=shuffle_buffer_size,
        resnet50_rotation_factor=resnet50_rotation_factor,
        resnet50_zoom_factor=resnet50_zoom_factor,
        drop_remainder=drop_remainder,
        weak_class_names=tuple(weak_classes or ()),
        weak_classes_path=Path(weak_classes_path) if weak_classes_path is not None else None,
        strict_validation=(
            normalized_model_type == "resnet50"
            if strict_validation is None
            else strict_validation
        ),
        validate_split_overlap=validate_split_overlap,
        data_cleaning_log_path=Path(data_cleaning_log_path),
    )


def _load_raw_split_class_names(
    config: DataLoaderConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Nạp và kiểm tra danh sách raw class folders của train/val/test."""

    train_dir = _get_split_dir(config, config.train_dir_name)
    val_dir = _get_split_dir(config, config.val_dir_name)
    test_dir = _get_split_dir(config, config.test_dir_name)

    raw_train_class_names = _list_class_directories(train_dir)
    raw_val_class_names = _list_class_directories(val_dir)
    raw_test_class_names = _list_class_directories(test_dir)

    _validate_split_class_alignment(
        train_classes=raw_train_class_names,
        val_classes=raw_val_class_names,
        test_classes=raw_test_class_names,
    )
    return raw_train_class_names, raw_val_class_names, raw_test_class_names


def _collect_cleaned_splits_if_needed(
    config: DataLoaderConfig,
    raw_class_names: list[str],
) -> dict[str, CleanedSplitData]:
    """Chạy strict validation nếu model/config yêu cầu."""

    cleaned_splits: dict[str, CleanedSplitData] = {}
    if not _is_strict_validation_enabled(config):
        return cleaned_splits

    cleaning_logger = _setup_data_cleaning_logger(config.data_cleaning_log_path)
    cleaning_logger.info(
        "Bat dau strict data cleaning | model_type=%s | image_size=%s | dataset_root=%s",
        config.model_type,
        config.image_size,
        config.dataset_root,
    )
    for split_name in (
        config.train_dir_name,
        config.val_dir_name,
        config.test_dir_name,
    ):
        cleaned_splits[split_name] = _validate_and_collect_split_files(
            split_name=split_name,
            raw_class_names=raw_class_names,
            config=config,
            logger=cleaning_logger,
        )
    if config.validate_split_overlap:
        _validate_no_split_overlap(list(cleaned_splits.values()))
    cleaning_logger.info(
        "Hoan tat strict data cleaning | final_sizes=%s | rejected=%s | duplicates=%s",
        {
            split_name: split_data.total_accepted
            for split_name, split_data in cleaned_splits.items()
        },
        {
            split_name: split_data.total_rejected
            for split_name, split_data in cleaned_splits.items()
        },
        {
            split_name: split_data.total_duplicates
            for split_name, split_data in cleaned_splits.items()
        },
    )
    return cleaned_splits


def _build_three_split_datasets(
    config: DataLoaderConfig,
    raw_class_names: list[str],
    output_class_names: list[str],
    label_index_mapping: tuple[int, ...] | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train/val/test từ cùng một class mapping."""

    cleaned_splits = _collect_cleaned_splits_if_needed(
        config=config,
        raw_class_names=raw_class_names,
    )
    train_dataset = _build_split_dataset(
        split_name=config.train_dir_name,
        raw_class_names=raw_class_names,
        clean_class_names=output_class_names,
        config=config,
        training=True,
        cleaned_split_data=cleaned_splits.get(config.train_dir_name),
        label_index_mapping=label_index_mapping,
    )
    val_dataset = _build_split_dataset(
        split_name=config.val_dir_name,
        raw_class_names=raw_class_names,
        clean_class_names=output_class_names,
        config=config,
        training=False,
        cleaned_split_data=cleaned_splits.get(config.val_dir_name),
        label_index_mapping=label_index_mapping,
    )
    test_dataset = _build_split_dataset(
        split_name=config.test_dir_name,
        raw_class_names=raw_class_names,
        clean_class_names=output_class_names,
        config=config,
        training=False,
        cleaned_split_data=cleaned_splits.get(config.test_dir_name),
        label_index_mapping=label_index_mapping,
    )
    return train_dataset, val_dataset, test_dataset


def load_datasets(
    dataset_root: str | Path = "dataset",
    image_size: tuple[int, int] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    one_hot_labels: bool = False,
    cache_enabled: bool = True,
    cache_root: str | Path | None = None,
    shuffle_buffer_size: int = 1000,
    resnet50_rotation_factor: float = 0.1,
    resnet50_zoom_factor: float = 0.1,
    drop_remainder: bool = False,
    weak_classes: list[str] | tuple[str, ...] | None = None,
    weak_classes_path: str | Path | None = None,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
    test_dir_name: str = "test",
    model_type: str = "mobilenetv2",
    strict_validation: bool | None = None,
    validate_split_overlap: bool = True,
    data_cleaning_log_path: str | Path = "logs/data_cleaning.log",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Nạp train/val/test và trả về class_names sạch theo đúng thứ tự nhãn.

    ``dataset_root`` có thể trỏ tới:
    - ``dataset/``: pipeline cũ, có thể chứa class ``other``.
    - ``dataset_fruit_only/``: Stage B, bắt buộc không còn class ``other``.
    """

    normalized_model_type = _normalize_model_type(model_type)
    config = DataLoaderConfig(
        dataset_root=Path(dataset_root),
        model_type=normalized_model_type,
        image_size=image_size or get_default_image_size_for_model(normalized_model_type),
        batch_size=batch_size,
        seed=seed,
        one_hot_labels=one_hot_labels,
        train_dir_name=train_dir_name,
        val_dir_name=val_dir_name,
        test_dir_name=test_dir_name,
        cache_enabled=cache_enabled,
        cache_root=Path(cache_root) if cache_root is not None else None,
        shuffle_buffer_size=shuffle_buffer_size,
        resnet50_rotation_factor=resnet50_rotation_factor,
        resnet50_zoom_factor=resnet50_zoom_factor,
        drop_remainder=drop_remainder,
        weak_class_names=tuple(weak_classes or ()),
        weak_classes_path=Path(weak_classes_path) if weak_classes_path is not None else None,
        strict_validation=(
            normalized_model_type == "resnet50"
            if strict_validation is None
            else strict_validation
        ),
        validate_split_overlap=validate_split_overlap,
        data_cleaning_log_path=Path(data_cleaning_log_path),
    )

    _validate_config(config)
    tf.keras.utils.set_random_seed(config.seed)

    train_dir = _get_split_dir(config, config.train_dir_name)
    val_dir = _get_split_dir(config, config.val_dir_name)
    test_dir = _get_split_dir(config, config.test_dir_name)

    raw_train_class_names = _list_class_directories(train_dir)
    raw_val_class_names = _list_class_directories(val_dir)
    raw_test_class_names = _list_class_directories(test_dir)

    _validate_split_class_alignment(
        train_classes=raw_train_class_names,
        val_classes=raw_val_class_names,
        test_classes=raw_test_class_names,
    )

    clean_class_names = _build_clean_class_names(raw_train_class_names)
    if _is_fruit_only_dataset_root(config.dataset_root):
        _validate_fruit_only_class_names(
            raw_class_names=raw_train_class_names,
            clean_class_names=clean_class_names,
            dataset_root=config.dataset_root,
        )

    cleaned_splits: dict[str, CleanedSplitData] = {}
    if _is_strict_validation_enabled(config):
        cleaning_logger = _setup_data_cleaning_logger(config.data_cleaning_log_path)
        cleaning_logger.info(
            "Bat dau strict data cleaning | model_type=%s | image_size=%s | dataset_root=%s",
            config.model_type,
            config.image_size,
            config.dataset_root,
        )
        for split_name in (
            config.train_dir_name,
            config.val_dir_name,
            config.test_dir_name,
        ):
            cleaned_splits[split_name] = _validate_and_collect_split_files(
                split_name=split_name,
                raw_class_names=raw_train_class_names,
                config=config,
                logger=cleaning_logger,
            )
        if config.validate_split_overlap:
            _validate_no_split_overlap(list(cleaned_splits.values()))
        cleaning_logger.info(
            "Hoan tat strict data cleaning | final_sizes=%s | rejected=%s | duplicates=%s",
            {
                split_name: split_data.total_accepted
                for split_name, split_data in cleaned_splits.items()
            },
            {
                split_name: split_data.total_rejected
                for split_name, split_data in cleaned_splits.items()
            },
            {
                split_name: split_data.total_duplicates
                for split_name, split_data in cleaned_splits.items()
            },
        )

    train_dataset = _build_split_dataset(
        split_name=config.train_dir_name,
        raw_class_names=raw_train_class_names,
        clean_class_names=clean_class_names,
        config=config,
        training=True,
        cleaned_split_data=cleaned_splits.get(config.train_dir_name),
    )
    val_dataset = _build_split_dataset(
        split_name=config.val_dir_name,
        raw_class_names=raw_train_class_names,
        clean_class_names=clean_class_names,
        config=config,
        training=False,
        cleaned_split_data=cleaned_splits.get(config.val_dir_name),
    )
    test_dataset = _build_split_dataset(
        split_name=config.test_dir_name,
        raw_class_names=raw_train_class_names,
        clean_class_names=clean_class_names,
        config=config,
        training=False,
        cleaned_split_data=cleaned_splits.get(config.test_dir_name),
    )

    return train_dataset, val_dataset, test_dataset, clean_class_names


def load_binary_datasets(
    dataset_root: str | Path = "dataset",
    image_size: tuple[int, int] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    cache_enabled: bool = True,
    cache_root: str | Path | None = None,
    shuffle_buffer_size: int = 1000,
    resnet50_rotation_factor: float = 0.1,
    resnet50_zoom_factor: float = 0.1,
    drop_remainder: bool = False,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
    test_dir_name: str = "test",
    model_type: str = "resnet50",
    strict_validation: bool | None = None,
    validate_split_overlap: bool = True,
    data_cleaning_log_path: str | Path = "logs/data_cleaning.log",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load Stage A dataset: `other` vs `fruit`.

    Output label:
    - 0: other
    - 1: fruit

    Không tạo/copy folder mới. Loader giữ nguyên cấu trúc dataset/train|val|test
    và map nhãn trong tf.data, nhờ vậy không có rủi ro trộn class vật lý.
    """

    config = _build_loader_config(
        dataset_root=dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        one_hot_labels=False,
        cache_enabled=cache_enabled,
        cache_root=cache_root,
        shuffle_buffer_size=shuffle_buffer_size,
        resnet50_rotation_factor=resnet50_rotation_factor,
        resnet50_zoom_factor=resnet50_zoom_factor,
        drop_remainder=drop_remainder,
        weak_classes=(),
        weak_classes_path=None,
        train_dir_name=train_dir_name,
        val_dir_name=val_dir_name,
        test_dir_name=test_dir_name,
        model_type=model_type,
        strict_validation=strict_validation,
        validate_split_overlap=validate_split_overlap,
        data_cleaning_log_path=data_cleaning_log_path,
    )
    _validate_config(config)
    tf.keras.utils.set_random_seed(config.seed)

    raw_train_class_names, _, _ = _load_raw_split_class_names(config)
    output_class_names = ["other", "fruit"]
    label_index_mapping = tuple(
        0 if extract_clean_label(raw_class_name) == "other" else 1
        for raw_class_name in raw_train_class_names
    )

    train_dataset, val_dataset, test_dataset = _build_three_split_datasets(
        config=config,
        raw_class_names=raw_train_class_names,
        output_class_names=output_class_names,
        label_index_mapping=label_index_mapping,
    )
    return train_dataset, val_dataset, test_dataset, output_class_names


def load_fruit_datasets(
    dataset_root: str | Path = "dataset",
    image_size: tuple[int, int] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    one_hot_labels: bool = False,
    cache_enabled: bool = True,
    cache_root: str | Path | None = None,
    shuffle_buffer_size: int = 1000,
    resnet50_rotation_factor: float = 0.1,
    resnet50_zoom_factor: float = 0.1,
    drop_remainder: bool = False,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
    test_dir_name: str = "test",
    model_type: str = "resnet50",
    strict_validation: bool | None = None,
    validate_split_overlap: bool = True,
    data_cleaning_log_path: str | Path = "logs/data_cleaning.log",
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load Stage B dataset: chỉ các lớp nông sản, loại `other`.

    Không hardcode số class. Stage B đọc số lớp từ folder `dataset_fruit_only`
    sau khi taxonomy đã được gộp. Class `other` thuộc Stage A nên không được
    xuất hiện trong loader này.
    """

    config = _build_loader_config(
        dataset_root=dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        one_hot_labels=one_hot_labels,
        cache_enabled=cache_enabled,
        cache_root=cache_root,
        shuffle_buffer_size=shuffle_buffer_size,
        resnet50_rotation_factor=resnet50_rotation_factor,
        resnet50_zoom_factor=resnet50_zoom_factor,
        drop_remainder=drop_remainder,
        weak_classes=(),
        weak_classes_path=None,
        train_dir_name=train_dir_name,
        val_dir_name=val_dir_name,
        test_dir_name=test_dir_name,
        model_type=model_type,
        strict_validation=strict_validation,
        validate_split_overlap=validate_split_overlap,
        data_cleaning_log_path=data_cleaning_log_path,
    )
    _validate_config(config)
    tf.keras.utils.set_random_seed(config.seed)

    raw_train_class_names, _, _ = _load_raw_split_class_names(config)
    fruit_raw_class_names = [
        raw_class_name
        for raw_class_name in raw_train_class_names
        if extract_clean_label(raw_class_name) != "other"
    ]
    if not fruit_raw_class_names:
        raise ValueError("Khong tim thay class fruit nao sau khi loai 'other'.")

    fruit_class_names = _build_clean_class_names(fruit_raw_class_names)
    if _is_fruit_only_dataset_root(config.dataset_root):
        _validate_fruit_only_class_names(
            raw_class_names=fruit_raw_class_names,
            clean_class_names=fruit_class_names,
            dataset_root=config.dataset_root,
        )

    train_dataset, val_dataset, test_dataset = _build_three_split_datasets(
        config=config,
        raw_class_names=fruit_raw_class_names,
        output_class_names=fruit_class_names,
        label_index_mapping=None,
    )
    return train_dataset, val_dataset, test_dataset, fruit_class_names


def load_data(
    dataset_root: str | Path = "dataset",
    image_size: tuple[int, int] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    one_hot_labels: bool = False,
    cache_enabled: bool = True,
    cache_root: str | Path | None = None,
    shuffle_buffer_size: int = 1000,
    resnet50_rotation_factor: float = 0.1,
    resnet50_zoom_factor: float = 0.1,
    drop_remainder: bool = False,
    weak_classes: list[str] | tuple[str, ...] | None = None,
    weak_classes_path: str | Path | None = None,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
    test_dir_name: str = "test",
    model_type: str = "mobilenetv2",
    strict_validation: bool | None = None,
    validate_split_overlap: bool = True,
    data_cleaning_log_path: str | Path = "logs/data_cleaning.log",
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Alias tương thích cho code cũ: chỉ trả về train/val."""

    train_dataset, val_dataset, _, _ = load_datasets(
        dataset_root=dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        one_hot_labels=one_hot_labels,
        model_type=model_type,
        cache_enabled=cache_enabled,
        cache_root=cache_root,
        shuffle_buffer_size=shuffle_buffer_size,
        resnet50_rotation_factor=resnet50_rotation_factor,
        resnet50_zoom_factor=resnet50_zoom_factor,
        drop_remainder=drop_remainder,
        weak_classes=weak_classes,
        weak_classes_path=weak_classes_path,
        train_dir_name=train_dir_name,
        val_dir_name=val_dir_name,
        test_dir_name=test_dir_name,
        strict_validation=strict_validation,
        validate_split_overlap=validate_split_overlap,
        data_cleaning_log_path=data_cleaning_log_path,
    )
    return train_dataset, val_dataset
