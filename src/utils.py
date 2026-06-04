from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

try:
    from config import ClassMetadata
except ImportError:  # pragma: no cover
    from .config import ClassMetadata


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_LANCZOS = Image.LANCZOS


def set_global_seed(seed: int) -> None:
    """Cố định seed để các bước phụ trợ luôn cho kết quả ổn định."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        cv2.setRNGSeed(seed)
    except AttributeError:  # pragma: no cover
        pass


def setup_logger(
    log_file: Path, logger_name: str = "dataset_preprocess"
) -> logging.Logger:
    """Khởi tạo logger ghi ra file để theo dõi chi tiết cho từng pipeline."""

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Xóa handler cũ để tránh ghi log lặp nếu script được import/chạy nhiều lần.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


def prepare_output_directory(output_dir: Path, clean: bool = True) -> None:
    """Tạo hoặc làm sạch thư mục output mà không chạm vào dữ liệu gốc."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean:
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def list_class_directories(raw_dir: Path) -> list[Path]:
    """Lấy danh sách thư mục lớp và sắp xếp để pipeline có tính tái lập."""

    return sorted(
        [path for path in raw_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def list_image_files(
    directory: Path, valid_extensions: Sequence[str]
) -> list[Path]:
    """Liệt kê file ảnh hợp lệ theo thứ tự cố định để pipeline tái lập."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in valid_extension_set
        ],
        key=lambda path: path.name.lower(),
    )


def normalize_class_key(raw_name: str) -> str:
    """Chuẩn hóa tên lớp đầu vào về cùng một định dạng khóa."""

    normalized = raw_name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    normalized = re.sub(r"_c$", "", normalized)
    return normalized


def sanitize_folder_token(value: str) -> str:
    """Đảm bảo token trong tên thư mục đầu ra an toàn và nhất quán."""

    sanitized = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized


def build_stable_random(seed: int, namespace: str) -> random.Random:
    """Sinh RNG ổn định theo seed và tên lớp, tránh phụ thuộc hash runtime."""

    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def get_class_metadata(
    raw_class_name: str, class_name_map: dict[str, ClassMetadata]
) -> ClassMetadata:
    """Tra metadata tên lớp; fallback an toàn nếu lớp chưa có trong mapping."""

    key = normalize_class_key(raw_class_name)
    metadata = class_name_map.get(key)
    if metadata is not None:
        return metadata

    fallback_name = sanitize_folder_token(key) or "unknown"
    return ClassMetadata(english_name=fallback_name, vietnamese_name="khong_ro")


def build_output_folder_name(metadata: ClassMetadata, valid_count: int) -> str:
    """Sinh tên thư mục đầu ra theo đúng format yêu cầu."""

    english_name = sanitize_folder_token(metadata.english_name)
    vietnamese_name = sanitize_folder_token(metadata.vietnamese_name)
    return f"{english_name}_{vietnamese_name}_{valid_count}"


def compute_split_counts(
    total_images: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    require_non_empty: bool = True,
) -> dict[str, int]:
    """Tính số lượng mẫu cho từng split, vừa đúng tổng vừa tránh split rỗng."""

    split_order = ("train", "val", "test")
    ratios = {
        "train": float(train_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }

    if total_images <= 0:
        raise ValueError("Tong so anh cua mot lop phai lon hon 0.")

    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Tat ca ti le split phai lon hon 0.")

    total_ratio = sum(ratios.values())
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Tong ti le split phai bang 1.0, nhan duoc {total_ratio:.12f}."
        )

    if require_non_empty and total_images < len(split_order):
        raise ValueError(
            "Khong the tao train/val/test deu khong rong vi so anh cua lop < 3."
        )

    raw_counts = {
        split_name: total_images * ratios[split_name] for split_name in split_order
    }
    counts = {
        split_name: int(math.floor(raw_counts[split_name])) for split_name in split_order
    }

    remainder = total_images - sum(counts.values())
    priority_map = {"train": 2, "val": 1, "test": 0}
    allocation_order = sorted(
        split_order,
        key=lambda split_name: (
            raw_counts[split_name] - counts[split_name],
            ratios[split_name],
            priority_map[split_name],
        ),
        reverse=True,
    )

    for index in range(remainder):
        target_split = allocation_order[index % len(allocation_order)]
        counts[target_split] += 1

    if require_non_empty:
        zero_splits = [split_name for split_name in split_order if counts[split_name] == 0]
        while zero_splits:
            for zero_split in zero_splits:
                donor_candidates = [
                    split_name for split_name in split_order if counts[split_name] > 1
                ]
                if not donor_candidates:
                    raise ValueError(
                        "Khong tim duoc split cho du mau de sua trang thai split rong."
                    )

                donor_split = max(
                    donor_candidates,
                    key=lambda split_name: (
                        counts[split_name] - raw_counts[split_name],
                        counts[split_name],
                        ratios[split_name],
                        priority_map[split_name],
                    ),
                )
                counts[donor_split] -= 1
                counts[zero_split] += 1

            zero_splits = [
                split_name for split_name in split_order if counts[split_name] == 0
            ]

    if sum(counts.values()) != total_images:
        raise AssertionError("Tong so mau sau khi tinh split khong khop tong dau vao.")

    return counts


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Chuẩn hóa mọi ảnh về RGB, đồng thời xử lý alpha trên nền đen."""

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba_image = image.convert("RGBA")
        black_background = Image.new("RGBA", rgba_image.size, (0, 0, 0, 255))
        composited_image = Image.alpha_composite(black_background, rgba_image)
        return composited_image.convert("RGB")

    return image.convert("RGB")


def open_image_safely(image_path: Path) -> Image.Image:
    """Mở ảnh an toàn, verify file, sửa EXIF orientation và chuyển về RGB."""

    with Image.open(image_path) as probe_image:
        probe_image.verify()

    with Image.open(image_path) as image:
        fixed_image = ImageOps.exif_transpose(image)
        fixed_image.load()
        return ensure_rgb(fixed_image.copy())


def validate_image_quality(
    image: Image.Image,
    min_image_side: int,
    max_image_side: int,
    max_aspect_ratio: float,
) -> tuple[bool, str | None]:
    """Loại ảnh quá nhỏ, quá lớn hoặc có tỉ lệ bất thường."""

    width, height = image.size
    if width < min_image_side or height < min_image_side:
        return False, "image_too_small"

    if width > max_image_side or height > max_image_side:
        return False, "image_too_large"

    if height == 0 or width == 0:
        return False, "invalid_dimension"

    aspect_ratio = width / height
    if aspect_ratio > max_aspect_ratio or aspect_ratio < (1 / max_aspect_ratio):
        return False, "invalid_aspect_ratio"

    return True, None


def compute_blur_score(image: Image.Image) -> float:
    """Tính độ nét cơ bản bằng Laplacian variance."""

    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def resize_with_padding(
    image: Image.Image,
    target_size: tuple[int, int],
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize giữ đúng tỉ lệ, sau đó pad nền đen để đạt 224x224."""

    target_width, target_height = target_size
    source_width, source_height = image.size
    scale = min(target_width / source_width, target_height / source_height)

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized_image = image.resize((resized_width, resized_height), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", target_size, background_color)

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas.paste(resized_image, (offset_x, offset_y))
    return canvas


def compute_perceptual_hash(image: Image.Image) -> imagehash.ImageHash:
    """Tạo perceptual hash để phát hiện ảnh trùng lặp."""

    return imagehash.phash(image)


class HashRegistry:
    """Lưu hash đã gặp để loại ảnh trùng, mặc định so khớp exact hash."""

    def __init__(self, tolerance: int = 0) -> None:
        self.tolerance = max(0, tolerance)
        self._hashes: list[imagehash.ImageHash] = []
        self._hash_strings: set[str] = set()

    def contains(self, image_hash: imagehash.ImageHash) -> bool:
        if self.tolerance == 0:
            return str(image_hash) in self._hash_strings

        return any(image_hash - existing_hash <= self.tolerance for existing_hash in self._hashes)

    def add(self, image_hash: imagehash.ImageHash) -> None:
        self._hashes.append(image_hash)
        self._hash_strings.add(str(image_hash))


def save_image(image: Image.Image, destination: Path, jpeg_quality: int = 95) -> None:
    """Lưu ảnh đầu ra, giữ nguyên tên file nhưng chuẩn hóa nội dung ảnh."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        image.save(
            destination,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            subsampling=0,
        )
        return

    if suffix == ".png":
        image.save(destination, format="PNG", optimize=True)
        return

    raise ValueError(f"Khong ho tro dinh dang file: {destination.suffix}")


def validate_split_overlap(split_to_keys: Mapping[str, set[str]]) -> None:
    """Đảm bảo không có bất kỳ khóa file nào xuất hiện ở nhiều split."""

    split_names = list(split_to_keys.keys())
    for index, left_split in enumerate(split_names):
        for right_split in split_names[index + 1 :]:
            overlap = split_to_keys[left_split] & split_to_keys[right_split]
            if overlap:
                sample_overlap = sorted(overlap)[0]
                raise ValueError(
                    "Phat hien data leakage giua "
                    f"'{left_split}' va '{right_split}': {sample_overlap}"
                )


def prepare_staging_directory(dataset_root: Path, staging_dir_name: str) -> Path:
    """Tạo thư mục staging riêng cho split để tránh sinh output nửa chừng."""

    staging_root = dataset_root / staging_dir_name
    if staging_root.exists():
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        else:
            staging_root.unlink()

    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def replace_directory(source_dir: Path, destination_dir: Path) -> None:
    """Thay thế thư mục đích bằng thư mục nguồn theo cách rõ ràng, an toàn."""

    if destination_dir.exists():
        if destination_dir.is_dir():
            shutil.rmtree(destination_dir)
        else:
            destination_dir.unlink()

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    source_dir.rename(destination_dir)
