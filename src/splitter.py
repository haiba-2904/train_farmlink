from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_VALID_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class SourceImage:
    """Một ảnh nguồn sau khi quét từ `dataset/processed_crop`.

    class_name:
        Tên folder label. Đây là label chính thức, không được đổi trong bước split.
    path:
        Đường dẫn ảnh trong `processed_crop`.
    content_hash:
        SHA256 theo byte ảnh. Hash này dùng để tránh cùng một ảnh xuất hiện ở
        nhiều split, kể cả khi file bị đổi tên.
    """

    class_name: str
    path: Path
    content_hash: str


@dataclass(frozen=True)
class ClassSplitStats:
    """Thống kê split cho một class."""

    class_name: str
    input_images: int
    unique_images: int
    duplicates_removed: int
    train_count: int
    val_count: int
    test_count: int


@dataclass(frozen=True)
class DatasetSplitResult:
    """Kết quả tổng của bước split dataset chính."""

    created_at: str
    input_dir: Path
    dataset_root: Path
    class_names: list[str]
    class_count: int
    split_totals: dict[str, int]
    per_class_counts: dict[str, dict[str, int]]
    duplicate_counts: dict[str, int]
    duplicates_removed_total: int


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Config cho bước tạo `dataset/train`, `dataset/val`, `dataset/test`.

    Không hardcode số class. Danh sách class được đọc trực tiếp từ
    `dataset/processed_crop`.
    """

    input_dir: Path = Path("dataset/processed_crop")
    dataset_root: Path = Path("dataset")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    clean_output: bool = False
    valid_extensions: tuple[str, ...] = DEFAULT_VALID_EXTENSIONS
    staging_dir_name: str = "_split_staging"


def parse_args() -> argparse.Namespace:
    """CLI độc lập nếu cần chạy riêng splitter."""

    parser = argparse.ArgumentParser(
        description="Split dataset/processed_crop thanh dataset/train|val|test."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/processed_crop"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DatasetSplitConfig:
    """Ghép tham số CLI thành config."""

    return DatasetSplitConfig(
        input_dir=args.input_dir,
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        clean_output=args.clean_output,
    )


def setup_logger(log_file: Path = Path("logs/rebuild_dataset.log")) -> logging.Logger:
    """Logger dự phòng khi chạy file này độc lập."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dataset_split")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def list_class_dirs(root: Path) -> list[Path]:
    """Liệt kê folder class theo thứ tự ổn định."""

    if not root.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Duong dan khong phai thu muc: {root}")

    class_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    if not class_dirs:
        raise ValueError(f"Khong tim thay class folder nao trong: {root}")
    return class_dirs


def list_image_files(directory: Path, valid_extensions: tuple[str, ...]) -> list[Path]:
    """Lấy danh sách ảnh hợp lệ trong một class folder."""

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


def compute_sha256(file_path: Path) -> str:
    """Tính SHA256 theo nội dung file để phát hiện ảnh trùng thật sự."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_random(seed: int, namespace: str) -> random.Random:
    """Tạo RNG cố định theo seed + class name."""

    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def validate_config(config: DatasetSplitConfig) -> None:
    """Fail fast nếu config có nguy cơ ghi sai chỗ hoặc split sai tỉ lệ."""

    if not config.input_dir.exists():
        raise FileNotFoundError(f"Khong tim thay input_dir: {config.input_dir}")
    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"input_dir khong phai thu muc: {config.input_dir}")

    input_dir = config.input_dir.resolve()
    dataset_root = config.dataset_root.resolve()
    if input_dir == dataset_root:
        raise ValueError("dataset_root khong duoc trung voi input_dir.")

    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Tong ti le train/val/test phai bang 1.0, nhan duoc {total_ratio:.12f}."
        )
    if min(config.train_ratio, config.val_ratio, config.test_ratio) <= 0:
        raise ValueError("Tat ca ti le split phai lon hon 0.")

    destination_dirs = [config.dataset_root / split for split in SPLITS]
    existing_destinations = [path for path in destination_dirs if path.exists()]
    if existing_destinations and not config.clean_output:
        raise FileExistsError(
            "Output split da ton tai. Hay them --clean-output neu muon tao lai: "
            f"{existing_destinations}"
        )


def compute_split_counts(total_images: int, config: DatasetSplitConfig) -> dict[str, int]:
    """Tính số ảnh train/val/test cho một class.

    Quy tắc:
    - Tỉ lệ gần 70/15/15 nhất có thể.
    - Tổng count sau split phải bằng tổng ảnh class.
    - Mỗi split phải có ít nhất 1 ảnh để Stage A/Stage B đều có đủ class.
    """

    if total_images < len(SPLITS):
        raise ValueError(
            "Khong the split train/val/test deu khong rong khi class co it hon 3 anh."
        )

    ratios = {
        "train": config.train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
    }
    raw_counts = {split: total_images * ratios[split] for split in SPLITS}
    counts = {split: int(math.floor(raw_counts[split])) for split in SPLITS}

    remainder = total_images - sum(counts.values())
    allocation_order = sorted(
        SPLITS,
        key=lambda split: (raw_counts[split] - counts[split], ratios[split]),
        reverse=True,
    )
    for index in range(remainder):
        counts[allocation_order[index % len(allocation_order)]] += 1

    for split in SPLITS:
        if counts[split] > 0:
            continue
        donor = max(SPLITS, key=lambda item: counts[item])
        if counts[donor] <= 1:
            raise ValueError("Khong du anh de sua split rong.")
        counts[donor] -= 1
        counts[split] += 1

    return counts


def collect_unique_images_by_class(
    config: DatasetSplitConfig,
    logger: logging.Logger,
) -> tuple[dict[str, list[SourceImage]], dict[str, int], dict[str, int]]:
    """Quét `processed_crop` và loại exact duplicate toàn cục.

    Duplicate được loại ở đây chỉ ảnh hưởng output split, không sửa
    `processed_crop`. Mục tiêu là tránh data leakage khi một ảnh giống hệt rơi
    vào train và val/test.
    """

    class_dirs = list_class_dirs(config.input_dir)
    class_to_images: dict[str, list[SourceImage]] = {}
    input_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    seen_hash_to_image: dict[str, SourceImage] = {}
    duplicate_preview: list[str] = []

    for class_dir in tqdm(class_dirs, desc="Quet processed_crop", leave=True):
        image_files = list_image_files(class_dir, config.valid_extensions)
        input_counts[class_dir.name] = len(image_files)
        duplicate_counts[class_dir.name] = 0
        class_to_images[class_dir.name] = []

        if not image_files:
            raise ValueError(f"Class rong trong processed_crop: {class_dir}")

        for image_path in image_files:
            content_hash = compute_sha256(image_path)
            previous_image = seen_hash_to_image.get(content_hash)
            if previous_image is not None:
                duplicate_counts[class_dir.name] += 1
                if len(duplicate_preview) < 30:
                    duplicate_preview.append(
                        f"{image_path} == {previous_image.path} ({content_hash[:12]})"
                    )
                continue

            source_image = SourceImage(
                class_name=class_dir.name,
                path=image_path,
                content_hash=content_hash,
            )
            seen_hash_to_image[content_hash] = source_image
            class_to_images[class_dir.name].append(source_image)

        if len(class_to_images[class_dir.name]) < len(SPLITS):
            raise ValueError(
                "Class khong du anh unique de chia train/val/test deu khong rong: "
                f"{class_dir.name} | unique={len(class_to_images[class_dir.name])}"
            )

    logger.info("Duplicate exact removal preview=%s", duplicate_preview)
    logger.info("Duplicate counts by class=%s", duplicate_counts)
    return class_to_images, input_counts, duplicate_counts


def build_split_plan(
    class_to_images: dict[str, list[SourceImage]],
    input_counts: dict[str, int],
    duplicate_counts: dict[str, int],
    config: DatasetSplitConfig,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, list[SourceImage]]], list[ClassSplitStats]]:
    """Tạo kế hoạch split theo từng class, chưa copy file."""

    plan: dict[str, dict[str, list[SourceImage]]] = {split: {} for split in SPLITS}
    stats_list: list[ClassSplitStats] = []

    for class_name in sorted(class_to_images, key=str.lower):
        images = list(class_to_images[class_name])
        rng = stable_random(config.seed, class_name)
        rng.shuffle(images)

        counts = compute_split_counts(len(images), config)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        split_images = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split in SPLITS:
            if not split_images[split]:
                raise ValueError(f"Split rong: class={class_name}, split={split}")
            plan[split][class_name] = split_images[split]

        stats = ClassSplitStats(
            class_name=class_name,
            input_images=input_counts[class_name],
            unique_images=len(images),
            duplicates_removed=duplicate_counts[class_name],
            train_count=len(split_images["train"]),
            val_count=len(split_images["val"]),
            test_count=len(split_images["test"]),
        )
        stats_list.append(stats)
        logger.info(
            "[split] class=%s | input=%d | unique=%d | duplicate_removed=%d | "
            "train=%d | val=%d | test=%d",
            stats.class_name,
            stats.input_images,
            stats.unique_images,
            stats.duplicates_removed,
            stats.train_count,
            stats.val_count,
            stats.test_count,
        )

    validate_split_plan(plan)
    return plan, stats_list


def validate_split_plan(plan: dict[str, dict[str, list[SourceImage]]]) -> None:
    """Kiểm tra kế hoạch split không rỗng, class đồng nhất, không overlap."""

    reference_classes = sorted(plan["train"], key=str.lower)
    for split in SPLITS:
        class_names = sorted(plan[split], key=str.lower)
        if class_names != reference_classes:
            raise ValueError(f"Danh sach class cua split '{split}' khong khop train.")

    split_to_keys: dict[str, set[str]] = {}
    split_to_hashes: dict[str, set[str]] = {}
    for split, class_map in plan.items():
        keys: set[str] = set()
        hashes: set[str] = set()
        for class_name, images in class_map.items():
            if not images:
                raise ValueError(f"Class rong trong plan: split={split}, class={class_name}")
            for image in images:
                key = f"{class_name}/{image.path.name}"
                if key in keys:
                    raise ValueError(f"File bi lap trong split '{split}': {key}")
                keys.add(key)
                if image.content_hash in hashes:
                    raise ValueError(
                        f"Hash bi lap trong split '{split}', can kiem tra duplicate input."
                    )
                hashes.add(image.content_hash)
        split_to_keys[split] = keys
        split_to_hashes[split] = hashes

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            name_overlap = split_to_keys[left] & split_to_keys[right]
            if name_overlap:
                raise ValueError(
                    f"Phat hien file overlap giua {left} va {right}: "
                    f"{sorted(name_overlap)[0]}"
                )

            hash_overlap = split_to_hashes[left] & split_to_hashes[right]
            if hash_overlap:
                raise ValueError(
                    f"Phat hien anh trung noi dung giua {left} va {right}: "
                    f"{sorted(hash_overlap)[0][:12]}"
                )


def prepare_staging_root(config: DatasetSplitConfig) -> Path:
    """Tạo staging riêng cho lần chạy hiện tại."""

    staging_root = config.dataset_root / f"{config.staging_dir_name}_{os.getpid()}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def materialize_split_plan(
    plan: dict[str, dict[str, list[SourceImage]]],
    config: DatasetSplitConfig,
) -> Path:
    """Copy ảnh vào staging theo kế hoạch đã validate."""

    staging_root = prepare_staging_root(config)
    total_images = sum(len(images) for class_map in plan.values() for images in class_map.values())

    with tqdm(total=total_images, desc="Copy split", leave=True) as progress:
        for split, class_map in plan.items():
            for class_name, images in class_map.items():
                destination_class_dir = staging_root / split / class_name
                destination_class_dir.mkdir(parents=True, exist_ok=True)

                for image in images:
                    destination_path = destination_class_dir / image.path.name
                    if destination_path.exists():
                        raise FileExistsError(f"File dich da ton tai: {destination_path}")
                    shutil.copy2(image.path, destination_path)
                    progress.update(1)

    return staging_root


def validate_materialized_split(
    staging_root: Path,
    plan: dict[str, dict[str, list[SourceImage]]],
    config: DatasetSplitConfig,
) -> None:
    """Đối chiếu staging với plan trước khi publish ra dataset chính."""

    for split, class_map in plan.items():
        split_dir = staging_root / split
        if not split_dir.exists():
            raise ValueError(f"Thieu split staging: {split_dir}")

        for class_name, images in class_map.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Thieu class staging: {class_dir}")

            output_files = list_image_files(class_dir, config.valid_extensions)
            expected_names = sorted(image.path.name for image in images)
            actual_names = sorted(path.name for path in output_files)
            if actual_names != expected_names:
                raise ValueError(
                    f"File staging khong khop plan: split={split}, class={class_name}"
                )

            if not output_files:
                raise ValueError(f"Class staging rong: split={split}, class={class_name}")


def publish_split(staging_root: Path, config: DatasetSplitConfig) -> None:
    """Publish staging thành `dataset/train|val|test`.

    Nếu không có `clean_output=True`, hàm sẽ không ghi đè output đang tồn tại.
    """

    config.dataset_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        source_dir = staging_root / split
        destination_dir = config.dataset_root / split

        if destination_dir.exists():
            if not config.clean_output:
                raise FileExistsError(
                    f"Thu muc output da ton tai, can --clean-output: {destination_dir}"
                )
            shutil.rmtree(destination_dir)

        shutil.move(str(source_dir), str(destination_dir))


def build_result(
    config: DatasetSplitConfig,
    stats_list: list[ClassSplitStats],
) -> DatasetSplitResult:
    """Tổng hợp kết quả split thành object dễ ghi report/log."""

    per_class_counts = {
        stats.class_name: {
            "input_images": stats.input_images,
            "unique_images": stats.unique_images,
            "duplicates_removed": stats.duplicates_removed,
            "train": stats.train_count,
            "val": stats.val_count,
            "test": stats.test_count,
        }
        for stats in stats_list
    }
    split_totals = {
        "train": sum(stats.train_count for stats in stats_list),
        "val": sum(stats.val_count for stats in stats_list),
        "test": sum(stats.test_count for stats in stats_list),
    }
    duplicate_counts = {
        stats.class_name: stats.duplicates_removed for stats in stats_list
    }

    return DatasetSplitResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        input_dir=config.input_dir,
        dataset_root=config.dataset_root,
        class_names=[stats.class_name for stats in stats_list],
        class_count=len(stats_list),
        split_totals=split_totals,
        per_class_counts=per_class_counts,
        duplicate_counts=duplicate_counts,
        duplicates_removed_total=sum(duplicate_counts.values()),
    )


def split_dataset(
    config: DatasetSplitConfig,
    logger: logging.Logger | None = None,
) -> DatasetSplitResult:
    """Chạy toàn bộ bước split dataset chính."""

    active_logger = logger or setup_logger()
    validate_config(config)
    active_logger.info("Bat dau split dataset chinh | config=%s", asdict(config))

    staging_root: Path | None = None
    try:
        class_to_images, input_counts, duplicate_counts = collect_unique_images_by_class(
            config=config,
            logger=active_logger,
        )
        plan, stats_list = build_split_plan(
            class_to_images=class_to_images,
            input_counts=input_counts,
            duplicate_counts=duplicate_counts,
            config=config,
            logger=active_logger,
        )
        staging_root = materialize_split_plan(plan=plan, config=config)
        validate_materialized_split(staging_root=staging_root, plan=plan, config=config)
        publish_split(staging_root=staging_root, config=config)
        staging_root = None

        result = build_result(config=config, stats_list=stats_list)
        active_logger.info(
            "[split_total] classes=%d | train=%d | val=%d | test=%d | "
            "duplicates_removed=%d",
            result.class_count,
            result.split_totals["train"],
            result.split_totals["val"],
            result.split_totals["test"],
            result.duplicates_removed_total,
        )
        active_logger.info("Hoan tat split dataset chinh.")
        return result
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def result_to_dict(result: DatasetSplitResult) -> dict[str, Any]:
    """Chuyển result thành dict JSON-safe nếu cần dùng ở report khác."""

    payload = asdict(result)
    payload["input_dir"] = str(result.input_dir)
    payload["dataset_root"] = str(result.dataset_root)
    return payload


def main() -> None:
    """Entry point: `python src/splitter.py --clean-output`."""

    args = parse_args()
    config = build_config(args)
    split_dataset(config=config)


if __name__ == "__main__":
    main()
