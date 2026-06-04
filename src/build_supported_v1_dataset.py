from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


SPLITS: tuple[str, ...] = ("train", "val", "test")
OTHER_CLASS_NAME = "other"
VALID_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
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
class SupportedV1Config:
    """Config build dataset Stage B production v1.

    source_root:
        Dataset Stage B gốc đã split, chỉ đọc, không sửa.
    output_root:
        Dataset production v1 chỉ gồm class supported.
    clean_output:
        Nếu False thì không ghi đè output đang tồn tại.
    """

    source_root: Path = Path("dataset_fruit_only")
    output_root: Path = Path("dataset_fruit_supported_v1")
    log_file: Path = Path("logs/build_supported_v1_dataset.log")
    clean_output: bool = False
    valid_extensions: tuple[str, ...] = VALID_EXTENSIONS
    unsupported_classes: tuple[str, ...] = UNSUPPORTED_CLASSES_V1


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI."""

    parser = argparse.ArgumentParser(
        description="Build dataset_fruit_supported_v1 from dataset_fruit_only."
    )
    parser.add_argument("--source-root", type=Path, default=Path("dataset_fruit_only"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_fruit_supported_v1"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/build_supported_v1_dataset.log"),
    )
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SupportedV1Config:
    """Ghép CLI args thành config."""

    return SupportedV1Config(
        source_root=args.source_root,
        output_root=args.output_root,
        log_file=args.log_file,
        clean_output=args.clean_output,
    )


def setup_logger(log_file: Path) -> logging.Logger:
    """Tạo logger ghi cả terminal và file log."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_supported_v1_dataset")
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


def list_class_dirs(split_dir: Path) -> list[Path]:
    """Liệt kê class folder trong một split theo thứ tự ổn định."""

    if not split_dir.exists():
        raise FileNotFoundError(f"Khong tim thay split dir: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"Split dir khong phai thu muc: {split_dir}")

    class_dirs = sorted(
        [path for path in split_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    if not class_dirs:
        raise ValueError(f"Split dir khong co class folder: {split_dir}")
    return class_dirs


def list_class_names(split_dir: Path) -> list[str]:
    """Lấy danh sách tên class trong split."""

    return [path.name for path in list_class_dirs(split_dir)]


def list_image_files(class_dir: Path, valid_extensions: tuple[str, ...]) -> list[Path]:
    """Liệt kê file ảnh hợp lệ trong một class folder."""

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


def read_class_names_file(source_root: Path) -> list[str]:
    """Đọc class order từ `dataset_fruit_only/class_names.txt` nếu có."""

    class_names_path = source_root / "class_names.txt"
    if not class_names_path.exists():
        return []
    return [
        line.strip()
        for line in class_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_config(config: SupportedV1Config) -> None:
    """Validate config, đặc biệt là không được ghi vào dataset gốc/raw."""

    if not config.source_root.exists():
        raise FileNotFoundError(f"Khong tim thay source_root: {config.source_root}")
    if not config.source_root.is_dir():
        raise NotADirectoryError(f"source_root khong phai thu muc: {config.source_root}")

    source_root = config.source_root.resolve()
    output_root = config.output_root.resolve()
    raw_dir = Path("dataset/raw").resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError(f"Output khong duoc trung/nam trong source_root: {config.output_root}")
    if output_root == raw_dir or raw_dir in output_root.parents:
        raise ValueError(f"Output khong duoc trung/nam trong dataset/raw: {config.output_root}")

    if config.output_root.exists() and not config.clean_output:
        raise FileExistsError(
            "Output da ton tai. Hay them --clean-output neu muon tao lai: "
            f"{config.output_root}"
        )

    if len(set(config.unsupported_classes)) != len(config.unsupported_classes):
        raise ValueError(f"unsupported_classes bi trung: {config.unsupported_classes}")


def validate_source_dataset(config: SupportedV1Config) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Validate dataset_fruit_only trước khi copy.

    Điều kiện:
    - train/val/test có cùng class list.
    - không có class `other`.
    - không có class rỗng.
    - unsupported class phải thật sự tồn tại trong source để tránh typo.
    """

    reference_class_names: list[str] | None = None
    source_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}

    for split in SPLITS:
        split_dir = config.source_root / split
        class_names = list_class_names(split_dir)
        if reference_class_names is None:
            reference_class_names = class_names
        elif class_names != reference_class_names:
            raise ValueError(
                f"Class list split '{split}' khong khop train. "
                f"train={reference_class_names}, {split}={class_names}"
            )

        if OTHER_CLASS_NAME in class_names:
            raise ValueError(f"Stage B source khong duoc chua class other: {split_dir}")

        for class_name in class_names:
            image_count = len(
                list_image_files(split_dir / class_name, config.valid_extensions)
            )
            if image_count <= 0:
                raise ValueError(f"Source co class rong: {split}/{class_name}")
            source_counts[split][class_name] = image_count

    assert reference_class_names is not None
    class_names_file = read_class_names_file(config.source_root)
    if class_names_file and class_names_file != reference_class_names:
        raise ValueError(
            "class_names.txt cua source khong khop folder train. "
            f"class_names_txt={class_names_file}, train={reference_class_names}"
        )

    missing_unsupported = sorted(set(config.unsupported_classes) - set(reference_class_names))
    if missing_unsupported:
        raise ValueError(
            "Unsupported class v1 khong ton tai trong source, can kiem tra typo: "
            f"{missing_unsupported}"
        )

    return reference_class_names, source_counts


def build_supported_class_names(
    source_class_names: list[str],
    unsupported_classes: tuple[str, ...],
) -> list[str]:
    """Tạo danh sách class supported v1, tự tính số class."""

    unsupported_set = set(unsupported_classes)
    supported_class_names = [
        class_name for class_name in source_class_names if class_name not in unsupported_set
    ]
    if not supported_class_names:
        raise ValueError("Khong con class supported nao sau khi loai unsupported.")
    if OTHER_CLASS_NAME in supported_class_names:
        raise ValueError("supported_classes_v1 khong duoc chua other.")
    return supported_class_names


def prepare_staging_root(output_root: Path) -> Path:
    """Tạo staging output để tránh sinh dataset nửa chừng."""

    staging_root = output_root.with_name(f"{output_root.name}_tmp_{os.getpid()}")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def copy_supported_dataset(
    config: SupportedV1Config,
    supported_class_names: list[str],
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    """Copy các class supported từ source sang staging rồi publish."""

    staging_root = prepare_staging_root(config.output_root)
    output_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    total_class_copies = len(SPLITS) * len(supported_class_names)

    try:
        with tqdm(total=total_class_copies, desc="Copy supported v1", leave=True) as progress:
            for split in SPLITS:
                for class_name in supported_class_names:
                    source_class_dir = config.source_root / split / class_name
                    output_class_dir = staging_root / split / class_name
                    image_count = len(
                        list_image_files(source_class_dir, config.valid_extensions)
                    )
                    if image_count <= 0:
                        raise ValueError(f"Class supported rong truoc khi copy: {source_class_dir}")

                    shutil.copytree(source_class_dir, output_class_dir)
                    output_counts[split][class_name] = image_count
                    logger.info(
                        "[copy] split=%s | class=%s | images=%d",
                        split,
                        class_name,
                        image_count,
                    )
                    progress.update(1)

        write_text_list(staging_root / "supported_classes_v1.txt", supported_class_names)
        write_text_list(staging_root / "unsupported_classes_v1.txt", list(config.unsupported_classes))
        validate_output_dataset(
            output_root=staging_root,
            supported_class_names=supported_class_names,
            unsupported_classes=config.unsupported_classes,
            valid_extensions=config.valid_extensions,
        )

        if config.output_root.exists():
            shutil.rmtree(config.output_root)
        shutil.move(str(staging_root), str(config.output_root))
        return output_counts
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def write_text_list(output_path: Path, values: list[str]) -> None:
    """Ghi mỗi class một dòng."""

    output_path.write_text("\n".join(values) + "\n", encoding="utf-8")


def compute_sha256(file_path: Path) -> str:
    """Tính hash nội dung để phát hiện overlap thật giữa splits."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_dataset(
    output_root: Path,
    supported_class_names: list[str],
    unsupported_classes: tuple[str, ...],
    valid_extensions: tuple[str, ...],
) -> None:
    """Validate output production v1 sau khi copy."""

    unsupported_set = set(unsupported_classes)
    split_to_name_keys: dict[str, set[str]] = {}
    split_to_hashes: dict[str, set[str]] = {}

    for split in SPLITS:
        split_dir = output_root / split
        class_names = list_class_names(split_dir)
        if class_names != supported_class_names:
            raise ValueError(
                f"Output split '{split}' sai class list. "
                f"expected={supported_class_names}, actual={class_names}"
            )

        if OTHER_CLASS_NAME in class_names:
            raise ValueError(f"Output production v1 van co other: {split_dir}")

        leaked_unsupported = sorted(set(class_names) & unsupported_set)
        if leaked_unsupported:
            raise ValueError(
                f"Output production v1 van con unsupported class: {leaked_unsupported}"
            )

        name_keys: set[str] = set()
        content_hashes: set[str] = set()
        for class_name in class_names:
            image_files = list_image_files(split_dir / class_name, valid_extensions)
            if not image_files:
                raise ValueError(f"Output co class rong: {split}/{class_name}")

            for image_path in image_files:
                name_key = f"{class_name}/{image_path.name}"
                if name_key in name_keys:
                    raise ValueError(f"File lap trong split {split}: {name_key}")
                name_keys.add(name_key)

                content_hash = compute_sha256(image_path)
                if content_hash in content_hashes:
                    raise ValueError(
                        f"Duplicate hash trong cung split {split}: {content_hash[:12]}"
                    )
                content_hashes.add(content_hash)

        split_to_name_keys[split] = name_keys
        split_to_hashes[split] = content_hashes

    for index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[index + 1 :]:
            name_overlap = split_to_name_keys[left_split] & split_to_name_keys[right_split]
            if name_overlap:
                raise ValueError(
                    f"File overlap giua {left_split} va {right_split}: "
                    f"{sorted(name_overlap)[0]}"
                )

            hash_overlap = split_to_hashes[left_split] & split_to_hashes[right_split]
            if hash_overlap:
                raise ValueError(
                    f"Anh trung noi dung giua {left_split} va {right_split}: "
                    f"{sorted(hash_overlap)[0][:12]}"
                )


def build_report(
    config: SupportedV1Config,
    source_class_names: list[str],
    supported_class_names: list[str],
    source_counts: dict[str, dict[str, int]],
    output_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Tạo report JSON cho production v1."""

    split_totals = {
        split: sum(class_counts.values())
        for split, class_counts in output_counts.items()
    }
    per_class_counts = {
        class_name: {
            split: output_counts[split][class_name] for split in SPLITS
        }
        for class_name in supported_class_names
    }
    unsupported_source_counts = {
        class_name: {
            split: source_counts[split][class_name] for split in SPLITS
        }
        for class_name in config.unsupported_classes
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "build_supported_v1_dataset",
        "config": {
            **asdict(config),
            "source_root": str(config.source_root),
            "output_root": str(config.output_root),
            "log_file": str(config.log_file),
            "valid_extensions": list(config.valid_extensions),
        },
        "source_class_count": len(source_class_names),
        "supported_class_count": len(supported_class_names),
        "unsupported_class_count": len(config.unsupported_classes),
        "supported_classes": supported_class_names,
        "unsupported_classes": list(config.unsupported_classes),
        "split_totals": split_totals,
        "per_class_counts": per_class_counts,
        "unsupported_source_counts": unsupported_source_counts,
        "source_split_totals": {
            split: sum(source_counts[split].values()) for split in SPLITS
        },
        "note": (
            "Production v1 chi giu supported_classes de train Stage B. "
            "Unsupported classes duoc ghi vao unsupported_classes_v1.txt/manual_review list, "
            "khong copy vao train/val/test output."
        ),
    }


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_supported_v1_dataset(config: SupportedV1Config) -> Path:
    """Điều phối build dataset production v1, không train model."""

    validate_config(config)
    logger = setup_logger(config.log_file)
    logger.info("Bat dau build supported v1 dataset | config=%s", asdict(config))

    source_class_names, source_counts = validate_source_dataset(config)
    supported_class_names = build_supported_class_names(
        source_class_names=source_class_names,
        unsupported_classes=config.unsupported_classes,
    )

    logger.info("Source classes=%d | supported=%d | unsupported=%d", len(source_class_names), len(supported_class_names), len(config.unsupported_classes))
    logger.info("Unsupported classes v1=%s", list(config.unsupported_classes))
    logger.info("Supported classes v1=%s", supported_class_names)

    output_counts = copy_supported_dataset(
        config=config,
        supported_class_names=supported_class_names,
        logger=logger,
    )
    report = build_report(
        config=config,
        source_class_names=source_class_names,
        supported_class_names=supported_class_names,
        source_counts=source_counts,
        output_counts=output_counts,
    )

    report_path = config.output_root / "supported_v1_report.json"
    save_json(report_path, report)
    logger.info("Da luu report: %s", report_path)
    logger.info("Hoan tat supported v1 dataset | split_totals=%s", report["split_totals"])
    return report_path


def main() -> None:
    """Entry point: `python src/build_supported_v1_dataset.py --clean-output`."""

    args = parse_args()
    config = build_config(args)
    build_supported_v1_dataset(config)


if __name__ == "__main__":
    main()
