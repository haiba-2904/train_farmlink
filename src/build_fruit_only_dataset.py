from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from src.splitter import DEFAULT_VALID_EXTENSIONS, SPLITS, list_class_dirs, list_image_files
except ImportError:  # pragma: no cover
    from splitter import DEFAULT_VALID_EXTENSIONS, SPLITS, list_class_dirs, list_image_files


OTHER_CLASS_NAME = "other"


@dataclass(frozen=True)
class FruitOnlyConfig:
    """Config tạo dataset riêng cho Stage B.

    source_root:
        Dataset chính đã split, gồm `train`, `val`, `test` và vẫn có class `other`.
    output_root:
        Dataset Stage B, chỉ giữ class nông sản, loại bỏ `other`.
    clean_output:
        Nếu False và output đã tồn tại thì raise lỗi để tránh ghi đè nhầm.
    """

    source_root: Path = Path("dataset")
    output_root: Path = Path("dataset_fruit_only")
    clean_output: bool = False
    class_names_file: str = "class_names.txt"
    report_file_name: str = "dataset_fruit_only_report.json"
    valid_extensions: tuple[str, ...] = DEFAULT_VALID_EXTENSIONS
    staging_dir_suffix: str = "_tmp"


@dataclass(frozen=True)
class FruitOnlyResult:
    """Kết quả tạo dataset Stage B."""

    created_at: str
    source_root: Path
    output_root: Path
    class_names: list[str]
    class_count: int
    skipped_class: str
    split_totals: dict[str, int]
    per_class_counts: dict[str, dict[str, int]]
    class_names_path: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    """CLI độc lập nếu cần build riêng `dataset_fruit_only`."""

    parser = argparse.ArgumentParser(
        description="Tao dataset_fruit_only cho Stage B bang cach loai bo class other."
    )
    parser.add_argument("--source-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset_fruit_only"))
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def normalize_class_token(value: str) -> str:
    """Chuẩn hóa tên folder để nhận diện chính xác class `other`."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def is_other_class(class_name: str) -> bool:
    """True nếu folder là class `other`."""

    return normalize_class_token(class_name) == OTHER_CLASS_NAME


def validate_source_splits(config: FruitOnlyConfig) -> list[str]:
    """Kiểm tra dataset chính trước khi tạo Stage B.

    Điều kiện:
    - Đủ train/val/test.
    - Mỗi split có cùng class list.
    - Mỗi split còn class `other` để loại ra.
    - Sau khi bỏ `other`, class nông sản không rỗng.
    """

    reference_class_names: list[str] | None = None
    for split in SPLITS:
        split_dir = config.source_root / split
        class_names = [class_dir.name for class_dir in list_class_dirs(split_dir)]

        if not any(is_other_class(class_name) for class_name in class_names):
            raise ValueError(f"Split dataset chinh khong co class other: {split_dir}")

        if reference_class_names is None:
            reference_class_names = class_names
        elif class_names != reference_class_names:
            raise ValueError(
                f"Danh sach class cua split '{split}' khong khop voi train."
            )

        for class_name in class_names:
            image_count = len(
                list_image_files(split_dir / class_name, config.valid_extensions)
            )
            if image_count <= 0:
                raise ValueError(f"Class rong trong dataset chinh: {split}/{class_name}")

    assert reference_class_names is not None
    fruit_class_names = [
        class_name for class_name in reference_class_names if not is_other_class(class_name)
    ]
    if not fruit_class_names:
        raise ValueError("Khong tim thay class nong san nao sau khi bo other.")
    return fruit_class_names


def ensure_output_can_be_written(config: FruitOnlyConfig) -> None:
    """Không ghi đè `dataset_fruit_only` nếu người dùng chưa bật clean."""

    if config.output_root.exists() and not config.clean_output:
        raise FileExistsError(
            "dataset_fruit_only da ton tai. Hay them --clean-output neu muon tao lai: "
            f"{config.output_root}"
        )


def prepare_staging_root(config: FruitOnlyConfig) -> Path:
    """Tạo staging tạm để tránh output nửa chừng khi đang copy."""

    staging_root = config.output_root.with_name(
        f"{config.output_root.name}{config.staging_dir_suffix}_{os.getpid()}"
    )
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def copy_fruit_only_splits(
    config: FruitOnlyConfig,
    class_names: list[str],
    staging_root: Path,
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    """Copy từng split sang staging, bỏ hẳn class `other`."""

    per_class_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    total_classes = len(class_names) * len(SPLITS)

    with tqdm(total=total_classes, desc="Copy fruit-only", leave=True) as progress:
        for split in SPLITS:
            source_split_dir = config.source_root / split
            output_split_dir = staging_root / split
            output_split_dir.mkdir(parents=True, exist_ok=True)

            for class_name in class_names:
                source_class_dir = source_split_dir / class_name
                output_class_dir = output_split_dir / class_name
                if not source_class_dir.exists():
                    raise FileNotFoundError(f"Thieu class source: {source_class_dir}")

                image_count = len(
                    list_image_files(source_class_dir, config.valid_extensions)
                )
                if image_count <= 0:
                    raise ValueError(f"Class fruit-only rong: {source_class_dir}")

                shutil.copytree(source_class_dir, output_class_dir)
                per_class_counts[split][class_name] = image_count
                logger.info("[fruit_only] split=%s | class=%s | images=%d", split, class_name, image_count)
                progress.update(1)

    return per_class_counts


def validate_staging_output(
    staging_root: Path,
    class_names: list[str],
    config: FruitOnlyConfig,
) -> None:
    """Validate staging trước khi publish thành `dataset_fruit_only`."""

    for split in SPLITS:
        split_dir = staging_root / split
        actual_class_names = [class_dir.name for class_dir in list_class_dirs(split_dir)]
        if actual_class_names != class_names:
            raise ValueError(
                f"Class list fruit-only staging sai: split={split}, actual={actual_class_names}"
            )
        if any(is_other_class(class_name) for class_name in actual_class_names):
            raise ValueError(f"Fruit-only staging van con class other: {split_dir}")

        for class_name in actual_class_names:
            image_count = len(
                list_image_files(split_dir / class_name, config.valid_extensions)
            )
            if image_count <= 0:
                raise ValueError(f"Class staging fruit-only rong: {split}/{class_name}")


def save_class_names(staging_root: Path, class_names: list[str], config: FruitOnlyConfig) -> Path:
    """Ghi class order cho Stage B."""

    class_names_path = staging_root / config.class_names_file
    class_names_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return class_names_path


def save_report(
    staging_root: Path,
    config: FruitOnlyConfig,
    class_names: list[str],
    per_class_counts: dict[str, dict[str, int]],
) -> Path:
    """Ghi JSON report để audit dataset Stage B."""

    split_totals = {
        split: sum(class_counts.values()) for split, class_counts in per_class_counts.items()
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "build_fruit_only_dataset",
        "config": {
            **asdict(config),
            "source_root": str(config.source_root),
            "output_root": str(config.output_root),
            "valid_extensions": list(config.valid_extensions),
        },
        "skipped_class": OTHER_CLASS_NAME,
        "class_count": len(class_names),
        "class_names": class_names,
        "split_totals": split_totals,
        "per_class_counts": per_class_counts,
        "note": "Dataset Stage B khong chua class other. So class duoc tu tinh tu dataset chinh.",
    }
    report_path = staging_root / config.report_file_name
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def publish_output(staging_root: Path, config: FruitOnlyConfig) -> None:
    """Publish staging ra `dataset_fruit_only`."""

    if config.output_root.exists():
        if not config.clean_output:
            raise FileExistsError(
                f"Output da ton tai, can --clean-output: {config.output_root}"
            )
        shutil.rmtree(config.output_root)
    shutil.move(str(staging_root), str(config.output_root))


def build_result(
    config: FruitOnlyConfig,
    class_names: list[str],
    per_class_counts: dict[str, dict[str, int]],
) -> FruitOnlyResult:
    """Tạo result sau khi publish output."""

    split_totals = {
        split: sum(class_counts.values()) for split, class_counts in per_class_counts.items()
    }
    return FruitOnlyResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        source_root=config.source_root,
        output_root=config.output_root,
        class_names=class_names,
        class_count=len(class_names),
        skipped_class=OTHER_CLASS_NAME,
        split_totals=split_totals,
        per_class_counts=per_class_counts,
        class_names_path=config.output_root / config.class_names_file,
        report_path=config.output_root / config.report_file_name,
    )


def build_fruit_only_dataset(
    config: FruitOnlyConfig,
    logger: logging.Logger,
) -> FruitOnlyResult:
    """Chạy toàn bộ bước tạo `dataset_fruit_only` cho Stage B."""

    ensure_output_can_be_written(config)
    class_names = validate_source_splits(config)
    logger.info(
        "Bat dau tao dataset_fruit_only | source=%s | output=%s | fruit_classes=%d",
        config.source_root,
        config.output_root,
        len(class_names),
    )

    staging_root: Path | None = None
    try:
        staging_root = prepare_staging_root(config)
        per_class_counts = copy_fruit_only_splits(
            config=config,
            class_names=class_names,
            staging_root=staging_root,
            logger=logger,
        )
        validate_staging_output(
            staging_root=staging_root,
            class_names=class_names,
            config=config,
        )
        save_class_names(
            staging_root=staging_root,
            class_names=class_names,
            config=config,
        )
        save_report(
            staging_root=staging_root,
            config=config,
            class_names=class_names,
            per_class_counts=per_class_counts,
        )
        publish_output(staging_root=staging_root, config=config)
        staging_root = None

        result = build_result(
            config=config,
            class_names=class_names,
            per_class_counts=per_class_counts,
        )
        logger.info(
            "[fruit_only_total] classes=%d | train=%d | val=%d | test=%d",
            result.class_count,
            result.split_totals["train"],
            result.split_totals["val"],
            result.split_totals["test"],
        )
        logger.info("Hoan tat tao dataset_fruit_only.")
        return result
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def result_to_dict(result: FruitOnlyResult) -> dict[str, Any]:
    """Chuyển result thành dict JSON-safe."""

    payload = asdict(result)
    payload["source_root"] = str(result.source_root)
    payload["output_root"] = str(result.output_root)
    payload["class_names_path"] = str(result.class_names_path)
    payload["report_path"] = str(result.report_path)
    return payload


def main() -> None:
    """Entry point: `python src/build_fruit_only_dataset.py --clean-output`."""

    args = parse_args()
    config = FruitOnlyConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        clean_output=args.clean_output,
    )
    logger = logging.getLogger("build_fruit_only_dataset")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    build_fruit_only_dataset(config=config, logger=logger)


if __name__ == "__main__":
    main()
