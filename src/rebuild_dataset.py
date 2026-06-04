from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.build_fruit_only_dataset import FruitOnlyConfig, build_fruit_only_dataset
    from src.preprocess import (
        PreprocessResult,
        RebuildPreprocessConfig,
        dataclass_to_report_dict,
        preprocess_dataset,
    )
    from src.smart_crop import ClassCropStats, SmartCropConfig, run_smart_crop
    from src.splitter import DatasetSplitConfig, split_dataset
    from src.taxonomy import build_taxonomy_plan
    from src.validate_dataset import (
        DatasetValidationConfig,
        build_raw_snapshot,
        validate_rebuild_split_outputs,
    )
except ImportError:  # pragma: no cover
    from build_fruit_only_dataset import FruitOnlyConfig, build_fruit_only_dataset
    from preprocess import (
        PreprocessResult,
        RebuildPreprocessConfig,
        dataclass_to_report_dict,
        preprocess_dataset,
    )
    from smart_crop import ClassCropStats, SmartCropConfig, run_smart_crop
    from splitter import DatasetSplitConfig, split_dataset
    from taxonomy import build_taxonomy_plan
    from validate_dataset import (
        DatasetValidationConfig,
        build_raw_snapshot,
        validate_rebuild_split_outputs,
    )


@dataclass(frozen=True)
class RebuildDatasetConfig:
    """Config tổng cho rebuild dataset, không train model."""

    raw_dir: Path = Path("dataset/raw")
    dataset_root: Path = Path("dataset")
    processed_clean_dir: Path = Path("dataset/processed_clean")
    processed_crop_dir: Path = Path("dataset/processed_crop")
    fruit_only_root: Path = Path("dataset_fruit_only")
    log_file: Path = Path("logs/rebuild_dataset.log")
    report_file: Path = Path("dataset/dataset_report.json")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_size: tuple[int, int] = (320, 320)
    valid_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
    )
    min_image_side: int = 100
    max_image_side: int = 5000
    max_aspect_ratio: float = 4.0
    blur_threshold: float = 100.0
    duplicate_hash_tolerance: int = 0
    random_seed: int = 42
    clean_output: bool = False


def parse_args() -> argparse.Namespace:
    """CLI chính cho rebuild dataset."""

    parser = argparse.ArgumentParser(
        description=(
            "Rebuild dataset pipeline: raw -> processed_clean -> processed_crop "
            "hoac split processed_crop -> train/val/test -> dataset_fruit_only."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("preprocess", "crop", "split"),
        default="preprocess",
        help=(
            "preprocess: build processed_clean va processed_crop tu raw. "
            "crop: chi build processed_crop tu processed_clean hien co. "
            "split: build dataset/train|val|test va dataset_fruit_only."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--processed-clean-dir",
        type=Path,
        default=Path("dataset/processed_clean"),
    )
    parser.add_argument(
        "--processed-crop-dir",
        type=Path,
        default=Path("dataset/processed_crop"),
    )
    parser.add_argument(
        "--fruit-only-root",
        type=Path,
        default=Path("dataset_fruit_only"),
    )
    parser.add_argument("--log-file", type=Path, default=Path("logs/rebuild_dataset.log"))
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("dataset/dataset_report.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument("--min-image-side", type=int, default=100)
    parser.add_argument("--max-image-side", type=int, default=5000)
    parser.add_argument("--max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--duplicate-hash-tolerance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help=(
            "Cho phep tao lai output cua stage dang chay. Stage preprocess/crop se "
            "clean processed output; stage split se clean dataset/train|val|test "
            "va dataset_fruit_only."
        ),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RebuildDatasetConfig:
    """Ghép CLI args thành config."""

    return RebuildDatasetConfig(
        raw_dir=args.raw_dir,
        dataset_root=args.dataset_root,
        processed_clean_dir=args.processed_clean_dir,
        processed_crop_dir=args.processed_crop_dir,
        fruit_only_root=args.fruit_only_root,
        log_file=args.log_file,
        report_file=args.report_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        target_size=(args.target_size, args.target_size),
        min_image_side=args.min_image_side,
        max_image_side=args.max_image_side,
        max_aspect_ratio=args.max_aspect_ratio,
        blur_threshold=args.blur_threshold,
        duplicate_hash_tolerance=args.duplicate_hash_tolerance,
        random_seed=args.seed,
        clean_output=args.clean_output,
    )


def setup_rebuild_logger(log_file: Path) -> logging.Logger:
    """Logger ghi cả file và terminal cho một lần rebuild."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rebuild_dataset")
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


def make_json_safe(value: Any) -> Any:
    """Convert Path/Counter/dataclass thành JSON-safe."""

    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "__dataclass_fields__"):
        return make_json_safe(asdict(value))
    return value


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 indent rõ ràng."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_preprocess_config(config: RebuildDatasetConfig) -> RebuildPreprocessConfig:
    """Config riêng cho bước processed_clean."""

    return RebuildPreprocessConfig(
        raw_dir=config.raw_dir,
        output_dir=config.processed_clean_dir,
        target_size=config.target_size,
        valid_extensions=config.valid_extensions,
        min_image_side=config.min_image_side,
        max_image_side=config.max_image_side,
        max_aspect_ratio=config.max_aspect_ratio,
        blur_threshold=config.blur_threshold,
        duplicate_hash_tolerance=config.duplicate_hash_tolerance,
        random_seed=config.random_seed,
        clean_output=config.clean_output,
    )


def build_crop_config(config: RebuildDatasetConfig) -> SmartCropConfig:
    """Config riêng cho bước processed_crop."""

    return SmartCropConfig(
        input_dir=config.processed_clean_dir,
        output_dir=config.processed_crop_dir,
        log_file=config.log_file,
        target_size=config.target_size,
        valid_extensions=(".jpg", ".jpeg", ".png"),
        min_image_side=config.min_image_side,
        max_image_side=config.max_image_side,
        max_aspect_ratio=config.max_aspect_ratio,
        blur_threshold=config.blur_threshold,
        clean_output_dir=config.clean_output,
    )


def build_split_config(config: RebuildDatasetConfig) -> DatasetSplitConfig:
    """Config riêng cho bước split dataset chính.

    Input luôn là `processed_crop`; output là `dataset/train|val|test`.
    """

    return DatasetSplitConfig(
        input_dir=config.processed_crop_dir,
        dataset_root=config.dataset_root,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.random_seed,
        clean_output=config.clean_output,
        valid_extensions=config.valid_extensions,
    )


def build_fruit_only_config(config: RebuildDatasetConfig) -> FruitOnlyConfig:
    """Config tạo dataset riêng cho Stage B, bỏ class `other`."""

    return FruitOnlyConfig(
        source_root=config.dataset_root,
        output_root=config.fruit_only_root,
        clean_output=config.clean_output,
        valid_extensions=config.valid_extensions,
    )


def build_validation_config(config: RebuildDatasetConfig) -> DatasetValidationConfig:
    """Config validation tổng sau stage split."""

    return DatasetValidationConfig(
        raw_dir=config.raw_dir,
        processed_clean_dir=config.processed_clean_dir,
        processed_crop_dir=config.processed_crop_dir,
        dataset_root=config.dataset_root,
        fruit_only_root=config.fruit_only_root,
        valid_extensions=config.valid_extensions,
    )


def crop_stats_to_dict(crop_stats: list[ClassCropStats]) -> dict[str, dict[str, Any]]:
    """Convert crop stats thành dict theo class."""

    return {
        stats.class_name: {
            "total_images": stats.total,
            "crop_success": stats.cropped,
            "kept_original": stats.kept_original,
            "failed": stats.failed,
            "reasons": dict(sorted(stats.reasons.items())),
        }
        for stats in crop_stats
    }


def build_report(
    config: RebuildDatasetConfig,
    preprocess_result: PreprocessResult | None,
    crop_stats: list[ClassCropStats],
) -> dict[str, Any]:
    """Tạo dataset_report.json theo đúng thông tin cần audit."""

    crop_by_class = crop_stats_to_dict(crop_stats)
    if preprocess_result is None:
        plan = build_taxonomy_plan(config.raw_dir)
        raw_class_mapping = [
            {
                "raw_class": mapping.raw_class_name,
                "class_after_taxonomy": mapping.taxonomy_class_name,
                "raw_images": None,
                "valid_images": None,
                "skipped_images": None,
                "skip_reasons": {},
            }
            for mapping in plan.raw_mappings
        ]
        class_report = {
            class_name: {
                "source_raw_classes": list(plan.class_to_raw_classes[class_name]),
                "raw_images": None,
                "valid_images": None,
                "skipped_images": None,
                "skip_reasons": {},
                "crop_success": crop_by_class.get(class_name, {}).get("crop_success", 0),
                "kept_original": crop_by_class.get(class_name, {}).get("kept_original", 0),
                "crop_failed": crop_by_class.get(class_name, {}).get("failed", 0),
                "crop_reasons": crop_by_class.get(class_name, {}).get("reasons", {}),
            }
            for class_name in plan.output_class_names
        }
        output_class_names = list(plan.output_class_names)
        merge_rules = plan.merge_rules
    else:
        plan = preprocess_result.taxonomy_plan
        raw_class_mapping = [
            {
                "raw_class": stats.raw_class_name,
                "class_after_taxonomy": stats.taxonomy_class_name,
                "raw_images": stats.raw_images,
                "valid_images": stats.valid_images,
                "skipped_images": stats.skipped_images,
                "skip_reasons": stats.skip_reasons,
            }
            for stats in preprocess_result.raw_class_stats.values()
        ]
        class_report = {}
        for class_name, stats in preprocess_result.class_stats.items():
            crop_item = crop_by_class.get(class_name, {})
            class_report[class_name] = {
                "source_raw_classes": stats.source_raw_classes,
                "raw_images": stats.raw_images,
                "valid_images": stats.valid_images,
                "skipped_images": stats.skipped_images,
                "skip_reasons": stats.skip_reasons,
                "crop_success": crop_item.get("crop_success", 0),
                "kept_original": crop_item.get("kept_original", 0),
                "crop_failed": crop_item.get("failed", 0),
                "crop_reasons": crop_item.get("reasons", {}),
            }
        output_class_names = list(plan.output_class_names)
        merge_rules = plan.merge_rules

    totals = {
        "raw_images": sum(
            item.get("raw_images") or 0 for item in class_report.values()
        ),
        "valid_images": sum(
            item.get("valid_images") or 0 for item in class_report.values()
        ),
        "skipped_images": sum(
            item.get("skipped_images") or 0 for item in class_report.values()
        ),
        "crop_success": sum(item["crop_success"] for item in class_report.values()),
        "kept_original": sum(item["kept_original"] for item in class_report.values()),
        "crop_failed": sum(item["crop_failed"] for item in class_report.values()),
        "raw_class_count": len(raw_class_mapping),
        "output_class_count": len(output_class_names),
    }

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "rebuild_dataset",
        "config": dataclass_to_report_dict(config),
        "taxonomy": {
            "merge_rules": merge_rules,
            "output_class_names": output_class_names,
            "output_class_count": len(output_class_names),
            "note": "Class other duoc giu lai cho Stage A; cac class cu da merge khong xuat hien o output.",
        },
        "totals": totals,
        "raw_class_mapping": raw_class_mapping,
        "classes": class_report,
    }


def validate_raw_is_read_only(config: RebuildDatasetConfig) -> None:
    """Chặn config có nguy cơ ghi vào dataset/raw."""

    raw_dir = config.raw_dir.resolve()
    output_dirs = (
        config.processed_clean_dir,
        config.processed_crop_dir,
        config.dataset_root / "train",
        config.dataset_root / "val",
        config.dataset_root / "test",
        config.fruit_only_root,
    )
    for output_dir in output_dirs:
        resolved_output = output_dir.resolve()
        if resolved_output == raw_dir:
            raise ValueError(f"Output dir khong duoc trung raw_dir: {output_dir}")
        if raw_dir in resolved_output.parents:
            raise ValueError(f"Output dir khong duoc nam trong raw_dir: {output_dir}")


def run_rebuild(stage: str, config: RebuildDatasetConfig) -> Path:
    """Điều phối rebuild dataset, tuyệt đối không train model."""

    validate_raw_is_read_only(config)
    logger = setup_rebuild_logger(config.log_file)
    logger.info("Bat dau rebuild dataset | stage=%s | config=%s", stage, config)

    if stage == "split":
        raw_snapshot_before = build_raw_snapshot(config.raw_dir)
        split_result = split_dataset(
            config=build_split_config(config),
            logger=logger,
        )
        fruit_only_result = build_fruit_only_dataset(
            config=build_fruit_only_config(config),
            logger=logger,
        )
        validation_summary = validate_rebuild_split_outputs(
            config=build_validation_config(config),
            logger=logger,
            raw_snapshot_before=raw_snapshot_before,
        )

        logger.info(
            "Split dataset chinh xong | classes=%d | totals=%s",
            split_result.class_count,
            split_result.split_totals,
        )
        logger.info(
            "Dataset fruit-only xong | classes=%d | totals=%s | report=%s",
            fruit_only_result.class_count,
            fruit_only_result.split_totals,
            fruit_only_result.report_path,
        )
        logger.info(
            "Validation summary | main_classes=%d | fruit_classes=%d",
            validation_summary["main_dataset"]["class_count"],
            validation_summary["fruit_only_dataset"]["class_count"],
        )
        logger.info("Hoan tat rebuild dataset stage split.")
        return fruit_only_result.report_path

    preprocess_result: PreprocessResult | None = None
    if stage == "preprocess":
        preprocess_result = preprocess_dataset(
            config=build_preprocess_config(config),
            logger=logger,
        )

    crop_stats = run_smart_crop(
        config=build_crop_config(config),
        logger=logger,
    )
    report = build_report(
        config=config,
        preprocess_result=preprocess_result,
        crop_stats=crop_stats,
    )
    save_json(config.report_file, report)

    logger.info("Da luu report: %s", config.report_file)
    logger.info("Hoan tat rebuild dataset.")
    return config.report_file


def main() -> None:
    """Entry point chính.

    Ví dụ:
    - `python src/rebuild_dataset.py --stage preprocess --clean-output`
    - `python src/rebuild_dataset.py --stage split --clean-output`
    """

    args = parse_args()
    config = build_config(args)
    run_rebuild(stage=args.stage, config=config)


if __name__ == "__main__":
    main()
