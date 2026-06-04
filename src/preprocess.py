from __future__ import annotations

import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from src.taxonomy import TaxonomyPlan, build_taxonomy_plan, normalize_class_name
    from src.utils import (
        HashRegistry,
        compute_blur_score,
        compute_perceptual_hash,
        open_image_safely,
        resize_with_padding,
        save_image,
        set_global_seed,
        validate_image_quality,
    )
except ImportError:  # pragma: no cover
    from taxonomy import TaxonomyPlan, build_taxonomy_plan, normalize_class_name
    from utils import (
        HashRegistry,
        compute_blur_score,
        compute_perceptual_hash,
        open_image_safely,
        resize_with_padding,
        save_image,
        set_global_seed,
        validate_image_quality,
    )


@dataclass(frozen=True)
class RebuildPreprocessConfig:
    """Config cho bước raw -> processed_clean."""

    raw_dir: Path = Path("dataset/raw")
    output_dir: Path = Path("dataset/processed_clean")
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
    jpeg_quality: int = 95
    random_seed: int = 42
    clean_output: bool = True
    padding_color: tuple[int, int, int] = (0, 0, 0)


@dataclass
class RawClassPreprocessStats:
    """Thống kê cho một class folder gốc trong dataset/raw."""

    raw_class_name: str
    taxonomy_class_name: str
    raw_images: int = 0
    valid_images: int = 0
    skipped_images: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class OutputClassPreprocessStats:
    """Thống kê sau khi đã gom các raw class về class taxonomy."""

    class_name: str
    source_raw_classes: list[str] = field(default_factory=list)
    raw_images: int = 0
    valid_images: int = 0
    skipped_images: int = 0
    skip_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class PreprocessResult:
    """Kết quả trả về để rebuild_dataset.py ghi dataset_report.json."""

    config: RebuildPreprocessConfig
    taxonomy_plan: TaxonomyPlan
    raw_class_stats: dict[str, RawClassPreprocessStats]
    class_stats: dict[str, OutputClassPreprocessStats]


def dataclass_to_report_dict(value: Any) -> Any:
    """Convert dataclass/Counter/Path thành object ghi JSON được."""

    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [dataclass_to_report_dict(item) for item in value]
    if isinstance(value, list):
        return [dataclass_to_report_dict(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): dataclass_to_report_dict(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "__dataclass_fields__"):
        return dataclass_to_report_dict(asdict(value))
    return value


def validate_preprocess_config(config: RebuildPreprocessConfig) -> None:
    """Fail fast nếu config có nguy cơ ghi nhầm raw hoặc sai tham số."""

    if not config.raw_dir.exists():
        raise FileNotFoundError(f"Khong tim thay raw_dir: {config.raw_dir}")
    if not config.raw_dir.is_dir():
        raise NotADirectoryError(f"raw_dir khong phai thu muc: {config.raw_dir}")
    if config.output_dir.resolve() == config.raw_dir.resolve():
        raise ValueError("output_dir khong duoc trung raw_dir.")
    if config.raw_dir.resolve() in config.output_dir.resolve().parents:
        raise ValueError("output_dir khong duoc nam ben trong raw_dir.")
    if config.target_size[0] <= 0 or config.target_size[1] <= 0:
        raise ValueError("target_size phai lon hon 0.")
    if config.min_image_side <= 0:
        raise ValueError("min_image_side phai lon hon 0.")
    if config.max_image_side <= config.min_image_side:
        raise ValueError("max_image_side phai lon hon min_image_side.")
    if config.max_aspect_ratio <= 1.0:
        raise ValueError("max_aspect_ratio phai lon hon 1.0.")
    if config.blur_threshold < 0:
        raise ValueError("blur_threshold khong duoc am.")
    if config.duplicate_hash_tolerance < 0:
        raise ValueError("duplicate_hash_tolerance khong duoc am.")


def prepare_output_dir(output_dir: Path, clean_output: bool) -> None:
    """Tạo hoặc làm sạch output. Hàm này không bao giờ thao tác dataset/raw."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean_output:
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def list_raw_files(class_dir: Path) -> list[Path]:
    """Liệt kê file trong raw class, bỏ hidden file."""

    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ],
        key=lambda path: path.name.lower(),
    )


def sanitize_file_token(value: str) -> str:
    """Tạo token an toàn cho tên file output."""

    sanitized = normalize_class_name(value)
    return sanitized or "image"


def build_unique_output_path(
    output_class_dir: Path,
    raw_class_name: str,
    image_path: Path,
    used_output_names: set[str],
) -> Path:
    """Sinh tên file output không overwrite khi nhiều raw class được gộp."""

    raw_token = sanitize_file_token(raw_class_name)
    stem_token = sanitize_file_token(image_path.stem)
    base_name = f"{raw_token}__{stem_token}"
    candidate_name = f"{base_name}.jpg"
    index = 2
    while candidate_name in used_output_names or (output_class_dir / candidate_name).exists():
        candidate_name = f"{base_name}__{index:04d}.jpg"
        index += 1

    used_output_names.add(candidate_name)
    return output_class_dir / candidate_name


def add_skip_reason(
    raw_stats: RawClassPreprocessStats,
    reason: str,
    logger: logging.Logger,
    image_path: Path,
    detail: str | None = None,
) -> None:
    """Cập nhật thống kê ảnh bị bỏ và ghi log."""

    raw_stats.skipped_images += 1
    raw_stats.skip_reasons[reason] += 1
    if detail:
        logger.warning(
            "[%s -> %s] skip %s | reason=%s | detail=%s",
            raw_stats.raw_class_name,
            raw_stats.taxonomy_class_name,
            image_path.name,
            reason,
            detail,
        )
        return

    logger.warning(
        "[%s -> %s] skip %s | reason=%s",
        raw_stats.raw_class_name,
        raw_stats.taxonomy_class_name,
        image_path.name,
        reason,
    )


def process_raw_image(
    image_path: Path,
    output_class_dir: Path,
    raw_stats: RawClassPreprocessStats,
    config: RebuildPreprocessConfig,
    hash_registry: HashRegistry,
    used_output_names: set[str],
    logger: logging.Logger,
) -> None:
    """Xử lý một ảnh raw: open -> quality -> blur -> resize/pad -> pHash -> save."""

    if image_path.suffix.lower() not in config.valid_extensions:
        add_skip_reason(raw_stats, "unsupported_extension", logger, image_path)
        return

    try:
        image = open_image_safely(image_path)
    except Exception as exc:  # noqa: BLE001
        add_skip_reason(raw_stats, "corrupted_image", logger, image_path, str(exc))
        return

    quality_ok, quality_reason = validate_image_quality(
        image=image,
        min_image_side=config.min_image_side,
        max_image_side=config.max_image_side,
        max_aspect_ratio=config.max_aspect_ratio,
    )
    if not quality_ok:
        add_skip_reason(
            raw_stats,
            quality_reason or "invalid_image",
            logger,
            image_path,
            detail=f"size={image.size}",
        )
        return

    blur_score = compute_blur_score(image)
    if blur_score < config.blur_threshold:
        add_skip_reason(
            raw_stats,
            "blurry_image",
            logger,
            image_path,
            detail=f"laplacian_variance={blur_score:.4f}",
        )
        return

    standardized_image = resize_with_padding(
        image=image,
        target_size=config.target_size,
        background_color=config.padding_color,
    )
    perceptual_hash = compute_perceptual_hash(standardized_image)
    if hash_registry.contains(perceptual_hash):
        add_skip_reason(
            raw_stats,
            "duplicate_image",
            logger,
            image_path,
            detail=f"phash={perceptual_hash}",
        )
        return

    output_path = build_unique_output_path(
        output_class_dir=output_class_dir,
        raw_class_name=raw_stats.raw_class_name,
        image_path=image_path,
        used_output_names=used_output_names,
    )
    save_image(
        image=standardized_image,
        destination=output_path,
        jpeg_quality=config.jpeg_quality,
    )
    hash_registry.add(perceptual_hash)
    raw_stats.valid_images += 1


def build_initial_output_stats(plan: TaxonomyPlan) -> dict[str, OutputClassPreprocessStats]:
    """Tạo stats rỗng cho từng output class taxonomy."""

    return {
        class_name: OutputClassPreprocessStats(
            class_name=class_name,
            source_raw_classes=list(plan.class_to_raw_classes[class_name]),
        )
        for class_name in plan.output_class_names
    }


def aggregate_raw_stats(
    raw_class_stats: dict[str, RawClassPreprocessStats],
    output_class_stats: dict[str, OutputClassPreprocessStats],
) -> None:
    """Gộp stats raw folder thành stats theo class taxonomy."""

    for raw_stats in raw_class_stats.values():
        class_stats = output_class_stats[raw_stats.taxonomy_class_name]
        class_stats.raw_images += raw_stats.raw_images
        class_stats.valid_images += raw_stats.valid_images
        class_stats.skipped_images += raw_stats.skipped_images
        class_stats.skip_reasons.update(raw_stats.skip_reasons)


def validate_output_taxonomy(plan: TaxonomyPlan, output_dir: Path) -> None:
    """Đảm bảo output chỉ có class taxonomy mới, không còn class cũ đã merge."""

    output_folder_names = {
        path.name for path in output_dir.iterdir() if path.is_dir()
    }
    expected_folder_names = set(plan.output_class_names)
    unexpected_folders = sorted(output_folder_names - expected_folder_names)
    missing_folders = sorted(expected_folder_names - output_folder_names)
    if unexpected_folders:
        raise ValueError(f"processed_clean co folder ngoai taxonomy: {unexpected_folders}")
    if missing_folders:
        raise ValueError(f"processed_clean thieu folder taxonomy: {missing_folders}")


def preprocess_dataset(
    config: RebuildPreprocessConfig,
    logger: logging.Logger,
) -> PreprocessResult:
    """Build `dataset/processed_clean` từ `dataset/raw` theo taxonomy mới."""

    validate_preprocess_config(config)
    set_global_seed(config.random_seed)
    plan = build_taxonomy_plan(config.raw_dir)
    logger.info("Taxonomy output classes=%s", list(plan.output_class_names))
    logger.info("Taxonomy class_to_raw=%s", plan.class_to_raw_classes)

    prepare_output_dir(config.output_dir, clean_output=config.clean_output)

    hash_registry_by_class = {
        class_name: HashRegistry(tolerance=config.duplicate_hash_tolerance)
        for class_name in plan.output_class_names
    }
    used_output_names_by_class: dict[str, set[str]] = defaultdict(set)
    raw_class_stats: dict[str, RawClassPreprocessStats] = {}

    for class_name in plan.output_class_names:
        (config.output_dir / class_name).mkdir(parents=True, exist_ok=True)

    for mapping in tqdm(plan.raw_mappings, desc="Preprocess raw classes", leave=True):
        output_class_dir = config.output_dir / mapping.taxonomy_class_name
        raw_stats = RawClassPreprocessStats(
            raw_class_name=mapping.raw_class_name,
            taxonomy_class_name=mapping.taxonomy_class_name,
        )
        raw_files = list_raw_files(mapping.raw_dir)
        raw_stats.raw_images = len(raw_files)
        raw_class_stats[mapping.raw_class_name] = raw_stats
        logger.info(
            "[%s -> %s] start | raw_images=%d",
            mapping.raw_class_name,
            mapping.taxonomy_class_name,
            raw_stats.raw_images,
        )

        for image_path in tqdm(
            raw_files,
            desc=f"{mapping.raw_class_name} -> {mapping.taxonomy_class_name}",
            leave=False,
        ):
            process_raw_image(
                image_path=image_path,
                output_class_dir=output_class_dir,
                raw_stats=raw_stats,
                config=config,
                hash_registry=hash_registry_by_class[mapping.taxonomy_class_name],
                used_output_names=used_output_names_by_class[mapping.taxonomy_class_name],
                logger=logger,
            )

        logger.info(
            (
                "[%s -> %s] done | raw=%d | valid=%d | skipped=%d | "
                "skip_reasons=%s"
            ),
            raw_stats.raw_class_name,
            raw_stats.taxonomy_class_name,
            raw_stats.raw_images,
            raw_stats.valid_images,
            raw_stats.skipped_images,
            dict(sorted(raw_stats.skip_reasons.items())),
        )

    output_class_stats = build_initial_output_stats(plan)
    aggregate_raw_stats(
        raw_class_stats=raw_class_stats,
        output_class_stats=output_class_stats,
    )
    validate_output_taxonomy(plan=plan, output_dir=config.output_dir)

    total_raw = sum(item.raw_images for item in output_class_stats.values())
    total_valid = sum(item.valid_images for item in output_class_stats.values())
    total_skipped = sum(item.skipped_images for item in output_class_stats.values())
    logger.info(
        (
            "PREPROCESS SUMMARY | raw=%d | valid=%d | skipped=%d | "
            "raw_classes=%d | output_classes=%d | output_dir=%s"
        ),
        total_raw,
        total_valid,
        total_skipped,
        len(raw_class_stats),
        len(output_class_stats),
        config.output_dir,
    )

    return PreprocessResult(
        config=config,
        taxonomy_plan=plan,
        raw_class_stats=raw_class_stats,
        class_stats=output_class_stats,
    )
