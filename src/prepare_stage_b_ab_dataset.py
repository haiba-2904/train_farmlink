from __future__ import annotations

import argparse
import hashlib
import json
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
OTHER_CLASS_NAME = "other"
OLD_MERGED_CLASS_NAMES = frozenset(
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
REQUIRED_MERGED_CLASS_NAMES = frozenset(
    {"mulberry", "jackfruit_cempedak", "gourd"}
)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


@dataclass(frozen=True)
class CompareDatasetConfig:
    """Config chuẩn bị dataset A/B cho Stage B.

    clean_input_dir:
        Nguồn A: ảnh đã preprocess/pad nhưng chưa smart crop.
    crop_input_dir:
        Nguồn B: ảnh sau smart crop nhẹ.
    clean_output_root:
        Dataset fruit-only từ nguồn clean.
    crop_output_root:
        Dataset fruit-only từ nguồn crop.
    clean_output:
        Nếu False thì không ghi đè output đang tồn tại. Đây là chốt an toàn để
        tránh xóa nhầm dataset đã train trước đó.
    """

    clean_input_dir: Path = Path("dataset/processed_clean")
    crop_input_dir: Path = Path("dataset/processed_crop")
    clean_output_root: Path = Path("dataset_fruit_only_clean")
    crop_output_root: Path = Path("dataset_fruit_only_crop")
    log_file: Path = Path("logs/compare_clean_crop_dataset.log")
    report_file: Path = Path("logs/compare_clean_crop_report.json")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    clean_output: bool = False
    valid_extensions: tuple[str, ...] = VALID_EXTENSIONS


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho script A/B dataset."""

    parser = argparse.ArgumentParser(
        description="Prepare Stage B A/B datasets from processed_clean and processed_crop."
    )
    parser.add_argument("--clean-input-dir", type=Path, default=Path("dataset/processed_clean"))
    parser.add_argument("--crop-input-dir", type=Path, default=Path("dataset/processed_crop"))
    parser.add_argument(
        "--clean-output-root",
        type=Path,
        default=Path("dataset_fruit_only_clean"),
    )
    parser.add_argument(
        "--crop-output-root",
        type=Path,
        default=Path("dataset_fruit_only_crop"),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/compare_clean_crop_dataset.log"),
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("logs/compare_clean_crop_report.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CompareDatasetConfig:
    """Ghép CLI args thành config bất biến."""

    return CompareDatasetConfig(
        clean_input_dir=args.clean_input_dir,
        crop_input_dir=args.crop_input_dir,
        clean_output_root=args.clean_output_root,
        crop_output_root=args.crop_output_root,
        log_file=args.log_file,
        report_file=args.report_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        clean_output=args.clean_output,
    )


def setup_logger(log_file: Path) -> logging.Logger:
    """Logger ghi cả terminal và file log riêng cho bước A/B dataset."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("compare_clean_crop_dataset")
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


def list_class_names(root: Path) -> list[str]:
    """Liệt kê class folder theo thứ tự ổn định."""

    if not root.exists():
        raise FileNotFoundError(f"Khong tim thay input dir: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input khong phai thu muc: {root}")

    class_names = sorted(
        [path.name for path in root.iterdir() if path.is_dir()],
        key=str.lower,
    )
    if not class_names:
        raise ValueError(f"Khong tim thay class folder trong: {root}")
    return class_names


def list_image_files(class_dir: Path, valid_extensions: tuple[str, ...]) -> dict[str, Path]:
    """Map file_name -> path cho một class folder."""

    valid_extension_set = {extension.lower() for extension in valid_extensions}
    return {
        path.name: path
        for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in valid_extension_set
    }


def validate_config(config: CompareDatasetConfig) -> None:
    """Validate config để không ghi nhầm và không sửa raw."""

    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"Tong ti le split phai bang 1.0, nhan duoc {total_ratio:.12f}"
        )

    if min(config.train_ratio, config.val_ratio, config.test_ratio) <= 0:
        raise ValueError("Tat ca ti le train/val/test phai lon hon 0.")

    raw_dir = Path("dataset/raw").resolve()
    for output_root in (config.clean_output_root, config.crop_output_root):
        resolved_output = output_root.resolve()
        if resolved_output == raw_dir or raw_dir in resolved_output.parents:
            raise ValueError(f"Output khong duoc trung/nam trong dataset/raw: {output_root}")

        if output_root.exists() and not config.clean_output:
            raise FileExistsError(
                "Output da ton tai. Hay them --clean-output neu muon tao lai: "
                f"{output_root}"
            )


def validate_taxonomy(config: CompareDatasetConfig, logger: logging.Logger) -> list[str]:
    """Validate taxonomy của hai nguồn processed trước khi tạo A/B dataset."""

    clean_classes = list_class_names(config.clean_input_dir)
    crop_classes = list_class_names(config.crop_input_dir)
    if clean_classes != crop_classes:
        raise ValueError(
            "processed_clean va processed_crop khong co cung class list. "
            f"clean_only={sorted(set(clean_classes) - set(crop_classes))}, "
            f"crop_only={sorted(set(crop_classes) - set(clean_classes))}"
        )

    class_set = set(clean_classes)
    old_folders = sorted(class_set & OLD_MERGED_CLASS_NAMES)
    if old_folders:
        raise ValueError(f"Output processed van con folder cu da gop: {old_folders}")

    missing_new_folders = sorted(REQUIRED_MERGED_CLASS_NAMES - class_set)
    if missing_new_folders:
        raise ValueError(f"Output processed thieu folder moi sau gop: {missing_new_folders}")

    if OTHER_CLASS_NAME not in class_set:
        logger.warning("processed dataset khong co class other; Stage B van tiep tuc.")

    fruit_class_names = [class_name for class_name in clean_classes if class_name != OTHER_CLASS_NAME]
    if not fruit_class_names:
        raise ValueError("Khong co class nong san nao sau khi loai other.")

    logger.info(
        "Taxonomy hop le | source_classes=%d | fruit_classes=%d",
        len(clean_classes),
        len(fruit_class_names),
    )
    return fruit_class_names


def compute_sha256(file_path: Path) -> str:
    """Tính SHA256 nội dung file để loại exact duplicate khỏi manifest."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_random(seed: int, namespace: str) -> random.Random:
    """Tạo random generator cố định theo seed và class name."""

    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def compute_split_counts(total_images: int, config: CompareDatasetConfig) -> dict[str, int]:
    """Tính số lượng train/val/test cho một class, đảm bảo không split rỗng."""

    if total_images < len(SPLITS):
        raise ValueError(
            f"Khong du anh de chia train/val/test deu khong rong: total={total_images}"
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


def collect_matched_files(
    config: CompareDatasetConfig,
    class_names: list[str],
    logger: logging.Logger,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Tạo danh sách file khớp giữa clean/crop theo từng class.

    Nếu một file chỉ có ở một nguồn, file đó bị bỏ khỏi cả hai dataset để A/B
    giữ cùng sample set. Exact duplicate theo nội dung clean cũng bị bỏ khỏi
    manifest để tránh cùng ảnh xuất hiện ở nhiều split.
    """

    matched_files: dict[str, list[str]] = {}
    seen_hash_to_key: dict[str, str] = {}
    report_classes: dict[str, dict[str, Any]] = {}
    total_missing_clean = 0
    total_missing_crop = 0
    total_duplicate_skipped = 0

    for class_name in tqdm(class_names, desc="Match clean/crop", leave=True):
        clean_files = list_image_files(
            config.clean_input_dir / class_name,
            config.valid_extensions,
        )
        crop_files = list_image_files(
            config.crop_input_dir / class_name,
            config.valid_extensions,
        )
        clean_names = set(clean_files)
        crop_names = set(crop_files)
        missing_in_clean = sorted(crop_names - clean_names)
        missing_in_crop = sorted(clean_names - crop_names)
        common_names = sorted(clean_names & crop_names, key=str.lower)

        kept_names: list[str] = []
        duplicate_skipped: list[str] = []
        for file_name in common_names:
            content_hash = compute_sha256(clean_files[file_name])
            previous_key = seen_hash_to_key.get(content_hash)
            current_key = f"{class_name}/{file_name}"
            if previous_key is not None:
                duplicate_skipped.append(current_key)
                logger.info(
                    "Bo duplicate exact | current=%s | previous=%s | hash=%s",
                    current_key,
                    previous_key,
                    content_hash[:12],
                )
                continue
            seen_hash_to_key[content_hash] = current_key
            kept_names.append(file_name)

        if len(kept_names) < len(SPLITS):
            raise ValueError(
                "Class khong du file matched unique de split train/val/test: "
                f"{class_name} | matched_unique={len(kept_names)}"
            )

        matched_files[class_name] = kept_names
        total_missing_clean += len(missing_in_clean)
        total_missing_crop += len(missing_in_crop)
        total_duplicate_skipped += len(duplicate_skipped)

        if missing_in_clean:
            logger.warning("File co o crop nhung thieu o clean | class=%s | files=%s", class_name, missing_in_clean[:30])
        if missing_in_crop:
            logger.warning("File co o clean nhung thieu o crop | class=%s | files=%s", class_name, missing_in_crop[:30])

        report_classes[class_name] = {
            "clean_images": len(clean_files),
            "crop_images": len(crop_files),
            "matched_images": len(common_names),
            "matched_unique_images": len(kept_names),
            "missing_in_clean": len(missing_in_clean),
            "missing_in_crop": len(missing_in_crop),
            "missing_in_clean_preview": missing_in_clean[:30],
            "missing_in_crop_preview": missing_in_crop[:30],
            "duplicate_skipped": len(duplicate_skipped),
            "duplicate_skipped_preview": duplicate_skipped[:30],
        }

    mismatch_classes = [
        class_name
        for class_name, stats in report_classes.items()
        if stats["missing_in_clean"] > 0 or stats["missing_in_crop"] > 0
    ]
    duplicate_classes = [
        class_name
        for class_name, stats in report_classes.items()
        if stats["duplicate_skipped"] > 0
    ]
    matching_report = {
        "classes": report_classes,
        "total_missing_in_clean": total_missing_clean,
        "total_missing_in_crop": total_missing_crop,
        "total_file_mismatch": total_missing_clean + total_missing_crop,
        "classes_with_missing_files": mismatch_classes,
        "total_duplicate_skipped": total_duplicate_skipped,
        "classes_with_duplicate_skipped": duplicate_classes,
    }
    return matched_files, matching_report


def build_split_manifest(
    matched_files: dict[str, list[str]],
    config: CompareDatasetConfig,
) -> dict[str, dict[str, list[str]]]:
    """Sinh manifest split dùng chung cho cả clean và crop."""

    manifest: dict[str, dict[str, list[str]]] = {split: {} for split in SPLITS}
    for class_name, file_names in matched_files.items():
        shuffled_names = list(file_names)
        stable_random(config.seed, class_name).shuffle(shuffled_names)
        counts = compute_split_counts(len(shuffled_names), config)

        train_end = counts["train"]
        val_end = train_end + counts["val"]
        manifest["train"][class_name] = shuffled_names[:train_end]
        manifest["val"][class_name] = shuffled_names[train_end:val_end]
        manifest["test"][class_name] = shuffled_names[val_end:]

    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, dict[str, list[str]]]) -> None:
    """Đảm bảo manifest không rỗng và không overlap file giữa split."""

    reference_classes = sorted(manifest["train"], key=str.lower)
    split_to_keys: dict[str, set[str]] = {}

    for split in SPLITS:
        class_names = sorted(manifest[split], key=str.lower)
        if class_names != reference_classes:
            raise ValueError(f"Class list cua split {split} khong khop train.")

        keys: set[str] = set()
        for class_name, file_names in manifest[split].items():
            if not file_names:
                raise ValueError(f"Split rong: split={split}, class={class_name}")
            for file_name in file_names:
                key = f"{class_name}/{file_name}"
                if key in keys:
                    raise ValueError(f"File lap trong split {split}: {key}")
                keys.add(key)
        split_to_keys[split] = keys

    for index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[index + 1 :]:
            overlap = split_to_keys[left_split] & split_to_keys[right_split]
            if overlap:
                raise ValueError(
                    f"Phat hien file overlap giua {left_split} va {right_split}: "
                    f"{sorted(overlap)[0]}"
                )


def prepare_staging_root(output_root: Path) -> Path:
    """Tạo staging folder riêng để tránh output nửa chừng."""

    staging_root = output_root.with_name(f"{output_root.name}_tmp_ab_{os.getpid()}")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    return staging_root


def materialize_dataset(
    source_root: Path,
    output_root: Path,
    manifest: dict[str, dict[str, list[str]]],
    class_names: list[str],
    config: CompareDatasetConfig,
    logger: logging.Logger,
) -> dict[str, dict[str, int]]:
    """Copy dataset theo manifest vào staging rồi publish ra output."""

    staging_root = prepare_staging_root(output_root)
    split_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    total_files = sum(
        len(file_names)
        for split_map in manifest.values()
        for file_names in split_map.values()
    )

    try:
        with tqdm(total=total_files, desc=f"Copy {output_root.name}", leave=True) as progress:
            for split in SPLITS:
                for class_name in class_names:
                    file_names = manifest[split][class_name]
                    destination_class_dir = staging_root / split / class_name
                    destination_class_dir.mkdir(parents=True, exist_ok=True)

                    for file_name in file_names:
                        source_path = source_root / class_name / file_name
                        if not source_path.exists():
                            raise FileNotFoundError(f"Thieu file source khi copy: {source_path}")
                        destination_path = destination_class_dir / file_name
                        shutil.copy2(source_path, destination_path)
                        progress.update(1)

                    split_counts[split][class_name] = len(file_names)

        write_class_names(staging_root, class_names)
        validate_materialized_dataset(staging_root, class_names, config)

        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.move(str(staging_root), str(output_root))
        logger.info("Da tao output dataset: %s", output_root)
        return split_counts
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def write_class_names(output_root: Path, class_names: list[str]) -> None:
    """Ghi class_names.txt để Stage B train dùng đúng class order."""

    (output_root / "class_names.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )


def validate_materialized_dataset(
    output_root: Path,
    class_names: list[str],
    config: CompareDatasetConfig,
) -> None:
    """Validate output fruit-only sau khi copy."""

    for split in SPLITS:
        split_dir = output_root / split
        actual_classes = list_class_names(split_dir)
        if actual_classes != class_names:
            raise ValueError(
                f"Output {output_root} split {split} sai class list: {actual_classes}"
            )
        if OTHER_CLASS_NAME in actual_classes:
            raise ValueError(f"Output Stage B khong duoc co other: {output_root}/{split}")

        for class_name in actual_classes:
            image_count = len(list_image_files(split_dir / class_name, config.valid_extensions))
            if image_count <= 0:
                raise ValueError(f"Output co class rong: {output_root}/{split}/{class_name}")


def build_report(
    config: CompareDatasetConfig,
    class_names: list[str],
    matching_report: dict[str, Any],
    manifest: dict[str, dict[str, list[str]]],
    clean_split_counts: dict[str, dict[str, int]],
    crop_split_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Tổng hợp report JSON cho A/B dataset."""

    split_totals = {
        split: sum(len(file_names) for file_names in manifest[split].values())
        for split in SPLITS
    }
    per_class_split_counts = {
        class_name: {
            split: len(manifest[split][class_name]) for split in SPLITS
        }
        for class_name in class_names
    }

    classes_missing_images = [
        class_name
        for class_name, stats in matching_report["classes"].items()
        if stats["missing_in_clean"] > 0 or stats["missing_in_crop"] > 0
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "prepare_stage_b_ab_dataset",
        "config": {
            **asdict(config),
            "clean_input_dir": str(config.clean_input_dir),
            "crop_input_dir": str(config.crop_input_dir),
            "clean_output_root": str(config.clean_output_root),
            "crop_output_root": str(config.crop_output_root),
            "log_file": str(config.log_file),
            "report_file": str(config.report_file),
            "valid_extensions": list(config.valid_extensions),
        },
        "class_count": len(class_names),
        "class_names": class_names,
        "excluded_class": OTHER_CLASS_NAME,
        "taxonomy": {
            "old_merged_class_names_absent": sorted(OLD_MERGED_CLASS_NAMES),
            "required_merged_class_names_present": sorted(REQUIRED_MERGED_CLASS_NAMES),
        },
        "matching": {
            "total_missing_in_clean": matching_report["total_missing_in_clean"],
            "total_missing_in_crop": matching_report["total_missing_in_crop"],
            "total_file_mismatch": matching_report["total_file_mismatch"],
            "classes_with_missing_files": classes_missing_images,
            "total_duplicate_skipped": matching_report["total_duplicate_skipped"],
            "classes_with_duplicate_skipped": matching_report["classes_with_duplicate_skipped"],
        },
        "split_totals": split_totals,
        "per_class_split_counts": per_class_split_counts,
        "per_class_source_counts": matching_report["classes"],
        "output_counts": {
            "clean": clean_split_counts,
            "crop": crop_split_counts,
        },
        "note": (
            "Clean va crop dung chung manifest split. File khong khop giua hai "
            "nguon bi bo qua de dam bao so sanh A/B cong bang."
        ),
    }


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Ghi report JSON UTF-8."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare_stage_b_ab_dataset(config: CompareDatasetConfig) -> Path:
    """Điều phối toàn bộ bước chuẩn bị A/B dataset, không train model."""

    validate_config(config)
    logger = setup_logger(config.log_file)
    logger.info("Bat dau prepare Stage B A/B dataset | config=%s", asdict(config))

    class_names = validate_taxonomy(config, logger)
    matched_files, matching_report = collect_matched_files(config, class_names, logger)
    manifest = build_split_manifest(matched_files, config)

    clean_counts = materialize_dataset(
        source_root=config.clean_input_dir,
        output_root=config.clean_output_root,
        manifest=manifest,
        class_names=class_names,
        config=config,
        logger=logger,
    )
    crop_counts = materialize_dataset(
        source_root=config.crop_input_dir,
        output_root=config.crop_output_root,
        manifest=manifest,
        class_names=class_names,
        config=config,
        logger=logger,
    )

    report = build_report(
        config=config,
        class_names=class_names,
        matching_report=matching_report,
        manifest=manifest,
        clean_split_counts=clean_counts,
        crop_split_counts=crop_counts,
    )
    save_json(config.report_file, report)

    logger.info("Da luu report: %s", config.report_file)
    logger.info(
        "Hoan tat A/B dataset | class_count=%d | split_totals=%s | mismatches=%d",
        report["class_count"],
        report["split_totals"],
        report["matching"]["total_file_mismatch"],
    )
    return config.report_file


def main() -> None:
    """Entry point: `python src/prepare_stage_b_ab_dataset.py --clean-output`."""

    args = parse_args()
    config = build_config(args)
    prepare_stage_b_ab_dataset(config)


if __name__ == "__main__":
    main()
