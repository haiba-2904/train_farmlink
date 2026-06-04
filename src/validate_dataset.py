from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from src.build_fruit_only_dataset import is_other_class
    from src.splitter import DEFAULT_VALID_EXTENSIONS, SPLITS, list_class_dirs, list_image_files
    from src.taxonomy import MERGED_CLASS_GROUPS, MERGED_SOURCE_TO_TARGET
except ImportError:  # pragma: no cover
    from build_fruit_only_dataset import is_other_class
    from splitter import DEFAULT_VALID_EXTENSIONS, SPLITS, list_class_dirs, list_image_files
    from taxonomy import MERGED_CLASS_GROUPS, MERGED_SOURCE_TO_TARGET


OLD_MERGED_CLASS_NAMES = frozenset(
    source_name
    for source_name, target_name in MERGED_SOURCE_TO_TARGET.items()
    if source_name != target_name
)
REQUIRED_MERGED_CLASS_NAMES = frozenset(MERGED_CLASS_GROUPS)


@dataclass(frozen=True)
class RawSnapshot:
    """Snapshot nhẹ để chứng minh `dataset/raw` không bị thay đổi trong pipeline.

    digest dùng relative path + size + mtime_ns. Pipeline này chỉ đọc raw nên nếu
    digest trước/sau khác nhau, có thể raw đã bị tác động bởi thao tác ngoài ý
    muốn hoặc process khác.
    """

    raw_dir: Path
    class_count: int
    file_count: int
    total_bytes: int
    digest: str


@dataclass(frozen=True)
class DatasetValidationConfig:
    """Config validation sau khi split và tạo fruit-only."""

    raw_dir: Path = Path("dataset/raw")
    processed_clean_dir: Path = Path("dataset/processed_clean")
    processed_crop_dir: Path = Path("dataset/processed_crop")
    dataset_root: Path = Path("dataset")
    fruit_only_root: Path = Path("dataset_fruit_only")
    valid_extensions: tuple[str, ...] = DEFAULT_VALID_EXTENSIONS


def compute_sha256(file_path: Path) -> str:
    """Tính hash nội dung để kiểm tra overlap theo ảnh thật."""

    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_snapshot(raw_dir: Path) -> RawSnapshot:
    """Tạo snapshot trước/sau cho `dataset/raw`."""

    if not raw_dir.exists():
        raise FileNotFoundError(f"dataset/raw khong ton tai: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"dataset/raw khong phai thu muc: {raw_dir}")

    class_dirs = [path for path in raw_dir.iterdir() if path.is_dir()]
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0

    for path in sorted(raw_dir.rglob("*"), key=lambda item: str(item.relative_to(raw_dir))):
        relative_path = path.relative_to(raw_dir).as_posix()
        if path.is_dir():
            digest.update(f"D|{relative_path}\n".encode("utf-8"))
            continue
        if not path.is_file():
            continue

        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size
        digest.update(
            f"F|{relative_path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8")
        )

    return RawSnapshot(
        raw_dir=raw_dir,
        class_count=len(class_dirs),
        file_count=file_count,
        total_bytes=total_bytes,
        digest=digest.hexdigest(),
    )


def assert_raw_unchanged(before: RawSnapshot, after: RawSnapshot) -> None:
    """Raise nếu snapshot raw trước/sau không khớp."""

    if before.digest != after.digest:
        raise ValueError(
            "dataset/raw co dau hieu bi thay doi trong qua trinh split. "
            f"before={asdict(before)}, after={asdict(after)}"
        )


def count_images(class_dir: Path, valid_extensions: tuple[str, ...]) -> int:
    """Đếm ảnh hợp lệ trong một class folder."""

    return len(list_image_files(class_dir, valid_extensions))


def direct_class_names(root: Path) -> list[str]:
    """Lấy class folder trực tiếp dưới root theo thứ tự ổn định."""

    return [class_dir.name for class_dir in list_class_dirs(root)]


def assert_no_old_merged_folders(root: Path, description: str) -> None:
    """Không cho output còn folder class cũ đã merge."""

    class_names = set(direct_class_names(root))
    old_names = sorted(class_names & OLD_MERGED_CLASS_NAMES)
    if old_names:
        raise ValueError(f"{description} van con folder class cu da merge: {old_names}")


def assert_required_merged_folders(root: Path, description: str) -> None:
    """Output phải có các folder class mới sau merge."""

    class_names = set(direct_class_names(root))
    missing_names = sorted(REQUIRED_MERGED_CLASS_NAMES - class_names)
    if missing_names:
        raise ValueError(f"{description} thieu folder class moi sau merge: {missing_names}")


def summarize_split_root(
    root: Path,
    valid_extensions: tuple[str, ...],
    require_other: bool,
    forbid_other: bool,
    description: str,
) -> dict[str, Any]:
    """Validate một dataset có cấu trúc train/val/test."""

    reference_classes: list[str] | None = None
    per_class_counts: dict[str, dict[str, int]] = {}
    split_totals: dict[str, int] = {}

    for split in SPLITS:
        split_dir = root / split
        class_dirs = list_class_dirs(split_dir)
        class_names = [class_dir.name for class_dir in class_dirs]

        if reference_classes is None:
            reference_classes = class_names
        elif class_names != reference_classes:
            raise ValueError(f"{description}: class list split '{split}' khong khop train.")

        has_other = any(is_other_class(class_name) for class_name in class_names)
        if require_other and not has_other:
            raise ValueError(f"{description}: split '{split}' thieu class other.")
        if forbid_other and has_other:
            raise ValueError(f"{description}: split '{split}' van con class other.")

        old_names = sorted(set(class_names) & OLD_MERGED_CLASS_NAMES)
        if old_names:
            raise ValueError(f"{description}: split '{split}' con folder cu: {old_names}")

        missing_new_names = sorted(REQUIRED_MERGED_CLASS_NAMES - set(class_names))
        if missing_new_names:
            raise ValueError(
                f"{description}: split '{split}' thieu folder moi: {missing_new_names}"
            )

        split_counts: dict[str, int] = {}
        for class_dir in class_dirs:
            image_count = count_images(class_dir, valid_extensions)
            if image_count <= 0:
                raise ValueError(f"{description}: class rong: {split}/{class_dir.name}")
            split_counts[class_dir.name] = image_count

        per_class_counts[split] = split_counts
        split_totals[split] = sum(split_counts.values())

    assert reference_classes is not None
    assert_no_overlap_between_splits(
        root=root,
        valid_extensions=valid_extensions,
        description=description,
    )
    return {
        "class_count": len(reference_classes),
        "class_names": reference_classes,
        "split_totals": split_totals,
        "per_class_counts": per_class_counts,
    }


def assert_no_overlap_between_splits(
    root: Path,
    valid_extensions: tuple[str, ...],
    description: str,
) -> None:
    """Kiểm tra không có ảnh xuất hiện ở nhiều split.

    Kiểm tra cả:
    - key theo `class_name/file_name`
    - SHA256 nội dung ảnh
    """

    split_to_names: dict[str, set[str]] = {}
    split_to_hashes: dict[str, set[str]] = {}
    hash_preview: dict[str, str] = {}

    for split in SPLITS:
        split_dir = root / split
        name_keys: set[str] = set()
        content_hashes: set[str] = set()

        for class_dir in list_class_dirs(split_dir):
            for image_path in list_image_files(class_dir, valid_extensions):
                name_key = f"{class_dir.name}/{image_path.name}"
                if name_key in name_keys:
                    raise ValueError(f"{description}: file lap trong split {split}: {name_key}")
                name_keys.add(name_key)

                content_hash = compute_sha256(image_path)
                if content_hash in content_hashes:
                    raise ValueError(
                        f"{description}: duplicate hash trong cung split {split}: "
                        f"{content_hash[:12]}"
                    )
                content_hashes.add(content_hash)
                hash_preview.setdefault(content_hash, str(image_path))

        split_to_names[split] = name_keys
        split_to_hashes[split] = content_hashes

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            name_overlap = split_to_names[left] & split_to_names[right]
            if name_overlap:
                raise ValueError(
                    f"{description}: file overlap giua {left} va {right}: "
                    f"{sorted(name_overlap)[0]}"
                )

            hash_overlap = split_to_hashes[left] & split_to_hashes[right]
            if hash_overlap:
                overlap_hash = sorted(hash_overlap)[0]
                raise ValueError(
                    f"{description}: anh trung noi dung giua {left} va {right}: "
                    f"{overlap_hash[:12]} | vi_du={hash_preview.get(overlap_hash)}"
                )


def validate_class_names_file(
    fruit_only_root: Path,
    fruit_summary: dict[str, Any],
) -> None:
    """Đảm bảo `class_names.txt` khớp class train của Stage B."""

    class_names_path = fruit_only_root / "class_names.txt"
    if not class_names_path.exists():
        raise FileNotFoundError(f"Thieu class_names.txt: {class_names_path}")

    class_names = [
        line.strip()
        for line in class_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if class_names != fruit_summary["class_names"]:
        raise ValueError("class_names.txt khong khop danh sach class trong train.")


def validate_fruit_only_report(fruit_only_root: Path) -> None:
    """Đảm bảo report Stage B đã được tạo và là JSON hợp lệ."""

    report_path = fruit_only_root / "dataset_fruit_only_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Thieu dataset_fruit_only_report.json: {report_path}")
    json.loads(report_path.read_text(encoding="utf-8"))


def validate_rebuild_split_outputs(
    config: DatasetValidationConfig,
    logger: logging.Logger,
    raw_snapshot_before: RawSnapshot | None = None,
) -> dict[str, Any]:
    """Validation bắt buộc sau khi chạy stage split."""

    if raw_snapshot_before is not None:
        raw_snapshot_after = build_raw_snapshot(config.raw_dir)
        assert_raw_unchanged(raw_snapshot_before, raw_snapshot_after)
    else:
        raw_snapshot_after = build_raw_snapshot(config.raw_dir)

    assert_no_old_merged_folders(config.processed_clean_dir, "processed_clean")
    assert_no_old_merged_folders(config.processed_crop_dir, "processed_crop")
    assert_required_merged_folders(config.processed_clean_dir, "processed_clean")
    assert_required_merged_folders(config.processed_crop_dir, "processed_crop")

    main_summary = summarize_split_root(
        root=config.dataset_root,
        valid_extensions=config.valid_extensions,
        require_other=True,
        forbid_other=False,
        description="dataset chinh",
    )
    fruit_summary = summarize_split_root(
        root=config.fruit_only_root,
        valid_extensions=config.valid_extensions,
        require_other=False,
        forbid_other=True,
        description="dataset_fruit_only",
    )
    validate_class_names_file(config.fruit_only_root, fruit_summary)
    validate_fruit_only_report(config.fruit_only_root)

    summary = {
        "raw_snapshot": {
            **asdict(raw_snapshot_after),
            "raw_dir": str(raw_snapshot_after.raw_dir),
        },
        "processed_clean_class_count": len(direct_class_names(config.processed_clean_dir)),
        "processed_crop_class_count": len(direct_class_names(config.processed_crop_dir)),
        "main_dataset": main_summary,
        "fruit_only_dataset": fruit_summary,
        "checks": {
            "raw_exists_and_unchanged": raw_snapshot_before is not None,
            "processed_outputs_do_not_have_old_merged_folders": True,
            "main_dataset_has_other": True,
            "fruit_only_has_no_other": True,
            "required_merged_classes_exist": sorted(REQUIRED_MERGED_CLASS_NAMES),
            "old_merged_classes_absent": sorted(OLD_MERGED_CLASS_NAMES),
            "no_empty_split_or_class": True,
            "no_overlap_between_splits": True,
        },
    }
    logger.info(
        "Validation split thanh cong | main_classes=%d | fruit_classes=%d",
        main_summary["class_count"],
        fruit_summary["class_count"],
    )
    return summary
