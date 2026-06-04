from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from src.config import DEFAULT_CLASS_METADATA
except ImportError:  # pragma: no cover
    try:
        from config import DEFAULT_CLASS_METADATA
    except ImportError:  # pragma: no cover
        DEFAULT_CLASS_METADATA = {}


def _basic_normalize(value: str) -> str:
    """Normalize token thô, chưa áp dụng alias/prefix taxonomy."""

    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if normalized.endswith("_c"):
        normalized = normalized[:-2]
    return normalized


MERGED_CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "mulberry": ("black_mulberry", "red_mulberry", "mulberry", "mullberry"),
    "jackfruit_cempedak": ("cempedak", "jackfruit", "jackfruit_cempedak"),
    "gourd": ("bitter_gourd", "ridged_gourd", "gourd"),
}

MERGED_SOURCE_TO_TARGET: dict[str, str] = {
    source_name: target_name
    for target_name, source_names in MERGED_CLASS_GROUPS.items()
    for source_name in source_names
}

MERGED_OUTPUT_CLASSES = frozenset(MERGED_CLASS_GROUPS)

KNOWN_CLASS_NAMES = frozenset(
    {
        _basic_normalize(class_name)
        for class_name in (
            *(MERGED_SOURCE_TO_TARGET.keys()),
            *(MERGED_SOURCE_TO_TARGET.values()),
            *(
                metadata.english_name
                for metadata in DEFAULT_CLASS_METADATA.values()
            ),
            *(DEFAULT_CLASS_METADATA.keys()),
        )
        if class_name
    }
)
KNOWN_CLASS_PREFIXES = tuple(sorted(KNOWN_CLASS_NAMES, key=len, reverse=True))


@dataclass(frozen=True)
class RawClassMapping:
    """Mapping từ một folder raw sang class sau taxonomy."""

    raw_class_name: str
    clean_raw_class_name: str
    taxonomy_class_name: str
    raw_dir: Path


@dataclass(frozen=True)
class TaxonomyPlan:
    """Kế hoạch taxonomy sau khi quét `dataset/raw`."""

    raw_mappings: tuple[RawClassMapping, ...]
    class_to_raw_classes: dict[str, tuple[str, ...]]
    output_class_names: tuple[str, ...]
    merge_rules: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(MERGED_CLASS_GROUPS)
    )


def normalize_class_name(value: str) -> str:
    """Chuẩn hóa tên folder class về token ổn định.

    Ví dụ:
    - `bell pepper` -> `bell_pepper`
    - `ridged gourd` -> `ridged_gourd`
    - `mango_c` -> `mango`
    - `apple_tao_123` -> `apple_tao` không được dùng cho output mới, nhưng vẫn
      được normalize an toàn nếu dataset/raw còn tên kiểu cũ.
    """

    normalized = _basic_normalize(value)
    if normalized in KNOWN_CLASS_NAMES:
        return normalized

    parts = normalized.split("_")
    stem = "_".join(parts[:-1]) if parts and parts[-1].isdigit() else normalized
    if stem in KNOWN_CLASS_NAMES:
        return stem

    for known_prefix in KNOWN_CLASS_PREFIXES:
        if stem == known_prefix or stem.startswith(f"{known_prefix}_"):
            return known_prefix

    return normalized


def resolve_taxonomy_class(raw_class_name: str) -> str:
    """Map raw class sang class mới sau taxonomy."""

    clean_name = normalize_class_name(raw_class_name)
    return MERGED_SOURCE_TO_TARGET.get(clean_name, clean_name)


def list_raw_class_dirs(raw_dir: Path) -> list[Path]:
    """Liệt kê class folder trong raw theo thứ tự cố định."""

    if not raw_dir.exists():
        raise FileNotFoundError(f"Khong tim thay raw_dir: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"raw_dir khong phai thu muc: {raw_dir}")

    class_dirs = sorted(
        [path for path in raw_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    if not class_dirs:
        raise ValueError(f"Khong tim thay class folder nao trong raw_dir: {raw_dir}")
    return class_dirs


def build_taxonomy_plan(raw_dir: Path) -> TaxonomyPlan:
    """Tạo plan taxonomy từ `dataset/raw` và validate duplicate class.

    Duplicate không chủ ý rất nguy hiểm vì làm trộn nhãn. Vì vậy nếu nhiều raw
    folder cùng map về một output class, chỉ cho phép khi output class đó là class
    merge chính thức trong `MERGED_CLASS_GROUPS`.
    """

    mappings: list[RawClassMapping] = []
    class_to_raw_classes: dict[str, list[str]] = defaultdict(list)

    for raw_class_dir in list_raw_class_dirs(raw_dir):
        clean_name = normalize_class_name(raw_class_dir.name)
        if not clean_name:
            raise ValueError(f"Ten class raw khong hop le: {raw_class_dir}")

        taxonomy_name = resolve_taxonomy_class(raw_class_dir.name)
        mappings.append(
            RawClassMapping(
                raw_class_name=raw_class_dir.name,
                clean_raw_class_name=clean_name,
                taxonomy_class_name=taxonomy_name,
                raw_dir=raw_class_dir,
            )
        )
        class_to_raw_classes[taxonomy_name].append(raw_class_dir.name)

    invalid_duplicates = {
        class_name: raw_names
        for class_name, raw_names in class_to_raw_classes.items()
        if len(raw_names) > 1 and class_name not in MERGED_OUTPUT_CLASSES
    }
    if invalid_duplicates:
        raise ValueError(
            "Phat hien nhieu raw folder map ve cung mot class nhung khong nam "
            f"trong merge rule chinh thuc: {invalid_duplicates}"
        )

    frozen_class_to_raw = {
        class_name: tuple(sorted(raw_names, key=str.lower))
        for class_name, raw_names in class_to_raw_classes.items()
    }
    output_class_names = tuple(sorted(frozen_class_to_raw, key=str.lower))

    old_class_names_in_output = [
        old_name
        for target_name, source_names in MERGED_CLASS_GROUPS.items()
        for old_name in source_names
        if old_name != target_name and old_name in output_class_names
    ]
    if old_class_names_in_output:
        raise ValueError(
            "Output taxonomy khong duoc con class cu da merge: "
            f"{old_class_names_in_output}"
        )

    return TaxonomyPlan(
        raw_mappings=tuple(mappings),
        class_to_raw_classes=frozen_class_to_raw,
        output_class_names=output_class_names,
    )
