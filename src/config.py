from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ClassMetadata:
    """Metadata cho từng lớp để đổi tên thư mục đầu ra nhất quán."""

    english_name: str
    vietnamese_name: str


# Ghi rõ mapping theo khóa đã được chuẩn hóa để:
# - hỗ trợ cả tên gốc có hậu tố "_c"
# - giữ tên thư mục output ổn định giữa các lần chạy
DEFAULT_CLASS_METADATA: Dict[str, ClassMetadata] = {
    "ambarella": ClassMetadata("ambarella", "coc"),
    "apple": ClassMetadata("apple", "tao"),
    "avocado": ClassMetadata("avocado", "bo"),
    "banana": ClassMetadata("banana", "chuoi"),
    "bell_pepper": ClassMetadata("bell_pepper", "ot_chuong"),
    "bitter_gourd": ClassMetadata("bitter_gourd", "kho_qua"),
    "black_mullberry": ClassMetadata("black_mulberry", "dau_tam_den"),
    "burmese_grape": ClassMetadata("burmese_grape", "dau_da"),
    "caimito": ClassMetadata("caimito", "vu_sua"),
    "canistel": ClassMetadata("canistel", "le_kima"),
    "cantaloupe": ClassMetadata("cantaloupe", "dua_luoi"),
    "cape_gooseberry": ClassMetadata("cape_gooseberry", "tam_bop"),
    "carambola": ClassMetadata("carambola", "khe"),
    "cashew": ClassMetadata("cashew", "dieu"),
    "cempedak": ClassMetadata("cempedak", "mit_to_nu"),
    "coconut": ClassMetadata("coconut", "dua"),
    "coffee": ClassMetadata("coffee", "ca_phe"),
    "dragonfruit": ClassMetadata("dragonfruit", "thanh_long"),
    "durian": ClassMetadata("durian", "sau_rieng"),
    "eggplant": ClassMetadata("eggplant", "ca_tim"),
    "grape": ClassMetadata("grape", "nho"),
    "guanabana": ClassMetadata("guanabana", "mang_cau_xiem"),
    "guava": ClassMetadata("guava", "oi"),
    "jackfruit": ClassMetadata("jackfruit", "mit"),
    "lime": ClassMetadata("lime", "chanh"),
    "longan": ClassMetadata("longan", "nhan"),
    "mango": ClassMetadata("mango", "xoai"),
    "mangosteen": ClassMetadata("mangosteen", "mang_cut"),
    "otaheite_apple": ClassMetadata("otaheite_apple", "man_roi"),
    "other": ClassMetadata("other", "khac"),
    "papaya": ClassMetadata("papaya", "du_du"),
    "passion_fruit": ClassMetadata("passion_fruit", "chanh_day"),
    "peanut": ClassMetadata("peanut", "dau_phong"),
    "pineapple": ClassMetadata("pineapple", "dua"),
    "pomegranate": ClassMetadata("pomegranate", "luu"),
    "pomelo": ClassMetadata("pomelo", "buoi"),
    "red_mulberry": ClassMetadata("red_mulberry", "dau_tam_do"),
    "ridged_gourd": ClassMetadata("ridged_gourd", "muop_khia"),
    "sapodilla": ClassMetadata("sapodilla", "sa_po_che"),
    "strawberry": ClassMetadata("strawberry", "dau_tay"),
    "sugar_apple": ClassMetadata("sugar_apple", "na"),
    "tomato": ClassMetadata("tomato", "ca_chua"),
    "watermelon": ClassMetadata("watermelon", "dua_hau"),
    "zucchini": ClassMetadata("zucchini", "bi_ngoi"),
}


@dataclass(frozen=True)
class PreprocessConfig:
    """Cấu hình trung tâm cho pipeline tiền xử lý ảnh."""

    raw_dir: Path = Path("dataset/raw")
    processed_dir: Path = Path("dataset/processed")
    log_file: Path = Path("logs/preprocess.log")
    target_size: Tuple[int, int] = (224, 224)
    min_image_side: int = 100
    max_image_side: int = 5000
    max_aspect_ratio: float = 4.0
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    blur_threshold: float = 100.0
    duplicate_hash_tolerance: int = 0
    jpeg_quality: int = 95
    random_seed: int = 42
    keep_empty_class_dirs: bool = True
    clean_processed_dir: bool = True
    black_padding_color: Tuple[int, int, int] = (0, 0, 0)
    class_name_map: Dict[str, ClassMetadata] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_METADATA)
    )


def get_default_config(**overrides: object) -> PreprocessConfig:
    """Cho phép override cấu hình mà vẫn giữ default tập trung tại một nơi."""

    return replace(PreprocessConfig(), **overrides)


@dataclass(frozen=True)
class SplitConfig:
    """Cấu hình cho pipeline chia dữ liệu train/val/test."""

    processed_dir: Path = Path("dataset/processed")
    dataset_root: Path = Path("dataset")
    log_file: Path = Path("logs/split.log")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    clean_output_dirs: bool = True
    require_non_empty_splits: bool = True
    staging_dir_name: str = "_split_staging"
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def get_default_split_config(**overrides: object) -> SplitConfig:
    """Cho phép override cấu hình split mà vẫn giữ default tập trung."""

    return replace(SplitConfig(), **overrides)
