from __future__ import annotations

"""Wrapper tương thích cho script cũ `split_after_crop.py`.

Luồng chính khuyến nghị:

    python src/rebuild_dataset.py --stage split --clean-output

File này chỉ gọi splitter mới để tránh script cũ còn hardcode taxonomy/số class.
Nó không tạo `dataset_fruit_only`; nếu cần đủ pipeline Stage B, hãy dùng
`rebuild_dataset.py --stage split`.
"""

import argparse
import logging
from pathlib import Path

try:
    from src.splitter import DatasetSplitConfig, split_dataset
except ImportError:  # pragma: no cover
    from splitter import DatasetSplitConfig, split_dataset


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho wrapper split sau crop."""

    parser = argparse.ArgumentParser(
        description="Split dataset/processed_crop bang splitter moi, khong hardcode so class."
    )
    parser.add_argument("--processed-crop-dir", type=Path, default=Path("dataset/processed_crop"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/rebuild_dataset.log"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    """Logger dùng chung với log rebuild dataset."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("split_after_crop")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main() -> None:
    """Entry point cũ, gọi splitter mới."""

    args = parse_args()
    logger = setup_logger(args.log_file)
    split_dataset(
        config=DatasetSplitConfig(
            input_dir=args.processed_crop_dir,
            dataset_root=args.dataset_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            clean_output=args.clean_output,
        ),
        logger=logger,
    )


if __name__ == "__main__":
    main()
