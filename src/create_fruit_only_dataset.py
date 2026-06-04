from __future__ import annotations

"""Wrapper tương thích cho script cũ `create_fruit_only_dataset.py`.

Luồng chính mới nằm ở `src/build_fruit_only_dataset.py` và được gọi tự động bởi:

    python src/rebuild_dataset.py --stage split --clean-output

File này chỉ giữ lại để tránh người dùng chạy nhầm script cũ hardcode số class.
Nó tự tính số class từ dataset chính.
"""

import argparse
import logging
from pathlib import Path

try:
    from src.build_fruit_only_dataset import FruitOnlyConfig, build_fruit_only_dataset
except ImportError:  # pragma: no cover
    from build_fruit_only_dataset import FruitOnlyConfig, build_fruit_only_dataset


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI cho wrapper fruit-only."""

    parser = argparse.ArgumentParser(
        description="Tao dataset_fruit_only bang pipeline moi, khong hardcode so class."
    )
    parser.add_argument("--source-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset_fruit_only"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/rebuild_dataset.log"))
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    """Logger dùng chung với log rebuild dataset."""

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("create_fruit_only_dataset")
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
    """Entry point cũ, gọi builder mới."""

    args = parse_args()
    logger = setup_logger(args.log_file)
    build_fruit_only_dataset(
        config=FruitOnlyConfig(
            source_root=args.source_root,
            output_root=args.output_root,
            clean_output=args.clean_output,
        ),
        logger=logger,
    )


if __name__ == "__main__":
    main()
