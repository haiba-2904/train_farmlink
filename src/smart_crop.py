from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from utils import (
        compute_blur_score,
        list_class_directories,
        list_image_files,
        open_image_safely,
        resize_with_padding,
        save_image,
        setup_logger,
        validate_image_quality,
    )
except ImportError:  # pragma: no cover
    from .utils import (
        compute_blur_score,
        list_class_directories,
        list_image_files,
        open_image_safely,
        resize_with_padding,
        save_image,
        setup_logger,
        validate_image_quality,
    )


@dataclass(frozen=True)
class SmartCropConfig:
    """Cấu hình cho bước smart crop offline.

    Pipeline này cố tình dùng heuristic đơn giản bằng edge/contrast, không dùng
    model detector. Mục tiêu là giảm nền thừa nhưng luôn ưu tiên an toàn: crop
    sai nguy hiểm hơn không crop, nên bất kỳ điều kiện nào không chắc chắn đều
    fallback về ảnh gốc đã resize/pad.
    """

    input_dir: Path = Path("dataset/processed_clean")
    output_dir: Path = Path("dataset/processed_crop")
    log_file: Path = Path("logs/smart_crop.log")
    target_size: tuple[int, int] = (320, 320)
    valid_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    min_image_side: int = 100
    max_image_side: int = 5000
    max_aspect_ratio: float = 4.0
    blur_threshold: float = 100.0
    min_crop_area_ratio: float = 0.20
    max_crop_area_ratio: float = 0.80
    min_bbox_side_ratio: float = 0.25
    min_bbox_side_pixels: int = 100
    max_bbox_aspect_ratio: float = 2.6
    crop_padding_ratio: float = 0.16
    min_texture_score: float = 25.0
    min_color_contrast: float = 12.0
    min_edge_density: float = 0.006
    jpeg_quality: int = 95
    clean_output_dir: bool = True


@dataclass
class ClassCropStats:
    """Thống kê crop cho một class để ghi log cuối pipeline."""

    class_name: str
    total: int = 0
    cropped: int = 0
    kept_original: int = 0
    failed: int = 0
    reasons: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class CropDecision:
    """Kết quả quyết định crop cho một ảnh.

    `applied=False` nghĩa là pipeline giữ ảnh gốc. `reason` giúp audit nhanh vì
    sao một ảnh không được crop: bbox quá nhỏ, quá lớn, thiếu texture, v.v.
    """

    applied: bool
    bbox: tuple[int, int, int, int] | None
    reason: str
    area_ratio: float = 0.0
    texture_score: float = 0.0
    color_contrast: float = 0.0
    edge_density: float = 0.0


def parse_args() -> argparse.Namespace:
    """Đọc tham số CLI để script chạy được độc lập từ terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Smart crop dataset/processed_clean sang dataset/processed_crop "
            "bang edge/contrast heuristic an toan cho ResNet50."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/processed_clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed_crop"))
    parser.add_argument("--log-file", type=Path, default=Path("logs/smart_crop.log"))
    parser.add_argument("--target-size", type=int, default=320)
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Khong xoa output-dir cu truoc khi chay.",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Nguong Laplacian variance de xem anh qua mo.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SmartCropConfig:
    """Ghép CLI args vào cấu hình mặc định."""

    return SmartCropConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        target_size=(args.target_size, args.target_size),
        blur_threshold=args.blur_threshold,
        clean_output_dir=not args.no_clean_output,
    )


def prepare_output_directory(output_dir: Path, clean: bool) -> None:
    """Tạo output folder và tùy chọn làm sạch dữ liệu cũ.

    Hàm này chỉ thao tác trên `dataset/processed_crop`, tuyệt đối không đụng
    `dataset/processed_clean`. Đây là nguyên tắc quan trọng để luôn có thể quay
    lại dataset sạch ban đầu nếu crop không cải thiện kết quả.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    if not clean:
        return

    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def validate_config(config: SmartCropConfig) -> None:
    """Fail fast nếu cấu hình không hợp lệ."""

    if not config.input_dir.exists():
        raise FileNotFoundError(f"Khong tim thay input_dir: {config.input_dir}")
    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"input_dir khong phai thu muc: {config.input_dir}")
    if config.input_dir.resolve() == config.output_dir.resolve():
        raise ValueError("output_dir khong duoc trung input_dir de tranh ghi de du lieu.")
    if config.target_size[0] <= 0 or config.target_size[1] <= 0:
        raise ValueError("target_size phai lon hon 0.")
    if not 0 < config.min_crop_area_ratio < config.max_crop_area_ratio < 1:
        raise ValueError("crop area ratio phai nam trong khoang 0..1 va min < max.")


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Chuyển PIL image sang numpy RGB chắc chắn ở dạng uint8."""

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def build_foreground_mask(image: Image.Image) -> np.ndarray:
    """Tìm vùng foreground bằng tín hiệu đơn giản: contrast nền + saturation + edge.

    Ý tưởng:
    - Nền thường xuất hiện nhiều ở viền ảnh, nên lấy màu median của viền làm
      màu nền gần đúng.
    - Pixel khác màu nền đủ mạnh có khả năng thuộc object.
    - Trái cây/nông sản thường có saturation/texture/edge rõ hơn nền.
    - Kết hợp các tín hiệu rồi dùng morphology để mask bớt nhiễu.
    """

    rgb = image_to_rgb_array(image)
    height, width = rgb.shape[:2]

    # Lấy dải viền ảnh làm mẫu nền. Tỉ lệ 4% đủ nhỏ để không nuốt object ở giữa,
    # nhưng đủ lớn để nền median ổn định hơn vài pixel ngoài cùng.
    border = max(2, int(round(min(width, height) * 0.04)))
    border_pixels = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)

    # Mask 1: pixel khác màu nền. Dùng Otsu để tự chọn ngưỡng theo từng ảnh.
    color_distance = np.linalg.norm(rgb.astype(np.float32) - background_color, axis=2)
    color_distance_u8 = np.clip(color_distance, 0, 255).astype(np.uint8)
    _, contrast_mask = cv2.threshold(
        color_distance_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Mask 2: saturation. Tín hiệu này giúp bắt quả có màu nổi bật trên nền xám.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    saturation_threshold = max(
        35,
        int(float(np.mean(saturation)) + 0.35 * float(np.std(saturation))),
    )
    saturation_mask = np.where(saturation > saturation_threshold, 255, 0).astype(np.uint8)

    # Mask 3: edge. Canny giúp bắt biên vật thể khi màu object gần màu nền.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median_gray = float(np.median(gray))
    lower = int(max(0, 0.66 * median_gray))
    upper = int(min(255, 1.33 * median_gray))
    edges = cv2.Canny(gray, lower, upper)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=2)

    # Gộp mask. Dùng OR để không quá phụ thuộc vào một tín hiệu duy nhất.
    mask = cv2.bitwise_or(contrast_mask, saturation_mask)
    mask = cv2.bitwise_or(mask, edge_mask)

    # Morphology giúp nối các vùng foreground bị đứt và loại nhiễu nhỏ.
    kernel_size = max(3, int(round(min(width, height) * 0.018)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def bbox_from_mask(mask: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Lấy bbox bao quanh các connected components foreground đáng kể.

    Không lấy component lớn nhất duy nhất vì một quả có thể bị tách thành nhiều
    vùng do highlight/bóng đổ. Ta lấy union của các component đủ lớn, sau đó
    validate nghiêm ngặt ở bước sau.
    """

    if mask.ndim != 2:
        raise ValueError("mask phai la anh grayscale 2D.")

    width, height = image_size
    image_area = max(1, width * height)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        # Loại đốm nhiễu rất nhỏ. Ngưỡng 0.15% ảnh đủ thấp để giữ chi tiết vật
        # thể, nhưng tránh để vài edge nền kéo bbox ra quá rộng.
        if area / image_area < 0.0015:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Nới bbox để tránh crop quá sát làm mất cuống/vỏ/biên dạng quả."""

    x1, y1, x2, y2 = bbox
    width, height = image_size
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def compute_bbox_signals(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """Tính texture, contrast và edge density trong vùng crop ứng viên.

    Các tín hiệu này không dùng để phân loại, chỉ để quyết định crop có đáng tin
    hay không. Nếu crop không có texture/màu khác nền, khả năng cao bbox đang
    bắt nhầm nền hoặc vùng không chứa object.
    """

    rgb = image_to_rgb_array(image)
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    crop = rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0, 0.0, 0.0

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    texture_score = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

    border = max(2, int(round(min(width, height) * 0.04)))
    border_pixels = np.concatenate(
        [
            rgb[:border, :, :].reshape(-1, 3),
            rgb[-border:, :, :].reshape(-1, 3),
            rgb[:, :border, :].reshape(-1, 3),
            rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    crop_color = np.median(crop.reshape(-1, 3), axis=0)
    color_contrast = float(np.linalg.norm(crop_color - background_color))

    edges = cv2.Canny(cv2.GaussianBlur(crop_gray, (5, 5), 0), 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    return texture_score, color_contrast, edge_density


def validate_crop_candidate(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    config: SmartCropConfig,
) -> CropDecision:
    """Kiểm tra bbox có đủ an toàn để crop hay không.

    Các điều kiện bám sát yêu cầu:
    - Diện tích vùng crop trong khoảng 20%-80% ảnh.
    - Bbox không quá nhỏ.
    - Bbox không chiếm gần như toàn ảnh.
    - Tỉ lệ bbox không quá méo.
    - Vùng crop có texture/màu/edge đủ khác nền.
    - Nếu không chắc chắn thì giữ nguyên ảnh.
    """

    if bbox is None:
        return CropDecision(applied=False, bbox=None, reason="no_bbox")

    image_width, image_height = image.size
    x1, y1, x2, y2 = bbox
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    area_ratio = (box_width * box_height) / image_area

    if box_width <= 0 or box_height <= 0:
        return CropDecision(False, bbox, "empty_bbox", area_ratio)
    if area_ratio < config.min_crop_area_ratio:
        return CropDecision(False, bbox, "bbox_area_too_small", area_ratio)
    if area_ratio > config.max_crop_area_ratio:
        return CropDecision(False, bbox, "bbox_area_too_large", area_ratio)
    if box_width < config.min_bbox_side_pixels or box_height < config.min_bbox_side_pixels:
        return CropDecision(False, bbox, "bbox_pixel_side_too_small", area_ratio)
    if (
        box_width / image_width < config.min_bbox_side_ratio
        or box_height / image_height < config.min_bbox_side_ratio
    ):
        return CropDecision(False, bbox, "bbox_ratio_side_too_small", area_ratio)

    bbox_aspect_ratio = box_width / max(1, box_height)
    if (
        bbox_aspect_ratio > config.max_bbox_aspect_ratio
        or bbox_aspect_ratio < 1 / config.max_bbox_aspect_ratio
    ):
        return CropDecision(False, bbox, "bbox_aspect_too_skewed", area_ratio)

    texture_score, color_contrast, edge_density = compute_bbox_signals(image, bbox)

    # Chỉ cần một trong các tín hiệu foreground đủ rõ. Điều này tránh reject ảnh
    # quả trơn ít texture nhưng màu nổi bật, hoặc ảnh màu gần nền nhưng biên rõ.
    has_enough_foreground_signal = (
        texture_score >= config.min_texture_score
        or color_contrast >= config.min_color_contrast
        or edge_density >= config.min_edge_density
    )
    if not has_enough_foreground_signal:
        return CropDecision(
            applied=False,
            bbox=bbox,
            reason="weak_foreground_signal",
            area_ratio=area_ratio,
            texture_score=texture_score,
            color_contrast=color_contrast,
            edge_density=edge_density,
        )

    return CropDecision(
        applied=True,
        bbox=bbox,
        reason="crop_applied",
        area_ratio=area_ratio,
        texture_score=texture_score,
        color_contrast=color_contrast,
        edge_density=edge_density,
    )


def decide_smart_crop(image: Image.Image, config: SmartCropConfig) -> CropDecision:
    """Tìm bbox object và quyết định có crop hay fallback ảnh gốc."""

    mask = build_foreground_mask(image)
    bbox = bbox_from_mask(mask, image.size)
    if bbox is not None:
        bbox = expand_bbox(
            bbox=bbox,
            image_size=image.size,
            padding_ratio=config.crop_padding_ratio,
        )
    return validate_crop_candidate(image=image, bbox=bbox, config=config)


def standardize_output_image(
    image: Image.Image,
    decision: CropDecision,
    config: SmartCropConfig,
) -> Image.Image:
    """Tạo ảnh output cuối cùng: crop nếu chắc, sau đó resize/pad 320x320."""

    if decision.applied and decision.bbox is not None:
        image = image.crop(decision.bbox)

    # ResNet50 pipeline chính dùng 320x320. Resize giữ tỉ lệ + padding để không
    # làm méo hình dạng quả, đúng tinh thần không crop quá sát/không bóp méo ảnh.
    return resize_with_padding(
        image=image,
        target_size=config.target_size,
        background_color=(0, 0, 0),
    )


def process_image(
    image_path: Path,
    output_path: Path,
    config: SmartCropConfig,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """Xử lý một ảnh và trả về `(cropped?, reason)`.

    Nếu ảnh lỗi hoặc không đạt chất lượng, pipeline vẫn cố gắng fallback về ảnh
    gốc đã resize/pad khi có thể. Nếu không mở được ảnh thì ghi log lỗi và bỏ qua.
    """

    try:
        image = open_image_safely(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SKIP corrupted_image | path=%s | error=%s", image_path, exc)
        return False, "corrupted_image"

    quality_ok, quality_reason = validate_image_quality(
        image=image,
        min_image_side=config.min_image_side,
        max_image_side=config.max_image_side,
        max_aspect_ratio=config.max_aspect_ratio,
    )
    if not quality_ok:
        output_image = resize_with_padding(
            image=image,
            target_size=config.target_size,
            background_color=(0, 0, 0),
        )
        save_image(output_image, output_path, jpeg_quality=config.jpeg_quality)
        logger.info(
            "KEEP quality_failed | path=%s | reason=%s | size=%s",
            image_path,
            quality_reason,
            image.size,
        )
        return False, quality_reason or "quality_failed"

    blur_score = compute_blur_score(image)
    if blur_score < config.blur_threshold:
        output_image = resize_with_padding(
            image=image,
            target_size=config.target_size,
            background_color=(0, 0, 0),
        )
        save_image(output_image, output_path, jpeg_quality=config.jpeg_quality)
        logger.info(
            "KEEP blurry_image | path=%s | blur_score=%.4f",
            image_path,
            blur_score,
        )
        return False, "blurry_image"

    decision = decide_smart_crop(image=image, config=config)
    output_image = standardize_output_image(
        image=image,
        decision=decision,
        config=config,
    )
    save_image(output_image, output_path, jpeg_quality=config.jpeg_quality)

    logger.info(
        (
            "%s | path=%s | output=%s | reason=%s | bbox=%s | area=%.4f | "
            "texture=%.4f | contrast=%.4f | edge_density=%.6f"
        ),
        "CROP" if decision.applied else "KEEP",
        image_path,
        output_path,
        decision.reason,
        decision.bbox,
        decision.area_ratio,
        decision.texture_score,
        decision.color_contrast,
        decision.edge_density,
    )
    return decision.applied, decision.reason


def process_class_dir(
    class_dir: Path,
    config: SmartCropConfig,
    logger: logging.Logger,
) -> ClassCropStats:
    """Xử lý toàn bộ ảnh trong một class, giữ nguyên tên folder label."""

    stats = ClassCropStats(class_name=class_dir.name)
    output_class_dir = config.output_dir / class_dir.name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(class_dir, config.valid_extensions)
    stats.total = len(image_files)
    logger.info("[%s] bat dau | total=%d", class_dir.name, stats.total)

    for image_path in tqdm(image_files, desc=f"Smart crop {class_dir.name}", leave=False):
        output_path = output_class_dir / image_path.name
        cropped, reason = process_image(
            image_path=image_path,
            output_path=output_path,
            config=config,
            logger=logger,
        )
        stats.reasons[reason] += 1
        if reason == "corrupted_image":
            stats.failed += 1
        elif cropped:
            stats.cropped += 1
        else:
            stats.kept_original += 1

    logger.info(
        "[%s] ket thuc | total=%d | cropped=%d | kept_original=%d | failed=%d | reasons=%s",
        stats.class_name,
        stats.total,
        stats.cropped,
        stats.kept_original,
        stats.failed,
        dict(stats.reasons),
    )
    return stats


def log_final_summary(
    class_stats: list[ClassCropStats],
    config: SmartCropConfig,
    logger: logging.Logger,
) -> None:
    """Ghi summary cuối cùng đúng các thông tin yêu cầu."""

    total_images = sum(stats.total for stats in class_stats)
    total_cropped = sum(stats.cropped for stats in class_stats)
    total_kept = sum(stats.kept_original for stats in class_stats)
    total_failed = sum(stats.failed for stats in class_stats)
    all_reasons: Counter[str] = Counter()
    for stats in class_stats:
        all_reasons.update(stats.reasons)

    logger.info("===== SMART CROP SUMMARY =====")
    logger.info("input_dir=%s", config.input_dir)
    logger.info("output_dir=%s", config.output_dir)
    logger.info("target_size=%s", config.target_size)
    logger.info("total_images=%d", total_images)
    logger.info("cropped_images=%d", total_cropped)
    logger.info("kept_original_images=%d", total_kept)
    logger.info("failed_images=%d", total_failed)
    logger.info("reason_counts=%s", dict(all_reasons))
    logger.info("----- per class -----")
    for stats in class_stats:
        logger.info(
            (
                "class=%s | total=%d | cropped=%d | kept_original=%d | "
                "failed=%d | reasons=%s"
            ),
            stats.class_name,
            stats.total,
            stats.cropped,
            stats.kept_original,
            stats.failed,
            dict(stats.reasons),
        )


def run_smart_crop(
    config: SmartCropConfig,
    logger: logging.Logger | None = None,
) -> list[ClassCropStats]:
    """Entry chính cho pipeline smart crop."""

    validate_config(config)
    if logger is None:
        logger = setup_logger(config.log_file, logger_name="smart_crop")
    logger.info("Bat dau smart crop voi config: %s", config)

    prepare_output_directory(config.output_dir, clean=config.clean_output_dir)
    class_dirs = list_class_directories(config.input_dir)
    if not class_dirs:
        raise ValueError(f"Khong tim thay class folder nao trong: {config.input_dir}")

    class_stats: list[ClassCropStats] = []
    for class_dir in tqdm(class_dirs, desc="Smart crop classes", leave=True):
        class_stats.append(
            process_class_dir(
                class_dir=class_dir,
                config=config,
                logger=logger,
            )
        )

    log_final_summary(class_stats=class_stats, config=config, logger=logger)
    logger.info("Hoan tat smart crop.")
    return class_stats


def main() -> None:
    """Cho phép chạy script trực tiếp bằng `python src/smart_crop.py`."""

    args = parse_args()
    config = build_config(args)
    run_smart_crop(config)


if __name__ == "__main__":
    main()
