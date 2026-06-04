from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

CropMode = Literal["none", "contour", "saliency", "auto"]
SUPPORTED_CROP_MODES: tuple[str, ...] = ("none", "contour", "saliency", "auto")


@dataclass(frozen=True)
class CropResult:
    """Metadata của bước smart-crop để debug inference trên website.

    `applied=False` nghĩa là crop không đủ tin cậy và pipeline đã dùng ảnh gốc.
    Điều này rất quan trọng vì crop sai còn nguy hiểm hơn không crop.
    """

    requested_mode: str
    method: str
    applied: bool
    bbox: tuple[int, int, int, int] | None
    original_size: tuple[int, int]
    cropped_size: tuple[int, int]
    box_area_ratio: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        payload["original_size"] = list(self.original_size)
        payload["cropped_size"] = list(self.cropped_size)
        return payload


def no_crop_result(image: Image.Image, requested_mode: str, reason: str) -> CropResult:
    width, height = image.size
    return CropResult(
        requested_mode=requested_mode,
        method="none",
        applied=False,
        bbox=None,
        original_size=(width, height),
        cropped_size=(width, height),
        box_area_ratio=1.0,
        reason=reason,
    )


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Nới bbox để không cắt mất viền/vùng đặc trưng của nông sản."""

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


def _validate_bbox(
    bbox: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_side_ratio: float,
) -> tuple[bool, float, str]:
    """Chỉ chấp nhận crop khi bbox vừa đủ lớn và không gần như toàn ảnh."""

    if bbox is None:
        return False, 1.0, "no_bbox"

    x1, y1, x2, y2 = bbox
    image_width, image_height = image_size
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    image_area = max(1, image_width * image_height)
    box_area_ratio = (box_width * box_height) / image_area

    if box_width <= 0 or box_height <= 0:
        return False, box_area_ratio, "empty_bbox"
    if box_area_ratio < min_area_ratio:
        return False, box_area_ratio, "bbox_too_small"
    if box_area_ratio > max_area_ratio:
        return False, box_area_ratio, "bbox_too_large"
    if box_width / image_width < min_side_ratio or box_height / image_height < min_side_ratio:
        return False, box_area_ratio, "bbox_side_too_small"
    return True, box_area_ratio, "ok"


def _bbox_from_mask(mask: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Lấy bbox của connected component foreground lớn nhất."""

    if mask.ndim != 2:
        raise ValueError("mask phai la anh grayscale 2D.")

    mask = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    image_width, image_height = image_size
    image_area = max(1, image_width * image_height)
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area and area / image_area >= 0.002:
            best_area = area
            best_label = label

    if best_label is None:
        return None

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def _postprocess_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Làm sạch mask foreground bằng morphology."""

    width, height = image_size
    kernel_size = max(3, int(round(min(width, height) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def contour_foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Mức 1: crop foreground bằng contour/segmentation đơn giản.

    Phù hợp ảnh nền sạch: vật thể khác nền rõ màu/biên. Hàm kết hợp 3 tín hiệu:
    khác màu nền viền ảnh, saturation HSV và edge Canny.
    """

    image = image.convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    if width < 20 or height < 20:
        return None

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
    color_distance = np.linalg.norm(rgb.astype(np.float32) - background_color, axis=2)
    color_distance_u8 = np.clip(color_distance, 0, 255).astype(np.uint8)
    _, bg_mask = cv2.threshold(
        color_distance_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Nếu ngưỡng Otsu quá thấp, saturation mask thường phủ cả nền xám/nhiễu.
    sat_mask = np.where(saturation > max(35, int(np.mean(saturation) + 0.4 * np.std(saturation))), 255, 0).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=2)

    mask = cv2.bitwise_or(bg_mask, sat_mask)
    mask = cv2.bitwise_or(mask, edge_mask)
    mask = _postprocess_mask(mask, image_size=(width, height))
    return _bbox_from_mask(mask, image_size=(width, height))


def saliency_foreground_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Mức 2: crop foreground bằng spectral-residual saliency.

    OpenCV saliency module thường cần opencv-contrib. Để project chạy ổn trên
    macOS hiện tại, ta dùng spectral residual tự triển khai bằng FFT.
    """

    image = image.convert("RGB")
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    if width < 20 or height < 20:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.fft.fft2(resized)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    log_amplitude = np.log(amplitude + 1e-8)
    average_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - average_log_amplitude
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    saliency = cv2.resize(saliency, (width, height), interpolation=cv2.INTER_CUBIC)

    percentile_threshold = int(np.percentile(saliency, 78))
    otsu_threshold, _ = cv2.threshold(
        saliency,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = max(35, percentile_threshold, int(otsu_threshold))
    mask = np.where(saliency >= threshold, 255, 0).astype(np.uint8)
    mask = _postprocess_mask(mask, image_size=(width, height))
    return _bbox_from_mask(mask, image_size=(width, height))


def crop_image_by_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    requested_mode: str,
    method: str,
    box_area_ratio: float,
) -> tuple[Image.Image, CropResult]:
    crop = image.crop(bbox)
    return crop, CropResult(
        requested_mode=requested_mode,
        method=method,
        applied=True,
        bbox=bbox,
        original_size=image.size,
        cropped_size=crop.size,
        box_area_ratio=box_area_ratio,
        reason="ok",
    )


def try_crop_method(
    image: Image.Image,
    requested_mode: str,
    method: str,
    padding_ratio: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_side_ratio: float,
) -> tuple[Image.Image, CropResult]:
    """Chạy một method crop và fallback ảnh gốc nếu bbox không đạt chuẩn."""

    if method == "contour":
        bbox = contour_foreground_bbox(image)
    elif method == "saliency":
        bbox = saliency_foreground_bbox(image)
    else:
        raise ValueError(f"Unsupported crop method: {method}")

    if bbox is not None:
        bbox = _expand_bbox(bbox, image.size, padding_ratio=padding_ratio)

    is_valid, box_area_ratio, reason = _validate_bbox(
        bbox=bbox,
        image_size=image.size,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_side_ratio=min_side_ratio,
    )
    if not is_valid or bbox is None:
        return image, no_crop_result(image, requested_mode, f"{method}:{reason}")
    return crop_image_by_bbox(
        image=image,
        bbox=bbox,
        requested_mode=requested_mode,
        method=method,
        box_area_ratio=box_area_ratio,
    )


def smart_crop_image(
    image: Image.Image,
    mode: str = "none",
    padding_ratio: float = 0.12,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.92,
    min_side_ratio: float = 0.12,
) -> tuple[Image.Image, CropResult]:
    """Crop ảnh upload trước inference theo mức 1/2.

    `auto` ưu tiên contour vì ổn với nền sạch; nếu contour không đáng tin thì
    fallback sang saliency. Nếu cả hai đều không ổn, trả ảnh gốc.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_CROP_MODES:
        raise ValueError(
            f"crop_mode khong hop le: {mode}. Hop le: {', '.join(SUPPORTED_CROP_MODES)}"
        )

    image = image.convert("RGB")
    if normalized_mode == "none":
        return image, no_crop_result(image, requested_mode=normalized_mode, reason="disabled")

    if normalized_mode in {"contour", "saliency"}:
        return try_crop_method(
            image=image,
            requested_mode=normalized_mode,
            method=normalized_mode,
            padding_ratio=padding_ratio,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_side_ratio=min_side_ratio,
        )

    contour_image, contour_result = try_crop_method(
        image=image,
        requested_mode=normalized_mode,
        method="contour",
        padding_ratio=padding_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_side_ratio=min_side_ratio,
    )
    if contour_result.applied:
        return contour_image, contour_result

    saliency_image, saliency_result = try_crop_method(
        image=image,
        requested_mode=normalized_mode,
        method="saliency",
        padding_ratio=padding_ratio,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_side_ratio=min_side_ratio,
    )
    if saliency_result.applied:
        return saliency_image, saliency_result

    return image, no_crop_result(
        image,
        requested_mode=normalized_mode,
        reason=f"auto_failed:{contour_result.reason};{saliency_result.reason}",
    )
