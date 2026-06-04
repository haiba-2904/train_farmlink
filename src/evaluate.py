from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from src.utils import open_image_safely, resize_with_padding, validate_image_quality
except ImportError:  # pragma: no cover
    from utils import open_image_safely, resize_with_padding, validate_image_quality


@dataclass(frozen=True)
class InferenceConfig:
    """Cấu hình inference cho ảnh người dùng upload từ website hoặc CLI local."""

    model_path: Path = Path("models/mobilenetv2_farm.keras")
    label_manifest_path: Path = Path("models/mobilenetv2_farm.labels.json")
    image_path: Path | None = None
    confidence_threshold: float = 0.6
    top_k: int = 5
    strict_quality_check: bool = False
    min_image_side: int = 100
    max_image_side: int = 5000
    max_aspect_ratio: float = 4.0
    background_color: tuple[int, int, int] = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    """Nhận tham số từ CLI để tiện test nhanh trên macOS local."""

    parser = argparse.ArgumentParser(
        description="Du doan anh nong san bang model MobileNetV2 da huan luyen."
    )
    parser.add_argument("--image", type=Path, required=True, help="Duong dan anh can du doan.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/mobilenetv2_farm.keras"),
        help="Duong dan toi model .keras da huan luyen.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("models/mobilenetv2_farm.labels.json"),
        help="Duong dan toi label manifest JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Nguong confidence de fallback neu model khong du tu tin.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="So luong du doan xac suat cao nhat can tra ve.",
    )
    parser.add_argument(
        "--strict-quality-check",
        action="store_true",
        help="Bat che do fail fast neu anh upload qua nho/qua lech ti le.",
    )
    return parser.parse_args()


def load_label_manifest(manifest_path: Path) -> dict[str, Any]:
    """Đọc label manifest đã export lúc train để giữ đúng thứ tự nhãn."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Khong tim thay label manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    class_names = manifest.get("class_names", [])
    if not class_names:
        raise ValueError("Label manifest khong chua class_names hop le.")

    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names trong label manifest dang bi trung.")

    return manifest


def load_inference_model(model_path: Path) -> tf.keras.Model:
    """Nạp model đã huấn luyện để phục vụ inference."""

    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model: {model_path}")
    return tf.keras.models.load_model(str(model_path))


def validate_model_and_labels(
    model: tf.keras.Model,
    class_names: list[str],
) -> None:
    """Đảm bảo output của model khớp số lượng nhãn trong label manifest."""

    output_units = model.output_shape[-1]
    if output_units != len(class_names):
        raise ValueError(
            "So output cua model khong khop so class trong manifest: "
            f"{output_units} != {len(class_names)}"
        )


def prepare_image_tensor_for_inference(
    image_path: Path,
    image_size: tuple[int, int],
    background_color: tuple[int, int, int],
    min_image_side: int,
    max_image_side: int,
    max_aspect_ratio: float,
    strict_quality_check: bool,
) -> tuple[np.ndarray, list[str]]:
    """Tiền xử lý ảnh upload giống lúc train: EXIF -> RGB -> resize/pad -> preprocess_input."""

    warnings: list[str] = []
    image = open_image_safely(image_path)

    quality_ok, quality_reason = validate_image_quality(
        image=image,
        min_image_side=min_image_side,
        max_image_side=max_image_side,
        max_aspect_ratio=max_aspect_ratio,
    )
    if not quality_ok and quality_reason is not None:
        if strict_quality_check:
            raise ValueError(
                f"Anh upload khong dat tieu chi chat luong de suy luan: {quality_reason}"
            )
        warnings.append(quality_reason)

    standardized_image = resize_with_padding(
        image=image,
        target_size=image_size,
        background_color=background_color,
    )
    image_array = np.asarray(standardized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array, warnings


def predict_with_confidence_fallback(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_names: list[str],
    confidence_threshold: float,
    top_k: int,
    fallback_label: str,
) -> dict[str, Any]:
    """Sinh top-k prediction và fallback nếu model không đủ tự tin."""

    probabilities = model.predict(image_tensor, verbose=0)[0]
    if probabilities.ndim != 1:
        raise ValueError("Dau ra model khong dung dinh dang softmax 1 chieu.")

    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = [
        {
            "class_index": int(index),
            "label": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    best_prediction = top_predictions[0]
    best_label = best_prediction["label"]
    best_confidence = best_prediction["confidence"]
    use_fallback = best_confidence < confidence_threshold

    final_label = fallback_label if use_fallback else best_label

    return {
        "predicted_label": final_label,
        "raw_best_label": best_label,
        "predicted_class_index": int(best_prediction["class_index"]),
        "confidence": float(best_confidence),
        "used_fallback": bool(use_fallback),
        "confidence_threshold": float(confidence_threshold),
        "top_k_predictions": top_predictions,
    }


def predict_image_file(
    image_path: Path,
    model_path: Path = Path("models/mobilenetv2_farm.keras"),
    label_manifest_path: Path = Path("models/mobilenetv2_farm.labels.json"),
    confidence_threshold: float = 0.6,
    top_k: int = 5,
    strict_quality_check: bool = False,
) -> dict[str, Any]:
    """Hàm tiện ích cho backend website: truyền path ảnh vào và nhận JSON kết quả."""

    manifest = load_label_manifest(label_manifest_path)
    class_names = manifest["class_names"]
    model = load_inference_model(model_path)
    validate_model_and_labels(model, class_names)

    image_size = tuple(manifest.get("image_size", [224, 224]))
    serving_config = manifest.get("serving", {})
    fallback_label = serving_config.get("fallback_label", "unknown")

    image_tensor, warnings = prepare_image_tensor_for_inference(
        image_path=image_path,
        image_size=image_size,
        background_color=(0, 0, 0),
        min_image_side=100,
        max_image_side=5000,
        max_aspect_ratio=4.0,
        strict_quality_check=strict_quality_check,
    )

    prediction = predict_with_confidence_fallback(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        top_k=top_k,
        fallback_label=fallback_label,
    )
    prediction["image_path"] = str(image_path)
    prediction["warnings"] = warnings
    return prediction


def main() -> None:
    """Entry point CLI để test nhanh model bằng ảnh local trên macOS."""

    args = parse_args()
    config = InferenceConfig(
        model_path=args.model,
        label_manifest_path=args.labels,
        image_path=args.image,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
        strict_quality_check=args.strict_quality_check,
    )

    prediction = predict_image_file(
        image_path=config.image_path if config.image_path is not None else Path(),
        model_path=config.model_path,
        label_manifest_path=config.label_manifest_path,
        confidence_threshold=config.confidence_threshold,
        top_k=config.top_k,
        strict_quality_check=config.strict_quality_check,
    )
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
