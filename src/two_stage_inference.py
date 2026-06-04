from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

try:
    from src.utils import open_image_safely, resize_with_padding, validate_image_quality
except ImportError:  # pragma: no cover
    from utils import open_image_safely, resize_with_padding, validate_image_quality


def parse_args() -> argparse.Namespace:
    """CLI test nhanh router 2-stage bằng một ảnh local."""

    parser = argparse.ArgumentParser(description="Two-stage fruit classifier inference.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True, help="router_manifest.json")
    parser.add_argument("--fruit-threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--strict-quality-check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Đọc JSON manifest an toàn."""

    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest khong phai JSON object: {path}")
    return payload


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve path trong router manifest theo experiment_dir."""

    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve() if not path.exists() else path.resolve()


def prepare_image_tensor(
    image_path: Path,
    manifest: dict[str, Any],
    strict_quality_check: bool,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess ảnh upload giống train: EXIF -> RGB -> resize/pad -> normalize."""

    warnings: list[str] = []
    image = open_image_safely(image_path)
    quality_ok, quality_reason = validate_image_quality(
        image=image,
        min_image_side=100,
        max_image_side=5000,
        max_aspect_ratio=4.0,
    )
    if not quality_ok and quality_reason is not None:
        if strict_quality_check:
            raise ValueError(f"Anh upload khong dat chat luong: {quality_reason}")
        warnings.append(quality_reason)

    image_size = tuple(manifest.get("image_size", [320, 320]))
    standardized_image = resize_with_padding(
        image=image,
        target_size=(int(image_size[1]), int(image_size[0])),
        background_color=(0, 0, 0),
    )
    image_array = np.asarray(standardized_image, dtype=np.float32)

    preprocess_config = manifest.get("preprocess", {})
    normalization_location = preprocess_config.get("normalization_location", "dataloader")
    model_type = manifest.get("model_type", "resnet50")
    if normalization_location == "dataloader":
        if model_type == "resnet50":
            image_array = tf.keras.applications.resnet.preprocess_input(image_array)
        elif model_type == "mobilenetv2":
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        else:
            raise ValueError(f"model_type khong duoc ho tro: {model_type}")

    return np.expand_dims(image_array, axis=0), warnings


def top_k_predictions(
    probabilities: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Format top-k softmax predictions của Stage B."""

    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "class_index": int(index),
            "label": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]


def predict_two_stage(
    image_path: Path,
    router_manifest_path: Path,
    fruit_threshold_override: float | None = None,
    top_k: int = 5,
    strict_quality_check: bool = False,
) -> dict[str, Any]:
    """Inference 2-stage cho backend website.

    Logic:
    - Stage A trả xác suất ảnh là fruit.
    - Nếu fruit_probability < threshold: trả `other`.
    - Nếu qua Stage A: gọi Stage B để phân loại fruit class cụ thể.
    """

    router_manifest = load_json(router_manifest_path)
    base_dir = router_manifest_path.parent

    stage_a_model_path = resolve_path(base_dir, router_manifest["stage_a_model_path"])
    stage_a_labels_path = resolve_path(base_dir, router_manifest["stage_a_label_manifest_path"])
    stage_b_model_path = resolve_path(base_dir, router_manifest["stage_b_model_path"])
    stage_b_labels_path = resolve_path(base_dir, router_manifest["stage_b_label_manifest_path"])

    stage_a_manifest = load_json(stage_a_labels_path)
    stage_b_manifest = load_json(stage_b_labels_path)
    stage_a_model = tf.keras.models.load_model(str(stage_a_model_path), compile=False)
    stage_b_model = tf.keras.models.load_model(str(stage_b_model_path), compile=False)

    image_tensor, warnings = prepare_image_tensor(
        image_path=image_path,
        manifest=stage_a_manifest,
        strict_quality_check=strict_quality_check,
    )

    fruit_threshold = (
        float(fruit_threshold_override)
        if fruit_threshold_override is not None
        else float(router_manifest.get("fruit_threshold", 0.5))
    )
    fruit_probability = float(stage_a_model.predict(image_tensor, verbose=0).reshape(-1)[0])

    if fruit_probability < fruit_threshold:
        return {
            "predicted_label": "other",
            "confidence": float(1.0 - fruit_probability),
            "route": "stage_a_other",
            "stage_a": {
                "fruit_probability": fruit_probability,
                "fruit_threshold": fruit_threshold,
            },
            "stage_b": None,
            "warnings": warnings,
            "image_path": str(image_path),
        }

    stage_b_probabilities = stage_b_model.predict(image_tensor, verbose=0)[0]
    fruit_class_names = [str(class_name) for class_name in stage_b_manifest["class_names"]]
    predictions = top_k_predictions(stage_b_probabilities, fruit_class_names, top_k)
    best_prediction = predictions[0]
    stage_b_threshold = float(router_manifest.get("stage_b_confidence_threshold", 0.6))
    used_low_confidence = best_prediction["confidence"] < stage_b_threshold

    return {
        "predicted_label": best_prediction["label"],
        "confidence": float(best_prediction["confidence"]),
        "route": "stage_b_fruit",
        "low_confidence": bool(used_low_confidence),
        "stage_a": {
            "fruit_probability": fruit_probability,
            "fruit_threshold": fruit_threshold,
        },
        "stage_b": {
            "confidence_threshold": stage_b_threshold,
            "top_k_predictions": predictions,
        },
        "warnings": warnings,
        "image_path": str(image_path),
    }


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    result = predict_two_stage(
        image_path=args.image,
        router_manifest_path=args.router,
        fruit_threshold_override=args.fruit_threshold,
        top_k=args.top_k,
        strict_quality_check=args.strict_quality_check,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
