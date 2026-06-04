from __future__ import annotations

import tensorflow as tf

from .mobilenetv2 import build_mobilenetv2_classifier
from .resnet50 import build_resnet50_classifier


def build_model(
    model_type: str,
    num_classes: int,
    input_shape: tuple[int, int, int],
    dropout_rate: float,
    head_units: int | None = None,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Factory tạo model theo tên kiến trúc, giúp mở rộng thêm model sau này."""

    normalized_model_type = model_type.strip().lower()

    if normalized_model_type == "mobilenetv2":
        return build_mobilenetv2_classifier(
            num_classes=num_classes,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
        )

    if normalized_model_type == "resnet50":
        resolved_head_units = 512 if head_units is None else head_units
        return build_resnet50_classifier(
            num_classes=num_classes,
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            head_units=resolved_head_units,
        )

    raise ValueError("Unsupported model")
