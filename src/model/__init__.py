"""Package chứa các kiến trúc model để dễ mở rộng nhiều model trong tương lai."""

from .binary import (
    build_resnet50_binary_classifier,
    compile_binary_model,
    set_fine_tuning_binary_resnet,
)
from .mobilenetv2 import (
    build_mobilenetv2_classifier,
    compile_model,
    set_fine_tuning,
)
from .resnet50 import build_resnet50_classifier, set_fine_tuning_resnet

__all__ = [
    "build_resnet50_binary_classifier",
    "build_mobilenetv2_classifier",
    "build_resnet50_classifier",
    "compile_binary_model",
    "compile_model",
    "set_fine_tuning_binary_resnet",
    "set_fine_tuning",
    "set_fine_tuning_resnet",
]
