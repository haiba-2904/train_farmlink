from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="farmlink")
def resnet50_preprocess_input(tensor: tf.Tensor) -> tf.Tensor:
    """Wrapper có thể serialize cho ResNet50 preprocess_input."""

    return tf.keras.applications.resnet.preprocess_input(tensor)


def build_resnet50_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int] = (320, 320, 3),
    dropout_rate: float = 0.5,
    head_units: int = 512,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Xây dựng classifier dựa trên ResNet50 pretrained ImageNet."""

    if num_classes <= 0:
        raise ValueError("num_classes phai lon hon 0.")
    if head_units <= 0:
        raise ValueError("head_units phai lon hon 0.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate phai nam trong khoang [0.0, 1.0).")

    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")

    # ResNet50 baseline hiện nhận tensor đã qua
    # `tf.keras.applications.resnet.preprocess_input` từ dataloader. Tách
    # preprocessing khỏi model giúp flow train rõ ràng:
    # load -> augment -> preprocess_input -> batch -> prefetch.
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(head_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_farm")
    return model, base_model


def set_fine_tuning_resnet(
    base_model: tf.keras.Model,
    fine_tune_last_layers: int = 50,
) -> None:
    """Mở fine-tune phần cuối ResNet50 nhưng luôn freeze BatchNorm layers."""

    if fine_tune_last_layers <= 0:
        raise ValueError("fine_tune_last_layers phai lon hon 0.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    fine_tune_from = max(0, total_layers - fine_tune_last_layers)

    for layer_index, layer in enumerate(base_model.layers):
        # BatchNorm giữ moving mean/variance; nếu train lại với batch nhỏ sẽ dễ
        # làm lệch thống kê ImageNet và khiến fine-tuning mất ổn định.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue

        layer.trainable = layer_index >= fine_tune_from
