from __future__ import annotations

import tensorflow as tf


def build_mobilenetv2_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Xây dựng classifier dựa trên MobileNetV2 pretrained ImageNet."""

    if num_classes <= 0:
        raise ValueError("num_classes phai lon hon 0.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")

    # Dataloader đã preprocess_input sẵn nên model chỉ nhận tensor đã chuẩn hóa.
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_farm")
    return model, base_model


def set_fine_tuning(
    base_model: tf.keras.Model,
    fine_tune_last_layers: int = 40,
) -> None:
    """Mở fine-tune phần đuôi của base_model nhưng luôn giữ BatchNorm frozen."""

    if fine_tune_last_layers <= 0:
        raise ValueError("fine_tune_last_layers phai lon hon 0.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    fine_tune_from = max(0, total_layers - fine_tune_last_layers)

    for layer_index, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue

        layer.trainable = layer_index >= fine_tune_from


def compile_model(
    model: tf.keras.Model,
    learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule = 3e-4,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> None:
    """Compile model bằng Focal Loss để tập trung hơn vào hard samples.

    Lưu ý:
    - Pipeline train/eval phải cung cấp label dạng one-hot.
    - Không dùng `class_weight` cùng lúc với focal loss để tránh chồng trọng số
      theo hai cơ chế khác nhau, dễ làm gradient lệch mạnh trên các lớp hiếm.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        ),
        metrics=["accuracy"],
    )
