from __future__ import annotations

import tensorflow as tf


def build_resnet50_binary_classifier(
    input_shape: tuple[int, int, int] = (320, 320, 3),
    dropout_rate: float = 0.5,
    head_units: int = 256,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Tạo Stage A classifier: phân biệt `other` và `fruit`.

    Model nhận tensor đã qua `tf.keras.applications.resnet.preprocess_input`
    từ dataloader/inference. Output là sigmoid 1 chiều: xác suất ảnh thuộc
    nhóm `fruit`. Nếu xác suất này thấp hơn threshold, router trả về `other`.
    """

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
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(head_units, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fruit_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_fruit_vs_other")
    return model, base_model


def set_fine_tuning_binary_resnet(
    base_model: tf.keras.Model,
    fine_tune_last_layers: int = 50,
) -> None:
    """Fine-tune phần cuối ResNet50 cho Stage A, luôn freeze BatchNorm."""

    if fine_tune_last_layers <= 0:
        raise ValueError("fine_tune_last_layers phai lon hon 0.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    fine_tune_from = max(0, total_layers - fine_tune_last_layers)

    for layer_index, layer in enumerate(base_model.layers):
        # BatchNorm giữ thống kê pretrained; train lại với batch nhỏ dễ làm lệch
        # phân phối và khiến fine-tune kém ổn định.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = layer_index >= fine_tune_from


def compile_binary_model(
    model: tf.keras.Model,
    learning_rate: float | tf.keras.optimizers.schedules.LearningRateSchedule = 3e-4,
    label_smoothing: float = 0.0,
) -> None:
    """Compile Stage A bằng binary crossentropy.

    Stage A là bài toán nhị phân rõ ràng:
    - label 0: other
    - label 1: fruit/nông sản

    Mặc định không dùng label smoothing để đúng yêu cầu baseline và để xác suất
    sigmoid dễ diễn giải khi tune threshold.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
