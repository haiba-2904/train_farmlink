# So sánh kết quả huấn luyện model

Generated at: 2026-06-06T15:20:00

## Model được so sánh

| Model | Run | Ghi chú |
|---|---|---|
| ResNet50 supported v1 | `experiments/stage_b_supported_v1_resnet50_20260602_223530` | Model hien tai dang dung cho Stage B production v1, 32 supported classes, khong co class other. |
| MobileNetV2 best old run | `archive_unused/resnet50_pipeline_cleanup_20260602_0100/logs/log_l1` | Run MobileNetV2 cu tot nhat tim thay trong archive, dataset/class taxonomy khac pipeline supported v1 hien tai. |

## Metric chính trên test set

| Metric | ResNet50 supported v1 | MobileNetV2 best old run | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.8089 | 0.7503 | +0.0586 |
| Macro Precision | 0.8699 | 0.7818 | +0.0881 |
| Macro Recall | 0.7939 | 0.7234 | +0.0705 |
| Macro F1 | 0.8023 | 0.7113 | +0.0910 |
| Weighted F1 | 0.8141 | 0.7417 | +0.0724 |

## Kết luận nhanh

- ResNet50 supported v1 có Macro F1 cao hơn MobileNetV2 cũ khoảng `+0.0910`.
- ResNet50 supported v1 có Accuracy cao hơn MobileNetV2 cũ khoảng `+0.0586`.
- Lưu ý: hai model không hoàn toàn cùng taxonomy. MobileNetV2 cũ train trên 45 class gồm `other` và taxonomy cũ; ResNet50 supported v1 train trên 32 supported classes, không gồm `other` và đã loại unsupported classes.

## File biểu đồ

- `training_curves_resnet50_supported_v1.png`
- `training_curves_mobilenetv2_best_old_run.png`
- `comparison_test_metrics.png`
- `comparison_validation_curves.png`
- `comparison_best_validation_summary.png`
- `confusion_matrix_resnet50.png`
- `confusion_matrix_mobilenetv2.png`
