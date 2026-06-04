# ResNet50 Two-stage Folder Contract

Tài liệu này chốt vai trò folder cho pipeline chính mới của project.

Quyết định hiện tại:

- Số class chính thức: 44 class, bao gồm `other`.
- Dữ liệu gốc: `dataset/raw`.
- Backbone chính: ResNet50.
- `other` chỉ dùng cho Stage A.
- Stage B chỉ phân loại nông sản, không chứa `other`.
- Không dùng MobileNetV2 trong pipeline chính.
- Không dùng pairwise classifier hoặc hard mining trong pipeline chính lúc này.

## 1. Folder Dataset Chính

### `dataset/raw`

Vai trò: dữ liệu gốc ban đầu.

Quy tắc:

- Không chỉnh sửa trực tiếp.
- Không resize/crop/rename thủ công trong folder này.
- Là nguồn sự thật để tái tạo toàn bộ dataset.
- Hiện có 44 class folder, bao gồm `other`.

### `dataset/processed`

Vai trò: output preprocess cũ.

Quy tắc:

- Xem như artifact trung gian/cũ.
- Không dùng làm input chính cho pipeline ResNet50 two-stage mới nếu đã có `processed_clean` hoặc `processed_crop`.
- Không xóa vội vì còn dùng để đối chiếu.

### `dataset/processed_clean`

Vai trò: dataset đã preprocess và clean.

Quy tắc:

- Là input chuẩn cho bước smart crop.
- Có thể dùng làm baseline nếu muốn so sánh trước/sau crop.
- Không chứa split train/val/test; đây vẫn là dataset theo class.

### `dataset/processed_crop`

Vai trò: output của bước smart crop nhẹ.

Quy tắc:

- Đây là dataset ứng viên cho pipeline chính mới.
- Được sinh từ `dataset/processed_clean`.
- Nếu crop tự tin thì lưu ảnh crop.
- Nếu crop không chắc thì giữ ảnh gốc đã chuẩn hóa.
- Không ghi đè `processed_clean`.
- Sau khi tạo xong, dùng folder này để split ra train/val/test.

### `dataset/train`, `dataset/val`, `dataset/test`

Vai trò: split chính cho Stage A.

Quy tắc:

- Dùng cho bài toán binary Stage A: `fruit` vs `other`.
- Vẫn chứa đủ 44 class folder vật lý.
- Khi train Stage A, dataloader map:
  - `other` -> label `other`
  - tất cả class còn lại -> label `fruit`
- Không dùng trực tiếp cho Stage B nếu còn chứa `other`.

### `dataset_fruit_only/train`, `dataset_fruit_only/val`, `dataset_fruit_only/test`

Vai trò: split chính cho Stage B.

Quy tắc:

- Chỉ chứa 43 class nông sản.
- Tuyệt đối không chứa `other`.
- Dùng cho bài toán multi-class fruit classification.
- Được tạo từ split chính bằng cách loại `other`.
- Sau cleanup ngày 2026-06-02, folder `dataset_fruit_only` cũ đã được đưa vào
  archive vì được tạo trước khi split lại từ `processed_crop`. Cần tạo lại
  `dataset_fruit_only` từ `dataset/train`, `dataset/val`, `dataset/test` mới
  trước khi train Stage B.

## 2. Folder Không Thuộc Pipeline Chính

### `dataset/hard_examples`

Vai trò: artifact của hard example mining cũ.

Quy tắc:

- Không dùng trong pipeline chính hiện tại.
- Chỉ giữ để tham khảo/phân tích.

### `dataset/train_hard_mining`

Vai trò: train set phụ của hard mining cũ.

Quy tắc:

- Không dùng trong pipeline chính hiện tại.
- Không dùng để train Stage A hoặc Stage B mới.

### `dataset/other_candidates`

Vai trò: nguồn hoặc ứng viên ảnh `other`.

Quy tắc:

- Có thể dùng để bổ sung/làm sạch class `other`.
- Không đưa thẳng vào train nếu chưa qua preprocess, clean, split.

## 3. Folder Experiment

Tất cả kết quả train/evaluate mới phải lưu trong `experiments/`.

Quy ước đặt tên:

```text
experiments/stage_a_resnet50_crop_YYYYMMDD_HHMMSS/
experiments/stage_b_resnet50_crop_YYYYMMDD_HHMMSS/
experiments/two_stage_resnet50_crop_YYYYMMDD_HHMMSS/
```

Mỗi experiment nên có:

- `config.json`
- `train.log`
- `history.json`
- `test_results.txt`
- `classification_report.txt`
- `confusion_matrix.png`
- `model.keras`
- `labels.json`

Với full two-stage evaluation, nên có thêm:

- `router_manifest.json`
- `full_test_results.txt`
- `full_classification_report.txt`
- `full_confusion_matrix.png`
- `error_cases.csv`

## 4. Folder Logs

### `logs`

Vai trò: log dùng chung hoặc log chạy nhanh.

Quy tắc:

- Pipeline mới nên ưu tiên log theo từng experiment trong `experiments/`.
- `logs/` chỉ dùng cho preprocess, crop, split hoặc chạy debug.

Log quan trọng sắp tới:

- `logs/smart_crop.log`
- `logs/split.log`
- `logs/data_cleaning.log`

## 5. Folder Models

### `models`

Vai trò: model export chung/cũ.

Quy tắc:

- Với pipeline mới, model chính nên nằm trong từng experiment.
- Chỉ copy model sang `models/` nếu đã chốt bản deploy cuối.

## 6. Pipeline Folder Chuẩn

Luồng chính từ bây giờ:

```text
dataset/raw
-> dataset/processed_clean
-> dataset/processed_crop
-> dataset/train, dataset/val, dataset/test
-> dataset_fruit_only/train, dataset_fruit_only/val, dataset_fruit_only/test
-> experiments/stage_a_resnet50_crop_*
-> experiments/stage_b_resnet50_crop_*
-> experiments/two_stage_resnet50_crop_*
```

## 7. Checklist Trước Khi Train

Trước Stage A:

- `dataset/train`, `dataset/val`, `dataset/test` tồn tại.
- Mỗi split có 44 class folder.
- Có class `other`.
- Tên folder class phải khớp chính xác giữa train/val/test.
- Không có split rỗng.

Trước Stage B:

- `dataset_fruit_only/train`, `dataset_fruit_only/val`, `dataset_fruit_only/test` tồn tại.
- Mỗi split có 43 class folder.
- Không có class `other`.
- Class order phải nhất quán giữa train/val/test.

Trước full two-stage evaluation:

- Có model Stage A.
- Có labels Stage A.
- Có model Stage B.
- Có labels Stage B.
- Có router manifest trỏ đúng các file trên.

## 8. Audit Hiện Tại

Trạng thái kiểm tra trước cleanup:

- `dataset/raw`: 44 class folder.
- `dataset/train`: 44 class folder.
- `dataset/val`: 44 class folder.
- `dataset/test`: 44 class folder.
- `dataset_fruit_only/train`: 43 class folder, không có `other`.
- `dataset_fruit_only/val`: 43 class folder, không có `other`.
- `dataset_fruit_only/test`: 43 class folder, không có `other`.

Vấn đề cần xử lý trước khi train lại Stage A:

- `dataset/train` hiện không khớp tên class folder với `dataset/val` và `dataset/test`.
- Ví dụ `dataset/train` dùng tên như `apple_tao`, còn `dataset/val` và `dataset/test` dùng tên như `apple`.
- Vì dataloader yêu cầu train/val/test có cùng danh sách folder và cùng thứ tự, split Stage A hiện tại nên được sinh lại từ `dataset/processed_crop` sau khi smart crop hoàn tất.

Kết luận audit:

- Vai trò folder đã được chốt.
- `dataset_fruit_only` hiện đúng tinh thần Stage B.
- `dataset/train`, `dataset/val`, `dataset/test` cần được tái tạo lại sau bước smart crop để đảm bảo class folder đồng nhất.

Trạng thái sau cleanup ngày 2026-06-02:

- Đã split lại `dataset/train`, `dataset/val`, `dataset/test` từ `dataset/processed_crop`.
- Mỗi split hiện có 44 class folder và tên folder đã khớp nhau.
- Tổng ảnh official sau crop: 41,064.
- Train: 28,746 ảnh.
- Val: 6,167 ảnh.
- Test: 6,151 ảnh.
- Folder crop ngoài danh sách official `chico_hong_xiem_903` không được đưa vào split.
- Các nhánh cũ như pairwise/hard mining/old experiments/MobileNetV2 artifacts
  đã được đưa vào `archive_unused/resnet50_pipeline_cleanup_20260602_0100`.
