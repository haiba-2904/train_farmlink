# Giải Thích Hệ Thống Phân Loại Ảnh Nông Sản Farmlink

Tài liệu này mô tả đầy đủ project `train_farmlink`: mục tiêu, dữ liệu, flow xử lý, công nghệ sử dụng, cách train, cách inference và các artifact quan trọng. Mục tiêu là để một dev mới, người review đồ án, hoặc người vận hành hệ thống có thể đọc một lần và hiểu project đang làm gì, chạy như thế nào, và vì sao pipeline được thiết kế như hiện tại.

## 1. Tổng Quan Project

`train_farmlink` là project machine learning dùng để huấn luyện và kiểm thử hệ thống phân loại ảnh nông sản. Đầu vào là một ảnh do người dùng upload. Đầu ra là nhãn dự đoán, ví dụ `mango`, `banana`, `durian`, `tomato`, hoặc `other` nếu ảnh không phải nông sản trong phạm vi nhận diện.

Project hiện ưu tiên pipeline ResNet50 two-stage:

1. Stage A kiểm tra ảnh có phải nông sản hay không.
2. Nếu Stage A kết luận không phải nông sản, hệ thống trả về `other`.
3. Nếu Stage A kết luận là nông sản, ảnh được đưa sang Stage B.
4. Stage B phân loại ảnh vào một trong các class nông sản cụ thể.

Thiết kế two-stage giúp hệ thống phù hợp hơn với upload thực tế. Người dùng có thể upload ảnh bất kỳ, không chỉ ảnh trái cây hoặc rau củ. Nếu chỉ dùng một model multi-class có class `other`, model dễ bị nhiễu vì `other` là một nhóm rất rộng. Tách Stage A thành cổng kiểm tra `fruit` vs `other` giúp Stage B tập trung vào phân biệt các nông sản cụ thể.

## 2. Bài Toán Cần Giải Quyết

Trong thực tế, hệ thống cần xử lý hai câu hỏi khác nhau:

1. Ảnh này có thuộc miền nông sản cần nhận diện không?
2. Nếu có, ảnh này là loại nông sản nào?

Stage A trả lời câu hỏi thứ nhất. Stage B trả lời câu hỏi thứ hai.

Ví dụ flow khi người dùng upload ảnh xoài:

```text
upload image
-> open image safely
-> fix EXIF orientation
-> convert RGB
-> resize + padding về 320x320
-> ResNet50 preprocess_input
-> Stage A: fruit_probability = 0.94
-> fruit_probability >= threshold
-> Stage B: mango = 0.87, papaya = 0.05, ...
-> result: mango
```

Ví dụ flow khi người dùng upload ảnh không liên quan:

```text
upload image
-> preprocess giống train
-> Stage A: fruit_probability = 0.12
-> fruit_probability < threshold
-> result: other
```

## 3. Phạm Vi Class

Project đang chốt dữ liệu theo taxonomy mới:

- Stage A dùng 44 class folder vật lý trong `dataset/train`, `dataset/val`, `dataset/test`.
- Trong Stage A, `other` được map thành label `0`, toàn bộ class còn lại được map thành label `1` tức `fruit`.
- Stage B dùng dataset riêng `dataset_fruit_only`, chỉ chứa class nông sản, không chứa `other`.
- Stage B hiện kỳ vọng 43 class nông sản ở các script cũ, nhưng taxonomy mới có thể gộp một số nhóm thành class đầu ra chung. Khi train lại, luôn kiểm tra class count thật trong `labels.json` hoặc label manifest của experiment.

Các rule gộp class quan trọng nằm trong `src/taxonomy.py`:

```text
black_mulberry, red_mulberry, mulberry, mullberry -> mulberry
cempedak, jackfruit, jackfruit_cempedak -> jackfruit_cempedak
bitter_gourd, ridged_gourd, gourd -> gourd
```

Lý do gộp class:

- Một số class có hình ảnh rất giống nhau hoặc tên raw bị lẫn giữa các nguồn.
- Gộp class giúp giảm nhiễu nhãn và làm bài toán nhất quán hơn.
- Taxonomy được áp dụng trong bước rebuild/preprocess để output class ổn định giữa các lần chạy.

## 4. Cấu Trúc Thư Mục

Các thư mục chính của project:

```text
train_farmlink/
├── src/                         # Source code train, preprocess, evaluate, inference
├── docs/                        # Tài liệu kỹ thuật
├── requirements-macos.txt       # Dependency cho macOS
├── setup_macos.sh               # Script setup môi trường
├── dataset/                     # Dataset gốc và split Stage A, không push git
├── dataset_fruit_only/          # Dataset Stage B, không push git
├── experiments/                 # Kết quả train/evaluate, không push git
├── logs/                        # Log preprocess/split/debug, không push git
└── models/                      # Model export cuối nếu có, không push git
```

Các folder dữ liệu và model rất nặng nên đã được đưa vào `.gitignore`. GitHub repo chỉ nên chứa source code, docs, config và script setup. Dataset, model `.keras`, log và experiment artifact nên lưu bằng release artifact, cloud storage, hoặc Git LFS nếu thật sự cần version hóa.

## 5. Dataset Contract

### 5.1 `dataset/raw`

`dataset/raw` là dữ liệu gốc ban đầu. Mỗi folder con là một class hoặc một nhóm raw class.

Quy tắc:

- Không chỉnh sửa thủ công nếu có thể tránh.
- Không resize/crop trực tiếp trong raw.
- Không đổi tên lẻ tẻ sau khi đã train, vì sẽ làm khó tái lập kết quả.
- Đây là nguồn để rebuild toàn bộ dataset.

Ví dụ:

```text
dataset/raw/
├── mango/
├── banana/
├── dragonfruit/
├── tomato/
└── other/
```

### 5.2 `dataset/processed_clean`

Đây là output của bước preprocess/rebuild từ raw. Ảnh đã được:

- mở an toàn,
- sửa hướng EXIF,
- convert RGB,
- kiểm tra chất lượng,
- loại ảnh hỏng/quá nhỏ/quá lớn/tỉ lệ quá lệch,
- lọc ảnh quá mờ,
- lọc trùng bằng perceptual hash,
- resize giữ tỉ lệ,
- padding về kích thước chuẩn.

Folder này là dataset sạch trước khi crop.

### 5.3 `dataset/processed_crop`

Đây là output của bước smart crop từ `processed_clean`. Mục tiêu là giảm nền thừa quanh vật thể nhưng vẫn an toàn:

- Nếu heuristic crop đủ tự tin, ảnh được crop rồi resize/pad lại.
- Nếu crop không chắc, ảnh gốc đã chuẩn hóa được giữ lại.
- Không ghi đè `processed_clean`.

### 5.4 `dataset/train`, `dataset/val`, `dataset/test`

Đây là split chính cho Stage A.

Quy tắc:

- Có class `other`.
- Có cùng danh sách class folder giữa train/val/test.
- Split theo từng class để mỗi class đều có mẫu ở train/val/test.
- Stage A không dùng trực tiếp nhãn multi-class; nó map `other` thành `0`, class còn lại thành `1`.

### 5.5 `dataset_fruit_only/train`, `dataset_fruit_only/val`, `dataset_fruit_only/test`

Đây là split chính cho Stage B.

Quy tắc:

- Không có class `other`.
- Chỉ chứa các class nông sản cụ thể.
- Class order phải nhất quán giữa train/val/test.
- Đây là dữ liệu để train multi-class classifier.

## 6. Flow Tổng Thể Từ Raw Data Đến Model

Flow chuẩn của project:

```text
dataset/raw
-> rebuild/preprocess
-> dataset/processed_clean
-> smart crop
-> dataset/processed_crop
-> split train/val/test
-> dataset/train, dataset/val, dataset/test
-> build fruit-only dataset
-> dataset_fruit_only/train, val, test
-> train Stage A
-> train Stage B
-> evaluate two-stage
-> inference bằng router_manifest.json
```

Mỗi bước đều có output rõ ràng để dễ audit. Không nên train trực tiếp từ `dataset/raw` vì raw có thể chứa ảnh lỗi, ảnh trùng, ảnh lệch kích thước hoặc tên class chưa chuẩn hóa.

## 7. Preprocess Và Rebuild Dataset

Code liên quan:

- `src/preprocess.py`: pipeline preprocess cũ/tổng quát.
- `src/rebuild_dataset.py`: orchestration rebuild dataset theo pipeline mới.
- `src/taxonomy.py`: chuẩn hóa class name và merge taxonomy.
- `src/utils.py`: helper mở ảnh, resize/pad, validate chất lượng, hash ảnh.

Các kiểm tra chính:

- Extension hợp lệ: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp` tùy script.
- Ảnh phải mở được bằng PIL.
- Ảnh được fix EXIF orientation để tránh xoay sai.
- Ảnh được convert sang RGB.
- Cạnh nhỏ nhất không được quá nhỏ, mặc định tối thiểu 100 px.
- Cạnh lớn nhất không được quá lớn, mặc định tối đa 5000 px.
- Tỉ lệ ảnh không được quá lệch, mặc định max aspect ratio là 4.0.
- Blur score được tính bằng Laplacian variance, mặc định threshold là 100.0.
- Perceptual hash dùng để loại ảnh trùng hoặc gần trùng theo ngưỡng cấu hình.

Resize/padding:

- Ảnh không bị kéo méo.
- Ảnh được resize giữ tỉ lệ.
- Phần dư được padding màu đen.
- Với ResNet50, kích thước chuẩn hiện dùng là `320x320`.

Vì sao không kéo ảnh về thẳng `320x320` bằng stretch:

- Trái cây/rau củ có hình dạng là tín hiệu quan trọng.
- Stretch làm quả tròn thành bầu dục hoặc làm vật thể dài/ngắn bất thường.
- Padding giữ hình học gốc tốt hơn.

## 8. Smart Crop

Code chính: `src/smart_crop.py`.

Smart crop là bước offline để giảm nền thừa. Script không dùng object detector mà dùng heuristic an toàn dựa trên:

- khác biệt màu so với nền viền ảnh,
- saturation,
- edge bằng Canny,
- morphology để làm sạch mask,
- area ratio của bounding box,
- texture score,
- color contrast,
- edge density.

Nguyên tắc thiết kế:

- Crop sai nguy hiểm hơn không crop.
- Nếu object detection heuristic không đủ chắc, giữ ảnh gốc.
- Luôn ghi log lý do crop hoặc fallback.

Output:

```text
dataset/processed_crop/
```

Log:

```text
logs/smart_crop.log
```

## 9. Split Dataset

Code chính: `src/splitter.py` và một số script hỗ trợ như `src/split_after_crop.py`.

Dataset được chia thành:

- `train`: để học trọng số.
- `val`: để chọn checkpoint, early stopping và tune threshold.
- `test`: để đánh giá cuối cùng.

Tỉ lệ mặc định:

```text
train = 70%
val   = 15%
test  = 15%
seed  = 42
```

Split theo từng class, không split ngẫu nhiên toàn bộ ảnh một lần. Cách này giúp mỗi split vẫn có phân phối class ổn định hơn.

Yêu cầu bắt buộc:

- `dataset/train`, `dataset/val`, `dataset/test` phải có cùng class folder.
- Không split rỗng.
- Không để cùng một ảnh hoặc ảnh trùng nội dung xuất hiện ở nhiều split.

Dataloader ResNet50 còn có cơ chế strict validation và kiểm tra overlap bằng content hash giữa các split để giảm nguy cơ data leakage.

## 10. Dataloader

Code chính: `src/dataloader.py`.

Dataloader có nhiệm vụ biến file ảnh trên ổ đĩa thành `tf.data.Dataset`.

Luồng xử lý:

1. Đọc folder `train`, `val`, `test`.
2. Liệt kê class folder theo thứ tự ổn định.
3. Chuẩn hóa tên label bằng `extract_clean_label`.
4. Kiểm tra train/val/test có class list nhất quán.
5. Scan ảnh hợp lệ theo extension.
6. Kiểm tra chất lượng ảnh khi strict validation bật.
7. Map label folder thành integer label hoặc one-hot label.
8. Decode ảnh bằng TensorFlow.
9. Resize/pad về `image_size`.
10. Augment train set.
11. Áp dụng preprocessing theo backbone.
12. Batch và prefetch.

ResNet50 mặc định dùng:

```text
image_size = 320x320
preprocess = tf.keras.applications.resnet.preprocess_input
```

MobileNetV2 vẫn còn code hỗ trợ legacy:

```text
image_size = 224x224
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
```

Pipeline chính hiện tại dùng ResNet50.

## 11. Kiến Trúc Model

### 11.1 Backbone

Backbone chính: `tf.keras.applications.ResNet50`.

Cấu hình:

```text
weights     = "imagenet"
include_top = False
input_shape = (320, 320, 3)
```

Project dùng transfer learning vì dataset nông sản không đủ lớn để train ResNet50 từ đầu. ResNet50 pretrained trên ImageNet đã học được các feature nền tảng như cạnh, texture, màu sắc, hình khối. Project chỉ fine-tune để backbone phù hợp hơn với domain nông sản.

### 11.2 Stage A: Binary Classifier

Code chính: `src/model/binary.py`, train bằng `src/train_stage_a.py`.

Nhiệm vụ:

```text
other vs fruit
```

Label:

```text
0 = other
1 = fruit
```

Kiến trúc head:

```text
ResNet50 backbone
-> GlobalAveragePooling2D
-> BatchNormalization
-> Dense(head_units=256, relu)
-> Dropout(0.5)
-> Dense(1, sigmoid)
```

Output sigmoid là `fruit_probability`. Nếu xác suất này thấp hơn threshold, router trả về `other`.

Loss/metric:

- BinaryCrossentropy.
- Accuracy.
- AUC.
- Precision.
- Recall.

Stage A có class weight vì `fruit` là tổng của nhiều class nên thường nhiều mẫu hơn `other`.

### 11.3 Stage B: Multi-class Fruit Classifier

Code chính: `src/model/resnet50.py`, train bằng `src/train_stage_b.py`.

Nhiệm vụ:

```text
fruit image -> one specific fruit/agricultural class
```

Stage B không được chứa `other`.

Kiến trúc head:

```text
ResNet50 backbone
-> GlobalAveragePooling2D
-> BatchNormalization
-> Dense(head_units=512, relu)
-> Dropout(0.5)
-> Dense(num_classes, softmax)
```

Output softmax là xác suất cho từng class. Tổng xác suất bằng 1.

Metric chính:

- Accuracy.
- Classification report theo từng class.
- Macro precision/recall/F1.
- Weighted precision/recall/F1.
- Confusion matrix.
- Weak class report theo recall threshold.

## 12. Training Flow

Project train theo 2 phase trong mỗi stage:

### 12.1 Phase 1: Train Head

Backbone ResNet50 được freeze. Chỉ classifier head được train.

Mục tiêu:

- Học mapping từ feature ImageNet sang nhãn của project.
- Tránh làm hỏng pretrained weights ngay từ đầu.
- Tạo baseline ổn định trước khi fine-tune.

Learning rate mặc định:

```text
3e-4
```

### 12.2 Phase 2: Fine-tune Backbone

Một số layer cuối của ResNet50 được mở train.

Cấu hình thường dùng:

```text
fine_tune_last_layers = 50
learning_rate         = 1e-5
BatchNorm             = frozen
```

BatchNormalization được giữ frozen vì batch size local thường nhỏ. Nếu train lại BatchNorm, moving mean/variance dễ lệch và làm model kém ổn định.

Stage 2 dùng learning rate thấp và có thể dùng CosineDecay để fine-tune nhẹ hơn.

## 13. Script Chính Và Vai Trò

Các script quan trọng:

```text
src/rebuild_dataset.py
```

Rebuild dataset từ raw, áp dụng taxonomy, preprocess, ghi report.

```text
src/smart_crop.py
```

Tạo `dataset/processed_crop` từ `dataset/processed_clean`.

```text
src/splitter.py
src/split_after_crop.py
```

Chia dataset thành train/val/test.

```text
src/build_fruit_only_dataset.py
src/create_fruit_only_dataset.py
```

Tạo dataset Stage B bằng cách loại `other`.

```text
src/train_stage_a.py
```

Train riêng Stage A: `other` vs `fruit`.

```text
src/train_stage_b.py
```

Train riêng Stage B: phân loại class nông sản.

```text
src/train_two_stage.py
```

Train hoặc orchestration cho hệ thống two-stage.

```text
src/evaluate_upload_flow.py
src/evaluate_supported_v1_uploads.py
src/test_upload_flow_production.py
```

Đánh giá flow upload/inference trên dữ liệu test hoặc bộ upload đã gán nhãn.

```text
src/two_stage_inference.py
```

CLI và function inference two-stage dùng `router_manifest.json`.

## 14. Cách Chạy Tham Khảo

Tạo môi trường macOS:

```bash
./setup_macos.sh
```

Train Stage A:

```bash
.venv/bin/python src/train_stage_a.py \
  --dataset-root dataset \
  --experiment-root experiments \
  --image-size 320 \
  --batch-size 32
```

Train Stage B:

```bash
.venv/bin/python src/train_stage_b.py \
  --dataset-root dataset_fruit_only \
  --experiment-root experiments \
  --image-size 320 \
  --batch-size 16
```

Train orchestration two-stage:

```bash
.venv/bin/python src/train_two_stage.py \
  --stage full \
  --dataset-root dataset \
  --stage-b-dataset-root dataset_fruit_only \
  --experiment-root experiments \
  --image-size 320 \
  --batch-size 32
```

Inference một ảnh bằng router manifest:

```bash
.venv/bin/python src/two_stage_inference.py \
  --image path/to/upload.jpg \
  --router experiments/two_stage_xxx/router_manifest.json \
  --top-k 5
```

Nếu muốn bắt lỗi ảnh chất lượng thấp thay vì chỉ warning:

```bash
.venv/bin/python src/two_stage_inference.py \
  --image path/to/upload.jpg \
  --router experiments/two_stage_xxx/router_manifest.json \
  --strict-quality-check
```

## 15. Inference Router

Code chính: `src/two_stage_inference.py`.

Router cần một file:

```text
router_manifest.json
```

Manifest này cho biết:

- path tới model Stage A,
- path tới label manifest Stage A,
- path tới model Stage B,
- path tới label manifest Stage B,
- image size,
- model type,
- preprocess contract,
- `fruit_threshold`,
- `stage_b_confidence_threshold`.

Logic inference:

1. Load router manifest.
2. Resolve path model/label theo experiment directory.
3. Load model Stage A và Stage B.
4. Mở ảnh upload an toàn.
5. Validate chất lượng cơ bản.
6. Resize/pad về kích thước model.
7. Áp dụng `tf.keras.applications.resnet.preprocess_input`.
8. Chạy Stage A để lấy `fruit_probability`.
9. Nếu `fruit_probability < fruit_threshold`, trả `other`.
10. Nếu qua gate, chạy Stage B.
11. Lấy top-k class theo softmax.
12. Nếu confidence Stage B thấp hơn threshold, đánh dấu `low_confidence = true`.

Output ví dụ:

```json
{
  "predicted_label": "mango",
  "confidence": 0.87,
  "route": "stage_b_fruit",
  "low_confidence": false,
  "stage_a": {
    "fruit_probability": 0.94,
    "fruit_threshold": 0.5
  },
  "stage_b": {
    "confidence_threshold": 0.6,
    "top_k_predictions": [
      {
        "class_index": 12,
        "label": "mango",
        "confidence": 0.87
      }
    ]
  },
  "warnings": [],
  "image_path": "path/to/upload.jpg"
}
```

## 16. Artifact Trong Experiment

Mỗi lần train nên tạo folder riêng trong `experiments/`.

Ví dụ:

```text
experiments/stage_a_resnet50_YYYYMMDD_HHMMSS/
experiments/stage_b_resnet50_YYYYMMDD_HHMMSS/
experiments/two_stage_YYYYMMDD_HHMMSS_resnet50/
```

Artifact thường có:

```text
config.json
train.log
history.json
model.keras
labels.json hoặc label_manifest.json
classification_report.txt
confusion_matrix.npy hoặc confusion_matrix.png
test_results.txt
weak_classes.json
confused_pairs.json
router_manifest.json
error_cases.csv
```

Ý nghĩa:

- `config.json`: biết model được train với tham số nào.
- `train.log`: audit quá trình train theo epoch.
- `history.json`: loss/metric từng epoch để vẽ biểu đồ.
- `model.keras`: model đã lưu.
- `labels.json`: class order bắt buộc khi inference.
- `classification_report.txt`: precision/recall/F1 theo class.
- `confusion_matrix`: xem model nhầm class nào sang class nào.
- `weak_classes.json`: danh sách class có recall thấp hơn threshold.
- `router_manifest.json`: contract để backend/inference load đúng model và preprocess.

## 17. Metric Và Cách Đọc Kết Quả

### Accuracy

Accuracy là tỷ lệ dự đoán đúng trên tổng số mẫu. Metric này dễ hiểu nhưng không đủ trong bài toán nhiều class và mất cân bằng dữ liệu.

### Precision

Precision cho biết trong tất cả ảnh model dự đoán là class X, bao nhiêu ảnh thật sự là X.

Precision thấp nghĩa là model hay gán nhầm ảnh class khác vào class X.

### Recall

Recall cho biết trong tất cả ảnh thật sự thuộc class X, model bắt đúng được bao nhiêu.

Recall thấp nghĩa là model hay bỏ sót class X.

### F1-score

F1-score cân bằng precision và recall. Nếu một trong hai thấp, F1 cũng thấp.

### Macro Average

Macro average tính trung bình đều giữa các class. Mỗi class có trọng số như nhau, kể cả class ít ảnh. Đây là metric quan trọng để biết model có đang bỏ rơi class yếu không.

### Weighted Average

Weighted average tính trung bình theo số mẫu từng class. Class nhiều ảnh ảnh hưởng nhiều hơn. Metric này phản ánh performance tổng thể theo phân phối test set.

### Confusion Matrix

Confusion matrix cho biết class thật bị dự đoán nhầm sang class nào.

Cách đọc:

- Hàng là nhãn thật.
- Cột là nhãn dự đoán.
- Đường chéo là dự đoán đúng.
- Ô ngoài đường chéo là lỗi.

Nếu hàng `lime`, cột `pomelo` cao, model đang nhầm nhiều ảnh chanh thành bưởi. Đây là tín hiệu để kiểm tra dữ liệu, bổ sung ảnh, hoặc cân nhắc gộp/tách taxonomy.

## 18. Các Công Nghệ Sử Dụng

Ngôn ngữ:

- Python.

Deep learning:

- TensorFlow.
- Keras.
- ResNet50 pretrained ImageNet.
- MobileNetV2 còn trong code cho legacy/baseline, không phải pipeline chính hiện tại.

Xử lý ảnh:

- Pillow/PIL để mở ảnh, fix EXIF, convert RGB, lưu ảnh.
- OpenCV để xử lý edge/mask trong smart crop.
- NumPy để xử lý array.

Machine learning utility:

- scikit-learn cho classification report, confusion matrix, class weight.

Visualization/logging:

- Matplotlib.
- Seaborn.
- Python logging.

Hạ tầng dữ liệu:

- `tf.data.Dataset`.
- Batch, shuffle, prefetch.
- Deterministic seed để dễ tái lập.

Môi trường:

- macOS.
- Python virtual environment `.venv`.
- TensorFlow macOS/Metal tùy setup máy.

## 19. Những Nguyên Tắc Quan Trọng Khi Dev Tiếp

1. Không commit dataset/model/log nặng vào git.
2. Không sửa trực tiếp `dataset/raw` nếu không có lý do rõ ràng.
3. Khi đổi taxonomy, phải rebuild dataset và train lại.
4. Khi train Stage B, đảm bảo `dataset_fruit_only` không có `other`.
5. Khi inference, dùng đúng `labels.json` hoặc label manifest của model đã train.
6. Không thay đổi preprocessing giữa train và inference.
7. Không dùng accuracy một mình để kết luận model tốt.
8. Luôn xem confusion matrix và weak classes.
9. Nếu class folder train/val/test không khớp, phải split lại.
10. Nếu model chạy backend, dùng `router_manifest.json` làm nguồn cấu hình duy nhất.

## 20. Checklist Trước Khi Train Lại

Trước preprocess:

- `dataset/raw` tồn tại.
- Mỗi class có ảnh hợp lệ.
- Không có folder class bị đặt tên nhầm ngoài taxonomy.

Trước split:

- `dataset/processed_clean` hoặc `dataset/processed_crop` đã tạo xong.
- Log preprocess không có lỗi bất thường.
- Class count đúng với kỳ vọng.

Trước Stage A:

- `dataset/train`, `dataset/val`, `dataset/test` tồn tại.
- Có class `other`.
- Train/val/test có cùng class folder.
- Không có split rỗng.

Trước Stage B:

- `dataset_fruit_only/train`, `dataset_fruit_only/val`, `dataset_fruit_only/test` tồn tại.
- Không có class `other`.
- Class count đúng với taxonomy hiện tại.
- Class order nhất quán.

Trước deploy/inference:

- Có model Stage A.
- Có label manifest Stage A.
- Có model Stage B.
- Có label manifest Stage B.
- Có `router_manifest.json`.
- Test thử `src/two_stage_inference.py` với ảnh fruit và ảnh non-fruit.

## 21. Tóm Tắt Ngắn Gọn Cho Người Mới

Project này train hệ thống nhận diện ảnh nông sản bằng ResNet50. Hệ thống không dùng một model duy nhất mà dùng hai tầng:

- Stage A: ảnh có phải nông sản không?
- Stage B: nếu là nông sản thì là loại nào?

Dữ liệu đi từ raw -> clean -> crop -> split -> fruit-only. Train tạo ra model và manifest trong `experiments/`. Backend hoặc CLI inference dùng `router_manifest.json` để đảm bảo model, label order, image size và preprocessing khớp với lúc train.

Điểm quan trọng nhất khi làm tiếp project là giữ đúng contract giữa dữ liệu, model và inference. Nếu đổi class, đổi preprocessing, đổi image size hoặc đổi label order mà không cập nhật manifest/train lại, kết quả inference sẽ sai hoặc rất khó debug.
