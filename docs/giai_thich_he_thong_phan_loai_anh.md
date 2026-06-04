# 1. OVERVIEW (BIG PICTURE)

Dự án này là một hệ thống phân loại ảnh nông sản. Mục tiêu là nhận một ảnh đầu vào và dự đoán ảnh đó thuộc loại nông sản nào, ví dụ: xoài, nhãn, bưởi, cà chua, dưa hấu, sầu riêng, hoặc nhóm `other` nếu ảnh không phải nhóm sản phẩm cần nhận diện.

Luồng sử dụng thực tế có thể hiểu như sau:

1. Người dùng upload một ảnh lên hệ thống.
2. Ảnh được tiền xử lý giống lúc train: mở ảnh an toàn, sửa EXIF orientation, chuyển RGB, resize/padding về đúng kích thước.
3. Model trả về xác suất cho từng class.
4. Hệ thống chọn class có xác suất cao nhất.
5. Nếu xác suất thấp hơn ngưỡng tin cậy, hệ thống trả về `unknown` hoặc `other` tùy pipeline inference.

Ví dụ: người dùng upload ảnh quả xoài. Model tính xác suất cho tất cả class, thấy `mango` có xác suất cao nhất là 0.87, cao hơn threshold 0.60, nên kết quả cuối cùng là `mango`.

Lưu ý quan trọng: mô tả ban đầu nói 45 classes, nhưng code hiện tại đang cấu hình `expected_num_classes = 44`, và các thư mục `dataset/train`, `dataset/val`, `dataset/test` hiện có 44 class folder, bao gồm cả `other`. Khi đọc hoặc train lại dự án, nên kiểm tra lại danh sách class thực tế để tránh lỗi mismatch số lớp.

# 2. DATA PIPELINE

## 2.1 Raw Data

Dữ liệu gốc nằm trong `dataset/raw`.

Cấu trúc cơ bản:

- `dataset/raw/ambarella`
- `dataset/raw/apple`
- `dataset/raw/banana`
- `dataset/raw/longan_c`
- `dataset/raw/mango_c`
- `dataset/raw/other`
- ...

Mỗi thư mục con tương ứng với một class. Tên thư mục raw có thể hơi khác nhau, ví dụ có dấu cách, có hậu tố `_c`, hoặc dùng tên tiếng Anh. Code có phần metadata để chuẩn hóa tên class sang format ổn định, ví dụ `bell pepper` thành `bell_pepper`, `passion fruit_c` thành `passion_fruit`.

Class `other` là nhóm đặc biệt. Nó chứa ảnh không thuộc nông sản cần nhận diện, ví dụ ảnh xe, chó mèo, cảnh trong nhà, đồ vật, ảnh từ COCO hoặc các nguồn ngoài. Class này giúp model học được rằng không phải ảnh nào upload lên cũng là nông sản.

## 2.2 Preprocess

Pipeline preprocess nằm chính trong `src/preprocess.py`, cấu hình mặc định ở `src/config.py`.

Preprocess làm các việc chính:

- Đọc ảnh từ `dataset/raw`.
- Chỉ nhận các extension hợp lệ như `.jpg`, `.jpeg`, `.png`.
- Mở ảnh an toàn, xử lý lỗi ảnh hỏng.
- Kiểm tra chất lượng ảnh: ảnh quá nhỏ, quá lớn, tỉ lệ quá lệch sẽ bị loại.
- Tính độ mờ bằng Laplacian variance; ảnh quá mờ sẽ bị loại.
- Dùng perceptual hash để phát hiện ảnh trùng lặp.
- Resize ảnh về kích thước chuẩn, thường là `224x224` cho MobileNetV2 hoặc có thể lớn hơn với ResNet50.
- Giữ tỉ lệ ảnh bằng padding màu đen thay vì kéo méo ảnh.
- Lưu ảnh đã xử lý vào `dataset/processed`.
- Có pipeline làm sạch thêm từ `dataset/processed` sang `dataset/processed_clean`.

Vì sao preprocess quan trọng:

- Model cần ảnh đầu vào có kích thước cố định.
- Ảnh hỏng hoặc ảnh quá mờ làm model học sai.
- Ảnh trùng lặp có thể làm kết quả đánh giá ảo, vì model có thể gặp gần như cùng ảnh ở train và test.
- Resize có padding giúp giữ hình dạng thật của trái cây, không làm méo đối tượng.

## 2.3 Split

Pipeline split nằm trong `src/splitter.py`.

Dataset được chia thành:

- `dataset/train`: dùng để model học.
- `dataset/val`: dùng để kiểm tra trong lúc train, chọn checkpoint tốt và early stopping.
- `dataset/test`: dùng để đánh giá cuối cùng sau khi train xong.

Tỉ lệ mặc định:

- Train: 70%
- Validation: 15%
- Test: 15%

Việc split được làm theo từng class. Nghĩa là mỗi class đều được chia riêng thành train/val/test, thay vì trộn toàn bộ ảnh rồi chia ngẫu nhiên. Cách này giúp mỗi split vẫn có đủ mẫu của từng class.

Split rất quan trọng để tránh data leakage. Nếu cùng một ảnh xuất hiện ở cả train và test, kết quả test sẽ không còn trung thực. Trong dataloader của ResNet50 còn có kiểm tra hash nội dung file giữa các split để phát hiện trùng lặp thật giữa train/val/test.

# 3. DATALOADER PIPELINE

Dataloader nằm trong `src/dataloader.py`.

Nhiệm vụ của dataloader là biến ảnh trên ổ đĩa thành `tf.data.Dataset` để đưa vào model.

Luồng chính:

1. Đọc class folder từ `dataset/train`, `dataset/val`, `dataset/test`.
2. Kiểm tra train/val/test có cùng danh sách class và cùng thứ tự.
3. Chuẩn hóa tên folder thành clean label, ví dụ `banana_chuoi_725` thành `banana`.
4. Gán label dạng số nguyên: class đầu tiên là 0, class tiếp theo là 1, ...
5. Load ảnh, resize/padding, convert sang tensor.
6. Với train set, áp dụng augmentation.
7. Áp dụng preprocessing đúng theo backbone.
8. Nếu train dùng focal loss, label được chuyển sang one-hot.
9. Batch và prefetch để train nhanh hơn.

Augmentation là kỹ thuật tạo biến thể ảnh trong lúc train. Dự án đang dùng các phép như:

- Lật ngang.
- Xoay nhẹ.
- Zoom nhẹ.

Augmentation giúp model không học thuộc ảnh quá cứng. Ví dụ quả xoài có thể nằm lệch, bị xoay nhẹ, hoặc chụp ở nhiều góc khác nhau. Nếu model chỉ thấy ảnh quá sạch và cố định, khi gặp ảnh upload thực tế nó dễ sai.

Preprocessing theo backbone:

- Với MobileNetV2: dùng `tf.keras.applications.mobilenet_v2.preprocess_input`.
- Với ResNet50: dùng `tf.keras.applications.resnet.preprocess_input`.

Điểm cần chú ý: preprocessing được đặt ở dataloader, không đặt trong model. Vì vậy khi inference cũng phải preprocess ảnh theo cùng contract, nếu không phân phối input sẽ lệch so với lúc train.

# 4. MODEL ARCHITECTURE

Dự án hỗ trợ ít nhất hai backbone:

- MobileNetV2 trong `src/model/mobilenetv2.py`.
- ResNet50 trong `src/model/resnet50.py`.

Theo yêu cầu hiện tại, trọng tâm là ResNet50.

Kiến trúc ResNet50 classifier:

1. Input ảnh, ví dụ `320x320x3`.
2. ResNet50 pretrained ImageNet, bỏ phần classifier gốc bằng `include_top=False`.
3. `GlobalAveragePooling2D` để biến feature map thành vector đặc trưng.
4. `BatchNormalization`.
5. `Dense(512, activation="relu")`.
6. `Dropout(0.5)`.
7. `Dense(num_classes, activation="softmax")`.

Softmax ở layer cuối trả về xác suất cho từng class. Tổng xác suất của tất cả class bằng 1.

Transfer learning được dùng vì dataset nông sản không đủ lớn để train ResNet50 từ đầu. ResNet50 pretrained trên ImageNet đã biết các đặc trưng cơ bản như cạnh, màu, texture, hình khối. Dự án chỉ cần dạy thêm phần phân biệt các class nông sản cụ thể.

# 5. TRAINING FLOW

Training chính nằm trong `src/train.py`.

## Stage 1: frozen backbone, training classifier

Ở Stage 1, backbone ResNet50 hoặc MobileNetV2 được freeze. Nghĩa là các layer pretrained không thay đổi trọng số.

Chỉ phần classifier head được train:

- Global pooling.
- Dense layer.
- Dropout.
- Dense softmax cuối.

Mục tiêu của Stage 1 là để classifier head học cách map feature ImageNet sang class nông sản của dự án. Cách này ổn định hơn vì không làm hỏng feature pretrained ngay từ đầu.

Với ResNet50, cấu hình trong code hiện tại:

- Image size mặc định: `320x320`.
- Stage 1 learning rate: `3e-4`.
- Stage 1 epochs: `15`.
- Early stopping theo `val_loss`.
- Lưu checkpoint tốt nhất.

Loss đang dùng là `CategoricalFocalCrossentropy` với label smoothing. Focal loss giúp model chú ý hơn vào các mẫu khó, thay vì chỉ tối ưu tốt cho các class dễ hoặc class nhiều ảnh.

## Stage 2: fine-tuning

Ở Stage 2, dự án mở một phần cuối của backbone để fine-tune.

Với ResNet50:

- Mở khoảng 50 layer cuối.
- Vẫn giữ BatchNorm frozen.
- Learning rate thấp hơn, mặc định `1e-5`.
- Dùng CosineDecay learning rate schedule.

Vì sao unfreezing giúp tốt hơn:

- Stage 1 chỉ học classifier head, nên backbone vẫn là feature ImageNet chung chung.
- Stage 2 cho phép các layer cuối điều chỉnh theo domain nông sản: màu vỏ, texture, hình dạng trái, đặc điểm lá/cuống, bề mặt quả.
- Learning rate nhỏ giúp fine-tune nhẹ nhàng, tránh làm mất kiến thức pretrained.

Sau Stage 1 và Stage 2, code so sánh checkpoint theo `val_loss` và chọn model tốt nhất để evaluate trên test set.

Dự án còn có hard example mining cho MobileNetV2 trong một số cấu hình. Ý tưởng là tìm các ảnh model dự đoán sai hoặc không tự tin, sau đó dùng chúng để fine-tune thêm. Với ResNet50 baseline hiện tại, hard mining đang được tắt để kết quả dễ phân tích.

# 6. METRICS (CRITICAL)

## Accuracy

Accuracy là tỉ lệ dự đoán đúng trên tổng số mẫu.

Ví dụ: test set có 1000 ảnh, model đoán đúng 800 ảnh, accuracy = 80%.

Accuracy dễ hiểu, nhưng không đủ trong bài toán này. Nếu class `other` hoặc một vài class mạnh chiếm nhiều ảnh, model có thể đạt accuracy khá cao nhưng vẫn bỏ sót nhiều class yếu.

## Precision

Precision trả lời câu hỏi: trong những ảnh model dự đoán là class X, bao nhiêu ảnh thật sự là class X?

Ví dụ với class `mango`:

- Model dự đoán 100 ảnh là `mango`.
- Trong đó chỉ 70 ảnh thật sự là mango.
- Precision của `mango` là 70%.

Precision thấp nghĩa là model hay gán nhầm ảnh class khác vào class này.

## Recall

Recall trả lời câu hỏi: trong tất cả ảnh thật sự thuộc class X, model tìm đúng được bao nhiêu?

Ví dụ với class `mango`:

- Test set có 100 ảnh mango thật.
- Model chỉ nhận ra đúng 40 ảnh.
- Recall của `mango` là 40%.

Recall thấp nghĩa là model hay bỏ sót class đó.

## F1-score

F1-score là điểm cân bằng giữa precision và recall.

F1 cao khi cả precision và recall đều tốt. Nếu một trong hai thấp, F1 cũng thấp.

F1 rất quan trọng trong dự án này vì nhiều class có thể bị mất cân bằng. Một class có precision cao nhưng recall rất thấp vẫn là class yếu. Ví dụ model chỉ dám dự đoán `lime` khi cực kỳ chắc, nên precision có thể ổn, nhưng nếu nó bỏ sót gần hết ảnh lime thì recall và F1 vẫn thấp.

## Confusion matrix

Confusion matrix là bảng cho biết model nhầm class nào sang class nào.

Cách đọc:

- Trục dọc là nhãn thật.
- Trục ngang là nhãn model dự đoán.
- Ô trên đường chéo là dự đoán đúng.
- Ô ngoài đường chéo là dự đoán sai.

Ví dụ nếu hàng `lime`, cột `pomelo` có giá trị cao, nghĩa là nhiều ảnh chanh bị dự đoán thành bưởi. Đây là tín hiệu để thu thập thêm dữ liệu hoặc tạo classifier phụ cho cặp dễ nhầm.

## Vì sao accuracy không đủ

Accuracy không cho biết class nào đang bị bỏ sót.

Trong kết quả ResNet50 hiện có, accuracy khoảng 0.5917, nhưng macro recall chỉ khoảng 0.5433 và macro F1 khoảng 0.5614. Điều này nghĩa là nhìn tổng thể model đoán đúng một phần đáng kể, nhưng hiệu năng giữa các class không đều.

Một số class có recall rất thấp, ví dụ trong report hiện có:

- `lime` recall khoảng 0.0114.
- `coffee` recall khoảng 0.0488.
- `black_mulberry` recall khoảng 0.0526.
- `longan` recall khoảng 0.0682.
- `burmese_grape` recall khoảng 0.1358.

Trong khi đó `other` có recall 1.0000, nghĩa là tất cả ảnh `other` trong test đều được bắt đúng, nhưng điều này cũng có thể cho thấy class `other` đang quá mạnh hoặc model quá dễ nghiêng về `other`.

# 7. LOGGING SYSTEM

Dự án lưu nhiều artifact để debug và tái lập kết quả.

Các file log/artifact chính:

- `logs/preprocess.log`: log quá trình preprocess raw data.
- `logs/data_clean.log`: log quá trình làm sạch processed data.
- `logs/data_cleaning.log`: log strict cleaning khi dataloader chạy, đặc biệt với ResNet50.
- `logs/train.log`: log training nếu dùng cấu hình cũ.
- `logs/history.json`: lịch sử loss/accuracy theo epoch.
- `logs/test_results.txt`: kết quả test tổng hợp.
- `logs/classification_report.txt`: precision, recall, F1 của từng class.
- `logs/confusion_matrix.png`: ảnh confusion matrix.

Với pipeline training mới, mỗi lần train tạo một experiment riêng trong `experiments/exp_<timestamp>_<model_type>/`, ví dụ:

- `experiments/exp_20260426_094507_resnet50/train.log`
- `experiments/exp_20260426_094507_resnet50/config.json`
- `experiments/exp_20260426_094507_resnet50/history.json`
- `experiments/exp_20260426_094507_resnet50/test_results.txt`
- `experiments/exp_20260426_094507_resnet50/classification_report.txt`
- `experiments/exp_20260426_094507_resnet50/confusion_matrix.png`
- `experiments/exp_20260426_094507_resnet50/model.keras`
- `experiments/exp_20260426_094507_resnet50/labels.json`

Logs quan trọng vì:

- Biết model train có bị overfit không.
- Biết stage nào tốt hơn.
- Biết class nào yếu.
- Biết ảnh nào bị loại do mờ, lỗi, duplicate.
- Có thể tái lập experiment dựa trên `config.json`.
- Backend inference dùng `labels.json` để giữ đúng thứ tự class.

# 8. CURRENT PROBLEMS

Dựa trên kết quả hiện có và cấu trúc dữ liệu, các vấn đề chính là:

1. Class imbalance.

Class `other` có nhiều mẫu hơn nhiều class khác. Trong processed data, `other` có hơn 4000 ảnh, trong khi nhiều class nông sản chỉ khoảng 700-1000 ảnh. Nếu không xử lý kỹ, model sẽ học rất mạnh nhóm `other` và gây lệch kết quả.

2. Một số class có recall thấp.

Recall thấp nghĩa là ảnh thật của class đó thường bị dự đoán sang class khác. Các class như `lime`, `coffee`, `black_mulberry`, `longan`, `burmese_grape` đang là nhóm cần chú ý.

3. Model nhầm giữa các class nhìn giống nhau.

Một số nông sản có hình dạng hoặc màu sắc giống nhau:

- `lime` và `pomelo`.
- `longan` và `burmese_grape`.
- `black_mulberry` và `red_mulberry`.
- `apple`, `otaheite_apple`, một số quả tròn màu đỏ/xanh.
- `sapodilla`, `canistel`, `mango` trong một số góc chụp.

4. Accuracy tổng thể che giấu lỗi theo class.

Nếu chỉ nhìn accuracy, developer có thể nghĩ model tạm ổn. Nhưng classification report cho thấy nhiều class gần như chưa được nhận diện tốt.

5. Threshold inference có thể làm nhiều ảnh thành `unknown`.

Trong kết quả test hiện có, threshold deploy là 0.60 và có 1443 dự đoán low-confidence trên 4149 ảnh. Điều này cho thấy model chưa đủ tự tin với khá nhiều mẫu.

# 9. INFERENCE FLOW (DEPLOYMENT)

Inference một stage nằm trong `src/evaluate.py` và các hàm tiện ích prediction.

Luồng cơ bản:

1. Load model `.keras`.
2. Load `labels.json` để lấy đúng `class_names`.
3. Đọc ảnh upload.
4. Sửa EXIF orientation, convert RGB.
5. Kiểm tra chất lượng ảnh nếu bật strict mode.
6. Resize/padding về đúng `image_size`.
7. Áp dụng preprocessing đúng backbone.
8. Model trả về softmax probabilities.
9. Lấy class có xác suất cao nhất.
10. So sánh với confidence threshold.

Logic threshold:

- Nếu xác suất cao nhất >= threshold: trả về class đó.
- Nếu xác suất cao nhất < threshold: trả về `unknown`.

Ví dụ:

- `mango`: 0.87
- `papaya`: 0.05
- `banana`: 0.03
- các class khác thấp hơn

Threshold = 0.60, vậy kết quả là `mango`.

Ví dụ khác:

- `mango`: 0.31
- `papaya`: 0.24
- `sapodilla`: 0.19
- các class khác thấp hơn

Threshold = 0.60, model không đủ tự tin, kết quả là `unknown`.

Dự án cũng có pipeline two-stage trong `src/two_stage_inference.py`:

- Stage A: phân biệt `fruit` và `other`.
- Nếu xác suất fruit thấp hơn threshold, trả về `other`.
- Nếu qua Stage A, Stage B phân loại fruit cụ thể.

Two-stage hữu ích khi class `other` quá mạnh hoặc khi muốn tách bài toán "có phải nông sản không" khỏi bài toán "nông sản này là loại nào".

# 10. FINAL SUMMARY

Hệ thống hiện đã có pipeline khá đầy đủ: preprocess ảnh, split train/val/test, dataloader có augmentation và preprocessing theo backbone, model transfer learning, training 2 stage, evaluation bằng classification report và confusion matrix, logging theo từng experiment, và inference có threshold.

Phần đang hoạt động tốt:

- Cấu trúc pipeline rõ ràng.
- Có kiểm tra chất lượng ảnh và duplicate.
- Có split riêng train/val/test.
- Có transfer learning với ResNet50/MobileNetV2.
- Có lưu model, labels, config, history và metric.
- Có threshold để tránh trả kết quả khi model không tự tin.

Phần chưa tốt:

- Số class trong mô tả và code đang lệch: mô tả nói 45, code/dataset hiện là 44.
- Một số class có recall rất thấp.
- `other` đang rất mạnh.
- Một số class tương tự nhau bị nhầm nhiều.
- Accuracy chưa phản ánh đúng chất lượng từng class.

Nên cải thiện tiếp theo:

- Xác nhận lại danh sách class chính thức: 44 hay 45.
- Cân bằng lại dữ liệu, đặc biệt class `other` và các class yếu.
- Thu thập thêm ảnh thật cho các class recall thấp.
- Phân tích confusion matrix để tìm cặp class hay nhầm.
- Dùng hard example mining hoặc pairwise classifier cho các cặp khó.
- Tune threshold theo validation set thay vì dùng cố định 0.60.
- Theo dõi macro F1 và per-class recall như metric chính, không chỉ accuracy.
