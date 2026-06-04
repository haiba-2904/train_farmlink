# README ResNet50 Training

## 1. Thu tu tuning khuyen nghi

Danh sach duoi day la thu tu toi uu/thuc nghiem, khong phai thu tu runtime noi bo:

1. Label smoothing
2. Data augmentation
3. Stage 1 tuning
4. Stage 2 fine-tuning
5. Cosine learning rate
6. Head tuning
7. Image size
8. Class weight
9. Hard example mining

Runtime train thuc te se chay theo flow o muc 3.

## 2. Cach chay day du tu data den train

### Buoc 0: Chuan bi data raw

Dat anh goc theo cau truc:

```text
dataset/raw/
├── banana/
├── mango/
├── sapodilla/
└── ...
```

Luu y dataset moi la 44 class, class `chico` da gop vao `sapodilla`.

### Buoc 1: Preprocess raw -> processed

Voi ResNet50, hay preprocess cung kich thuoc voi luc train de tranh upsample tu anh da pad nho hon.

Preprocess cho ResNet50 320x320:

```bash
.venv/bin/python src/preprocess.py \
  --mode preprocess_raw \
  --raw-dir dataset/raw \
  --processed-dir dataset/processed \
  --target-size 320 \
  --log-file logs/preprocess.log
```

Neu muon train ResNet50 384x384, preprocess lai voi `--target-size 384`:

```bash
.venv/bin/python src/preprocess.py \
  --mode preprocess_raw \
  --raw-dir dataset/raw \
  --processed-dir dataset/processed \
  --target-size 384 \
  --log-file logs/preprocess.log
```

Neu ban da co `dataset/processed` san va chi muon lam sach lai:

```bash
.venv/bin/python src/preprocess.py \
  --mode clean_processed \
  --input-dir dataset/processed \
  --in-place \
  --log-file logs/data_clean.log
```

### Buoc 2: Kiem tra khong con chico

```bash
find dataset/processed -maxdepth 1 -type d -name '*chico*'
```

Lenh tren khong duoc in ra folder nao. Neu con `chico_*`, hay gop/xoa va chay lai preprocess/split.

### Buoc 3: Split train/val/test

```bash
.venv/bin/python src/splitter.py \
  --processed-dir dataset/processed \
  --dataset-root dataset \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --log-file logs/split.log
```

Output:

```text
dataset/train/
dataset/val/
dataset/test/
```

### Buoc 4: Kiem tra so class

```bash
find dataset/train -mindepth 1 -maxdepth 1 -type d | wc -l
find dataset/val -mindepth 1 -maxdepth 1 -type d | wc -l
find dataset/test -mindepth 1 -maxdepth 1 -type d | wc -l
```

Ket qua mong doi: moi split co `44` class.

### Buoc 5: Train ResNet50

Train voi anh 320x320:

```bash
.venv/bin/python src/train.py --model resnet50 --image-size 320
```

Khuyen nghi khi train lau tren macOS: dung `nohup` de khong bi tat khi dong terminal,
dung `caffeinate` de chan may sleep trong luc train, va ghi stdout/stderr ra file rieng.

```bash
mkdir -p logs/run_stdout
run_id=$(date +%Y%m%d_%H%M%S)
out_file="logs/run_stdout/resnet50_${run_id}.out"
pid_file="logs/run_stdout/resnet50_${run_id}.pid"

nohup caffeinate -dimsu .venv/bin/python -u src/train.py \
  --model resnet50 \
  --image-size 320 \
  > "$out_file" 2>&1 &

echo $! > "$pid_file"
echo "PID: $(cat "$pid_file")"
echo "Stdout/stderr: $out_file"
tail -f "$out_file"
```

Lenh nay tranh duoc idle sleep va terminal bi dong. Khong tranh duoc truong hop
nguoi dung logout, restart, shutdown, het pin, hoac he dieu hanh bi crash.

Train voi anh 384x384:

```bash
.venv/bin/python src/train.py --model resnet50 --image-size 384
```

Trong luc train, dataloader ResNet50 se tiep tuc strict cleaning theo split va ghi log tai:

```text
logs/data_cleaning.log
```

### Buoc 6: So sanh experiment

```bash
.venv/bin/python compare_experiments.py
```

## 3. Cach chay nhanh chi train

Train ResNet50 mac dinh voi anh 320x320:

```bash
.venv/bin/python src/train.py --model resnet50
```

Train ResNet50 voi do phan giai cao hon:

```bash
.venv/bin/python src/train.py --model resnet50 --image-size 384
```

So sanh cac experiment:

```bash
.venv/bin/python compare_experiments.py
```

## 4. Pipeline ResNet50

Flow tong quat:

```text
dataset/train,val,test
-> strict data cleaning
-> EXIF fix + RGB
-> resize giu ti le + padding den
-> duplicate/blur/quality filter
-> tf.data pipeline + augmentation train only + prefetch
-> ResNet50 model-owned preprocess_input
-> Stage 1 train head
-> Stage 2 fine-tune last layers
-> evaluate test
-> hard example mining optional
-> save experiment artifacts
```

Luu y: ResNet50 preprocess nam ben trong model, nen dataloader khong preprocess ResNet50 lan nua de tranh double-preprocess.
Source hien tai khong tao disk cache tf.data trong dataloader; cac file `cache/tfdata*`
neu con lai tu run cu co the xoa khi khong co training dang chay.

## 5. Model

Backbone:

```text
tf.keras.applications.ResNet50
weights="imagenet"
include_top=False
input_shape=(320, 320, 3) hoac (384, 384, 3)
```

Head:

```text
GlobalAveragePooling2D
BatchNormalization
Dense(head_units, relu)
Dropout(dropout_rate)
Dense(num_classes, softmax)
```

Tham so dang test:

```text
head_units: 512 / 768 / 1024
dropout_rate: 0.4 / 0.5 / 0.6
```

## 6. Training stages

Stage 1: train classification head

```text
base_model.trainable = False
learning_rate = 1e-3
epochs = 15
EarlyStopping patience = 5
ReduceLROnPlateau factor = 0.3, patience = 3
```

Stage 2: fine-tuning

```text
fine_tune_last_layers = 50
BatchNormalization always frozen
initial_learning_rate = 1e-5
schedule = CosineDecay
epochs = 20
```

## 7. Data augmentation

Chi ap dung cho train dataset:

```text
RandomFlip("horizontal")
RandomRotation(0.2)
RandomZoom(0.2)
RandomContrast(0.2)
RandomBrightness(0.2)
GaussianNoise(0.05)
```

Khong ap dung augmentation cho val/test.

## 8. Class imbalance va hard mining

Weak classes hien tai:

```text
sapodilla, mango, burmese
```

Class weight:

```text
weak_class_multiplier = 2.0
```

Hard example mining:

```text
duplicate_factor = 2
max_added_ratio_per_class = 0.25
```

## 9. Output artifacts

Moi lan train tao mot experiment rieng:

```text
experiments/exp_YYYYMMDD_HHMMSS_resnet50/
├── model.keras
├── config.json
├── history.json
├── classification_report.txt
├── confusion_matrix.png
├── test_results.txt
└── checkpoints/
```

Log data cleaning:

```text
logs/data_cleaning.log
```

## 10. Metrics can theo doi

Uu tien xem:

```text
val_loss
val_accuracy
test_accuracy
macro_f1
weighted_f1
per-class recall
confusion matrix
```

Neu muc tieu la nhan dien nong san tren website, nen uu tien macro F1 va recall cua cac class yeu thay vi chi nhin accuracy.

## 11. Cong nghe su dung

```text
Python
TensorFlow / Keras
tf.data
ResNet50 ImageNet pretrained
scikit-learn
NumPy
Pillow
OpenCV
imagehash
seaborn / matplotlib
```

## 12. Luu y du lieu

Dataset hien tai can dung 44 class. Class `chico` da duoc xoa va gop vao `sapodilla`, nen truoc khi train can dam bao `dataset/train`, `dataset/val`, `dataset/test` khong con folder `chico_*`.
