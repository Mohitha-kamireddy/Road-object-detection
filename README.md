# Road Object Detection

End-to-end object detection pipeline for autonomous-driving scenes, built on the BDD100K dataset using Ultralytics YOLO11.

## Overview

This project covers the full pipeline: dataset preparation, subset sampling, YOLO training (local CPU and Google Colab GPU), and evaluation — detecting 10 road object classes.

**Classes:** person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train

## Results

Evaluated on a 10,000-image validation subset (`road_object_detection-6` checkpoint):

| Metric | Value |
|---|---|
| Precision | 0.486 |
| Recall | 0.280 |
| mAP@50 | 0.287 |
| mAP@50:95 | 0.150 |

Full per-class breakdown available in `outputs/metrics/metrics.csv` after running `evaluate.py`.

## Project Structure

```
road-object-detection/
├── data/
│   ├── raw/                     # Original BDD100K data (not tracked in git)
│   └── yolo/                    # YOLO-format images + labels (not tracked in git)
│       ├── images/{train,val}/
│       └── labels/{train,val}/
├── src/
│   ├── prepare_data.py          # Convert BDD100K annotations to YOLO format
│   ├── subset_dataset.py        # Sample a smaller subset from a YOLO-format split
│   ├── generate_data_yaml.py    # Build/validate data.yaml from actual label files
│   ├── train.py                 # Train a YOLO model
│   └── evaluate.py              # Evaluate a checkpoint, save metrics to CSV
├── notebooks/
│   └── train_colab.ipynb        # GPU training on Google Colab, with resume support
├── outputs/
│   ├── weights/                 # Trained checkpoints (not tracked in git)
│   ├── predictions/             # Inference outputs (not tracked in git)
│   └── metrics/                 # Evaluation results (CSV)
├── data.yaml                    # YOLO dataset config (paths, classes)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

BDD100K is not downloaded automatically. Download it from [bdd100k.com](https://www.bdd100k.com/) or a mirror (e.g. Kaggle), then either:

- Use an already YOLO-formatted version directly, or
- Convert raw BDD100K annotations yourself:

```bash
python3 src/prepare_data.py \
    --images-dir /path/to/bdd100k/images/100k/train \
    --labels-json /path/to/bdd100k_labels_images_train.json \
    --output-dir data/yolo \
    --subset-size 2000 \
    --val-split 0.2
```

To sample a smaller subset from an existing YOLO-format split:

```bash
python3 src/subset_dataset.py \
    --src-images-dir data/yolo/images/train \
    --src-labels-dir data/yolo/labels/train \
    --dst-images-dir data/yolo_subset/images/train \
    --dst-labels-dir data/yolo_subset/labels/train \
    --subset-size 10000
```

Generate `data.yaml`, validated against the actual classes present in your labels:

```bash
python3 src/generate_data_yaml.py \
    --path data/yolo_subset \
    --train-labels-dir data/yolo_subset/labels/train \
    --val-labels-dir data/yolo_subset/labels/val \
    --class-names "person,rider,car,bus,truck,bike,motor,traffic light,traffic sign,train" \
    --output data.yaml
```

## Training

**Locally (CPU):**

```bash
python3 src/train.py --data data.yaml --epochs 10 --batch 8 --imgsz 640
```

**On Google Colab (GPU, recommended for the full dataset):**

Open `notebooks/train_colab.ipynb` in Colab, upload your zipped dataset to Google Drive, and run the cells top to bottom. Checkpoints save directly to Drive and the training cell auto-resumes from the last checkpoint if the session disconnects.

## Evaluation

```bash
python3 src/evaluate.py \
    --weights runs/detect/outputs/weights/road_object_detection-6/weights/best.pt \
    --data data.yaml
```

Prints precision, recall, mAP@50, and mAP@50:95 overall and saves a full per-class breakdown to `outputs/metrics/metrics.csv`.

## Notes

- Model checkpoints, raw data, and image/label files are excluded from version control (see `.gitignore`) due to size — only code, configs, and metrics are tracked.
- Trained on a subset of the full BDD100K dataset; results will improve with more training data and epochs.
