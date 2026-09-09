import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

FALLBACK_IMG_WIDTH = 1280
FALLBACK_IMG_HEIGHT = 720


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a subset of BDD100K detection annotations to YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Path to a directory of BDD100K images (e.g. bdd100k/images/100k/train).",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        required=True,
        help="Path to the BDD100K detection annotation JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo"),
        help="Root output directory for the YOLO-format dataset.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="Number of images to sample from the annotation file. "
        "Use -1 to use all annotated images.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of the subset to use for validation (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset sampling and train/val split.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of class names to keep, in YOLO class-index order. "
        "Defaults to the standard 10 BDD100K detection classes. "
        "Labels not in this list are skipped.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copying an image if it already exists in the output directory.",
    )
    return parser.parse_args()


def load_annotations(labels_json: Path):
    if not labels_json.is_file():
        sys.exit(f"[ERROR] Annotation file not found: {labels_json}")
    with open(labels_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"[ERROR] Unexpected annotation format in {labels_json} "
                  f"(expected a JSON list of image records).")
    return data


def select_subset(records, subset_size, seed):
    if subset_size is not None and subset_size >= 0 and subset_size < len(records):
        rng = random.Random(seed)
        return rng.sample(records, subset_size)
    return list(records)


def split_train_val(records, val_split, seed):
    if not 0.0 <= val_split < 1.0:
        sys.exit(f"[ERROR] --val-split must be in [0, 1), got {val_split}")
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_split))
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]
    return train_records, val_records


def get_image_size(image_path: Path):
    if Image is not None and image_path.is_file():
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            pass
    return FALLBACK_IMG_WIDTH, FALLBACK_IMG_HEIGHT


def box2d_to_yolo(box2d, img_w, img_h):
    x1 = max(0.0, min(box2d["x1"], img_w))
    y1 = max(0.0, min(box2d["y1"], img_h))
    x2 = max(0.0, min(box2d["x2"], img_w))
    y2 = max(0.0, min(box2d["y2"], img_h))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    norm_w = box_w / img_w
    norm_h = box_h / img_h
    return x_center, y_center, norm_w, norm_h


def process_split(split_name, records, images_dir, output_dir, classes, skip_existing):
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing_images = 0
    n_boxes_written = 0
    n_boxes_skipped = 0

    for record in tqdm(records, desc=f"Processing {split_name}", unit="img"):
        image_name = record.get("name")
        if not image_name:
            continue

        src_image_path = images_dir / image_name
        if not src_image_path.is_file():
            n_missing_images += 1
            continue

        dst_image_path = images_out / image_name
        if not (skip_existing and dst_image_path.is_file()):
            shutil.copy2(src_image_path, dst_image_path)
        n_copied += 1

        img_w, img_h = get_image_size(src_image_path)

        yolo_lines = []
        for label in record.get("labels", []):
            box2d = label.get("box2d")
            category = label.get("category")
            if box2d is None or category not in class_to_idx:
                if box2d is not None:
                    n_boxes_skipped += 1
                continue

            yolo_box = box2d_to_yolo(box2d, img_w, img_h)
            if yolo_box is None:
                n_boxes_skipped += 1
                continue

            class_idx = class_to_idx[category]
            x_center, y_center, norm_w, norm_h = yolo_box
            yolo_lines.append(
                f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )
            n_boxes_written += 1

        label_path = labels_out / (Path(image_name).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

    return {
        "images_copied": n_copied,
        "images_missing": n_missing_images,
        "boxes_written": n_boxes_written,
        "boxes_skipped": n_boxes_skipped,
    }


def main():
    args = parse_args()

    if not args.images_dir.is_dir():
        sys.exit(f"[ERROR] Images directory not found: {args.images_dir}")

    classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes
        else DEFAULT_CLASSES
    )

    print(f"[INFO] Loading annotations from {args.labels_json}")
    records = load_annotations(args.labels_json)
    print(f"[INFO] Loaded {len(records)} annotated image records")

    subset = select_subset(records, args.subset_size, args.seed)
    print(f"[INFO] Selected subset of {len(subset)} images (seed={args.seed})")

    train_records, val_records = split_train_val(subset, args.val_split, args.seed)
    print(
        f"[INFO] Split into {len(train_records)} train / {len(val_records)} val "
        f"(val_split={args.val_split})"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    summary["train"] = process_split(
        "train", train_records, args.images_dir, args.output_dir, classes, args.skip_existing
    )
    summary["val"] = process_split(
        "val", val_records, args.images_dir, args.output_dir, classes, args.skip_existing
    )

    print("\n[SUMMARY]")
    for split_name, stats in summary.items():
        print(
            f"  {split_name}: {stats['images_copied']} images copied, "
            f"{stats['images_missing']} images missing, "
            f"{stats['boxes_written']} boxes written, "
            f"{stats['boxes_skipped']} boxes skipped"
        )
    print(f"\n[INFO] Classes ({len(classes)}): {classes}")
    print(f"[INFO] Output written to: {args.output_dir.resolve()}")
    print(
        "[INFO] Update data.yaml with these paths and class names before training."
    )


if __name__ == "__main__":
    main()
