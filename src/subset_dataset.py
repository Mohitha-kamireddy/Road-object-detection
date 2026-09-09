import argparse
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a subset of image/label pairs from an existing YOLO-format split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-images-dir", type=Path, required=True)
    parser.add_argument("--src-labels-dir", type=Path, required=True)
    parser.add_argument("--dst-images-dir", type=Path, required=True)
    parser.add_argument("--dst-labels-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.src_images_dir.is_dir():
        sys.exit(f"[ERROR] Source images dir not found: {args.src_images_dir}")
    if not args.src_labels_dir.is_dir():
        sys.exit(f"[ERROR] Source labels dir not found: {args.src_labels_dir}")

    image_files = sorted(
        p for p in args.src_images_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not image_files:
        sys.exit(f"[ERROR] No image files found in {args.src_images_dir}")

    if args.subset_size > len(image_files):
        sys.exit(
            f"[ERROR] --subset-size {args.subset_size} exceeds available images "
            f"({len(image_files)}) in {args.src_images_dir}"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(image_files, args.subset_size)

    args.dst_images_dir.mkdir(parents=True, exist_ok=True)
    args.dst_labels_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_missing_labels = 0

    for image_path in sampled:
        label_path = args.src_labels_dir / (image_path.stem + ".txt")
        if not label_path.is_file():
            n_missing_labels += 1
            continue

        shutil.copy2(image_path, args.dst_images_dir / image_path.name)
        shutil.copy2(label_path, args.dst_labels_dir / label_path.name)
        n_copied += 1

    print(f"[INFO] Copied {n_copied} image/label pairs to {args.dst_images_dir} / {args.dst_labels_dir}")
    if n_missing_labels:
        print(f"[WARN] {n_missing_labels} sampled images had no matching label file and were skipped")


if __name__ == "__main__":
    main()
