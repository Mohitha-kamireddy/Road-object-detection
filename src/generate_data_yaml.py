import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a YOLO data.yaml, validated against actual label files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", type=str, default="data/yolo")
    parser.add_argument("--train", type=str, default="images/train")
    parser.add_argument("--val", type=str, default="images/val")
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--train-labels-dir", type=Path, default=Path("data/yolo/labels/train"))
    parser.add_argument("--val-labels-dir", type=Path, default=Path("data/yolo/labels/val"))
    parser.add_argument("--class-names", type=str, required=True,
                         help="Comma-separated class names in index order, e.g. "
                              "'person,rider,car,bus,truck,bike,motor,traffic light,traffic sign,train'")
    parser.add_argument("--output", type=Path, default=Path("data.yaml"))
    parser.add_argument("--skip-validation", action="store_true",
                         help="Skip scanning label files to confirm class indices match --class-names.")
    return parser.parse_args()


def collect_used_class_ids(labels_dir: Path):
    if not labels_dir.is_dir():
        sys.exit(f"[ERROR] Labels directory not found: {labels_dir}")
    used_ids = set()
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    class_id = int(parts[0])
                except (ValueError, IndexError):
                    sys.exit(f"[ERROR] Malformed label line in {label_file}: '{line}'")
                used_ids.add(class_id)
    return used_ids


def main():
    args = parse_args()

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    if not class_names:
        sys.exit("[ERROR] --class-names produced an empty list.")

    if not args.skip_validation:
        used_ids = collect_used_class_ids(args.train_labels_dir)
        used_ids |= collect_used_class_ids(args.val_labels_dir)

        if not used_ids:
            sys.exit("[ERROR] No class ids found in label files. Check --train-labels-dir / --val-labels-dir.")

        max_id = max(used_ids)
        if max_id >= len(class_names):
            sys.exit(
                f"[ERROR] Label files reference class id {max_id}, but only "
                f"{len(class_names)} class names were provided. Refusing to invent a name "
                f"for an unmapped class id."
            )

        unused = set(range(len(class_names))) - used_ids
        if unused:
            unused_names = [class_names[i] for i in sorted(unused)]
            print(f"[WARN] These provided classes never appear in the label files: {unused_names}")

        print(f"[INFO] Validated: class ids in label files are {sorted(used_ids)}, "
              f"all within the provided {len(class_names)} class names.")
    else:
        print("[INFO] Skipping validation against label files (--skip-validation set).")

    lines = [
        f"path: {args.path}",
        f"train: {args.train}",
        f"val: {args.val}",
    ]
    if args.test:
        lines.append(f"test: {args.test}")
    lines.append("")
    lines.append(f"nc: {len(class_names)}")
    lines.append("names:")
    for name in class_names:
        lines.append(f"  - {name}")

    args.output.write_text("\n".join(lines) + "\n")
    print(f"[INFO] Wrote {args.output} with {len(class_names)} classes.")


if __name__ == "__main__":
    main()
