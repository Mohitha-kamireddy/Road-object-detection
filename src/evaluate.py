import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO checkpoint and save metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/metrics.csv"))
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"[ERROR] Weights file not found: {args.weights}")
    if not args.data.is_file():
        raise SystemExit(f"[ERROR] data.yaml not found: {args.data}")

    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    overall = {
        "class": "all",
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }

    per_class_rows = []
    class_names = metrics.names
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        per_class_rows.append({
            "class": class_names[int(class_id)],
            "precision": metrics.box.p[idx],
            "recall": metrics.box.r[idx],
            "mAP50": metrics.box.ap50[idx],
            "mAP50-95": metrics.box.ap[idx],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerow(overall)
        for row in per_class_rows:
            writer.writerow(row)

    print("\n[RESULTS]")
    print(f"  precision: {overall['precision']:.4f}")
    print(f"  recall:    {overall['recall']:.4f}")
    print(f"  mAP50:     {overall['mAP50']:.4f}")
    print(f"  mAP50-95:  {overall['mAP50-95']:.4f}")
    print(f"\n[INFO] Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
