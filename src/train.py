import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on a BDD100K-derived dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="outputs/weights")
    parser.add_argument("--name", type=str, default="road_object_detection")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.is_file():
        raise SystemExit(f"[ERROR] data.yaml not found: {args.data}")

    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    metrics = model.val()

    best_ckpt = Path(results.save_dir) / "weights" / "best.pt"

    print("\n[SUMMARY]")
    print(f"  best checkpoint: {best_ckpt}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
