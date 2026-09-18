import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

from config import DEFAULT_MODEL, DEFAULT_CONFIDENCE, DEFAULT_IMAGE_SIZE
from detect import detect_image, detect_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using YOLO and OpenCV"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input source: webcam index (0), image path, or video path."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model name/path. Default: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Minimum confidence threshold. Default: {DEFAULT_CONFIDENCE}"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Inference image size. Default: {DEFAULT_IMAGE_SIZE}"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path for image/video results."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")

    model = YOLO(args.model)
    source = args.source

    # Webcam: "0", "1", etc.
    if source.isdigit():
        camera_index = int(source)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam {camera_index}.")
        detect_video(model, cap, args.conf, args.imgsz, args.output)
        return

    path = Path(source)
    if not path.exists():
        raise SystemExit(f"Input file not found: {source}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.suffix.lower() in image_extensions:
        detect_image(model, str(path), args.conf, args.imgsz, args.output)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {source}")
        detect_video(model, cap, args.conf, args.imgsz, args.output)


if __name__ == "__main__":
    main()
