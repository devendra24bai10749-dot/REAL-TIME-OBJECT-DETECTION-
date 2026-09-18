from pathlib import Path
import cv2


def draw_detections(frame, result):
    names = result.names

    if result.boxes is None:
        return frame

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{names[class_id]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y = max(y1, text_h + baseline + 4)

        cv2.rectangle(
            frame,
            (x1, text_y - text_h - baseline - 4),
            (x1 + text_w + 6, text_y),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def detect_image(model, image_path, conf, imgsz, output_path=None):
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
    annotated = draw_detections(frame.copy(), results[0])

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), annotated)
        print(f"Saved result to: {output}")
    else:
        cv2.imshow("YOLO Object Detection", annotated)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_video(model, cap, conf, imgsz, output_path=None):
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f"Could not create output video: {output}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame, conf=conf, imgsz=imgsz, verbose=False
            )
            annotated = draw_detections(frame, results[0])

            if writer:
                writer.write(annotated)

            cv2.imshow("YOLO Object Detection - Press Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if output_path:
        print(f"Saved result to: {output_path}")
