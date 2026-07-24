"""
Real-Time Object Detection and Tracking
----------------------------------------
- Video input: webcam or video file (OpenCV)
- Detection: pre-trained YOLOv8 (Ultralytics)
- Tracking: SORT (Kalman Filter + Hungarian algorithm)
- Output: live window with bounding boxes, class labels, and track IDs

Usage:
    python main.py                          # use default webcam (index 0)
    python main.py --source 0                # webcam index 0
    python main.py --source video.mp4         # video file
    python main.py --source video.mp4 --save out.mp4   # also save output
    python main.py --model yolov8n.pt --conf 0.4
"""

import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO

from sort import Sort


def get_color(track_id):
    """Deterministic color per track ID for consistent box colors."""
    np.random.seed(track_id * 7 + 3)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking (YOLO + SORT)")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: webcam index (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Path/name of YOLO model weights (auto-downloaded if not found)")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="Detection confidence threshold (raise this, e.g. 0.5-0.6, "
                              "if you see flickering/wrong labels on yolov8n)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for detector")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                         help="Restrict detection to specific class IDs (COCO), e.g. --classes 0 2 for person, car")
    parser.add_argument("--max-age", type=int, default=15, help="SORT: frames to keep a lost track alive")
    parser.add_argument("--min-hits", type=int, default=3, help="SORT: min detections before confirming a track")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="SORT: IoU threshold for matching")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save annotated output video")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window (headless mode)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Video source (webcam index or file path) ----
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))

    # ---- Load pre-trained detector ----
    print(f"Loading detector: {args.model} ...")
    model = YOLO(args.model)
    class_names = model.names  # dict: id -> class name (COCO by default)

    # ---- Tracker ----
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_thresh)

    prev_time = time.time()
    fps_smooth = 0.0

    print("Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream / cannot read frame.")
            break

        # ---- Detection ----
        results = model.predict(
            frame, conf=args.conf, iou=args.iou, classes=args.classes, verbose=False
        )[0]

        dets = []
        cls_ids = []
        cls_names_list = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                dets.append([x1, y1, x2, y2, score])
                cls_ids.append(int(cls))
                cls_names_list.append(class_names.get(int(cls), str(cls)))

        dets_np = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

        # ---- Tracking ----
        tracks = tracker.update(dets_np, cls_ids, cls_names_list)

        # ---- Draw results ----
        for trk in tracks:
            x1, y1, x2, y2, track_id, cls_id = trk
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label_name = class_names.get(cls_id, "obj") if cls_id in class_names else "obj"
            color = get_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_name} ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # ---- FPS overlay ----
        now = time.time()
        fps_curr = 1.0 / max(now - prev_time, 1e-6)
        fps_smooth = fps_curr if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps_curr)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}  Tracks: {len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Object Detection & Tracking (YOLO + SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
