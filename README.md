# 🎯 Real-Time Object Detection & Tracking

**YOLOv8 + SORT — live webcam or video, labeled boxes, persistent track IDs**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-00FFFF?logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/Video-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Tracker](https://img.shields.io/badge/Tracker-SORT-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What it does

Point it at a webcam or a video file, and it will detect every object frame
by frame, follow each one across time, and draw a live overlay with class
name, a stable ID, and a running FPS counter.

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌───────────────┐
│  Video Feed  │ ───▶ │  YOLOv8      │ ───▶ │  SORT        │ ───▶ │  Live Display │
│ (webcam/file)│      │  Detection   │      │  Tracking    │      │  boxes + IDs  │
└──────────────┘      └──────────────┘      └──────────────┘      └───────────────┘
```

## 🧩 How each requirement is met

| # | Requirement | Implementation |
|---|---|---|
| 1 | Real-time video input | `cv2.VideoCapture` — webcam index **or** video file path |
| 2 | Pre-trained detection model | **YOLOv8** (Ultralytics) — weights auto-download on first run |
| 3 | Per-frame detection + boxes | `model.predict()` every frame, boxes drawn with OpenCV |
| 4 | Object tracking | **SORT** — Kalman Filter per object + Hungarian algorithm on IoU |
| 5 | Real-time labeled display | OpenCV window: `ClassName ID:N` label + live FPS counter |

## 📦 Setup

```bash
pip install -r requirements.txt
```

> No manual downloads needed — the first run auto-fetches `yolov8n.pt` (~6 MB).

## 🚀 Usage

```bash
# Webcam (default camera)
python main.py

# Specific webcam index
python main.py --source 1

# Video file
python main.py --source path/to/video.mp4

# Save the annotated output while viewing it live
python main.py --source video.mp4 --save output.mp4

# Track only people and cars (COCO ids: 0=person, 2=car)
python main.py --classes 0 2

# Bigger, more accurate model
python main.py --model yolov8m.pt --conf 0.5
```

Press **`q`** in the video window to quit.

## ⚙️ Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLO weights — `n` (fastest) → `s` → `m` → `l` → `x` (most accurate) |
| `--conf` | `0.4` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold for the detector |
| `--classes` | *all* | Restrict to specific COCO class IDs |
| `--max-age` | `15` | Frames a track survives with no matching detection |
| `--min-hits` | `3` | Detections required before a track is confirmed/shown |
| `--iou-thresh` | `0.3` | SORT matching IoU threshold |
| `--save` | *none* | Path to write an annotated output video |
| `--no-display` | off | Headless mode (for servers without a display) |

## 📁 Project structure

```
object_tracking_project/
├── main.py            # capture loop, YOLO inference, drawing, CLI
├── sort.py             # SORT tracker: Kalman filter + Hungarian assignment
├── requirements.txt     # dependencies
└── README.md            # this file
```

## 💡 Notes

- Each track gets a **stable, deterministic color** and a `ClassName ID:N` label,
  so the same object keeps the same look across frames.
- **SORT vs. Deep SORT** — this project uses motion-only SORT (fast,
  dependency-light). IDs can swap when objects cross paths or fully occlude
  one another. To upgrade to appearance-based **Deep SORT**, swap the `Sort`
  class for `deep-sort-realtime`'s tracker — the detection loop stays
  identical, you'd just call `tracker.update_tracks(...)` instead.
- **GPU acceleration** — install a CUDA-enabled PyTorch build and Ultralytics
  will use it automatically if available.

---

<div align="center">
Built with YOLOv8 · OpenCV · SORT
</div>