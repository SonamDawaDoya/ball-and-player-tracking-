#!/usr/bin/env python3
"""
track.py
YOLOv8 + ByteTrack pipeline that:
- loads YOLOv8 weights
- uses ByteTrack tracker (yolox implementation)
- writes output video to outputs/
- prints a final JSON (output_video path + summary objects)

Usage:
    python track.py --video input_videos/input.mp4 --weights best.pt --outdir outputs --conf 0.35 --save-video
"""
#!/usr/bin/env python3
import argparse, json, os, sys, time
from pathlib import Path
from dataclasses import dataclass
import cv2, numpy as np
from ultralytics import YOLO
from onemetric.cv.utils.iou import box_iou_batch

try:
    sys.path.insert(0, 'ByteTrack')
    from yolox.tracker.byte_tracker import BYTETracker
except Exception as e:
    print(f"[ERROR] BYTETracker import failed: {e}", file=sys.stderr)
    sys.exit(2)

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def format_predictions(preds, with_conf=True):
    arr = []
    for p in preds:
        bbox = p.boxes.xyxy.int().tolist()[0]
        conf = p.boxes.conf.item()
        arr.append(bbox + [conf] if with_conf else bbox)
    return np.array(arr, float)

def match_detections_with_tracks(dets, tracks):
    det_b = format_predictions(dets, False)
    tr_b = np.array([t.tlbr for t in tracks], float)
    iou = box_iou_batch(tr_b, det_b)
    for ti, di in enumerate(np.argmax(iou, 1)):
        if iou[ti, di] != 0:
            dets[di].tracker_id = tracks[ti].track_id
    return dets

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--skip", type=int, default=1, help="Process every Nth frame")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    tracker = BYTETracker(BYTETrackerArgs())

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    
    # Downscale resolution if too large for speed
    if w > 1280 or h > 720:
        scale = min(1280/w, 720/h, 1.0)
        w, h = int(w * scale), int(h * scale)

    out_path = str(Path(args.outdir) / "processed_output.mp4")

    writer = None
    if args.save_video:
        # Try different codecs
        for codec in ["mp4v", "XVID", "MJPG"]:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if writer.isOpened():
                break
        if not writer.isOpened():
            print("[ERROR] VideoWriter failed with all codecs", file=sys.stderr)
            sys.exit(1)

    frame_idx, results, skip_counter = 0, [], 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        skip_counter += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        # Downscale frame if needed
        if w != frame.shape[1] or h != frame.shape[0]:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Skip frames for speed
        if skip_counter < args.skip:
            if writer:
                writer.write(frame)
            continue
        skip_counter = 0

        dets = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        det_list = []
        for d in dets:
            d.tracker_id = ""
            det_list.append(d)

        tracks = tracker.update(
            output_results=format_predictions(det_list, True),
            img_info=frame.shape, img_size=frame.shape
        ) if len(det_list) else []

        if det_list and tracks:
            det_list = match_detections_with_tracks(det_list, tracks)

        for d in det_list:
            if hasattr(d, "tracker_id") and d.tracker_id != "":
                x1, y1, x2, y2 = d.boxes.xyxy.int().tolist()[0]
                conf = d.boxes.conf.item()
                cid = int(d.boxes.cls.item())
                cls = model.names[cid]
                tid = d.tracker_id

                color_map = {
                    "player": (255, 0, 0),
                    "referee": (0, 0, 255),
                    "goalkeeper": (255, 0, 255),
                    "ball": (0, 200, 200)
                }
                color = color_map.get(cls.lower(), (255, 255, 0))

                if writer:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls} {conf*100:.1f}% ID:{tid}"
                    cv2.putText(frame, label, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                results.append({
                    "frame": frame_idx, "id": tid,
                    "bbox": [x1, y1, x2, y2], "class": cls,
                    "confidence": conf
                })

        if writer:
            # ✅ Prevent black frames
            if frame is not None and frame.size > 0:
                writer.write(frame)

    cap.release()
    if writer: writer.release()

    print(json.dumps({
        "output_video": out_path if args.save_video else None,
        "total_frames": frame_idx,
        "time_s": time.time() - t0,
        "objects_sample": results[:200]
    }))
    return 0

if __name__ == "__main__":
    main()
