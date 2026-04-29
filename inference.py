from ultralytics import YOLO
import cv2
import time
import numpy as np
import os
import json
from tracker import MAMOTTracker


def run_inference(video_path, model_path, meta_path=None, conf=0.3, output_path=None):
    print("Loading model...")
    model = YOLO(model_path)
    tracker = MAMOTTracker()

    metadata = {}
    if meta_path:
        with open(meta_path) as f:
            data = json.load(f)
        metadata = {m["frame"]: m for m in data}

    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    print("Video opened successfully")

    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in, (W, H)
        )

    frame_idx = 0
    times = []
    id_switches = 0
    prev_ids = set()
    colors = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        results = model(frame, conf=conf, verbose=False,)[0]
        if len(results.boxes):
            dets = results.boxes.xyxy.cpu().numpy().tolist()
        else:
            dets = []

        meta_curr = metadata.get(frame_idx, {})
        meta_prev = metadata.get(frame_idx - 1, {})
        dx, dy = 0.0, 0.0
        if meta_curr and meta_prev:
            alt = meta_curr.get("altitude_m", 100)
            scale = 800 / alt
            dx = (meta_curr["gps_lon"] - meta_prev["gps_lon"]) * 111320 * scale
            dy = -(meta_curr["gps_lat"] - meta_prev["gps_lat"]) * 111320 * scale

        tracks = tracker.update(dets, dx, dy)

        curr_ids = {tid for tid, _ in tracks}
        new_ids = curr_ids - prev_ids - {t.id for t in tracker.trackers if t.hits == 1}
        id_switches += len(new_ids)
        prev_ids = curr_ids

        inf_time = (time.time() - t0) * 1000
        times.append(inf_time)
        fps_val = 1000 / inf_time if inf_time > 0 else 0

        for tid, (x1, y1, x2, y2) in tracks:
            if tid not in colors:
                np.random.seed(tid)
                colors[tid] = tuple(np.random.randint(80, 220, 3).tolist())
            col = colors[tid]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(frame, "ID:" + str(tid),
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.putText(frame, "FPS: " + str(round(fps_val, 1)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Inf: " + str(round(inf_time, 1)) + "ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Tracks: " + str(len(tracks)),
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if out:
            out.write(frame)

        frame_idx += 1

        if frame_idx % 50 == 0:
            print("Frame " + str(frame_idx) + " | FPS:" + str(round(fps_val, 1)) + " | Tracks:" + str(len(tracks)))

    cap.release()
    if out:
        out.release()

    avg_fps = 1000 / np.mean(times)
    avg_inf = np.mean(times)

    print("Done!")
    print("Avg FPS: " + str(round(avg_fps, 2)))
    print("Avg Inf Time: " + str(round(avg_inf, 2)) + "ms")
    print("ID Switches: " + str(id_switches))
    if output_path:
        print("Output saved: " + output_path)

    return avg_inf, avg_fps, id_switches