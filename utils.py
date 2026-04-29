# import json, numpy as np

# def load_metadata(json_path):
#     with open(json_path) as f:
#         data = json.load(f)
#     return {m["frame"]: m for m in data}

# def compute_shift(meta_prev, meta_curr):
#     if not meta_prev or not meta_curr:
#         return 0.0, 0.0
#     alt   = meta_curr.get("altitude_m", 100)
#     scale = 800 / alt
#     dlat  = (meta_curr["gps_lat"] - meta_prev["gps_lat"]) * 111320
#     dlon  = (meta_curr["gps_lon"] - meta_prev["gps_lon"]) * 111320
#     dx_gps = dlon * scale
#     dy_gps = -dlat * scale
#     dyaw   = meta_curr["yaw_deg"] - meta_prev["yaw_deg"]
#     dx_yaw = dyaw * scale * 0.05
#     return dx_gps + dx_yaw, dy_gps

# def correct_bbox(bbox, dx, dy):
#     x1, y1, x2, y2 = bbox
#     return (x1-dx, y1-dy, x2-dx, y2-dy)

# def draw_tracks(frame, tracks, colors):
#     import cv2
#     for tid, (x1, y1, x2, y2) in tracks:
#         if tid not in colors:
#             np.random.seed(tid)
#             colors[tid] = tuple(np.random.randint(80, 220, 3).tolist())
#         col = colors[tid]
#         cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), col, 2)
#         cv2.putText(frame, f"ID:{tid}",
#                     (int(x1), int(y1)-8),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
#     return frame

import json
import numpy as np

def load_metadata(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {m['frame']: m for m in data}

def compute_shift(meta_prev, meta_curr):
    if not meta_prev or not meta_curr:
        return 0.0, 0.0
    alt = meta_curr.get('altitude_m', 100)
    scale = 800 / alt
    dlat = (meta_curr['gps_lat'] - meta_prev['gps_lat']) * 111320
    dlon = (meta_curr['gps_lon'] - meta_prev['gps_lon']) * 111320
    dx_gps = dlon * scale
    dy_gps = -dlat * scale
    dyaw = meta_curr['yaw_deg'] - meta_prev['yaw_deg']
    dx_yaw = dyaw * scale * 0.05
    return dx_gps + dx_yaw, dy_gps

def correct_bbox(bbox, dx, dy):
    x1, y1, x2, y2 = bbox
    return (x1-dx, y1-dy, x2-dx, y2-dy)

def draw_tracks(frame, tracks, colors):
    import cv2
    for tid, (x1, y1, x2, y2) in tracks:
        if tid not in colors:
            np.random.seed(tid)
            colors[tid] = tuple(np.random.randint(80, 220, 3).tolist())
        col = colors[tid]
        cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), col, 5)
        cv2.putText(frame, f"ID:{tid}",
                    (int(x1), int(y1)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 5)
    return frame