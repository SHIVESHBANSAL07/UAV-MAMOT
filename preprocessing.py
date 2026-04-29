import cv2, numpy as np

def load_frame(frame):
    """Resize and normalize frame for model input"""
    resized = cv2.resize(frame, (640, 640))
    return resized

def extract_features(frame):
    """Extract basic visual features from frame"""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    return edges

def extract_frames(video_path, output_dir, every_n=5):
    """Extract every Nth frame from a video file"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} FPS")
    frame_count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % every_n == 0:
            fname = f"{output_dir}/frame_{frame_count:06d}.jpg"
            cv2.imwrite(fname, frame)
            saved += 1
        frame_count += 1
    cap.release()
    print(f"Saved {saved} frames to {output_dir}")
    return saved