import streamlit as st
import cv2
import numpy as np
import json
import time
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAV MA-MOT Dashboard",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --primary: #00d4ff;
    --secondary: #ff6b35;
    --accent: #00ff88;
    --bg-dark: #0a0e1a;
    --bg-card: #0f1629;
    --bg-card2: #141d35;
    --border: #1e3a5f;
    --text: #c8d8e8;
    --text-dim: #5a7a9a;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a1020 100%);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main header */
.main-header {
    background: linear-gradient(90deg, rgba(0,212,255,0.1) 0%, rgba(0,255,136,0.05) 50%, rgba(255,107,53,0.1) 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 8px;
    padding: 20px 30px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
}
.main-header h1 {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: var(--primary);
    margin: 0;
    letter-spacing: 2px;
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
}
.main-header p {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-dim);
    margin: 6px 0 0;
    font-size: 0.85rem;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--primary);
    text-shadow: 0 0 10px rgba(0,212,255,0.4);
    line-height: 1;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
}
.metric-delta {
    font-size: 0.8rem;
    color: var(--accent);
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}
.badge-green { background: rgba(0,255,136,0.15); color: var(--accent); border: 1px solid rgba(0,255,136,0.3); }
.badge-blue  { background: rgba(0,212,255,0.15); color: var(--primary); border: 1px solid rgba(0,212,255,0.3); }
.badge-orange{ background: rgba(255,107,53,0.15); color: var(--secondary); border: 1px solid rgba(255,107,53,0.3); }

/* Info box */
.info-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
}

/* Comparison table */
.comp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}
.comp-table th {
    background: rgba(0,212,255,0.1);
    color: var(--primary);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
}
.comp-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(30,58,95,0.5);
    color: var(--text);
}
.comp-table tr:hover td { background: rgba(0,212,255,0.03); }
.highlight { color: var(--accent); font-weight: bold; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown {
    color: var(--text);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,255,136,0.1));
    color: var(--primary);
    border: 1px solid var(--primary);
    border-radius: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    padding: 10px 20px;
    transition: all 0.3s;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.25);
    box-shadow: 0 0 15px rgba(0,212,255,0.3);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 10px;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    color: var(--text-dim);
    background: transparent;
    border: none;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    background: rgba(0,212,255,0.05) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def load_metadata(path):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return {m['frame']: m for m in data}
    return {}

def compute_shift(meta_prev, meta_curr):
    if not meta_prev or not meta_curr:
        return 0.0, 0.0
    alt = meta_curr.get('altitude_m', 100)
    scale = 800 / alt
    dlat = (meta_curr['gps_lat'] - meta_prev['gps_lat']) * 111320
    dlon = (meta_curr['gps_lon'] - meta_prev['gps_lon']) * 111320
    dx = dlon * scale + (meta_curr['yaw_deg'] - meta_prev['yaw_deg']) * scale * 0.05
    dy = -dlat * scale
    return dx, dy

def correct_bbox(bbox, dx, dy):
    x1, y1, x2, y2 = bbox
    return (x1-dx, y1-dy, x2-dx, y2-dy)

def iou(b1, b2):
    x1 = max(b1[0],b2[0]); y1 = max(b1[1],b2[1])
    x2 = min(b1[2],b2[2]); y2 = min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-9)

class KalmanTracker:
    count = 0
    def __init__(self, bbox):
        KalmanTracker.count += 1
        self.id = KalmanTracker.count
        self.hits = 1
        self.no_det = 0
        x1,y1,x2,y2 = bbox
        cx,cy = (x1+x2)/2,(y1+y2)/2
        w,h = x2-x1,y2-y1
        self.state = np.array([cx,cy,w,h,0,0], dtype=float)
    def predict(self):
        self.state[0]+=self.state[4]; self.state[1]+=self.state[5]
        self.no_det+=1; return self.get_bbox()
    def update(self, bbox):
        x1,y1,x2,y2=bbox; cx,cy=(x1+x2)/2,(y1+y2)/2; w,h=x2-x1,y2-y1; a=0.6
        self.state[4]=a*(cx-self.state[0])+(1-a)*self.state[4]
        self.state[5]=a*(cy-self.state[1])+(1-a)*self.state[5]
        self.state[0],self.state[1]=cx,cy; self.state[2],self.state[3]=w,h
        self.hits+=1; self.no_det=0
    def get_bbox(self):
        cx,cy,w,h=self.state[:4]; return (cx-w/2,cy-h/2,cx+w/2,cy+h/2)

class MAMOTTracker:
    def __init__(self, max_age=10, min_hits=1, iou_thresh=0.3):
        self.trackers=[]; self.max_age=max_age
        self.min_hits=min_hits; self.iou_thresh=iou_thresh
        KalmanTracker.count=0
    def update(self, detections, dx=0, dy=0):
        from scipy.optimize import linear_sum_assignment
        predicted=[t.predict() for t in self.trackers]
        corrected=[correct_bbox(d,dx,dy) for d in detections]
        if self.trackers and corrected:
            iou_matrix=np.array([[iou(p,c) for c in corrected] for p in predicted])
            row_ind,col_ind=linear_sum_assignment(-iou_matrix)
            matched_t,matched_d=set(),set()
            for r,c in zip(row_ind,col_ind):
                if iou_matrix[r,c]>=self.iou_thresh:
                    self.trackers[r].update(corrected[c]); matched_t.add(r); matched_d.add(c)
            for i,det in enumerate(corrected):
                if i not in matched_d: self.trackers.append(KalmanTracker(det))
        else:
            for det in corrected: self.trackers.append(KalmanTracker(det))
        self.trackers=[t for t in self.trackers if t.no_det<=self.max_age]
        return [(t.id,t.get_bbox()) for t in self.trackers if t.hits>=self.min_hits]

def make_comparison_chart():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')
    metrics = ['Inf Time (ms)', 'Avg FPS', 'ID Switches']
    baseline = [30.67, 32.61, 86]
    mamot = [153.62, 6.51, 84]
    colors_b = '#1e3a5f'
    colors_m = '#00d4ff'
    for i, (ax, m, b, ma) in enumerate(zip(axes, metrics, baseline, mamot)):
        ax.set_facecolor('#0f1629')
        bars = ax.bar(['Baseline', 'MA-MOT'], [b, ma],
                      color=[colors_b, colors_m], width=0.5, edgecolor='none')
        ax.set_title(m, color='#00d4ff', fontsize=11,
                     fontfamily='monospace', pad=10)
        ax.tick_params(colors='#5a7a9a', labelsize=9)
        ax.spines['bottom'].set_color('#1e3a5f')
        ax.spines['left'].set_color('#1e3a5f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.label.set_color('#5a7a9a')
        for bar, val in zip(bars, [b, ma]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}', ha='center', va='bottom',
                    color='white', fontsize=9, fontfamily='monospace')
        ax.grid(axis='y', color='#1e3a5f', alpha=0.5, linewidth=0.5)
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0a0e1a')
    buf.seek(0)
    plt.close()
    return buf

def make_training_chart():
    epochs = list(range(1, 51))
    map50 = [0.096, 0.113, 0.139, 0.126, 0.142, 0.144, 0.160, 0.162,
             0.187, 0.183, 0.204, 0.208, 0.204, 0.209, 0.213, 0.222,
             0.223, 0.212, 0.230, 0.231, 0.238, 0.232, 0.238, 0.233,
             0.243, 0.245, 0.245, 0.253, 0.256, 0.256, 0.258, 0.257,
             0.257, 0.262, 0.255, 0.260, 0.262, 0.262, 0.264, 0.265,
             0.263, 0.262, 0.261, 0.265, 0.262, 0.265, 0.267, 0.267,
             0.266, 0.267]
    box_loss = [2.022, 1.931, 1.901, 1.871, 1.847, 1.847, 1.807, 1.816,
                1.799, 1.773, 1.766, 1.776, 1.771, 1.734, 1.742, 1.726,
                1.711, 1.714, 1.710, 1.713, 1.696, 1.689, 1.692, 1.683,
                1.669, 1.674, 1.655, 1.643, 1.656, 1.656, 1.644, 1.638,
                1.634, 1.641, 1.624, 1.635, 1.629, 1.630, 1.611, 1.601,
                1.569, 1.562, 1.531, 1.538, 1.548, 1.527, 1.536, 1.539,
                1.531, 1.530]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0a0e1a')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0f1629')
        ax.tick_params(colors='#5a7a9a', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#1e3a5f')
        ax.grid(color='#1e3a5f', alpha=0.4, linewidth=0.5)
    ax1.plot(epochs, box_loss, color='#ff6b35', linewidth=2, label='Box Loss')
    ax1.set_title('Box Loss vs Epoch', color='#00d4ff', fontsize=11, fontfamily='monospace')
    ax1.set_xlabel('Epoch', color='#5a7a9a', fontsize=9)
    ax1.set_ylabel('Loss', color='#5a7a9a', fontsize=9)
    ax1.fill_between(epochs, box_loss, alpha=0.1, color='#ff6b35')
    ax2.plot(epochs, map50, color='#00d4ff', linewidth=2, label='mAP50')
    ax2.set_title('mAP50 vs Epoch (YOLOv8n)', color='#00d4ff', fontsize=11, fontfamily='monospace')
    ax2.set_xlabel('Epoch', color='#5a7a9a', fontsize=9)
    ax2.set_ylabel('mAP50', color='#5a7a9a', fontsize=9)
    ax2.fill_between(epochs, map50, alpha=0.1, color='#00d4ff')
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0e1a')
    buf.seek(0)
    plt.close()
    return buf


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-family: Orbitron, monospace; font-size: 1.1rem;
                    color: #00d4ff; letter-spacing: 3px; font-weight: 900;'>
            🛸 UAV-MAMOT
        </div>
        <div style='font-family: Share Tech Mono, monospace; font-size: 0.7rem;
                    color: #5a7a9a; margin-top: 4px; letter-spacing: 2px;'>
            SURVEILLANCE SYSTEM
        </div>
    </div>
    <hr style='border-color: #1e3a5f; margin: 10px 0 20px;'>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:0">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠  Overview", "🎬  Run Inference", "📊  Training Results", "📈  Model Comparison", "📋  Logs", "ℹ️  About"], label_visibility="collapsed")

    st.markdown('<hr style="border-color: #1e3a5f; margin: 20px 0;">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">SYSTEM STATUS</div>', unsafe_allow_html=True)

    model_path = "models/maritime_v1/weights/best.pt"
    meta_path = "metadata/drone_meta.json"

    model_ok = os.path.exists(model_path)
    meta_ok = os.path.exists(meta_path)

    st.markdown(f"""
    <div class="info-box" style="padding: 10px 14px;">
        <div style="margin-bottom:6px;">
            {'🟢' if model_ok else '🔴'} Model Weights
            <span style="float:right; font-family: Share Tech Mono, monospace; font-size:0.75rem; color: {'#00ff88' if model_ok else '#ff4444'}">
                {'LOADED' if model_ok else 'MISSING'}
            </span>
        </div>
        <div>
            {'🟢' if meta_ok else '🔴'} Metadata JSON
            <span style="float:right; font-family: Share Tech Mono, monospace; font-size:0.75rem; color: {'#00ff88' if meta_ok else '#ff4444'}">
                {'LOADED' if meta_ok else 'MISSING'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">CONFIGURATION</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    use_metadata = st.toggle("Use Metadata Fusion", value=True)
    show_fps = st.toggle("Show FPS Overlay", value=True)


# ─── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛸 UAV METADATA-ASSISTED TRACKING</h1>
    <p>MA-MOT SYSTEM · YOLOV8s · KALMAN FILTER · HUNGARIAN ALGORITHM · THAPAR INSTITUTE</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    st.markdown('<div class="section-header">KEY PERFORMANCE METRICS</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, "0.878", "mAP50", "Cars Only"),
        (col2, "2.2ms", "Inf Time", "Per Frame"),
        (col3, "84", "ID Switches", "MA-MOT"),
        (col4, "6.5 FPS", "System FPS", "With Metadata"),
        (col5, "87.8%", "Accuracy", "YOLOv8s"),
    ]
    for col, val, label, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">SYSTEM PIPELINE</div>', unsafe_allow_html=True)
        steps = [
            ("01", "INPUT", "Video frames from drone camera (1920×1080 @ 30 FPS)"),
            ("02", "PREPROCESS", "Resize to 640×640, normalize pixel values"),
            ("03", "DETECTION", "YOLOv8s detects cars — outputs [x1,y1,x2,y2] + confidence"),
            ("04", "METADATA", "GPS/altitude/yaw loaded → pixel shift (dx,dy) computed"),
            ("05", "CORRECTION", "Bounding boxes corrected: x'=x-dx, y'=y-dy"),
            ("06", "TRACKING", "Kalman filter + Hungarian algorithm → stable Track IDs"),
            ("07", "OUTPUT", "Annotated video with IDs, FPS, inference time overlay"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="info-box" style="display:flex; gap:14px; align-items:flex-start; margin:6px 0;">
                <div style="font-family: Orbitron, monospace; font-size: 1rem;
                            color: #00d4ff; font-weight: 700; min-width: 28px;">{num}</div>
                <div>
                    <div style="font-family: Orbitron, monospace; font-size: 0.75rem;
                                color: #00ff88; letter-spacing: 2px; margin-bottom: 2px;">{title}</div>
                    <div style="font-size: 0.9rem; color: #c8d8e8;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">MODEL COMPARISON</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="comp-table">
            <tr>
                <th>PROPERTY</th>
                <th>YOLOV8n</th>
                <th>YOLOV8s</th>
            </tr>
            <tr>
                <td>Classes</td>
                <td>10 (all)</td>
                <td class="highlight">1 (car)</td>
            </tr>
            <tr>
                <td>mAP50</td>
                <td>0.267</td>
                <td class="highlight">0.878</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>0.409</td>
                <td class="highlight">0.869</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>0.292</td>
                <td class="highlight">0.815</td>
            </tr>
            <tr>
                <td>Inf Time</td>
                <td>2.2ms</td>
                <td>~5ms</td>
            </tr>
            <tr>
                <td>Model Size</td>
                <td>6.2MB</td>
                <td>21.5MB</td>
            </tr>
            <tr>
                <td>Epochs</td>
                <td>50</td>
                <td>80</td>
            </tr>
            <tr>
                <td>Image Size</td>
                <td>640</td>
                <td>1280</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">TRACKING COMPARISON</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="comp-table">
            <tr>
                <th>METRIC</th>
                <th>BASELINE</th>
                <th>MA-MOT</th>
            </tr>
            <tr>
                <td>Inf Time</td>
                <td>30.67ms</td>
                <td>153.62ms</td>
            </tr>
            <tr>
                <td>Avg FPS</td>
                <td>32.61</td>
                <td>6.51</td>
            </tr>
            <tr>
                <td>ID Switches</td>
                <td>86</td>
                <td class="highlight">84 ✓</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RUN INFERENCE
# ════════════════════════════════════════════════════════════════════════════════
elif "Inference" in page:

    st.markdown('<div class="section-header">RUN MA-MOT INFERENCE</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">INPUT SOURCE</div>', unsafe_allow_html=True)
        source = st.radio("", ["Upload Video", "Use Default Test Video"], label_visibility="collapsed")

        video_path = None
        if source == "Upload Video":
            uploaded = st.file_uploader("Upload drone video", type=['mp4', 'avi', 'mov'])
            if uploaded:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                video_path = tfile.name
                st.markdown('<span class="badge badge-green">VIDEO LOADED</span>', unsafe_allow_html=True)
        else:
            default_path = "videos/test.mp4"
            if os.path.exists(default_path):
                video_path = default_path
                st.markdown('<span class="badge badge-green">DEFAULT VIDEO READY</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-orange">videos/test.mp4 NOT FOUND</span>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">INFERENCE SETTINGS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <div style="margin-bottom:6px;">🎯 Confidence Threshold: <strong style="color:#00d4ff">{conf_threshold}</strong></div>
            <div style="margin-bottom:6px;">🔗 Metadata Fusion: <strong style="color:{'#00ff88' if use_metadata else '#ff6b35'}">{'ENABLED' if use_metadata else 'DISABLED'}</strong></div>
            <div style="margin-bottom:6px;">📡 Model: <strong style="color:#00d4ff">YOLOv8s (cars only)</strong></div>
            <div>🎬 FPS Overlay: <strong style="color:{'#00ff88' if show_fps else '#ff6b35'}">{'ON' if show_fps else 'OFF'}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀  LAUNCH INFERENCE PIPELINE"):
        if not os.path.exists(model_path):
            st.error("❌ Model not found at: " + model_path)
        elif video_path is None:
            st.error("❌ No video source selected")
        else:
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                metadata = load_metadata(meta_path) if use_metadata else {}
                tracker = MAMOTTracker()

                cap = cv2.VideoCapture(video_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps_in = cap.get(cv2.CAP_PROP_FPS)

                out_path = "outputs/dashboard_output.mp4"
                os.makedirs("outputs", exist_ok=True)
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (W, H))

                progress = st.progress(0)
                status = st.empty()
                col_fps, col_tracks, col_idsw = st.columns(3)
                fps_ph = col_fps.empty()
                trk_ph = col_tracks.empty()
                isw_ph = col_idsw.empty()
                frame_ph = st.empty()

                frame_idx, times, id_switches = 0, [], 0
                prev_ids, colors = set(), {}
                colors_pool = ['#00d4ff','#00ff88','#ff6b35','#ff4499','#ffdd00','#aa44ff']

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    t0 = time.time()
                    results = model(frame, conf=conf_threshold, verbose=False)[0]
                    dets = results.boxes.xyxy.cpu().numpy().tolist() if len(results.boxes) else []

                    meta_curr = metadata.get(frame_idx, {})
                    meta_prev = metadata.get(frame_idx-1, {})
                    dx, dy = compute_shift(meta_prev, meta_curr)
                    tracks = tracker.update(dets, dx, dy)

                    curr_ids = {tid for tid,_ in tracks}
                    id_switches += len(curr_ids - prev_ids - {t.id for t in tracker.trackers if t.hits==1})
                    prev_ids = curr_ids

                    inf_time = (time.time()-t0)*1000
                    times.append(inf_time)
                    fps_val = 1000/inf_time if inf_time > 0 else 0

                    for tid, (x1,y1,x2,y2) in tracks:
                        color_hex = colors_pool[tid % len(colors_pool)]
                        r,g,b = int(color_hex[1:3],16),int(color_hex[3:5],16),int(color_hex[5:7],16)
                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(b,g,r),2)
                        cv2.putText(frame,f"ID:{tid}",(int(x1),int(y1)-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(b,g,r),2)

                    if show_fps:
                        cv2.putText(frame,f"FPS:{fps_val:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,212,255),2)
                        cv2.putText(frame,f"Inf:{inf_time:.1f}ms",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,136),2)
                        cv2.putText(frame,f"Tracks:{len(tracks)}",(10,85),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,107,53),2)

                    out.write(frame)

                    if frame_idx % 10 == 0:
                        pct = frame_idx / max(total,1)
                        progress.progress(min(pct, 1.0))
                        status.markdown(f'<div class="info-box">⚡ Processing frame {frame_idx}/{total}</div>', unsafe_allow_html=True)
                        fps_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.5rem">{fps_val:.1f}</div><div class="metric-label">LIVE FPS</div></div>', unsafe_allow_html=True)
                        trk_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.5rem">{len(tracks)}</div><div class="metric-label">ACTIVE TRACKS</div></div>', unsafe_allow_html=True)
                        isw_ph.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.5rem">{id_switches}</div><div class="metric-label">ID SWITCHES</div></div>', unsafe_allow_html=True)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_ph.image(rgb, channels="RGB", use_container_width=True)

                    frame_idx += 1

                cap.release()
                out.release()
                progress.progress(1.0)

                avg_fps = 1000/np.mean(times)
                avg_inf = np.mean(times)

                st.markdown(f"""
                <div class="main-header" style="margin-top:20px;">
                    <h1 style="font-size:1.2rem; color: #00ff88;">✅ INFERENCE COMPLETE</h1>
                    <p>AVG FPS: {avg_fps:.2f} &nbsp;|&nbsp; AVG INF TIME: {avg_inf:.2f}ms &nbsp;|&nbsp; ID SWITCHES: {id_switches} &nbsp;|&nbsp; FRAMES: {frame_idx}</p>
                </div>
                """, unsafe_allow_html=True)

                if os.path.exists(out_path):
                    with open(out_path, 'rb') as f:
                        st.download_button("⬇️  DOWNLOAD OUTPUT VIDEO", f, "mamot_output.mp4", "video/mp4")

            except ImportError:
                st.error("❌ ultralytics not installed. Run: pip install ultralytics")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING RESULTS
# ════════════════════════════════════════════════════════════════════════════════
elif "Training" in page:

    st.markdown('<div class="section-header">TRAINING RESULTS — YOLOV8n (ALL CLASSES)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in zip([col1,col2,col3,col4],
                                ["0.267","0.409","0.292","2.2ms"],
                                ["mAP50","PRECISION","RECALL","INF TIME"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(make_training_chart(), use_container_width=True)

    st.markdown('<div class="section-header">PER-CLASS DETECTION (YOLOV8n)</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="comp-table">
        <tr><th>CLASS</th><th>mAP50</th><th>PRECISION</th><th>RECALL</th></tr>
        <tr><td>Car</td><td class="highlight">0.673</td><td>0.561</td><td>0.706</td></tr>
        <tr><td>Bus</td><td>0.379</td><td>0.513</td><td>0.371</td></tr>
        <tr><td>Motor</td><td>0.295</td><td>0.443</td><td>0.329</td></tr>
        <tr><td>Van</td><td>0.282</td><td>0.462</td><td>0.286</td></tr>
        <tr><td>Pedestrian</td><td>0.275</td><td>0.414</td><td>0.312</td></tr>
        <tr><td>Truck</td><td>0.230</td><td>0.413</td><td>0.253</td></tr>
        <tr><td>People</td><td>0.220</td><td>0.465</td><td>0.228</td></tr>
        <tr><td>Tricycle</td><td>0.189</td><td>0.386</td><td>0.230</td></tr>
        <tr><td>Bicycle</td><td>0.047</td><td>0.191</td><td>0.061</td></tr>
        <tr><td>Awning-tricycle</td><td>0.083</td><td>0.240</td><td>0.145</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<br><div class="section-header">YOLOV8s CARS ONLY — FINAL RESULTS</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in zip([col1,col2,col3,col4],
                                ["0.878","0.869","0.815","0.932"],
                                ["mAP50","PRECISION","RECALL","BOX LOSS"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem; color:#00ff88">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="margin-top:16px;">
        <strong style="color:#00ff88">229% improvement</strong> in mAP50 achieved by training YOLOv8s on
        cars-only subset of VisDrone (0.267 → 0.878). Single-class focused training
        with higher resolution input (1280×1280) and more epochs (80) drove the improvement.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif "Comparison" in page:

    st.markdown('<div class="section-header">BASELINE vs MA-MOT COMPARISON</div>', unsafe_allow_html=True)
    st.image(make_comparison_chart(), use_container_width=True)

    st.markdown('<div class="section-header">DETAILED METRICS</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="comp-table">
        <tr><th>METRIC</th><th>BASELINE</th><th>MA-MOT</th><th>DIFFERENCE</th></tr>
        <tr><td>Avg Inf Time</td><td>30.67ms</td><td>153.62ms</td><td style="color:#ff6b35">+122.95ms</td></tr>
        <tr><td>Average FPS</td><td>32.61</td><td>6.51</td><td style="color:#ff6b35">-26.1 FPS</td></tr>
        <tr><td>ID Switches</td><td>86</td><td class="highlight">84</td><td class="highlight">-2 (2.3% better)</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<br><div class="section-header">MODEL ARCHITECTURE COMPARISON</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="comp-table">
        <tr><th>PROPERTY</th><th>YOLOV8n</th><th>YOLOV8s</th></tr>
        <tr><td>Parameters</td><td>3,012,798</td><td class="highlight">11,136,374</td></tr>
        <tr><td>GFLOPs</td><td>8.2</td><td>28.6</td></tr>
        <tr><td>mAP50</td><td>0.267</td><td class="highlight">0.878</td></tr>
        <tr><td>Model Size</td><td>6.2MB</td><td>21.5MB</td></tr>
        <tr><td>Training Time</td><td>2.8 hrs</td><td>3.5 hrs</td></tr>
        <tr><td>Platform</td><td colspan="2" style="text-align:center">Google Colab Tesla T4 GPU</td></tr>
    </table>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ════════════════════════════════════════════════════════════════════════════════
elif "About" in page:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">PROJECT INFO</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <div style="font-family: Orbitron, monospace; font-size: 0.85rem;
                        color: #00d4ff; margin-bottom: 12px; letter-spacing: 2px;">
                SUBMITTED BY
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: white;">Shivesh Bansal</div>
            <div style="color: #5a7a9a; font-family: Share Tech Mono, monospace; font-size:0.85rem;">Roll: 1024240047</div>
            <div style="color: #5a7a9a; margin-top: 8px;">Department of Computer Science and Engineering</div>
            <div style="color: #5a7a9a;">Thapar Institute of Engineering & Technology</div>
            <div style="color: #5a7a9a;">Patiala · April 2026</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">TECH STACK</div>', unsafe_allow_html=True)
        techs = [
            ("YOLOv8s", "Object Detection", "badge-blue"),
            ("Kalman Filter", "Object Tracking", "badge-green"),
            ("Hungarian Algorithm", "Data Association", "badge-green"),
            ("OpenCV 4.x", "Vision Processing", "badge-blue"),
            ("PyTorch + CUDA", "Deep Learning", "badge-orange"),
            ("NumPy + SciPy", "Metadata Simulation", "badge-blue"),
            ("Google Colab T4", "Training Platform", "badge-orange"),
            ("Streamlit", "Dashboard", "badge-green"),
        ]
        for name, role, badge in techs:
            st.markdown(f"""
            <div class="info-box" style="padding: 8px 14px; margin: 4px 0; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:600">{name}</span>
                <span><span class="badge {badge}">{role}</span></span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">GITHUB REPOSITORY</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <div style="font-family: Share Tech Mono, monospace; color: #00d4ff; margin-bottom:8px;">
                🔗 github.com/SHIVESHBANSAL07/UAV-MAMOT
            </div>
            <div style="color: #5a7a9a; font-size:0.9rem;">
                Complete source code, trained weights, metadata, and documentation
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">DATASET INFO</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="comp-table">
            <tr><th>SPLIT</th><th>IMAGES</th><th>NOTES</th></tr>
            <tr><td>Train (all)</td><td>6,471</td><td>YOLOv8n</td></tr>
            <tr><td>Validation</td><td>548</td><td>mAP eval</td></tr>
            <tr><td>Cars Only Train</td><td>5,151</td><td>YOLOv8s</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.markdown('<br><div class="section-header">REFERENCES</div>', unsafe_allow_html=True)
        refs = [
            "Ultralytics YOLOv8, 2024",
            "Bewley et al. — SORT, IEEE ICIP 2016",
            "Wojke et al. — DeepSORT, IEEE ICIP 2017",
            "Zhu et al. — VisDrone 2018, ECCV",
            "Zhang et al. — ByteTrack, ECCV 2022",
        ]
        for ref in refs:
            st.markdown(f'<div class="info-box" style="padding:8px 14px; margin:4px 0; font-size:0.85rem; color:#5a7a9a;">{ref}</div>', unsafe_allow_html=True)
            # ════════════════════════════════════════════════════════════════════════════════
# PAGE: LOGS
# ════════════════════════════════════════════════════════════════════════════════
elif "Logs" in page:

    st.markdown('<div class="section-header">INFERENCE LOG VIEWER</div>', unsafe_allow_html=True)

    log_path = "logs/inference.log"

    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.toggle("Auto Refresh", value=False)
        if st.button("🔄  REFRESH LOG"):
            st.rerun()

    if not os.path.exists(log_path):
        st.markdown("""
        <div class="info-box" style="border-left-color: #ff6b35;">
            ⚠️ No log file found at logs/inference.log<br>
            <span style="color:#5a7a9a; font-size:0.85rem;">
            Run inference first using the Run Inference page or python main.py
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Stats from log
        info_count  = sum(1 for l in lines if '[INFO]' in l)
        error_count = sum(1 for l in lines if '[ERROR]' in l)
        warn_count  = sum(1 for l in lines if '[WARNING]' in l)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem">{len(lines)}</div><div class="metric-label">TOTAL LINES</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem; color:#00ff88">{info_count}</div><div class="metric-label">INFO</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem; color:#ffdd00">{warn_count}</div><div class="metric-label">WARNINGS</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.8rem; color:#ff4444">{error_count}</div><div class="metric-label">ERRORS</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Extract key results from log
        results = [l for l in lines if 'FINAL RESULTS' in l or 'Avg Inference' in l
                   or 'Average FPS' in l or 'ID Switches' in l or 'Output saved' in l]
        if results:
            st.markdown('<div class="section-header">LATEST INFERENCE RESULTS</div>', unsafe_allow_html=True)
            result_html = ""
            for line in results:
                line = line.strip()
                color = "#00ff88" if "FINAL" in line else "#00d4ff"
                result_html += f'<div style="font-family: Share Tech Mono, monospace; font-size: 0.85rem; color: {color}; padding: 4px 0; border-bottom: 1px solid #1e3a5f;">{line}</div>'
            st.markdown(f'<div class="info-box">{result_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter options
        st.markdown('<div class="section-header">FULL LOG</div>', unsafe_allow_html=True)
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search = st.text_input("🔍 Search log", placeholder="Type to filter...", label_visibility="collapsed")
        with col_f2:
            level_filter = st.selectbox("Level", ["ALL", "INFO", "ERROR", "WARNING"], label_visibility="collapsed")

        # Filter lines
        filtered = lines
        if search:
            filtered = [l for l in filtered if search.lower() in l.lower()]
        if level_filter != "ALL":
            filtered = [l for l in filtered if f'[{level_filter}]' in l]

        # Display log with color coding
        log_html = ""
        for line in filtered[-200:]:  # show last 200 lines
            line = line.strip()
            if '[ERROR]' in line:
                color = "#ff4444"
            elif '[WARNING]' in line:
                color = "#ffdd00"
            elif 'FINAL RESULTS' in line or 'complete' in line.lower():
                color = "#00ff88"
            elif '[INFO]' in line:
                color = "#c8d8e8"
            else:
                color = "#5a7a9a"
            log_html += f'<div style="font-family: Share Tech Mono, monospace; font-size: 0.78rem; color: {color}; padding: 2px 0; border-bottom: 1px solid rgba(30,58,95,0.3);">{line}</div>'

        st.markdown(f"""
        <div style="background: #0a0e1a; border: 1px solid #1e3a5f; border-radius: 8px;
                    padding: 16px; max-height: 450px; overflow-y: auto;
                    font-family: Share Tech Mono, monospace;">
            {log_html}
        </div>
        """, unsafe_allow_html=True)

        # Download log button
        st.markdown("<br>", unsafe_allow_html=True)
        with open(log_path, 'rb') as f:
            st.download_button(
                "⬇️  DOWNLOAD LOG FILE",
                f,
                "inference.log",
                "text/plain"
            )

        # Auto refresh
        if auto_refresh:
            time.sleep(3)
            st.rerun()