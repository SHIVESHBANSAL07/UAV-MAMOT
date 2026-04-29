import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model
MODEL_PATH     = os.path.join(BASE_DIR, 'models/maritime_v1/weights/best.pt')
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
CLASSES=[0]

# Input
INPUT_SOURCE = 'video'
INPUT_PATH   = os.path.join(BASE_DIR, 'videos/test.mp4')

# Output
OUTPUT_DIR   = os.path.join(BASE_DIR, 'outputs')
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, 'tracked_output.mp4')
SAVE_OUTPUT  = True

# Metadata
METADATA_PATH = os.path.join(BASE_DIR, 'metadata/drone_meta.json')
USE_METADATA  = True

# Tracker
MAX_TRACK_AGE = 10
MIN_HITS      = 2

# Logging
LOG_DIR  = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'inference.log')

# Image
IMG_SIZE = 640
DEVICE   = 'cpu'  # change to '0' if you have GPU