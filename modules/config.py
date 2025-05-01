"""
Configuration settings and constants for the YOLO11 Live Demo
"""

import torch
import numpy as np

# -- Device Configuration ------------------------------------------------
def get_available_devices():
    """Get list of available compute devices."""
    devices = ['cpu']
    
    # Check for CUDA
    if torch.cuda.is_available():
        devices.append('cuda')
        
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append('mps')
    
    return devices

# Default to best available device
AVAILABLE_DEVICES = get_available_devices()
if 'cuda' in AVAILABLE_DEVICES:
    DEFAULT_DEVICE = 'cuda'
elif 'mps' in AVAILABLE_DEVICES:
    DEFAULT_DEVICE = 'mps'
else:
    DEFAULT_DEVICE = 'cpu'

# -- Model Configuration ------------------------------------------------
MODEL_SIZES = ['n', 's', 'm', 'l', 'x']
TASKS = ['detect', 'segment', 'classify', 'pose', 'obb', 'track']

# -- Visualization Configuration -----------------------------------------
PALETTE = np.array([
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Green (dark)
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
], dtype=np.uint8)

# -- Motion Trail Configuration -----------------------------------------
TRAIL_DURATION = 1.5  # seconds
TRAIL_MAX_POINTS = 30  # maximum number of points to show in trail

# -- Pose Configuration ------------------------------------------------
POSE_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# -- Skeleton Configuration --------------------------------------------
SKELETON_PARTS = {
    'face': {
        'connections': [(0,1), (0,2), (1,3), (2,4)],  # nose to eyes to ears
        'color': (255, 192, 0)  # cyan
    },
    'torso': {
        'connections': [(5,6), (5,11), (6,12), (11,12)],  # shoulders and hips
        'color': (255, 0, 0)  # green
    },
    'right_arm': {
        'connections': [(6,8), (8,10)],  # right shoulder to elbow to wrist
        'color': (0, 255, 0)  # blue
    },
    'left_arm': {
        'connections': [(5,7), (7,9)],  # left shoulder to elbow to wrist
        'color': (0, 0, 255)  # red
    },
    'right_leg': {
        'connections': [(12,14), (14,16)],  # right hip to knee to ankle
        'color': (255, 255, 0)  # purple
    },
    'left_leg': {
        'connections': [(11,13), (13,15)],  # left hip to knee to ankle
        'color': (255, 0, 255)  # orange
    }
}

# COCO class names mapping
CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
} 