"""
Configuration settings and constants for the YOLO11 Live Demo
"""

import torch

# -- Device Configuration ------------------------------------------------
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# -- Model Configuration ------------------------------------------------
MODEL_SIZES = ['n', 's', 'm', 'l', 'x']
TASKS = ['detect', 'segment', 'classify', 'pose', 'obb', 'track']

# -- Visualization Configuration -----------------------------------------
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

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
        'color': (0, 255, 0)  # green
    },
    'right_arm': {
        'connections': [(6,8), (8,10)],  # right shoulder to elbow to wrist
        'color': (255, 0, 0)  # blue
    },
    'left_arm': {
        'connections': [(5,7), (7,9)],  # left shoulder to elbow to wrist
        'color': (0, 0, 255)  # red
    },
    'right_leg': {
        'connections': [(12,14), (14,16)],  # right hip to knee to ankle
        'color': (255, 128, 0)  # purple
    },
    'left_leg': {
        'connections': [(11,13), (13,15)],  # left hip to knee to ankle
        'color': (0, 128, 255)  # orange
    }
} 