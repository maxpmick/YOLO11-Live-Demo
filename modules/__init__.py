"""
YOLO11 Live Demo module initialization
"""

from .config import *
from .model import load_model, run_inference
from .visualization import process_results
from .tracking import TrackingManager 