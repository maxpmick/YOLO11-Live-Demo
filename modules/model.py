"""
Model loading and inference functionality for YOLO11 Live Demo
"""

import streamlit as st
from ultralytics import YOLO
from .config import DEVICE
import os
import torch

def download_model(task: str, size: str):
    """Download YOLO model weights if they don't exist."""
    suffix_map = {
        'detect': '',
        'segment': '-seg',
        'classify': '-cls',
        'pose': '-pose',
        'obb': '-obb',
        'track': ''
    }
    
    model_name = f"yolov8{size}{suffix_map[task]}.pt"
    weights_path = f"models/yolo11{size}{suffix_map[task]}.pt"
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    if not os.path.exists(weights_path):
        st.info(f"Downloading {model_name}...")
        try:
            model = YOLO(model_name)
            model.export(format="pt", imgsz=640)
            st.success(f"Downloaded {model_name} successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to download {model_name}: {str(e)}")
            return False
    return True

@st.cache_resource
def load_model(task: str, size: str):
    """Load a YOLO model with caching."""
    suffix_map = {
        'detect': '',
        'segment': '-seg',
        'classify': '-cls',
        'pose': '-pose',
        'obb': '-obb',
        'track': ''
    }
    
    # Try to download model if it doesn't exist
    if not download_model(task, size):
        return None
        
    weights = f"models/yolo11{size}{suffix_map[task]}.pt"
    
    try:
        model = YOLO(weights)
        model.to(DEVICE)
        model.fuse()
        model.model.eval()
        if DEVICE == 'mps':
            model.model.half()
        st.success(f"Model loaded successfully on {DEVICE}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def run_inference(model, frame, task, **kwargs):
    """Run model inference with appropriate settings."""
    if model is None:
        st.error("Model not loaded. Cannot run inference.")
        return []
        
    # Use CPU for pose detection due to MPS issues
    curr_dev = 'cpu' if task == 'pose' else DEVICE
    
    inference_kwargs = {
        'source': frame,
        'device': curr_dev,
        'half': (curr_dev=='mps'),
        'verbose': False,
        'conf': 0.25,  # Set confidence threshold
        'iou': 0.45,   # Set IoU threshold
        **kwargs
    }
    
    try:
        if task == 'segment' or (task == 'track' and kwargs.get('segment', False)):
            results = model.track(**inference_kwargs, tracker='bytetrack.yaml', task='segment')
        elif task == 'track':
            results = model.track(**inference_kwargs, tracker='bytetrack.yaml')
        else:
            if task != 'detect':
                inference_kwargs['task'] = task
            results = model.predict(**inference_kwargs)
            
        if not results:
            st.warning(f"No detections for {task}")
        return results
    except Exception as e:
        st.error(f"Inference error ({task}): {str(e)}")
        return [] 