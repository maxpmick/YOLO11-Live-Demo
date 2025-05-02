"""
Model loading and inference functionality for YOLO11 Live Demo
"""

import streamlit as st
from ultralytics import YOLO
import os
import torch
import shutil

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
    
    model_name = f"yolo11{size}{suffix_map[task]}.pt"
    weights_path = os.path.join('models', model_name)
    
    # First check if model exists in models directory
    if os.path.exists(weights_path):
        return True
        
    # If not, ensure models directory exists and download
    if not os.path.exists('models'):
        os.makedirs('models')
        
    st.info(f"Downloading {model_name}...")
    try:
        model = YOLO(model_name)
        shutil.copy2(model.ckpt_path, weights_path)
        st.success(f"Downloaded {model_name} to models directory")
        return True
    except Exception as e:
        st.error(f"Failed to download {model_name}: {str(e)}")
        return False

@st.cache_resource
def load_model(task: str, size: str, device: str = 'cpu'):
    """Load a YOLO model with caching."""
    suffix_map = {
        'detect': '',
        'segment': '-seg',
        'classify': '-cls',
        'pose': '-pose',
        'obb': '-obb',
        'track': ''
    }
    
    model_name = f"yolo11{size}{suffix_map[task]}.pt"
    weights_path = os.path.join('models', model_name)
    
    # Check if model exists first
    if not os.path.exists(weights_path):
        if not download_model(task, size):
            return None
    
    try:
        model = YOLO(weights_path)
        model.to(device)
        model.fuse()
        model.model.eval()
        if device in ['mps', 'cuda']:
            model.model.half()  # Use half precision for MPS and CUDA
        st.success(f"Model loaded successfully on {device}")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def run_inference(model, frame, task, device='cpu', **kwargs):
    """Run model inference with appropriate settings."""
    if model is None:
        st.error("Model not loaded. Cannot run inference.")
        return []
        
    # Use CPU for pose detection due to MPS issues
    curr_dev = 'mps' if task == 'pose' else device
    
    inference_kwargs = {
        'source': frame,
        'device': curr_dev,
        'half': (curr_dev in ['mps', 'cuda']),
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