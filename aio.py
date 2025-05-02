#!/usr/bin/env python3
import os
# Disable Streamlit's file-watcher to avoid torch.classes errors on macOS
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import cv2
import torch
import time
import pandas as pd
from ultralytics.utils.benchmarks import benchmark

# Import our modules after Streamlit configuration
from modules import (
    DEFAULT_DEVICE, MODEL_SIZES, TASKS, AVAILABLE_DEVICES,
    DEFAULT_CONF_THRESH, DEFAULT_IOU_THRESH,
    load_model, run_inference, process_results,
    TrackingManager, CLASSES
)

# -- Streamlit UI Setup --------------------------------------------------
st.set_page_config(layout="wide")
st.title("YOLO11 Live Demo")

# Create two columns for the main layout
col1, col2 = st.columns([2, 1])

with col1:
    frame_disp = st.empty()
    perf_disp = st.empty()

with col2:
    st.subheader("Detected Objects")
    table_disp = st.empty()
    debug_disp = st.empty()

# Sidebar controls
st.sidebar.header("Settings")

# Device selector
device = st.sidebar.selectbox(
    "Compute Device",
    options=AVAILABLE_DEVICES,
    index=AVAILABLE_DEVICES.index(DEFAULT_DEVICE),
    help="Select the compute device for inference"
)

st.sidebar.write(f"**Device:** {device.upper()} | Torch {torch.__version__}")

# Model settings
size = st.sidebar.selectbox("Model size", MODEL_SIZES, index=0)
options = {t: st.sidebar.checkbox(t.capitalize(), value=(t=='detect')) for t in TASKS}

# Detection settings
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_CONF_THRESH,
    step=0.05,
    help="Minimum confidence score for detections"
)

benchmark_enabled = st.sidebar.checkbox("Startup benchmark", False)
start, stop = st.sidebar.button("Start"), st.sidebar.button("Stop")

# -- Benchmark ---------------------------------------------------------
if benchmark_enabled:
    st.sidebar.write(f"Benchmarking detect/{size} on {device}…")
    bm = load_model('detect', size, device)
    if bm is not None:
        stats = benchmark(model=bm, data="coco8.yaml", imgsz=640,
                         device=device, half=(device in ['mps', 'cuda']))
        st.sidebar.write(stats)

# -- Main loop --------------------------------------------------------
if start:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("Cannot open webcam.")
        st.stop()

    # Initialize models and tracking
    active_tasks = [task for task, enabled in options.items() if enabled]
    if not active_tasks:
        st.warning("Please select at least one task.")
        st.stop()
        
    debug_disp.write(f"Active tasks: {', '.join(active_tasks)}")
    
    models = {}
    for task in active_tasks:
        try:
            models[task] = load_model(task, size, device)
            if models[task] is None:
                st.error(f"Failed to load model for {task}")
                continue
            debug_disp.write(f"Loaded model for {task}")
        except Exception as e:
            st.error(f"Error loading model for {task}: {str(e)}")
            continue

    if not models:
        st.error("No models loaded successfully.")
        st.stop()

    tracking = TrackingManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Frame capture failed.")
            break
        
        # Mirror the camera feed horizontally
        frame = cv2.flip(frame, 1)
        t0 = time.perf_counter()

        # Initialize detection data for table
        detection_data = []

        with torch.no_grad():
            for task in active_tasks:
                if task not in models:
                    continue
                
                # Run inference with confidence threshold
                results = run_inference(models[task], frame, task, device=device, conf=conf_thresh)
                
                if not results:  # Skip if no results
                    continue
                
                # Update tracking for tracked objects
                if task == 'track' and hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        
                        # Get track IDs if available
                        track_ids = None
                        if hasattr(boxes, 'id') and boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy()
                            
                            for i, (box, conf, cls) in enumerate(zip(xyxy, confs, clss)):
                                if conf < conf_thresh:  # Skip low confidence detections
                                    continue
                                    
                                track_id = int(track_ids[i])
                                x1, y1, x2, y2 = box.astype(int)
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                tracking.update_track(track_id, center_x, center_y)
                                
                                # Add to detection data
                                detection_data.append({
                                    'Task': 'Tracking',
                                    'ID': str(track_id),
                                    'Class': CLASSES.get(int(cls), f"class_{int(cls)}"),
                                    'Confidence': f"{conf:.2f}",
                                    'Position': f"({center_x}, {center_y})"
                                })
                            
                            # Cleanup old tracks
                            tracking.cleanup_old_tracks(set(track_ids))
                
                # Process and visualize results with confidence threshold
                frame = process_results(
                    frame, results, task,
                    track_colors=tracking.track_colors,
                    position_history=tracking.position_history,
                    current_time=time.time(),
                    trail_duration=1.5,
                    conf_thresh=conf_thresh
                )

                # Add detection data for other tasks
                if hasattr(results[0], 'boxes') and not task == 'track':
                    boxes = results[0].boxes
                    if hasattr(boxes, 'xyxy'):
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(xyxy, confs, clss):
                            x1, y1, x2, y2 = box.astype(int)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            detection_data.append({
                                'Task': task.capitalize(),
                                'ID': '-',
                                'Class': CLASSES.get(int(cls), f"class_{int(cls)}"),
                                'Confidence': f"{conf:.2f}",
                                'Position': f"({center_x}, {center_y})"
                            })

        # Update the table with detection data
        if detection_data:
            df = pd.DataFrame(detection_data)
            table_disp.dataframe(df, use_container_width=True)
        else:
            table_disp.write("No objects detected")

        dt = time.perf_counter() - t0
        fps = 1/dt if dt>0 else 0
        perf_disp.write(f"{device.upper()} — {dt*1000:.1f} ms/frame — {fps:.1f} FPS")
        frame_disp.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True)

        if stop:
            break

    cap.release()
