"""
Visualization functions for YOLO11 Live Demo
"""

import cv2
import numpy as np
import streamlit as st
from .config import PALETTE, SKELETON_PARTS

def draw_oriented_box(img, points, color, label=None):
    """Draw an oriented bounding box with its label."""
    points = points.astype(np.int32)
    # Draw the box
    cv2.polylines(img, [points], True, color, 2)
    
    if label:
        # Find top-left most point for label placement
        x = min(points[:, 0])
        y = min(points[:, 1])
        cv2.putText(img, label, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_detection_box(img, box, conf, cls, color=(0,255,0)):
    """Draw a standard detection box with label."""
    try:
        x1, y1, x2, y2 = box.astype(int)
        label = f"{cls}:{conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        st.error(f"Error drawing detection box: {str(e)}")

def draw_motion_trail(img, points, color, current_time, trail_duration):
    """Draw motion trail for tracked objects."""
    try:
        for j in range(1, len(points)):
            alpha = 0.7 * (1 - (current_time - points[j][2]) / trail_duration)
            if alpha > 0:
                pt1 = (int(points[j-1][0]), int(points[j-1][1]))
                pt2 = (int(points[j][0]), int(points[j][1]))
                cv2.line(img, pt1, pt2, 
                        tuple(int(c * alpha) for c in color), 2)
                cv2.circle(img, pt2, 2, 
                          tuple(int(c * alpha) for c in color), -1)
    except Exception as e:
        st.error(f"Error drawing motion trail: {str(e)}")

def draw_segmentation_mask(img, mask, color):
    """Draw segmentation mask with transparency."""
    try:
        color_mask = np.zeros_like(img, dtype=np.uint8)
        color_mask[mask == 1] = color
        return cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
    except Exception as e:
        st.error(f"Error drawing segmentation mask: {str(e)}")
        return img

def draw_pose_keypoints(img, person_keypoints):
    """Draw pose keypoints and skeleton."""
    try:
        # Draw joints
        for idx, (x, y, c) in enumerate(person_keypoints):
            if c > 0.3:
                cv2.circle(img, (int(x), int(y)), 4, (255,255,255), -1)
        
        # Draw bones by body part
        for part_name, part_info in SKELETON_PARTS.items():
            for a, b in part_info['connections']:
                xa, ya, ca = person_keypoints[a]
                xb, yb, cb = person_keypoints[b]
                if ca > 0.3 and cb > 0.3:
                    cv2.line(img,
                            (int(xa), int(ya)),
                            (int(xb), int(yb)),
                            part_info['color'], 2)
    except Exception as e:
        st.error(f"Error drawing pose keypoints: {str(e)}")

def process_results(img, results, task, track_colors={}, position_history={}, 
                   current_time=None, trail_duration=None):
    """Process and visualize detection results."""
    if not results:
        return img
        
    overlay = img.copy()
    
    try:
        for r in results:
            # Standard Boxes and Tracking
            if hasattr(r, 'boxes') and r.boxes is not None and not hasattr(r, 'obb'):
                boxes = r.boxes
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                
                # Get track IDs if available
                track_ids = None
                if hasattr(boxes, 'id'):
                    track_ids = boxes.id.cpu().numpy()
                
                for i, (xyxy, conf, cls) in enumerate(zip(xyxys, confs, clss)):
                    # Get tracking ID and color
                    track_id = int(track_ids[i]) if track_ids is not None else None
                    color = track_colors.get(track_id, (0,255,0)) if track_id else (0,255,0)
                    
                    # Draw box and label
                    draw_detection_box(overlay, xyxy, conf, int(cls), color)
                    
                    # Draw motion trail if tracking
                    if track_id and current_time and trail_duration:
                        if track_id in position_history:
                            draw_motion_trail(overlay, position_history[track_id], 
                                           color, current_time, trail_duration)
            
            # Oriented Bounding Boxes
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb
                points = boxes.xyxyxyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                for pts, conf, cls in zip(points, confs, classes):
                    pts = pts.reshape(-1, 2)
                    label = f"{int(cls)}:{conf:.2f}"
                    color = PALETTE[int(cls) % len(PALETTE)]
                    draw_oriented_box(overlay, pts, color, label)
            
            # Segmentation Masks
            if hasattr(r, 'masks') and r.masks is not None:
                masks = r.masks.data.cpu().numpy()
                for idx, mask in enumerate(masks):
                    try:
                        tid = int(r.boxes.id.cpu().numpy()[idx])
                    except:
                        tid = idx
                    if tid not in track_colors:
                        track_colors[tid] = PALETTE[len(track_colors) % len(PALETTE)]
                    color = track_colors[tid]
                    
                    m = cv2.resize(mask.astype('uint8'),
                                 (overlay.shape[1], overlay.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
                    overlay = draw_segmentation_mask(overlay, m, color)
            
            # Pose Keypoints
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kp = getattr(r.keypoints, 'data', r.keypoints)
                pts = kp.cpu().numpy()
                people = pts if pts.ndim==3 else [pts] if pts.ndim==2 else []
                for person in people:
                    draw_pose_keypoints(overlay, person)
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        return img
        
    return overlay 