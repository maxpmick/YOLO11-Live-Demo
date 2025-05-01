"""
Object tracking functionality for YOLO11 Live Demo
"""

import time
from .config import TRAIL_DURATION, TRAIL_MAX_POINTS, PALETTE

class TrackingManager:
    def __init__(self):
        self.position_history = {}  # {track_id: [(x, y, timestamp), ...]}
        self.track_colors = {}      # {track_id: (r,g,b)}
    
    def update_track(self, track_id, center_x, center_y):
        """Update position history for a tracked object."""
        current_time = time.time()
        
        if track_id not in self.position_history:
            self.position_history[track_id] = []
            self.track_colors[track_id] = PALETTE[len(self.track_colors) % len(PALETTE)]
        
        self.position_history[track_id].append((center_x, center_y, current_time))
        
        # Remove old points beyond TRAIL_DURATION
        self.position_history[track_id] = [
            p for p in self.position_history[track_id] 
            if current_time - p[2] <= TRAIL_DURATION
        ]
        
        # Limit number of points
        if len(self.position_history[track_id]) > TRAIL_MAX_POINTS:
            self.position_history[track_id] = self.position_history[track_id][-TRAIL_MAX_POINTS:]
    
    def cleanup_old_tracks(self, current_tracks):
        """Remove tracks that are no longer active."""
        current_time = time.time()
        for track_id in list(self.position_history.keys()):
            if track_id not in current_tracks:
                last_update = self.position_history[track_id][-1][2]
                if current_time - last_update > TRAIL_DURATION:
                    del self.position_history[track_id]
                    if track_id in self.track_colors:
                        del self.track_colors[track_id]
    
    def get_track_color(self, track_id):
        """Get color for a track ID."""
        if track_id not in self.track_colors:
            self.track_colors[track_id] = PALETTE[len(self.track_colors) % len(PALETTE)]
        return self.track_colors[track_id] 