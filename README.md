# YOLO11 Live Demo - M4 Optimized

A real-time object detection and tracking application using YOLO11 models, optimized for Apple Silicon (M4) and other platforms. This application supports multiple computer vision tasks including detection, tracking, segmentation, pose estimation, and oriented bounding boxes.

## Features

- 🎯 Real-time object detection
- 🎭 Instance segmentation
- 👤 Pose estimation
- 📦 Oriented bounding box detection
- 🎯 Object tracking with motion trails
- 🚀 Optimized for Apple Silicon
- 📊 Performance benchmarking
- 🎨 Beautiful Streamlit UI

## Requirements

- Python 3.8 or later
- macOS, Linux, or Windows
- Webcam
- For Apple Silicon Macs: macOS 12.0+
- For NVIDIA GPUs: CUDA 11.7+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/maxpmick/YOLO11-Live-Demo.git
cd YOLO11-Live-Demo
```

2. Make the setup script executable:
```bash
chmod +x setup.sh
```

3. Run the setup script:
```bash
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all required dependencies
- Create necessary directories
- Configure Streamlit settings

## Usage

1. Activate the virtual environment:
```bash
source yolo_env/bin/activate
```

2. Run the application:
```bash
streamlit run aio.py
```

3. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

4. Accept the webcam permission request from your OS, you may have to restart the program and run it again for the camera feed to become available.

## Interface Guide

### Sidebar Controls
- **Model Size**: Choose from 'n' (nano) to 'x' (extra large)
- **Task Selection**: Enable/disable different detection tasks
  - Detection
  - Segmentation
  - Classification
  - Pose Estimation
  - Oriented Bounding Boxes
  - Tracking
- **Benchmark**: Test model performance
- **Start/Stop**: Control webcam feed

### Main Display
- Live webcam feed with detection overlays
- Performance metrics (FPS, processing time)
- Debug information

## Features in Detail

### Object Detection
- Real-time bounding box detection
- Class labels and confidence scores
- Multiple object categories

### Tracking
- Object persistence across frames
- Colored motion trails (1.5 second history)
- Unique ID assignment

### Pose Estimation
- 17-point keypoint detection
- Skeletal visualization
- Color-coded body parts

### Segmentation
- Instance segmentation masks
- Transparent overlays
- Color-coded instances

### Oriented Bounding Boxes
- Rotated bounding boxes
- Precise object orientation
- Ideal for aerial imagery

## Performance

Performance varies based on:
- Selected model size
- Enabled tasks
- Hardware capabilities
- Input resolution

Typical performance on M4:
- Nano model: 30-60 FPS
- Base model: 20-40 FPS
- Large model: 10-20 FPS

## Troubleshooting

### Common Issues

1. **Webcam Access**
   - Ensure your webcam is connected
   - Grant necessary permissions
   - Check if other applications are using the camera

2. **Performance**
   - Reduce model size for better FPS
   - Disable unused tasks
   - Check CPU/GPU utilization

3. **Model Loading**
   - Ensure internet connection for first run
   - Check models directory permissions
   - Verify Python environment activation

### Error Messages

If you encounter errors:
1. Check the debug information in the UI
2. Verify virtual environment activation
3. Ensure all dependencies are installed
4. Check system requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [Streamlit](https://streamlit.io/) for the UI framework
- The computer vision community 