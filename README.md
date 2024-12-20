# Parking Lot Occupancy Tracker

## Overview
This project uses computer vision and object detection to track and count the number of cars in a designated parking area. It leverages YOLOv5 for real-time object detection and OpenCV for video processing.

## Features
- Real-time car detection in a predefined parking area
- Visualization of parking area boundaries
- Car count display on video frame
- Video output of processed frames

## Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- NumPy

## Installation
1. Clone the repository:
```bash
git clone https://github.com/AdbulrhmanEldeeb/parking_managment
cd parking_managment
```

2. Install required dependencies:
```bash
pip -r requirements.txt
```

## Usage
1. Prepare your input video:
   - Place your parking lot video in the `videos/` directory
   - Rename or update the video path in `manage.py`

2. Run the script:
```bash
python manage.py
```

## Configuration
- Modify `parking_area` coordinates in `manage.py` to match your specific parking lot layout
- Adjust video input/output settings as needed

## Output
- Processed video will be saved as `videos/output.avi`
- Real-time display shows:
  - Parking area boundary
  - Detected cars within the area
  - Current car count

## Troubleshooting
- Ensure all dependencies are correctly installed
- Check video file path and format
- Verify parking area coordinates match your specific use case

## License
MIT License

## Contributing
Contributions are welcome! Please submit pull requests or open issues to improve the project.

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics/yolov5) for YOLOv5
- [OpenCV](https://opencv.org/) for computer vision tools
