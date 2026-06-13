# 👥 People Counter - Real-Time Person Tracking with YOLOv8

A real-time people counting system that tracks individuals entering or exiting a space using YOLOv8 object detection and ByteTrack algorithm.

## 🎯 What It Does

- **Detects** people in real-time from webcam feed
- **Tracks** each person individually across frames
- **Counts** how many people go IN and OUT across a virtual line
- **Displays** live bounding boxes, tracking IDs, and counters

## 🚀 How It Works

Webcam → YOLOv8 Detection → ByteTrack Tracking → Line Crossing Logic → IN/OUT Counts

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8n (nano version) |
| Tracking | ByteTrack algorithm |
| Computer Vision | OpenCV |
| Framework | Ultralytics YOLO |

## 📋 Requirements

```bash
pip install ultralytics opencv-python

🏃 How to Run
Clone the repository

bash
git clone https://github.com/yosefdev798/SpotBox.git
cd SpotBox
Install dependencies

bash
pip install -r requirements.txt
Run the people counter

bash
python people_counter.py
Press q to quit the application

🎮 Features
Feature	Description
Real-time Processing	Processes webcam feed at 30+ FPS
Person Tracking	Unique ID assigned to each person
Line Crossing	Counts when center point crosses green line
Visual Feedback	Bounding boxes, center dots, and counters on screen
Confidence Filter	Only shows detections above 50% confidence
📸 Demo
text
┌─────────────────────────────────┐
│  IN: 5    OUT: 3                │
│                                 │
│      ┌─────┐                    │
│      │ 🧑  │  ─────── Green Line│
│      └─────┘                    │
│                                 │
│  [Bounding Box] [Center Dot]    │
└─────────────────────────────────┘
🔧 Configuration
You can adjust these parameters in the code:

Parameter	Default	Description
line_y	300	Y-position of counting line
conf	0.5	Minimum confidence threshold
classes	[0]	Only detect people (class 0)
alpha	1.2	Image brightness adjustment
📁 Project Structure
text
SpotBox/
├── people_counter.py   # Main application
├── yolov8n.pt          # YOLO model weights
└── README.md           # This file
🎯 Use Cases
Retail store footfall counting

Office occupancy monitoring

Gym or facility usage tracking

Event entrance/exit management

Social distancing monitoring

🧠 What I Learned
Real-time object detection with YOLOv8

Multi-object tracking algorithms (ByteTrack)

Line-crossing logic for counting systems

OpenCV video processing and annotations

Optimizing inference for live webcam feed

🔮 Future Improvements
Save daily counts to CSV file

Add a reset button for counters

Display counts on separate HTML dashboard

Send alerts when capacity is reached

Support for IP cameras and video files

📄 License
This project is for educational purposes.

👨‍💻 Author
Yosef Dev - GitHub

Built with YOLOv8, OpenCV, and Python

text

---

## Steps to add it:

1. Go to `https://github.com/yosefdev798/SpotBox`
2. Click **"Add file"** → **"Create new file"**
3. Name the file: `README.md`
4. **Copy and paste ONLY the code above**
5. Click **"Commit new file"**

That's it! ✅



