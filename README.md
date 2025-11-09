## ⚽ Player and Ball Tracking in Football using Computer Vision

### 📖 Overview

This project focuses on **automated player and ball tracking in football matches** using **computer vision** and **deep learning**. By combining YOLOv8 for object detection and ByteTrack for object tracking, the system identifies and follows each player and the football across video frames. This framework can serve as a foundation for performance analytics, tactical insights, and sports broadcasting automation.

---

### 🎯 Objectives

* Detect players and the football in each video frame.
* Assign unique IDs and track their motion over time.
* Visualize detections and trajectories on the processed video.
* Optionally, generate simple analytics such as ball possession and movement heatmaps.

---

### 🧠 System Architecture

The system follows a detection–tracking pipeline:

1. **Input Video** → captured or uploaded match footage.
2. **YOLOv8 Model** → detects players (`person`) and ball (`sports ball`).
3. **ByteTrack Algorithm** → assigns unique, consistent IDs for each detection.
4. **Tracking Visualization** → overlays bounding boxes and IDs.
5. **Output Video** → saved annotated footage with optional analytics.

```
Football Video → YOLOv8 (Detection) → ByteTrack (Tracking) → Processed Output
```

---

### ⚙️ Technologies Used

| Component                   | Description                              |
| --------------------------- | ---------------------------------------- |
| **Language**                | Python, JavaScript (Node.js for backend) |
| **Deep Learning Framework** | PyTorch                                  |
| **Detection Model**         | YOLOv8 (Ultralytics)                     |
| **Tracking Algorithm**      | ByteTrack                                |
| **Video Processing**        | OpenCV                                   |
| **Visualization**           | Matplotlib                               |
| **Web Integration**         | Node.js + Express server                 |
| **Frontend**                | HTML, CSS, JS (inside `public/` folder)  |

---

### 🧩 Folder Structure

```
TRACKING/
│
├── ByteTrack/                 # Tracker algorithm files
├── model_preparation/         # YOLO model setup
├── node_modules/              # Node dependencies
├── outputs/                   # Generated output videos
├── processed/                 # Processed frames
├── public/                    # Frontend files
├── uploads/                   # Uploaded input videos
│
├── best.pt                    # Trained YOLO model weights
├── requirements.txt           # Python dependencies
├── server.js                  # Node.js backend server
├── track.py                   # Main Python script for detection/tracking
├── README.md                  # Project documentation (this file)
├── TODO.md                    # Pending improvements
└── package.json               # Node.js project configuration
```

---

### ⚡ Installation & Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/ball-and-player-tracking.git
cd ball-and-player-tracking
```

#### 2️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Install Node.js Dependencies

```bash
npm install
```

#### 4️⃣ Run the Backend Server

```bash
node server.js
```

#### 5️⃣ Run Tracking Script

```bash
python track.py --source path_to_video.mp4 --weights best.pt
```

The processed video will be saved in the `outputs/` folder.

---

### 📊 Results

* Players and football are detected in real-time.
* Each player receives a unique ID for tracking across frames.
* Bounding boxes and labels are drawn on the output video.
* ByteTrack ensures stable IDs even under occlusion.
* Output examples and test results are stored in the `outputs/` directory.

---

### 🧪 Example Output

* **Input:** Raw football match footage
* **Output:** Annotated video with detected players and ball (with IDs)
* **Optional Analytics:** Movement heatmap, possession ratio, player paths

---

### 🧰 Requirements

**Python Libraries:**

```
opencv-python
torch
numpy
matplotlib
ultralytics
bytetrack
flask
```

**Node.js Modules:**

```
express
multer
path
child_process
```

---

### 📘 References

* Bewley, A. et al. (2016). *Simple Online and Realtime Tracking (SORT)*.
* Wojke, N. et al. (2017). *Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric.*
* Giancola, S. et al. (2018). *SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos.*
* Ultralytics YOLOv8 Documentation ([https://docs.ultralytics.com](https://docs.ultralytics.com))
* ByteTrack Official GitHub ([https://github.com/ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack))

---

### 🚀 Future Work

* Integrate a Re-ID model for consistent tracking across camera views.
* Implement live stream tracking using webcam or RTSP feed.
* Add automatic event detection (goals, passes, offsides).
* Expand analytics (heatmaps, team formations, player statistics).

---

### 🙌 Acknowledgement

Special thanks to the instructors and team members for their continuous guidance and support throughout the development of this project.

---
