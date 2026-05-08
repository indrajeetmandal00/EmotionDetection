# Facial Emotion Detection Studio

🌐 **Live Demo:** [https://emotiondetection-rr0w.onrender.com](https://emotiondetection-rr0w.onrender.com)

## Overview
This project is a real-time Facial Emotion Detection application. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to analyze facial expressions and classify them into one of seven emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.**

The backend is powered by Flask, and the video feed is processed using OpenCV to provide a sci-fi inspired "Neural Net Visualizer" overlay, complete with confidence graphing, target tracking, and simulated biometrics.

## Features
* **Real-Time Tracking:** Utilizes Haar Cascade face detection for locking onto targets.
* **Emotion Classification:** Custom-trained 30-epoch CNN model predicting 7 distinct emotions.
* **Sci-Fi UI Overlays:** Telemetry data, confidence line graphs, dynamic bounding boxes, and PIP (Picture-in-Picture) edge detection overlays.
* **Web Interface:** Streams the processed OpenCV video feed directly to a web browser via Flask.

## Tech Stack
* **Python 3.10**
* **Flask** (Web Server)
* **TensorFlow / Keras** (Deep Learning / Model Prediction)
* **OpenCV** (Image Processing and UI Drawing)
* **NumPy** (Matrix operations)

## Running Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Facial Emotion"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App:**
   ```bash
   python app.py
   ```

4. **Open in your Browser:** Navigate to `http://localhost:5000` to view the feed!