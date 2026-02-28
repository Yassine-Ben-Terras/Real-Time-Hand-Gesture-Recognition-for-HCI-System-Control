# Real-Time Hand Gesture Recognition for HCI System Control

## Project Overview
This project implements a real-time, contactless Human-Computer Interaction (HCI) system. It allows users to control system environments—specifically computer audio volume—using dynamic hand gestures recognized through a standard webcam. By leveraging computer vision and a lightweight neural network, the system translates physical gestures into instant digital commands.

## Key Features
* **Real-Time Processing:** Low-latency gesture detection and classification.
* **Contactless Control:** Adjust system volume without touching the keyboard or mouse (e.g., "Thumbs Up" to increase volume, "Fist" to mute).
* **Robust to Variance:** Custom mathematical preprocessing ensures the AI accurately recognizes gestures regardless of the hand's position on the screen or distance from the camera (zoom-proof).

## System Architecture Pipeline
Our system processes the video feed in five continuous steps:
1. **Manipulation (OpenCV):** Captures the live video feed, mirrors the image, and converts the color space (BGR to RGB).
2. **Processing (MediaPipe):** Applies noise reduction and detects the Region of Interest (ROI) containing the hand.
3. **Analysis:** Extracts 21 precise 3D skeletal landmarks from the detected hand.
4. **Geometric Preprocessing:** * **Translation Invariance:** Anchors the wrist to the `(0,0)` coordinate to remove positional bias.
    * **Scale Invariance:** Normalizes all coordinates to a `[-1, 1]` interval to remove size/distance bias.
5. **Classification (MLP):** Feeds the standardized geometric vector into a custom Multi-Layer Perceptron (MLP) neural network to predict the gesture class and execute the corresponding system command.

## Technology Stack
* **Language:** Python
* **Computer Vision:** OpenCV
* **Pose Estimation:** Google MediaPipe
* **Machine Learning:** Multi-Layer Perceptron (MLP) Architecture

## Author
* **Yassine Ben Terras**

## License
This project is licensed under the MIT License - see the LICENSE file for details.
