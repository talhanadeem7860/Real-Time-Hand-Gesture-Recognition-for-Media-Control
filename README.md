Project Overview

This project implements a real-time system that controls your computer's system volume using hand gestures captured through a webcam. It demonstrates a practical application of computer vision by translating physical actions into system commands.

The core of this project lies in its ability to accurately track hand landmarks and calculate the distance between specific points to control an output. The pipeline is as follows:

Hand Tracking: Utilizes Google's MediaPipe library to detect and track 21 keypoints of a hand in the video feed with high accuracy.

Gesture Recognition: The distance between the tip of the thumb (landmark 4) and the tip of the index finger (landmark 8) is calculated in real-time.

Volume Control: This calculated distance is mapped directly to the system's master volume. A smaller distance corresponds to lower volume, and a larger distance corresponds to higher volume. The script is cross-platform and will work on Windows, macOS, and Linux.

Visual Feedback: The application provides immediate visual feedback, drawing the hand landmarks, the line between the fingers, and a volume bar directly onto the video stream.
