---
# Leveraging Face Recognition Technology for Efficient Attendance Monitoring in Bina Nusantara (Binus) University | 2024

## Overview

This project is focused on implementing a robust and efficient attendance monitoring system using **face recognition technology** at **Binus Bina Nusantara University**. The solution leverages advanced machine learning techniques and computer vision to automate attendance taking, minimizing manual errors, and improving overall efficiency.

## Features

- **Real-time Face Detection and Recognition**: Uses webcam or external camera to capture and recognize faces in real-time.
- **Attendance Recording**: Automatically logs attendance into the system once a face is recognized.
- **Improved Accuracy**: Utilizes OpenCV and `face_recognition` libraries for highly accurate face matching.
- **Scalability**: Designed to accommodate a large number of students with a scalable backend system.
- **User-Friendly Interface**: Simple UI to track and manage attendance easily.
  
## Tech Stack

- **Programming Language**: Python
- **Libraries/Frameworks**: 
  - [OpenCV](https://opencv.org/)
  - [face_recognition](https://github.com/ageitgey/face_recognition)
  - [cvzone](https://github.com/cvzone/cvzone)
  - NumPy
- **Development Environment**: 
  - IDE: Visual Studio Code
  - Compiler: MinGW
- **Hardware**: Standard Webcam or external camera

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attendance-monitoring-binus.git
   cd attendance-monitoring-binus
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the system:
   ```bash
   python main.py
   ```

## Usage

1. Connect your webcam or external camera.
2. Run the application to start face detection.
3. The system will automatically record the attendance of recognized individuals in real-time.
4. Use the interface to view and manage attendance records.

## Future Improvements

- **Mobile Integration**: Develop a mobile app for easier access and notifications.
- **Cloud Storage**: Implement cloud-based storage to make attendance records more accessible.
- **Integration with Existing University Systems**: Sync the attendance data with university management systems like LMS or ERP.

---

