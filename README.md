Here's a comprehensive README for your GitHub repository:

---

# Real-Time Face Recognition and Attendance System

This project implements a real-time face recognition and attendance tracking system using OpenCV, Firebase, and the `face_recognition` library. The system captures video frames from a webcam, detects and recognizes faces, and uploads relevant entry and exit data to Firebase Realtime Database and Storage.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Project Details](#project-details)
  - [Firebase Setup](#firebase-setup)
  - [Face Detection and Recognition](#face-detection-and-recognition)
  - [Handling and Storing Data](#handling-and-storing-data)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The Real-Time Face Recognition and Attendance System uses computer vision techniques to identify and track the entry and exit of individuals from a video feed. It detects faces using Haar cascades and recognizes them with pre-stored encodings. This information is recorded in Firebase, including entry and exit images, timestamps, and attendance status.

## Features
- **Real-time face detection and recognition** using OpenCV and the `face_recognition` library.
- **Attendance tracking** with entry and exit times, images, and status classification.
- **Data storage** using Firebase Realtime Database and Firebase Storage.
- **Threaded processing** for encoding faces concurrently to improve performance.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/real-time-face-recognition-attendance.git
   cd real-time-face-recognition-attendance
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python cvzone firebase-admin face_recognition numpy
   ```

3. **Set up Firebase credentials:**
   - Place your `serviceAccKey.json` (downloaded from Firebase) in the root folder.
   - Update the `databaseURL` and `storageBucket` values in the code to match your Firebase project's settings.

4. **Add face data:**
   - Create and store your face encodings in the `EncodeFile.p` using the `AddDataToData.py` script or similar. Ensure that the `peopleFaceList` and `peopleID` lists are correctly formatted.

## Usage
1. **Run the main script:**
   ```bash
   python main_script.py
   ```
2. The webcam feed will open and start detecting faces. If a known face is detected, the system will log the entry and exit data to Firebase.

## File Structure
```
📦project-root
 ┣ 📜main_script.py       # Main script for running face recognition
 ┣ 📜AddDataToData.py     # Script to add and encode face data
 ┣ 📜EncodeFile.p         # Encoded face data
 ┣ 📜serviceAccKey.json   # Firebase service account credentials
 ┗ 📜README.md            # This README file
```

## Project Details

### Firebase Setup
The system uses Firebase for storing and retrieving face entry and exit data. The setup involves:
- **Firebase Admin SDK** for authentication.
- **Realtime Database** for storing attendance records.
- **Firebase Storage** for storing entry and exit images.

### Face Detection and Recognition
- **Haar Cascade Classifier:** Detects faces in the video feed.
- **Face Encoding:** Uses the `face_recognition` library to create unique encodings for detected faces.
- **Face Matching:** Compares the detected face encodings with known encodings to identify individuals.

### Handling and Storing Data
1. **Entry Logging:** When a known face is detected, the system logs the entry time and uploads an entry image to Firebase Storage.
2. **Exit Detection:** Tracks how long a face remains visible and logs the exit time and image when the face disappears.
3. **Data Recording:** Stores the recorded information in the Realtime Database, including entry and exit images, timestamps, and attendance status (on time or late).

## Contributing
If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License
This project is open-source and available under the MIT License.

---

Let me know if you need any further customization or details!
