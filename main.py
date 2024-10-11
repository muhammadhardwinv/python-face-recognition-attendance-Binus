import cv2
import os
import pickle
import face_recognition
import numpy as np
from encodeGenerator import peopleFaceListWithId
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, initialize_app, db
from firebase_admin import storage

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
        'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'
    })
# for MIL tracker
tracker = cv2.TrackerMIL_create()
tracking = False
bbox = None

now = datetime.now()
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
fps = 30
cam.set(cv2.CAP_PROP_FPS, 30)

recognized_faces_counter = 0
# imgBg = cv2.imread('library/halah.png')
folderModePath = 'Library/Modes'
modePathList = os.listdir(folderModePath)
imageModeList = []
for path in modePathList:
    imageModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    print(len(imageModeList))

# load process the encoding file
file = open('EncodeFile.p', 'rb')
peopleFaceListWithId = pickle.load(file)
file.close()
peopleFaceList, peopleID = peopleFaceListWithId
print(peopleID)


# Initialize face recognition counter and set for recognized faces
recognized_faces_counter = 0
recognized_faces_set = set()  # To store already recognized face IDs


while True:
    _, img = cam.read()
    object_detector = cv2.createBackgroundSubtractorMOG2()
    mask = object_detector.apply(img)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:  # You can adjust this threshold
            # Draw the contour on the original image
            # Green color for contours
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

    smallerImage = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    smallerImage = cv2.cvtColor(smallerImage, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(smallerImage)
    encodeCurrentFrame = face_recognition.face_encodings(
        smallerImage, faceCurrentFrame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('Face Detector(webcam)', img)

    # Compare image and camera
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        # matches = face_recognition.compare_faces(peopleFaceList, encodeFace)

        matches = face_recognition.compare_faces(
            peopleFaceList, encodeFace, tolerance=0.7)
        distanceComparison = face_recognition.face_distance(
            peopleFaceList, encodeFace)
        matches = [bool(match) for match in matches]
        # print("matches", matches)
        # print("distanceComparison", distanceComparison)

        matchIndex = np.argmin(distanceComparison)
        # print("Match Index", matchIndex)

        if matches[matchIndex]:
            recognized_id = peopleID[matchIndex]  # Get recognized face ID

            # Only increment counter if the face is not already recognized
            if recognized_id not in recognized_faces_set:
                print("New Known Face Detected")
                print(recognized_id)

                # Increment the counter and add the ID to the recognized set
                recognized_faces_counter += 1
                recognized_faces_set.add(recognized_id)

                print(
                    f"Total Recognized Faces: {recognized_faces_counter}")
                # Get the current date and time
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

                db_ref = db.reference("timestamps")
                # Write to Firebase Database
                db_ref.push({
                    'face_id': recognized_id,
                    'timestamp': timestamp
                })
                print(
                    f"Timestamp {timestamp} written to Firebase for face {recognized_id}")

    # Get the current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    # Display the counter and the date/time on the video feed
    cv2.putText(img, f"Recognized Faces: {recognized_faces_counter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Date and Time: {dt_string}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Detector(webcam)', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cam.release()
cv2.destroyAllWindows()
