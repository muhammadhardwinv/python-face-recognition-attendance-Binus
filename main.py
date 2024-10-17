import cv2
import os
import pickle
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, initialize_app, db, storage
from AddDataToData import data
# from AddDataToData import LectureData

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
        'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'
    })

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up the camera
cam = cv2.VideoCapture(0)

cam.set(3, 1280)
cam.set(4, 720)
fps = 30
cam.set(cv2.CAP_PROP_FPS, 30)

recognized_faces_counter = 0

# Load and process the encoding file
file = open('EncodeFile.p', 'rb')
peopleFaceListWithId = pickle.load(file)
file.close()
peopleFaceList, peopleID = peopleFaceListWithId

# Initialize a dictionary to store entry and exit times for recognized faces
face_times = {}
face_presence = {}  # Tracks the last time a face was detected
recognized_faces_set = set()  # To store already recognized face IDs
EXIT_DELAY = 5  # The delay in seconds to confirm that a face has exited the frame

bucket = storage.bucket()

# Function to upload image to Firebase


def upload_image_to_firebase(image, face_id, event_type):
    now = datetime.now()
    image_filename = f"{face_id}_{event_type}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"

    # Convert image to file-like object
    image_path = f"temp_images/{image_filename}"
    cv2.imwrite(image_path, image)  # Save the image locally first

    # Upload the image to Firebase storage
    blob = bucket.blob(f"face_images/{image_filename}")
    blob.upload_from_filename(image_path)

    # Get the URL of the uploaded image
    image_url = blob.generate_signed_url(
        expiration=datetime.now() + timedelta(days=365))
    print(f"Image uploaded to Firebase: {image_url}")

    return image_url


while True:
    _, img = cam.read()

    smallerImage = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    smallerImage = cv2.cvtColor(smallerImage, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(smallerImage)
    encodeCurrentFrame = face_recognition.face_encodings(
        smallerImage, faceCurrentFrame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detector(webcam)', img)

    # Compare image and camera
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(
            peopleFaceList, encodeFace, tolerance=0.7)
        distanceComparison = face_recognition.face_distance(
            peopleFaceList, encodeFace)
        matchIndex = np.argmin(distanceComparison)

        if matches[matchIndex]:
            recognized_id = peopleID[matchIndex]
            now = datetime.now()
            recognized_name = data[recognized_id]['name']

            if recognized_id not in recognized_faces_set:
                print(f"New Known Face Detected: {recognized_id}")

                # Record entry time for the recognized face
                face_times[recognized_id] = {'entry': now, 'exit': None}
                recognized_faces_set.add(recognized_id)
                recognized_faces_set.add(recognized_name)
                recognized_faces_counter += 1
                print(
                    f"{recognized_id} - {recognized_name} entered the room at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

                folder_name = f"{now.strftime('%Y-%m-%d %H:%M')} - {recognized_id}"

                # Capture the image for the recognized face (cropping it)
                top, right, bottom, left = faceLocation
                face_image = img
                face_image_resized = cv2.resize(
                    face_image, (1280, 720))  # Resize to 480p

                # Upload the entry image to Firebase storage and get the URL
                entry_image_url = upload_image_to_firebase(
                    face_image_resized, recognized_id, "entry")

                # Adding entry time and image URL to the Database
                db_ref = db.reference(f"attendance/{folder_name}")
                db_ref.push({
                    'face_id': recognized_id,
                    'face_name': recognized_name,
                    'entry_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': None,
                    'entry_image_url': entry_image_url,
                    'exit_image_url': None
                })

            # Update face last seen timestamps to track presence
            face_presence[recognized_id] = now

    # Handle exit times for faces that are no longer visible
    current_time = datetime.now()
    for face_id in list(face_presence.keys()):
        last_seen_time = face_presence[face_id]
        time_elapsed = (current_time - last_seen_time).total_seconds()

        if time_elapsed > EXIT_DELAY:
            # Update exit time in the dictionary and remove the face from tracking
            face_times[face_id]['exit'] = current_time
            recognized_faces_set.remove(face_id)

            # Log the exit time
            print(
                f"Face {face_id} - {recognized_name} exited the room at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Capture an exit image
            _, img_exit = cam.read()
            exit_image_resized = cv2.resize(
                img_exit, (640, 480))  # Resize to 480p

            # Upload the exit image to Firebase storage and get the URL
            exit_image_url = upload_image_to_firebase(
                exit_image_resized, face_id, "exit")

            # Calculate the duration the person stayed in the frame
            entry_time = face_times[face_id]['entry']
            duration_in_seconds = (current_time - entry_time).total_seconds()

            # Determine if the person is late or on time (threshold: 10 seconds)
            attendance_status = "ON TIME" if duration_in_seconds >= 10 else "LATE"

            # Push exit time, image URL, duration, and attendance status to the database
            db_ref.push({
                'face_id': recognized_id,
                'face_name': recognized_name,
                'entry_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': None,
                'entry_image_url': entry_image_url,
                'exit_image_url': None
            })

            # Remove face from face_presence to stop tracking
            del face_presence[face_id]

    # Display the counter and the date/time on the video feed
    dt_string = current_time.strftime("%d/%m/%Y %H:%M:%S")
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
