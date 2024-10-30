import cv2
import cvzone
import os
import pickle
import face_recognition
import numpy as np
import time
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, initialize_app, db, storage
from AddDataToData import data
from concurrent.futures import ThreadPoolExecutor

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
        'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'
    })

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up the video capture with a video file
# Replace with your video file path
video_path = 'Video/How does facial recognition work_.mp4'
cam = cv2.VideoCapture(video_path)
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
    image_filename = f"{face_id}_{event_type}_{
        now.strftime('%Y%m%d_%H%M%S')}.jpg"

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


def encode_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, face_locations)
    return face_locations, encodings


while True:
    # Read a frame from the video
    ret, img = cam.read()
    if not ret:
        print("End of video or cannot read the video file.")
        break  # Exit the loop if the video ends

    img_copy = img.copy()

    # Start timer for total latency measurement
    start_total_time = time.perf_counter()
    # Capture frame from video (Video Capture Latency)
    start_capture_time = time.perf_counter()
    capture_latency = time.perf_counter() - start_capture_time

    smallerImage = cv2.resize(img, (0, 0), None, 0.6, 0.6)
    smallerImage = cv2.cvtColor(smallerImage, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(smallerImage)
    encodeCurrentFrame = face_recognition.face_encodings(
        smallerImage, faceCurrentFrame)
    # Start face detection and recognition (Face Detection Latency)
    start_detection_time = time.perf_counter()
    detection_latency = time.perf_counter() - start_detection_time

    # Display the resulting image (Display Latency)
    start_display_time = time.perf_counter()
    display_latency = time.perf_counter() - start_display_time
    # Total latency
    total_latency = time.perf_counter() - start_total_time

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(encode_faces, smallerImage)
        faceCurrentFrame, encodeCurrentFrame = future.result()

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Create contours
        cv2.polylines(img_copy, [np.array([[x, y], [
                      x + w, y], [x + w, y + h], [x, y + h]])], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow('Webcam', img)
    recognized_name = None
    # Compare image and video frame
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(
            peopleFaceList, encodeFace, tolerance=0.5)
        distanceComparison = face_recognition.face_distance(
            peopleFaceList, encodeFace)
        matchIndex = np.argmin(distanceComparison)

        if matches[matchIndex]:
            recognized_id = peopleID[matchIndex]
            now = datetime.now()
            recognized_name = data.get(
                recognized_id, {}).get('name', 'Unknown')

            if recognized_id not in recognized_faces_set:
                print(f"New Known Face Detected: {recognized_id}")
                # Record entry time for the recognized face
                face_times[recognized_id] = {'entry': now, 'exit': None}
                recognized_faces_set.add(recognized_id)
                recognized_faces_counter += 1
                print(
                    f"{recognized_id} - {recognized_name} entered the room at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

                folder_name = f"{now.strftime(
                    '%Y-%m-%d %H:%M')} - {recognized_id}"

                # Calculate the total duration the person has been in the frame
                entry_time = face_times[recognized_id]['entry']

                # Capture the image for the recognized face (cropping it)
                top, right, bottom, left = faceLocation
                face_image = img
                face_image_resized = cv2.resize(
                    face_image, (1280, 720))

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
                    'exit_image_url': None,
                    'attendance_status': None
                })

            # Update face last seen timestamps to track presence
            face_presence[recognized_id] = now

        else:
            recognized_name = "Unknown"

    # Handle exit times for faces that are no longer visible
    current_time = datetime.now()
    for face_id in list(face_presence.keys()):
        last_seen_time = face_presence[face_id]
        time_elapsed = (current_time - last_seen_time).total_seconds()

        if time_elapsed > EXIT_DELAY:
            # Update exit time in the dictionary and remove the face from tracking
            face_times[face_id]['exit'] = current_time
            recognized_name = data.get(
                recognized_id, {}).get('name', 'Unknown')
            recognized_faces_set.remove(face_id)

            # Log the exit time
            print(
                f"Face {face_id} - {recognized_name} exited the room at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Capture an exit image
            _, img_exit = cam.read()
            exit_image_resized = cv2.resize(
                img_exit, (1280, 720))

            # Upload the exit image to Firebase storage and get the URL
            exit_image_url = upload_image_to_firebase(
                exit_image_resized, face_id, "exit")

            # Calculate the duration the person stayed in the frame
            entry_time = face_times[face_id]['entry']
            duration_in_seconds = (current_time - entry_time).total_seconds()

            # Determine if the person is late or on time (threshold: 10 seconds)
            attendance_status = "ON TIME" if duration_in_seconds >= 10 else "LATE"
            recognized_name = data.get(
                recognized_id, {}).get('name', 'Unknown')

            print(
                f"Face {face_id} - {recognized_name} stayed for {duration_in_seconds:.2f} seconds. Status: {attendance_status}")

            # Update the exit time and attendance status in the database
            folder_name = f"{entry_time.strftime('%d-%m-%Y')} - {face_id}"

            db_ref = db.reference(f"attendance/{folder_name}")

            db_ref.update({
                'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_image_url': exit_image_url,
                'duration': duration_in_seconds,
                'attendance_status': attendance_status
            })

            # Remove face from face_presence to stop tracking
            del face_presence[face_id]

    # Display the counter and the date/time on the video feed
    dt_string = current_time.strftime("%d/%m/%Y %H:%M:%S")
    cvzone.putTextRect(img, f"{dt_string}", (10, 20),
                       scale=1, thickness=2, colorR=(0, 0, 0))
    cvzone.putTextRect(img, f"Recognized Faces: {recognized_faces_counter}", (
        10, 80), scale=1, thickness=2, colorR=(0, 0, 0))

    # Only display the name if recognized_name is defined
    for (x, y, w, h) in faces:
        if recognized_name and recognized_name != "Unknown":
            # Case for Matched Face (green text with background)
            text_position_x = x
            text_position_y = y - 10  # Slightly above the face box
            cvzone.putTextRect(img, f"Name: {recognized_name} - {recognized_id}", (text_position_x, text_position_y),
                               scale=1, thickness=2, colorR=(0, 255, 0), colorT=(255, 255, 255), offset=10)
        else:
            # Case for Not Matched Face (red text with background)
            text_position_x = x
            text_position_y = y - 10  # Slightly above the face box
            cvzone.putTextRect(img, "Unknown Face", (text_position_x, text_position_y),
                               scale=1, thickness=2, colorR=(0, 0, 255), colorT=(255, 255, 255), offset=10)

    cv2.imshow('Webcam', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cam.release()
cv2.destroyAllWindows()
