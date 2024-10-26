import face_recognition
import cv2
import os
import pickle

# Directory where images are stored
IMAGE_DIR = 'Images/'

# Output file for face encodings
ENCODINGS_FILE = 'encodings.pickle'

# Initialize arrays to hold the encodings and the names of individuals
known_face_encodings = []
known_face_names = []

# Loop through all the images in the directory
for image_name in os.listdir(IMAGE_DIR):
    # Make sure the file is an image
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image file
        image_path = os.path.join(IMAGE_DIR, image_name)
        print(f"Processing {image_path}...")
        image = face_recognition.load_image_file(image_path)

        # Find the face locations and encodings in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Loop over each face found in the image
        for face_encoding in face_encodings:
            # Add the encoding to the list
            known_face_encodings.append(face_encoding)

            # Use the file name (without extension) as the person's name
            name = os.path.splitext(image_name)[0]
            known_face_names.append(name)

# Save the encodings and names to a file
print(f"Saving encodings to {ENCODINGS_FILE}...")
data = {"encodings": known_face_encodings, "names": known_face_names}
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump(data, f)

print("Encodings generated successfully.")
