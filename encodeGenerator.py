import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
    'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'

})

folderPath = 'ImageLib'
modePathList = os.listdir(folderPath)
imageList = []
peopleID = []
for path in modePathList:
    imageList.append(cv2.imread(os.path.join(folderPath, path)))
    peopleID.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # Untuk mengecek list file / gambar pada folder |
    #                                               |
    # print(path)                                   |
    # print(os.path.splitext(path)[0])              |
    #                                               |
    # Untuk mengecek list file / gambar pada folder |

print(peopleID)

# print("Encoded image -> successfully encoded")


def findEncodings(imageList):
    encodeList = []
    for faceImg in imageList:
        faceImg = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(faceImg)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding is starting ...")  # encodeListKnown = peopleFaceList
peopleFaceList = findEncodings(imageList)  # print(peopleFaceList)
print(peopleFaceList)
# sudah bisa jadi gapapa gak di print lagi, next taro di pickle
# ketika kita save ke file (1. encodingnya) (2. nama ID encodings)
peopleFaceListWithId = [peopleFaceList, peopleID]
print("Encoded successfully")

# generate pickle file
pickleFile = open("EncodeFile.p", 'wb')
pickle.dump(peopleFaceListWithId, pickleFile)
pickleFile.close()
print("Picked successfully saved")
