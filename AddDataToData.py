import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
    'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'

})
ref = db.reference('LectureData')
data = {
    "19980101": {
        "name": "John Doe",
        "attendance_status": []
    },
    "19980102": {
        "name": "Jane Doe",
        "attendance_status": []
    },
    "20000514": {
        "name": "M Hardwin",
        "attendance_status": []
    },
    "20010102": {
        "name": "Shannon Smith",
        "attendance_status": []
    }
}

for key, value in data.items():
    ref.child(key).set(value)
