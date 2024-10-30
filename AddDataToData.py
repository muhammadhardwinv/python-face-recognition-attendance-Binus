import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://realtimerecognizer-ac6e4-default-rtdb.firebaseio.com/',
    'storageBucket': 'realtimerecognizer-ac6e4.appspot.com'

})
ref = db.reference('Lecture Data')

data = {
    "19980101": {
        "name": "John Cena",
        "attendance": "Absent",
        "occupation": "Director",
        "last_attendance_time": "2023-10-06 07:54:32",
        "active": "inactive",
    },
    "19980102": {
        "name": "Jane Doe",
        "attendance": "Absent",
        "occupation": "Engineer",
        "last_attendance_time": "2024-10-03 08:54:32",
        "active": "active",
    },
    "20000514": {
        "name": "M Hardwin",
        "attendance": "Present",
        "occupation": "Programmer Intern",
        "last_attendance_time": "2024-10-07 07:20:25",
        "active": "active",
    },
    "20010102": {
        "name": "Shannon Smith",
        "attendance": "Absent",
        "occupation": "Analyst Intern",
        "last_attendance_time": "2024-10-07 09:30:25",
        "active": "Active",
    },
    "20010103": {
        "name": "Jaden Smith",
        "attendance": "Absent",
        "occupation": "Karate Kid",
        "last_attendance_time": "2024-10-07 09:30:25",
        "active": "Active",
    },
}


for key, value in data.items():
    ref.child(key).set(value)
