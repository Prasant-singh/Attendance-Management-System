
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import pyttsx3

# Paths
training_images_path = 'Training_images'
attendance_log_path = 'attendance.json'
attendance_csv_path = 'Attendance.csv'

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load and encode images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print(f"Warning: No face found in image")
    return encodeList

def loadTrainingImages(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    print("Training images:", myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print("Class names:", classNames)
    return images, classNames

def markAttendance(name):
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M')  # Time in HH:MM format
    
    # Check if attendance file exists and is not empty
    if not os.path.exists(attendance_log_path) or os.path.getsize(attendance_log_path) == 0:
        with open(attendance_log_path, 'w') as f:
            json.dump({}, f)
    
    # Read the attendance file
    try:
        with open(attendance_log_path, 'r') as f:
            attendance = json.load(f)
    except (json.JSONDecodeError, ValueError):
        # If the file is corrupted or invalid, initialize with an empty JSON object
        attendance = {}
        with open(attendance_log_path, 'w') as f:
            json.dump(attendance, f)
    
    # Initialize user attendance if not present
    if name not in attendance:
        attendance[name] = {}

    # Check if today is already in attendance
    if today not in attendance[name]:
        # First login for the day
        attendance[name][today] = [{'login': current_time}]
        print(f"{name} logged in at {current_time}")
        engine.say(f"{name} logged in successfully")
        engine.runAndWait()
    else:
        last_entry = attendance[name][today][-1]
        
        # Helper function to calculate time difference
        def time_diff_in_minutes(time1, time2):
            time1_dt = datetime.strptime(time1, '%H:%M')
            time2_dt = datetime.strptime(time2, '%H:%M')
            return (time2_dt - time1_dt).total_seconds() / 60  # Return difference in minutes

        # Handle Overtime or Logout logic based on the last recorded time
        if 'logout' in last_entry and time_diff_in_minutes(last_entry['logout'], current_time) >= 10:
            # Add Overtime login
            attendance[name][today].append({'Over Time login': current_time})
            print(f"{name} Over Time login at {current_time}")
            engine.say(f"{name} logged in successfully for overtime")
            engine.runAndWait()

        elif 'Over Time login' in last_entry and 'Over Time logout' not in last_entry and time_diff_in_minutes(last_entry['Over Time login'], current_time) >= 10:
            # Add Overtime logout
            last_entry['Over Time logout'] = current_time
            print(f"{name} Over Time logout at {current_time}")
            engine.say(f"{name} logged out successfully for overtime")
            engine.runAndWait()

        elif 'login' in last_entry and 'logout' not in last_entry and time_diff_in_minutes(last_entry['login'], current_time) >= 10:
            # Add Logout after 10 min
            last_entry['logout'] = current_time
            print(f"{name} logged out at {current_time}")
            engine.say(f"{name} logged out successfully")
            engine.runAndWait()

    # Save updated attendance
    with open(attendance_log_path, 'w') as f:
        json.dump(attendance, f, indent=4)

    # Update CSV
    update_csv(attendance)


def update_csv(attendance):
    data = []
    for name, days in attendance.items():
        for day, entries in days.items():
            for entry in entries:
                if 'login' in entry and 'logout' in entry:
                    data.append([name, day, entry['login'], entry['logout']])
                elif 'Over Time login' in entry and 'Over Time logout' in entry:
                    data.append([name, day, entry['Over Time login'], entry['Over Time logout']])
                elif 'login' in entry:
                    data.append([name, day, entry['login'], ''])
                elif 'Over Time login' in entry:
                    data.append([name, day, entry['Over Time login'], ''])

    df = pd.DataFrame(data, columns=['Name', 'Date', 'Login', 'Logout'])
    df.to_csv(attendance_csv_path, index=False)


# Load training images
images, classNames = loadTrainingImages(training_images_path)
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image from webcam")
        break

    # Resize the captured image to speed up processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(f"Match found: {name}")

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            markAttendance(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

