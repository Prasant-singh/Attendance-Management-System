
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
            print("Warning: No face found in image")
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

    try:
        with open(attendance_log_path, 'r') as f:
            attendance = json.load(f)
    except (json.JSONDecodeError, ValueError):
        # If the file is corrupted or invalid, initialize with an empty JSON object
        attendance = {}
        with open(attendance_log_path, 'w') as f:
            json.dump(attendance, f)

    # Check if attendance file exists and is not empty
    if not os.path.exists(attendance_log_path) or os.path.getsize(attendance_log_path) == 0:
        attendance = {}
    else:
        with open(attendance_log_path, 'r') as f:
            attendance = json.load(f)

    # Initialize user attendance if not present
    if name not in attendance:
        attendance[name] = {}

    # Check if today is already in attendance
    if today not in attendance[name]:
        attendance[name][today] = [{'login': current_time}]
        print(f"{name} logged in at {current_time}")
        engine.say(f"{name} logged in successfully")
        engine.runAndWait()
    else:
        last_entry = attendance[name][today][-1]
        
        # Helper function to calculate time difference in minutes
        def time_diff_in_minutes(time1, time2):
            time1_dt = datetime.strptime(time1, '%H:%M')
            time2_dt = datetime.strptime(time2, '%H:%M')
            return (time2_dt - time1_dt).total_seconds() / 60  # Return difference in minutes

        # Handle Overtime or Logout logic based on the last recorded time
        if 'logout' in last_entry and time_diff_in_minutes(last_entry['logout'], current_time) >= 10:
            # Add Over Time login
            attendance[name][today].append({'Over Time login': current_time})
            print(f"{name} Over Time login at {current_time}")
            engine.say(f"{name} logged in successfully for overtime")
            engine.runAndWait()

        elif 'Over Time login' in last_entry and 'Over Time logout' not in last_entry and time_diff_in_minutes(last_entry['Over Time login'], current_time) >= 10:
            # Add Over Time logout
            last_entry['Over Time logout'] = current_time
            print(f"{name} Over Time logout at {current_time}")
            engine.say(f"{name} logged out successfully for overtime")
            engine.runAndWait()

        elif 'login' in last_entry and 'logout' not in last_entry and time_diff_in_minutes(last_entry['login'], current_time) >= 10:
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
    merged_data = {}

    # Loop through the attendance records
    for name, days in attendance.items():
        for day, entries in days.items():
            # Initialize row data for each user on a given date
            if (name, day) not in merged_data:
                merged_data[(name, day)] = {'login': '', 'logout': '', 'overtime_login': '', 'overtime_logout': ''}
            
            for entry in entries:
                if 'login' in entry:
                    merged_data[(name, day)]['login'] = entry['login']
                if 'logout' in entry:
                    merged_data[(name, day)]['logout'] = entry['logout']
                if 'Over Time login' in entry:
                    merged_data[(name, day)]['overtime_login'] = entry['Over Time login']
                if 'Over Time logout' in entry:
                    merged_data[(name, day)]['overtime_logout'] = entry['Over Time logout']

    # Prepare data for writing to CSV
    data = []
    for (name, day), times in merged_data.items():
        row = [
            name,
            day,
            times['login'],
            times['logout'],
            times['overtime_login'],
            times['overtime_logout']
        ]
        data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['Name', 'Date', 'Login', 'Logout', 'Over Time Login', 'Over Time Logout'])
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
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.4:  # Stricter matching
            name = classNames[matchIndex].upper()
            print(f"Match found: {name} with distance: {faceDis[matchIndex]}")

            # Get coordinates and scale them up since the image was resized to 1/4th earlier
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw the rectangle around the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the rectangle for the name label
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            
            # Put the name on the label
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Call the markAttendance function for this recognized face
            markAttendance(name)
        else:
            print(f"No confident match found, minimum distance: {faceDis[matchIndex]}")

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
