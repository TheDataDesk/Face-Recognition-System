import cv2
import json
import tkinter as tk
from tkinter import messagebox
MODEL_PATH = 'trainer/lbph_model.yml'
LABEL_MAP_PATH = 'labels.json'
LICENSE_DATA_PATH = 'licenses.json'

def recognize_from_webcam():
    # Load trained model and label map
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

    with open(LICENSE_DATA_PATH, 'r') as f:
        license_db = json.load(f)

    # Init GUI
    root = tk.Tk()
    root.withdraw()

    cam = cv2.VideoCapture(1)  # Your working webcam index
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    shown_ids = set()  # To avoid spamming popup every frame

    print("🎥 Face recognition started. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("Could not read from webcam.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            label, confidence = model.predict(face_img)

            name = label_map.get(str(label), "Unknown")

            # Show license info popup if confidence is good
            if confidence < 50 and name in license_db and name not in shown_ids:
                license_info = license_db[name]
                msg = f"""
Name: {license_info['Name']}
License ID: {license_info['License ID']}
Valid Until: {license_info['Valid Until']}
Status: {license_info['Status']}
"""
                print(msg)
                messagebox.showinfo("License Verified", msg)
                shown_ids.add(name)

            # Draw face rectangle and name on frame
            color = (0, 255, 0) if confidence < 50 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_from_webcam()
