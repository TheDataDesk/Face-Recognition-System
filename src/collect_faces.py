import cv2
import os

def collect_faces(person_name, sample_count=500):
    cam = cv2.VideoCapture(1)  #Use the correct working camera index
    if not cam.isOpened():
        print("ERROR: Webcam not accessible.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    dataset_dir = 'dataset'
    person_path = os.path.join(dataset_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    # Get count of existing images to continue from there
    existing_images = [
        f for f in os.listdir(person_path)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    start_count = len(existing_images)
    count = 0

    print(f"\nCollecting {sample_count} face samples for '{person_name}' in '{person_path}'")
    print("➡ Press 'c' to capture a face")
    print("➡ Press 'q' to quit early\n")

    while count < sample_count:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("Could not read frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imshow("Detected Face", face_img)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == ord('c') and len(faces) > 0:
            image_id = start_count + count
            face_filename = os.path.join(person_path, f"{image_id}.jpg")
            cv2.imwrite(face_filename, face_img)
            print(f"Sample {image_id} saved at {face_filename}")
            count += 1
        elif key == ord('q'):
            print("Stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Collected {count} new sample(s) for '{person_name}'.")

if __name__ == "__main__":
    name = input("Enter person name (folder name under 'dataset/'): ").strip()
    if name:
        collect_faces(name, sample_count=500)
    else:
        print("⚠️ Name cannot be empty.")
