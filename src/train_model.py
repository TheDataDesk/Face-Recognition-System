# src/train_model.py
import os
import cv2
import numpy as np
import json
dataset_path = 'dataset'
model_save_path = 'trainer/lbph_model.yml'
label_map_path = 'labels.json'

def load_images_and_labels():
    images, labels = [], []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path): continue

        label_map[current_id] = person_name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(current_id)
        current_id += 1

    return images, np.array(labels), label_map

def train_lbph_model():
    images, labels, label_map = load_images_and_labels()
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)
    print(f"Model trained and saved to {model_save_path}")
    print(f"Label map saved to {label_map_path}")

if __name__ == "__main__":
    train_lbph_model()
