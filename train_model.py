import cv2
import os
import numpy as np

def train_model():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    base_dir = 'dataset'  # Carpeta que contiene subcarpetas de imágenes de cada persona
    for label_dir in os.listdir(base_dir):
        if not label_dir.isdigit():
            continue
        label = int(label_dir)
        label_path = os.path.join(base_dir, label_dir)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(image)
                labels.append(label)

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('face_model.xml')
    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train_model()
