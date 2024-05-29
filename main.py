import cv2
import numpy as np
from database import init_db, get_students, mark_attendance

# Inicialización de la base de datos
init_db()

def run_recognition():
    # Cargar modelo de reconocimiento facial de OpenCV
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_model.xml')

    # Intentar con diferentes backends de captura de video
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]

    video_capture = None
    for backend in backends:
        video_capture = cv2.VideoCapture(1, backend)
        if video_capture.isOpened():
            print(f"Using backend: {backend}")
            break

    if not video_capture or not video_capture.isOpened():
        print("No se puede acceder a la cámara con ninguno de los backends disponibles.")
        return

    # Captura y procesamiento de imágenes en tiempo real
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se puede capturar el frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) == 0:
            print("No se detectaron rostros.")
        else:
            print(f"Rostros detectados: {len(faces)}")

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)
            print(f"Rostro detectado con etiqueta {label} y confianza {confidence}")

            if confidence > 50:  # Umbral de coincidencia
                student_id = label
                students = get_students()
                for student in students:
                    if student[0] == student_id:
                        mark_attendance(student[0])
                        cv2.putText(frame, student[1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print(f"Estudiante {student[1]} reconocido y asistencia marcada.")
                        break
            else:
                print("Rostro no reconocido o confianza insuficiente.")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
