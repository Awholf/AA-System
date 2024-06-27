import cv2
import numpy as np
import time
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
        video_capture = cv2.VideoCapture(0, backend)
        if video_capture.isOpened():
            print(f"Using backend: {backend}")
            break

    if not video_capture or not video_capture.isOpened():
        print("No se puede acceder a la cámara con ninguno de los backends disponibles.")
        return

    # Diccionario para rastrear la última marca de asistencia por estudiante
    last_marked = {}
    printed_messages = set()  # Set para rastrear impresiones ya realizadas

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
            continue

        students = get_students()
        current_time = time.time()
        for (x, y, w, h) in faces:
            # Asegurarse de que los índices no excedan los límites de la imagen
            y_end = min(y + h, gray.shape[0])
            x_end = min(x + w, gray.shape[1])
            face = gray[y:y_end, x:x_end]
            
            label, confidence = face_recognizer.predict(face)

            # Verificar la confianza antes de proceder
            if confidence > 50:
                continue

            student_name = "Desconocido"
            for student in students:
                if student[0] == label:
                    student_name = student[1]
                    last_marked_time = last_marked.get(label, 0)
                    # Marcar asistencia si ha pasado más de 1 minuto (60 segundos)
                    if current_time - last_marked_time > 60:
                        mark_attendance(student[0])
                        last_marked[label] = current_time
                        print(f"Asistencia marcada para {student_name} a las {time.ctime(current_time)}")
                        printed_messages.add(label)
                    break
            
            # Mostrar el nombre del estudiante en el frame
            cv2.putText(frame, student_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
