import cv2
import tkinter as tk
from tkinter import messagebox
from train_model import train_model
from main import run_recognition
from database import init_db, add_student, generate_excel
import numpy as np

# Inicializar la base de datos
init_db()

def capture_face():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    cap = None

    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Using backend: {backend}")
            break

    if not cap or not cap.isOpened():
        print("No se puede acceder a la cámara con ninguno de los backends disponibles.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede capturar el frame.")
            break

        cv2.imshow('Captura de Rostro', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Presiona 's' para capturar la imagen
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                cap.release()
                cv2.destroyAllWindows()
                return face
            else:
                messagebox.showerror("Error", "Se detectó más de un rostro o ninguno. Inténtalo de nuevo.")
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    return None

def train_model_callback():
    try:
        train_model()
        messagebox.showinfo("Éxito", "Modelo entrenado y guardado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")

def run_recognition_callback():
    try:
        run_recognition()
    except Exception as e:
        messagebox.showerror("Error", f"Error al iniciar el reconocimiento: {e}")

def add_student_callback():
    name = name_entry.get()
    if not name:
        messagebox.showerror("Error", "Por favor, ingrese el nombre del estudiante.")
        return

    face = capture_face()
    if face is None:
        return

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train([face], np.array([0]))  # Entrenar temporalmente con un solo rostro
    face_encoding = face_recognizer.getHistograms()[0]

    try:
        add_student(name, face_encoding)
        messagebox.showinfo("Éxito", "Estudiante agregado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al agregar estudiante: {e}")

# Crear la ventana principal
root = tk.Tk()
root.title("Sistema de Reconocimiento Facial")
root.geometry("800x600")
root.configure(bg="#8B0000")  # Fondo rojo vino

# Estilos personalizados
button_style = {
    "font": ("Helvetica", 14, "bold"),
    "bg": "#1A237E",  # Azul oscuro
    "fg": "white",
    "relief": "flat",
    "borderwidth": 0,
    "highlightthickness": 0,
    "width": 25
}
label_style = {
    "font": ("Helvetica", 12, "bold"),
    "bg": "#8B0000",
    "fg": "white"
}
entry_style = {
    "font": ("Helvetica", 12),
    "bg": "#ffffff",
    "fg": "#000000",
    "relief": "flat",
    "borderwidth": 1,
    "highlightthickness": 1,
    "width": 30
}

# Crear widgets con estilo mejorado
title_label = tk.Label(root, text="Sistema de Reconocimiento Facial", font=("Helvetica", 18, "bold"), bg="#8B0000", fg="white")
title_label.pack(pady=20)

instruction_label = tk.Label(root, text="Ingrese nombre del estudiante:", **label_style)
instruction_label.pack()

name_entry = tk.Entry(root, **entry_style)
name_entry.pack(pady=10)

add_student_button = tk.Button(root, text="Agregar Estudiante", **button_style, command=add_student_callback)
add_student_button.pack(pady=10)

train_button = tk.Button(root, text="Entrenar Modelo", **button_style, command=train_model_callback)
train_button.pack(pady=10)

recognize_button = tk.Button(root, text="Iniciar Reconocimiento", **button_style, command=run_recognition_callback)
recognize_button.pack(pady=10)

generate_excel_button = tk.Button(root, text="Generar Reporte de Asistencia", **button_style, command=generate_excel)
generate_excel_button.pack(pady=10)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
