import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from train_model import train_model
from main import run_recognition
from database import init_db, add_student, generate_excel
import numpy as np

# Inicializar la base de datos
init_db()

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("1000x600")
        self.root.configure(bg="#8B0000")  # Fondo rojo vino

        # Intentar diferentes backends
        self.cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
        for backend in backends:
            self.cap = cv2.VideoCapture(0, backend)
            if self.cap.isOpened():
                print(f"Using backend: {backend}")
                break

        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "No se puede acceder a la cámara.")
            self.root.destroy()
            return

        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.RIGHT, padx=10, pady=10)

        self.create_widgets()
        self.update_frame()

    def create_widgets(self):
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

        title_label = tk.Label(self.root, text="Sistema de Reconocimiento Facial", font=("Helvetica", 18, "bold"), bg="#8B0000", fg="white")
        title_label.pack(pady=20)

        instruction_label = tk.Label(self.root, text="Ingrese nombre del estudiante:", **label_style)
        instruction_label.pack()

        self.name_entry = tk.Entry(self.root, **entry_style)
        self.name_entry.pack(pady=10)

        add_student_button = tk.Button(self.root, text="Agregar Estudiante", **button_style, command=self.add_student_callback)
        add_student_button.pack(pady=10)

        train_button = tk.Button(self.root, text="Entrenar Modelo", **button_style, command=self.train_model_callback)
        train_button.pack(pady=10)

        recognize_button = tk.Button(self.root, text="Iniciar Reconocimiento", **button_style, command=self.run_recognition_callback)
        recognize_button.pack(pady=10)

        generate_excel_button = tk.Button(self.root, text="Generar Reporte de Asistencia", **button_style, command=generate_excel)
        generate_excel_button.pack(pady=10)

        self.recognition_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"), bg="#8B0000", fg="white")
        self.recognition_label.pack(pady=20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def capture_face(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "No se puede capturar el frame.")
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        print(f"Rostros detectados: {len(faces)}")  # Agregamos una impresión para ver cuántos rostros se detectan

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = gray[y:y + h, x:x + w]
            return face
        else:
            if len(faces) == 0:
                messagebox.showerror("Error", "No se detectó ningún rostro. Inténtalo de nuevo.")
            else:
                messagebox.showerror("Error", "Se detectó más de un rostro. Inténtalo de nuevo.")
            return None

    def train_model_callback(self):
        try:
            train_model()
            messagebox.showinfo("Éxito", "Modelo entrenado y guardado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")

    def run_recognition_callback(self):
        try:
            name = run_recognition(self.cap)  # Pasar la captura de video a la función de reconocimiento
            self.recognition_label.config(text=f"Reconocido: {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar el reconocimiento: {e}")

    def add_student_callback(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Por favor, ingrese el nombre del estudiante.")
            return

        face = self.capture_face()
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

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()

add_student_button.pack(pady=10)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
