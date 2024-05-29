import sqlite3
from datetime import datetime

# Configuración de la base de datos
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT,
            face_encoding BLOB
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY,
            student_id INTEGER,
            timestamp TEXT,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    ''')
    conn.commit()
    conn.close()

# Función para agregar un nuevo estudiante a la base de datos
def add_student(name, face_encoding):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('INSERT INTO students (name, face_encoding) VALUES (?, ?)', (name, face_encoding.tobytes()))
    conn.commit()
    conn.close()

# Función para registrar la asistencia
def mark_attendance(student_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('INSERT INTO attendance (student_id, timestamp) VALUES (?, ?)', (student_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

# Función para obtener los estudiantes de la base de datos
def get_students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM students')
    students = c.fetchall()
    conn.close()
    return students
