import sqlite3
import pandas as pd
from datetime import datetime
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl import Workbook


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


# Función para ver la lista de estudiantes
def view_students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM students')
    students = c.fetchall()
    print("Estudiantes:")
    for student in students:
        print(student)
    conn.close()


# Función para ver los registros de asistencia
def view_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM attendance')
    attendance_records = c.fetchall()
    print("Asistencia:")
    for record in attendance_records:
        print(record)
    conn.close()


# Función para generar el archivo Excel
def generate_excel():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    query = '''
        SELECT students.id, students.name, attendance.timestamp 
        FROM attendance
        JOIN students ON attendance.student_id = students.id
        ORDER BY students.name, attendance.timestamp
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Separar fecha y hora
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    # Crear un libro de Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance Report"

    # Escribir los encabezados
    headers = ['ID', 'Name', 'Date', 'Time']
    ws.append(headers)

    # Agrupar por nombre y fecha, y escribir los datos
    grouped = df.groupby(['id', 'name', 'date'])

    for (student_id, name, date), group in grouped:
        first_time = group['time'].iloc[0]
        ws.append([student_id, name, date, first_time])

    # Guardar el libro de Excel
    wb.save("Reporte de asistencia.xlsx")
    print("Archivo Excel generado exitosamente.")
