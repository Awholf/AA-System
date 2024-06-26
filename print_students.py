import sqlite3

def print_students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM students')
    students = c.fetchall()
    conn.close()
    
    print("Estudiantes en la base de datos:")
    for student in students:
        print(f"ID: {student[0]}, Nombre: {student[1]}")

if __name__ == "__main__":
    print_students()
