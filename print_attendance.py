import sqlite3

def print_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('SELECT * FROM attendance')
    attendance_records = c.fetchall()
    conn.close()
    
    print("Registros de asistencia en la base de datos:")
    for record in attendance_records:
        print(f"ID: {record[0]}, Student ID: {record[1]}, Timestamp: {record[2]}")

if __name__ == "__main__":
    #db_path = r'/mnt/data/AA-System-main/attendance.db'  # Ruta completa a la base de datos
    print_attendance()

