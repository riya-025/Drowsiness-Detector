import sqlite3
from datetime import datetime

DATABASE = "drowsiness.db"


def create_database():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            eye_status TEXT,
            ear_value REAL,
            mouth_status TEXT,
            mar_value REAL,
            head_status TEXT,
            alert_status TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_detection(
    eye_status,
    ear_value,
    mouth_status,
    mar_value,
    head_status,
    alert_status
):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    cursor.execute("""
        INSERT INTO detection_logs
        (
            date,
            time,
            eye_status,
            ear_value,
            mouth_status,
            mar_value,
            head_status,
            alert_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date,
        time,
        eye_status,
        ear_value,
        mouth_status,
        mar_value,
        head_status,
        alert_status
    ))

    conn.commit()
    conn.close()

    print("Detection saved successfully!")