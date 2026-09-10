import sqlite3

DATABASE = "drowsiness.db"


def show_history():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            date,
            time,
            eye_status,
            ear_value,
            mouth_status,
            mar_value,
            head_status,
            alert_status
        FROM detection_logs
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    print("\n==============================================")
    print("        DRIVER DROWSINESS HISTORY")
    print("==============================================")

    if not records:
        print("No detection records found.")
        return

    for record in records:

        print(f"""
ID       : {record[0]}
Date     : {record[1]}
Time     : {record[2]}
Eyes     : {record[3]}
EAR      : {record[4]:.2f}
Mouth    : {record[5]}
MAR      : {record[6]:.2f}
Head     : {record[7]}
Status   : {record[8]}
----------------------------------------------
""")


if __name__ == "__main__":
    show_history()