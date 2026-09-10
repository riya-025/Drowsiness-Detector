from datetime import datetime


class EventLogger:

    def __init__(self, filename="drowsiness_log.txt"):
        self.filename = filename

    def log_event(self, duration):

        current_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(self.filename, "a") as file:

            file.write(
                f"Drowsiness detected | "
                f"Time: {current_time} | "
                f"Duration: {duration:.2f} seconds\n"
            )

        print("📝 Drowsiness event logged.")
