import time
import winsound


class AlertSystem:

    def __init__(self):
        self.last_alert_time = 0
        self.alert_cooldown = 3

    def warning(self):
        print("WARNING: Driver is drowsy!")

    def alarm(self):
        current_time = time.time()

        if current_time - self.last_alert_time >= self.alert_cooldown:

            print("ALARM! WAKE UP!")

            # Play alarm sound
            winsound.Beep(1000, 500)
            winsound.Beep(1200, 500)
            winsound.Beep(1000, 500)

            self.last_alert_time = current_time

    def trigger_alert(self):
        self.warning()
        self.alarm()
