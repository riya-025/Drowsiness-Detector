import time


class DrowsinessMonitor:

    def __init__(self, threshold=2):
        self.threshold = threshold
        self.drowsy_count = 0
        self.total_events = 0
        self.drowsy_start_time = None

    def update(self, is_drowsy):

        if is_drowsy:

            self.drowsy_count += 1

            if self.drowsy_start_time is None:
                self.drowsy_start_time = time.time()

        else:

            self.drowsy_count = 0
            self.drowsy_start_time = None

    def is_drowsy(self):

        return self.drowsy_count >= self.threshold

    def get_duration(self):

        if self.drowsy_start_time is None:
            return 0

        return time.time() - self.drowsy_start_time

    def register_event(self):

        self.total_events += 1
