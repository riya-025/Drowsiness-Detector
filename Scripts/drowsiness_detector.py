class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=20):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.closed_frames = 0

    def update(self, ear):
        if ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        return self.closed_frames >= self.consecutive_frames

    def reset(self):
        self.closed_frames = 0