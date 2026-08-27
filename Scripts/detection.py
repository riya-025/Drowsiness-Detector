from Scripts.face_landmarks import FaceLandmarks
from Scripts.eye_detector import get_both_eyes
from Scripts.eye_aspect_ratio import eye_aspect_ratio
from Scripts.drowsiness_detector import DrowsinessDetector


class DrowsinessDetection:
    def __init__(self, ear_threshold=0.25, consecutive_frames=20):
        self.face_landmarks = FaceLandmarks()

        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            consecutive_frames=consecutive_frames
        )

    def process_frame(self, frame):
        height, width = frame.shape[:2]

        # Detect facial landmarks
        landmarks = self.face_landmarks.detect(frame)

        if landmarks is None:
            return {
                "ear": None,
                "drowsy": False,
                "face_detected": False
            }

        # Extract left and right eye points
        left_eye, right_eye = get_both_eyes(
            landmarks,
            width,
            height
        )

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average EAR
        ear = (left_ear + right_ear) / 2.0

        # Determine drowsiness
        drowsy = self.drowsiness_detector.update(ear)

        return {
            "ear": ear,
            "drowsy": drowsy,
            "face_detected": True
        }

    def close(self):
        self.face_landmarks.close()