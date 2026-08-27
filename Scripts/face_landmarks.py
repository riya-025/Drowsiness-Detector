import cv2
import mediapipe as mp
from pathlib import Path


class FaceLandmarks:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "face_landmarker.task"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {model_path}"
            )

        self.base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                self.options
            )
        )

        self.timestamp_ms = 0

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp_ms += 1

        results = self.face_landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        if not results.face_landmarks:
            return None

        return results.face_landmarks[0]

    def close(self):
        self.face_landmarker.close()