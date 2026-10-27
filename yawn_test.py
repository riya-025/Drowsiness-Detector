
import cv2
import time
import math
import mediapipe as mp


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


# ==========================================
# SETTINGS
# ==========================================

MAR_THRESHOLD = 0.20
YAWN_DURATION = 0.8


# ==========================================
# DISTANCE FUNCTION
# ==========================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# MOUTH ASPECT RATIO
# ==========================================


def calculate_mar(landmarks):
    # Vertical distance between upper and lower inner lips
    vertical = distance(landmarks[13], landmarks[14])

    # Horizontal distance between left and right mouth corners
    horizontal = distance(landmarks[78], landmarks[308])

    if horizontal == 0:
        return 0.0

    return vertical / horizontal

# ==========================================
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

print("Yawn Detection Started. Press Q to quit.")

yawn_start = None
last_timestamp = 0



# ==========================================
# MAIN PROGRAM
# ==========================================

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Could not read camera frame.")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # MediaPipe VIDEO timestamps must increase in milliseconds
            timestamp = int(time.monotonic() * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            # Detect facial landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            status = "NO FACE DETECTED"
            color = (0, 255, 255)
            mar = 0.0

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Calculate MAR
                mar = calculate_mar(landmarks)
                print(f"MAR: {mar:.3f}")

                # Check whether mouth remains open
                if mar > MAR_THRESHOLD:

                    if yawn_start is None:
                        yawn_start = time.monotonic()

                    duration = time.monotonic() - yawn_start

                    if duration >= YAWN_DURATION:
                        status = "YAWNING"
                        color = (0, 0, 255)
                    else:
                        status = "MOUTH OPEN"
                        color = (0, 165, 255)

                else:

                    yawn_start = None
                    status = "MOUTH NORMAL"
                    color = (0, 255, 0)

            else:
                yawn_start = None

            # Display MAR value
            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # Display yawn status
            cv2.putText(
                frame,
                status,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

            # Display threshold for debugging
            cv2.putText(
                frame,
                f"Threshold: {MAR_THRESHOLD:.2f}",
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Yawn Detection System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Yawn Detection Stopped.")
