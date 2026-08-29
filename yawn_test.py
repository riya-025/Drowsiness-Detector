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
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

time.sleep(2)

print("Yawn detector started...")


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

    # Vertical mouth opening
    vertical = distance(
        landmarks[13],
        landmarks[14]
    )

    # Horizontal mouth width
    horizontal = distance(
        landmarks[78],
        landmarks[308]
    )

    if horizontal == 0:
        return 0

    return vertical / horizontal


# ==========================================
# SETTINGS
# ==========================================

YAWN_THRESHOLD = 0.25


# ==========================================
# MAIN PROGRAM
# ==========================================

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
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

        timestamp += 1


        # Detect face landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Calculate Mouth Aspect Ratio
            mar = calculate_mar(landmarks)


            # Display MAR value
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # Detect yawning
            if mar > YAWN_THRESHOLD:

                status = "YAWNING"

                color = (0, 0, 255)

            else:

                status = "MOUTH NORMAL"

                color = (0, 255, 0)


            # Display status
            cv2.putText(
                frame,
                status,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )


        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )


        # Show camera
        cv2.imshow(
            "Yawn Detection Test",
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Yawn detector stopped.")