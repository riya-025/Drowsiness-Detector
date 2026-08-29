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

print("Head movement detector started...")


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

        # Get frame dimensions
        height, width, _ = frame.shape

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

            # Important facial landmarks
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]

            # Convert normalized coordinates to pixels
            nose_x = int(nose.x * width)

            left_eye_x = int(left_eye.x * width)
            right_eye_x = int(right_eye.x * width)

            # Calculate eye midpoint
            eye_center_x = (left_eye_x + right_eye_x) / 2

            # Compare nose position with eye center
            head_offset = nose_x - eye_center_x

            # Display value
            cv2.putText(
                frame,
                f"Head Offset: {head_offset:.0f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Detect head direction
            if head_offset > 25:

                status = "HEAD TURNED LEFT"
                color = (0, 0, 255)

            elif head_offset < -25:

                status = "HEAD TURNED RIGHT"
                color = (0, 0, 255)

            else:

                status = "HEAD FORWARD"
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
            "Head Movement Test",
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

print("Head movement detector stopped.")