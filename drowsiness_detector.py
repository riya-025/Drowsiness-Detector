import cv2
import time
import math
import threading
import winsound
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

EAR_THRESHOLD = 0.20
EYES_CLOSED_TIME = 2.0

YAWN_THRESHOLD = 0.25

HEAD_THRESHOLD = 25


# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ==========================================
# ALARM CONTROL
# ==========================================

alarm_playing = False
alarm_stop_event = threading.Event()


def play_alarm():

    global alarm_playing

    alarm_playing = True
    alarm_stop_event.clear()

    while not alarm_stop_event.is_set():

        winsound.PlaySound(
            "sounds/alarm.wav",
            winsound.SND_FILENAME
        )

    alarm_playing = False


def start_alarm():

    global alarm_playing

    if not alarm_playing:

        alarm_thread = threading.Thread(
            target=play_alarm,
            daemon=True
        )

        alarm_thread.start()


def stop_alarm():

    global alarm_playing

    if alarm_playing:

        alarm_stop_event.set()


# ==========================================
# CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

time.sleep(2)

print("Combined Drowsiness Detector started...")


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(landmarks, eye_points):

    vertical_1 = distance(
        landmarks[eye_points[1]],
        landmarks[eye_points[5]]
    )

    vertical_2 = distance(
        landmarks[eye_points[2]],
        landmarks[eye_points[4]]
    )

    horizontal = distance(
        landmarks[eye_points[0]],
        landmarks[eye_points[3]]
    )

    if horizontal == 0:
        return 0

    return (
        vertical_1 + vertical_2
    ) / (2 * horizontal)


def calculate_mar(landmarks):

    vertical = distance(
        landmarks[13],
        landmarks[14]
    )

    horizontal = distance(
        landmarks[78],
        landmarks[308]
    )

    if horizontal == 0:
        return 0

    return vertical / horizontal


# ==========================================
# MAIN PROGRAM
# ==========================================

eyes_closed_start = None


with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break


        height, width, _ = frame.shape


        # Convert frame for MediaPipe
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1


        # Detect face
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # ==========================================
            # EYE DETECTION
            # ==========================================

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear + right_ear
            ) / 2


            if ear < EAR_THRESHOLD:

                eyes_status = "CLOSED"

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()

                closed_duration = (
                    time.time() -
                    eyes_closed_start
                )

            else:

                eyes_status = "OPEN"

                eyes_closed_start = None

                closed_duration = 0

                stop_alarm()


            # ==========================================
            # DROWSINESS DETECTION
            # ==========================================

            if closed_duration >= EYES_CLOSED_TIME:

                drowsiness_status = (
                    "DROWSINESS DETECTED!"
                )

                start_alarm()

            else:

                drowsiness_status = (
                    "DRIVER ALERT"
                )


            # ==========================================
            # YAWN DETECTION
            # ==========================================

            mar = calculate_mar(
                landmarks
            )


            if mar > YAWN_THRESHOLD:

                mouth_status = "YAWNING"

            else:

                mouth_status = "NORMAL"


            # ==========================================
            # HEAD DETECTION
            # ==========================================

            nose = landmarks[1]

            left_eye = landmarks[33]

            right_eye = landmarks[263]


            nose_x = int(
                nose.x * width
            )

            left_eye_x = int(
                left_eye.x * width
            )

            right_eye_x = int(
                right_eye.x * width
            )


            eye_center_x = (
                left_eye_x +
                right_eye_x
            ) / 2


            head_offset = (
                nose_x -
                eye_center_x
            )


            if head_offset > HEAD_THRESHOLD:

                head_status = "TURNED LEFT"

            elif head_offset < -HEAD_THRESHOLD:

                head_status = "TURNED RIGHT"

            else:

                head_status = "FORWARD"


            # ==========================================
            # DISPLAY INFORMATION
            # ==========================================

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Eyes: {eyes_status}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Mouth: {mouth_status}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Head: {head_status}",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                drowsiness_status,
                (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
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


        # Show camera window
        cv2.imshow(
            "Driver Drowsiness Detection System",
            frame
        )


        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# CLEANUP
# ==========================================

stop_alarm()

cap.release()

cv2.destroyAllWindows()

print("Drowsiness Detector stopped.")