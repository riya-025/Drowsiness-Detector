import cv2
import time
import math
import mediapipe as mp
import subprocess
import threading
import os


# ==========================================
# MEDIAPIPE
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
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

time.sleep(2)

print("Drowsiness detector started...")


# ==========================================
# SOUND
# ==========================================

sound_file = os.path.abspath("sounds\\alarm.wav")

alarm_running = False


# ==========================================
# CONTINUOUS ALARM
# ==========================================

def continuous_alarm():

    global alarm_running

    while alarm_running:

        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(New-Object Media.SoundPlayer '{sound_file}').PlaySync()"
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# ==========================================
# DISTANCE
# ==========================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ==========================================
# EAR
# ==========================================

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


# ==========================================
# EYE LANDMARKS
# ==========================================

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]


# ==========================================
# SETTINGS
# ==========================================

EAR_THRESHOLD = 0.20
CLOSED_TIME = 2.0


# ==========================================
# MAIN PROGRAM
# ==========================================

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    eyes_closed_start = None

    while True:

        # ======================================
        # READ CAMERA
        # ======================================

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read frame.")

            break


        # ======================================
        # CONVERT IMAGE
        # ======================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1


        # ======================================
        # FACE DETECTION
        # ======================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ======================================
        # FACE FOUND
        # ======================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # ==================================
            # CALCULATE EAR
            # ==================================

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


            # ==================================
            # DISPLAY EAR
            # ==================================

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # ==================================
            # EYES CLOSED
            # ==================================

            if ear < EAR_THRESHOLD:

                if eyes_closed_start is None:

                    eyes_closed_start = time.time()


                closed_duration = (
                    time.time()
                    - eyes_closed_start
                )


                cv2.putText(
                    frame,
                    f"Eyes closed: {closed_duration:.1f}s",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )


                # ==================================
                # DROWSINESS DETECTED
                # ==================================

                if closed_duration >= CLOSED_TIME:

                    cv2.putText(
                        frame,
                        "DROWSINESS DETECTED!",
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "PLEASE STAY ALERT!",
                        (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


                    # ==================================
                    # START CONTINUOUS ALARM
                    # ==================================

                    if not alarm_running:

                        alarm_running = True

                        print(
                            "DROWSINESS DETECTED - ALARM STARTED"
                        )

                        threading.Thread(
                            target=continuous_alarm,
                            daemon=True
                        ).start()


                else:

                    cv2.putText(
                        frame,
                        "EYES CLOSED",
                        (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )


            # ==================================
            # EYES OPEN
            # ==================================

            else:

                eyes_closed_start = None

                # Stop continuous alarm
                alarm_running = False


                cv2.putText(
                    frame,
                    "EYES OPEN",
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


        # ======================================
        # NO FACE
        # ======================================

        else:

            eyes_closed_start = None

            alarm_running = False

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


        # ======================================
        # DISPLAY
        # ======================================

        cv2.imshow(
            "Drowsiness Detector",
            frame
        )


        # ======================================
        # QUIT
        # ======================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            alarm_running = False

            break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Drowsiness detector stopped.")