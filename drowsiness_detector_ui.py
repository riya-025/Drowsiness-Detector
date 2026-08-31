import cv2
import time
import math
import threading
import winsound
import mediapipe as mp
import numpy as np
from collections import deque

from database import create_database, save_detection


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

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


# ============================================================
# DETECTION SETTINGS
# ============================================================

EAR_THRESHOLD = 0.20
EYES_CLOSED_TIME = 2.0
YAWN_THRESHOLD = 0.25
HEAD_THRESHOLD = 25

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# DASHBOARD SETTINGS
# ============================================================

WIDTH = 1280
HEIGHT = 800

BG = (12, 12, 18)
PANEL = (22, 22, 31)
PANEL2 = (28, 28, 40)

WHITE = (235, 235, 245)
TEXT = (175, 180, 195)
BLUE = (255, 170, 40)
LIGHT_BLUE = (255, 210, 90)

SAFE = (80, 210, 120)
WARNING = (70, 80, 235)
ORANGE = (60, 150, 245)
GRID = (30, 30, 42)


# ============================================================
# GLOBAL STATE
# ============================================================

alarm_playing = False
alarm_stop_event = threading.Event()

night_mode = False
alarm_muted = False

total_blinks = 0
blink_history = deque(maxlen=100)

previous_eyes_status = "OPEN"
last_blink_time = 0

last_saved_status = None
last_save_time = 0
SAVE_INTERVAL = 5


# ============================================================
# ALARM
# ============================================================

def play_alarm():

    global alarm_playing

    alarm_playing = True
    alarm_stop_event.clear()

    while not alarm_stop_event.is_set():

        try:
            winsound.PlaySound(
                "sounds/alarm.wav",
                winsound.SND_FILENAME
            )
        except Exception:
            break

    alarm_playing = False


def start_alarm():

    global alarm_playing

    if alarm_muted:
        return

    if not alarm_playing:

        threading.Thread(
            target=play_alarm,
            daemon=True
        ).start()


def stop_alarm():

    global alarm_playing

    alarm_stop_event.set()

    try:
        winsound.PlaySound(
            None,
            winsound.SND_PURGE
        )
    except Exception:
        pass

    alarm_playing = False


# ============================================================
# MATH
# ============================================================

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


# ============================================================
# UI FUNCTIONS
# ============================================================

def text(
    img,
    value,
    position,
    size=0.6,
    color=WHITE,
    thickness=1
):

    cv2.putText(
        img,
        str(value),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def rounded_panel(
    img,
    x,
    y,
    w,
    h
):

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        PANEL,
        -1
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (55, 55, 70),
        1
    )


def draw_grid(img):

    for x in range(0, WIDTH, 40):

        cv2.line(
            img,
            (x, 0),
            (x, HEIGHT),
            GRID,
            1
        )

    for y in range(0, HEIGHT, 40):

        cv2.line(
            img,
            (0, y),
            (WIDTH, y),
            GRID,
            1
        )


def draw_metric(
    img,
    label,
    value,
    x,
    y,
    color=WHITE
):

    text(
        img,
        label,
        (x, y),
        0.40,
        TEXT,
        1
    )

    text(
        img,
        value,
        (x, y + 31),
        0.70,
        color,
        2
    )


def draw_button(
    img,
    x,
    y,
    w,
    h,
    label,
    active=False
):

    fill = PANEL2 if not active else (45, 45, 60)

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        fill,
        -1
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        LIGHT_BLUE if active else (60, 60, 75),
        1
    )

    size = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        1
    )[0]

    tx = x + (w - size[0]) // 2
    ty = y + (h + size[1]) // 2

    text(
        img,
        label,
        (tx, ty),
        0.48,
        LIGHT_BLUE if active else TEXT,
        1
    )


def draw_blink_graph(img):

    rounded_panel(
        img,
        40,
        650,
        800,
        120
    )

    text(
        img,
        "BLINK ACTIVITY",
        (60, 680),
        0.48,
        LIGHT_BLUE,
        1
    )

    text(
        img,
        f"{total_blinks:03d}",
        (60, 725),
        0.75,
        WHITE,
        2
    )

    gx = 140
    gy = 675
    gw = 670
    gh = 70

    points = list(blink_history)

    if len(points) < 2:
        return

    max_value = max(
        5,
        max(v for _, v in points) + 1
    )

    first_time = points[0][0]
    last_time = max(
        points[-1][0],
        first_time + 1
    )

    graph = []

    for timestamp, value in points:

        px = gx + int(
            (timestamp - first_time) /
            (last_time - first_time) *
            gw
        )

        py = gy + gh - int(
            value / max_value * gh
        )

        graph.append(
            (px, py)
        )

    for a, b in zip(graph, graph[1:]):

        cv2.line(
            img,
            a,
            b,
            LIGHT_BLUE,
            2,
            cv2.LINE_AA
        )


# ============================================================
# DATABASE
# ============================================================

create_database()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

time.sleep(2)


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "Driver Drowsiness Detection"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    WIDTH,
    HEIGHT
)


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global alarm_muted
    global night_mode

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # MUTE
    if 875 <= x <= 1235 and 525 <= y <= 570:

        alarm_muted = not alarm_muted

        if alarm_muted:
            stop_alarm()

    # NIGHT MODE
    elif 875 <= x <= 1235 and 585 <= y <= 630:

        night_mode = not night_mode


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# DETECTION
# ============================================================

eyes_closed_start = None

print("Driver Drowsiness Detection started...")
print("Press Q to exit.")


with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Could not read frame.")
            break


        frame = cv2.flip(
            frame,
            1
        )

        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        ear = 0.0
        mar = 0.0

        eyes_status = "NO FACE"
        mouth_status = "NO FACE"
        head_status = "NO FACE"

        drowsiness_status = "SEARCHING"


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # ------------------------------------------------
            # EAR
            # ------------------------------------------------

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                left_ear +
                right_ear
            ) / 2


            # ------------------------------------------------
            # EYES
            # ------------------------------------------------

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


            # ------------------------------------------------
            # DROWSINESS
            # ------------------------------------------------

            if closed_duration >= EYES_CLOSED_TIME:

                drowsiness_status = "DROWSINESS DETECTED"

                start_alarm()

            else:

                drowsiness_status = "DRIVER ALERT"


            # ------------------------------------------------
            # BLINK
            # ------------------------------------------------

            if (
                previous_eyes_status == "CLOSED"
                and eyes_status == "OPEN"
            ):

                current = time.time()

                if current - last_blink_time > 0.20:

                    total_blinks += 1
                    last_blink_time = current

            previous_eyes_status = eyes_status


            # ------------------------------------------------
            # YAWN
            # ------------------------------------------------

            mar = calculate_mar(
                landmarks
            )

            if mar > YAWN_THRESHOLD:
                mouth_status = "YAWNING"
            else:
                mouth_status = "NORMAL"


            # ------------------------------------------------
            # HEAD
            # ------------------------------------------------

            nose = landmarks[1]

            left_eye = landmarks[33]

            right_eye = landmarks[263]


            nose_x = int(
                nose.x * frame_width
            )

            left_eye_x = int(
                left_eye.x * frame_width
            )

            right_eye_x = int(
                right_eye.x * frame_width
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


            # ------------------------------------------------
            # DATABASE
            # ------------------------------------------------

            current_time = time.time()

            if (
                drowsiness_status != last_saved_status
                or
                current_time - last_save_time >= SAVE_INTERVAL
            ):

                save_detection(
                    eye_status=eyes_status,
                    ear_value=ear,
                    mouth_status=mouth_status,
                    mar_value=mar,
                    head_status=head_status,
                    alert_status=drowsiness_status
                )

                last_saved_status = drowsiness_status
                last_save_time = current_time


        else:

            previous_eyes_status = "NO FACE"

            stop_alarm()


        # ====================================================
        # BLINK GRAPH
        # ====================================================

        current = time.time()

        if (
            not blink_history
            or current - blink_history[-1][0] >= 0.25
        ):

            blink_history.append(
                (current, total_blinks)
            )


        # ====================================================
        # DASHBOARD
        # ====================================================

        dashboard = np.zeros(
            (
                HEIGHT,
                WIDTH,
                3
            ),
            dtype=np.uint8
        )

        dashboard[:] = BG

        draw_grid(
            dashboard
        )


        # ====================================================
        # HEADER
        # ====================================================

        text(
            dashboard,
            "DRIVER",
            (40, 55),
            1.05,
            WHITE,
            2
        )

        text(
            dashboard,
            "DROWSINESS DETECTION",
            (175, 55),
            0.68,
            LIGHT_BLUE,
            2
        )

        text(
            dashboard,
            "AI DRIVER MONITORING SYSTEM",
            (42, 82),
            0.38,
            TEXT,
            1
        )


        # ====================================================
        # STATUS
        # ====================================================

        if drowsiness_status == "DROWSINESS DETECTED":

            state = "WARNING"
            state_color = WARNING

        elif drowsiness_status == "DRIVER ALERT":

            state = "SAFE"
            state_color = SAFE

        else:

            state = "SCANNING"
            state_color = LIGHT_BLUE


        cv2.circle(
            dashboard,
            (1000, 45),
            7,
            state_color,
            -1
        )

        text(
            dashboard,
            state,
            (1020, 53),
            0.55,
            state_color,
            2
        )

        text(
            dashboard,
            "LIVE",
            (1160, 53),
            0.38,
            TEXT,
            1
        )


        # ====================================================
        # CAMERA
        # ====================================================

        rounded_panel(
            dashboard,
            40,
            110,
            800,
            520
        )

        text(
            dashboard,
            "LIVE CAMERA",
            (60, 140),
            0.48,
            LIGHT_BLUE,
            1
        )

        text(
            dashboard,
            "MEDIAPIPE FACE LANDMARKER",
            (590, 140),
            0.36,
            TEXT,
            1
        )


        cam_x = 55
        cam_y = 155
        cam_w = 770
        cam_h = 455


        camera_view = cv2.resize(
            frame,
            (cam_w, cam_h),
            interpolation=cv2.INTER_AREA
        )


        dashboard[
            cam_y:cam_y + cam_h,
            cam_x:cam_x + cam_w
        ] = camera_view


        cv2.rectangle(
            dashboard,
            (cam_x, cam_y),
            (cam_x + cam_w, cam_y + cam_h),
            LIGHT_BLUE,
            1
        )


        # ====================================================
        # RIGHT PANEL
        # ====================================================

        rounded_panel(
            dashboard,
            860,
            110,
            375,
            380
        )

        text(
            dashboard,
            "DRIVER TELEMETRY",
            (885, 140),
            0.48,
            LIGHT_BLUE,
            1
        )


        text(
            dashboard,
            state,
            (885, 190),
            0.90,
            state_color,
            2
        )

        text(
            dashboard,
            "CURRENT DRIVER STATE",
            (885, 215),
            0.34,
            TEXT,
            1
        )


        draw_metric(
            dashboard,
            "EYES",
            eyes_status,
            885,
            255,
            SAFE if eyes_status == "OPEN" else WARNING
        )

        draw_metric(
            dashboard,
            "EAR",
            f"{ear:.2f}",
            1080,
            255,
            LIGHT_BLUE
        )


        draw_metric(
            dashboard,
            "MOUTH",
            mouth_status,
            885,
            345,
            WARNING if mouth_status == "YAWNING" else SAFE
        )

        draw_metric(
            dashboard,
            "MAR",
            f"{mar:.2f}",
            1080,
            345,
            LIGHT_BLUE
        )


        draw_metric(
            dashboard,
            "HEAD",
            head_status,
            885,
            435,
            SAFE if head_status == "FORWARD" else WARNING
        )


        # ====================================================
        # CONTROLS
        # ====================================================

        draw_button(
            dashboard,
            875,
            525,
            360,
            45,
            "ALARM MUTED" if alarm_muted else "MUTE ALARM",
            alarm_muted
        )

        draw_button(
            dashboard,
            875,
            585,
            360,
            45,
            "NIGHT MODE: ON" if night_mode else "NIGHT MODE: OFF",
            night_mode
        )


        # ====================================================
        # ALERT
        # ====================================================

        if drowsiness_status == "DROWSINESS DETECTED":

            cv2.rectangle(
                dashboard,
                (40, 595),
                (840, 630),
                WARNING,
                -1
            )

            text(
                dashboard,
                "DROWSINESS DETECTED  //  ALARM ACTIVE",
                (260, 619),
                0.46,
                WHITE,
                1
            )

        else:

            cv2.rectangle(
                dashboard,
                (40, 595),
                (840, 630),
                (30, 80, 45),
                -1
            )

            text(
                dashboard,
                "DRIVER ALERT  //  CONTINUOUS MONITORING",
                (245, 619),
                0.46,
                SAFE,
                1
            )


        # ====================================================
        # BLINK GRAPH
        # ====================================================

        draw_blink_graph(
            dashboard
        )


        # ====================================================
        # SYSTEM INFO
        # ====================================================

        rounded_panel(
            dashboard,
            860,
            650,
            375,
            120
        )

        text(
            dashboard,
            "SYSTEM",
            (885, 680),
            0.42,
            LIGHT_BLUE,
            1
        )

        text(
            dashboard,
            f"BLINK COUNT     {total_blinks:03d}",
            (885, 720),
            0.43,
            WHITE,
            1
        )

        text(
            dashboard,
            "Q  =  EXIT",
            (1080, 720),
            0.43,
            TEXT,
            1
        )


        # ====================================================
        # NIGHT MODE
        # ====================================================

        if night_mode:

            dashboard = (
                dashboard.astype(np.float32) * 0.55
            ).astype(np.uint8)


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            dashboard
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

stop_alarm()

cap.release()

cv2.destroyAllWindows()

print("Driver Drowsiness Detection stopped.")