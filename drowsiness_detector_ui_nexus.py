# NEXUS // Driver Intelligence
# Futuristic dashboard UI for the existing MediaPipe + OpenCV + SQLite project.
# No new Python packages are required.

import cv2
import time
import math
import threading
import winsound
from collections import deque
import mediapipe as mp

from database import create_database, save_detection

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
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
# NEXUS UI SETTINGS
# ============================================================
DASH_W = 1280
DASH_H = 800
CAM_X, CAM_Y = 40, 145
CAM_W, CAM_H = 760, 430
GRAPH_X, GRAPH_Y = 40, 610
GRAPH_W, GRAPH_H = 760, 145
RIGHT_X = 835

BG = (25, 15, 11)          # BGR for #0B0F19
CYAN = (255, 220, 0)       # electric cyan
CYAN2 = (255, 150, 0)
GREEN = (60, 255, 80)
RED = (60, 60, 255)
WHITE = (235, 245, 250)
MUTED = (120, 145, 160)
PANEL = (32, 24, 20)
GRID = (45, 38, 30)

night_mode = False
alarm_muted = False
running = True

# Blink graph: store (time, cumulative blink count) points.
blink_history = deque(maxlen=90)
total_blinks = 0

# ============================================================
# ALARM CONTROL
# ============================================================
alarm_playing = False
alarm_stop_event = threading.Event()


def play_alarm():
    global alarm_playing
    alarm_playing = True
    alarm_stop_event.clear()

    while not alarm_stop_event.is_set():
        try:
            winsound.PlaySound("sounds/alarm.wav", winsound.SND_FILENAME)
        except Exception:
            break

    alarm_playing = False


def start_alarm():
    global alarm_playing
    if alarm_muted:
        return
    if not alarm_playing:
        threading.Thread(target=play_alarm, daemon=True).start()


def stop_alarm():
    global alarm_playing
    alarm_stop_event.set()
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass
    alarm_playing = False


# ============================================================
# MATH HELPERS
# ============================================================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_ear(landmarks, eye_points):
    vertical_1 = distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    vertical_2 = distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    horizontal = distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    if horizontal == 0:
        return 0
    return (vertical_1 + vertical_2) / (2 * horizontal)


def calculate_mar(landmarks):
    vertical = distance(landmarks[13], landmarks[14])
    horizontal = distance(landmarks[78], landmarks[308])
    if horizontal == 0:
        return 0
    return vertical / horizontal


# ============================================================
# UI HELPERS
# ============================================================
def draw_text(img, text, xy, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def panel(img, x, y, w, h, title=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), CYAN, 1)
    # Neon corner brackets
    L = 18
    T = 2
    for sx, sy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        dx = 1 if sx == x else -1
        dy = 1 if sy == y else -1
        cv2.line(img, (sx, sy), (sx + dx * L, sy), CYAN, T)
        cv2.line(img, (sx, sy), (sx, sy + dy * L), CYAN, T)
    if title:
        draw_text(img, title, (x + 18, y + 30), 0.55, CYAN, 1)


def button(img, x, y, w, h, label, active=False, danger=False):
    fill = (40, 30, 26) if not active else (65, 45, 30)
    border = RED if danger else CYAN
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    tx = x + (w - size[0]) // 2
    ty = y + (h + size[1]) // 2
    draw_text(img, label, (tx, ty), 0.52, border, 1)


def draw_grid(img):
    # Subtle HUD grid
    for x in range(0, DASH_W, 40):
        cv2.line(img, (x, 0), (x, DASH_H), GRID, 1)
    for y in range(0, DASH_H, 40):
        cv2.line(img, (0, y), (DASH_W, y), GRID, 1)


def draw_status_led(img, x, y, color):
    cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 12, color, 1, cv2.LINE_AA)


def draw_blink_graph(img):
    panel(img, GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H, "LIVE BLINK TELEMETRY")

    gx, gy = GRAPH_X + 55, GRAPH_Y + 45
    gw, gh = GRAPH_W - 75, GRAPH_H - 60

    # Graph grid
    for i in range(1, 5):
        yy = gy + int(gh * i / 5)
        cv2.line(img, (gx, yy), (gx + gw, yy), GRID, 1)
    for i in range(1, 7):
        xx = gx + int(gw * i / 7)
        cv2.line(img, (xx, gy), (xx, gy + gh), GRID, 1)

    draw_text(img, "BLINKS", (GRAPH_X + 10, GRAPH_Y + 63), 0.38, MUTED, 1)
    draw_text(img, str(total_blinks), (GRAPH_X + 10, GRAPH_Y + 105), 0.8, CYAN, 1)

    points = list(blink_history)
    if len(points) >= 2:
        max_value = max(5, max(v for _, v in points) + 1)
        t0 = points[0][0]
        t1 = max(points[-1][0], t0 + 1)
        graph_points = []
        for t, value in points:
            px = gx + int((t - t0) / (t1 - t0) * gw)
            py = gy + gh - int(value / max_value * gh)
            graph_points.append((px, py))
        for a, b in zip(graph_points, graph_points[1:]):
            cv2.line(img, a, b, CYAN, 2, cv2.LINE_AA)
        cv2.circle(img, graph_points[-1], 3, CYAN, -1, cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    global alarm_muted, night_mode
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Mute alarm
    if 850 <= x <= 1225 and 555 <= y <= 600:
        alarm_muted = not alarm_muted
        if alarm_muted:
            stop_alarm()
        return

    # Night mode
    if 850 <= x <= 1225 and 620 <= y <= 665:
        night_mode = not night_mode
        return

    # Calibrate camera: reset camera exposure/size and restart the capture settings.
    if 850 <= x <= 1225 and 490 <= y <= 535:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass


# ============================================================
# DATABASE + CAMERA
# ============================================================
create_database()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(1)

cv2.namedWindow("NEXUS // Driver Intelligence", cv2.WINDOW_NORMAL)
cv2.resizeWindow("NEXUS // Driver Intelligence", DASH_W, DASH_H)
cv2.setMouseCallback("NEXUS // Driver Intelligence", mouse_callback)

print("NEXUS // Driver Intelligence started...")
print("Click buttons inside the dashboard. Press Q to exit.")

# Detection state
closed_start = None
last_saved_status = None
last_save_time = 0
SAVE_INTERVAL = 5
previous_eyes_status = "OPEN"

with FaceLandmarker.create_from_options(options) as landmarker:
    timestamp = 0
    last_blink_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp += 1
        result = landmarker.detect_for_video(mp_image, timestamp)

        ear = 0.0
        mar = 0.0
        eyes_status = "NO FACE"
        mouth_status = "NO FACE"
        head_status = "NO FACE"
        drowsiness_status = "SEARCHING FOR DRIVER"

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            if ear < EAR_THRESHOLD:
                eyes_status = "CLOSED"
                if closed_start is None:
                    closed_start = time.time()
                closed_duration = time.time() - closed_start
            else:
                eyes_status = "OPEN"
                closed_start = None
                closed_duration = 0
                stop_alarm()

            # Blink = a closed->open transition that was not already counted very recently.
            if previous_eyes_status == "CLOSED" and eyes_status == "OPEN":
                now = time.time()
                if now - last_blink_time > 0.20:
                    total_blinks += 1
                    last_blink_time = now
            previous_eyes_status = eyes_status

            if closed_duration >= EYES_CLOSED_TIME:
                drowsiness_status = "DROWSINESS DETECTED"
                start_alarm()
            else:
                drowsiness_status = "DRIVER ALERT"

            mar = calculate_mar(landmarks)
            mouth_status = "YAWNING" if mar > YAWN_THRESHOLD else "NORMAL"

            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_x = int(nose.x * width)
            left_eye_x = int(left_eye.x * width)
            right_eye_x = int(right_eye.x * width)
            eye_center_x = (left_eye_x + right_eye_x) / 2
            head_offset = nose_x - eye_center_x

            if head_offset > HEAD_THRESHOLD:
                head_status = "TURNED LEFT"
            elif head_offset < -HEAD_THRESHOLD:
                head_status = "TURNED RIGHT"
            else:
                head_status = "FORWARD"

            current_time = time.time()
            if drowsiness_status != last_saved_status or current_time - last_save_time >= SAVE_INTERVAL:
                save_detection(
                    eye_status=eyes_status,
                    ear_value=ear,
                    mouth_status=mouth_status,
                    mar_value=mar,
                    head_status=head_status,
                    alert_status=drowsiness_status,
                )
                last_saved_status = drowsiness_status
                last_save_time = current_time

        else:
            previous_eyes_status = "NO FACE"
            stop_alarm()

        # Keep the graph moving in real time.
        now = time.time()
        if not blink_history or now - blink_history[-1][0] >= 0.25:
            blink_history.append((now, total_blinks))

        # ========================================================
        # BUILD DASHBOARD
        # ========================================================
        dashboard = BG * 0 + BG
        dashboard = __import__('numpy').zeros((DASH_H, DASH_W, 3), dtype='uint8')
        dashboard[:] = BG
        draw_grid(dashboard)

        # Header
        draw_text(dashboard, "NEXUS", (40, 55), 1.25, WHITE, 2)
        draw_text(dashboard, "// DRIVER INTELLIGENCE", (190, 55), 0.72, CYAN, 2)
        draw_text(dashboard, "AI DRIVER MONITORING SYSTEM", (40, 85), 0.38, MUTED, 1)

        # Right top status
        active = drowsiness_status != "DROWSINESS DETECTED"
        status_text = "SYSTEM ACTIVE" if active else "WARNING"
        status_color = GREEN if active else RED
        blink_on = int(time.time() * 3) % 2 == 0
        if blink_on or active:
            draw_status_led(dashboard, 855, 48, status_color)
        draw_text(dashboard, status_text, (875, 55), 0.62, status_color, 2)
        draw_text(dashboard, "LIVE / ONLINE", (1080, 55), 0.38, MUTED, 1)

        # Camera panel
        panel(dashboard, CAM_X, CAM_Y, CAM_W, CAM_H, "LIVE CAMERA // OPTICAL SENSOR")
        inner_x, inner_y = CAM_X + 12, CAM_Y + 45
        inner_w, inner_h = CAM_W - 24, CAM_H - 58
        cam = cv2.resize(frame, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        dashboard[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w] = cam

        # Add small HUD marks over camera
        cv2.rectangle(dashboard, (inner_x, inner_y), (inner_x + inner_w, inner_y + inner_h), CYAN, 1)
        draw_text(dashboard, "FACE TRACKING: ON", (inner_x + 12, inner_y + 25), 0.4, GREEN if result.face_landmarks else RED, 1)
        draw_text(dashboard, "MEDIAPIPE // FACE LANDMARKER", (inner_x + 12, inner_y + inner_h - 12), 0.35, CYAN, 1)

        # Right telemetry panel
        panel(dashboard, RIGHT_X, 145, 405, 330, "DRIVER TELEMETRY")

        if drowsiness_status == "DROWSINESS DETECTED":
            main_color = RED
            state = "WARNING"
        elif drowsiness_status == "DRIVER ALERT":
            main_color = GREEN
            state = "SAFE"
        else:
            main_color = CYAN
            state = "SCANNING"

        draw_text(dashboard, state, (RIGHT_X + 25, 205), 1.05, main_color, 2)
        draw_text(dashboard, "DRIVER STATE", (RIGHT_X + 25, 230), 0.36, MUTED, 1)

        def telemetry_row(label, value, yy, value_color=WHITE):
            draw_text(dashboard, label, (RIGHT_X + 25, yy), 0.45, MUTED, 1)
            draw_text(dashboard, value, (RIGHT_X + 180, yy), 0.52, value_color, 1)
            cv2.line(dashboard, (RIGHT_X + 25, yy + 12), (RIGHT_X + 375, yy + 12), GRID, 1)

        telemetry_row("EYES", eyes_status, 265, GREEN if eyes_status == "OPEN" else RED)
        telemetry_row("EAR", f"{ear:.2f}", 305, CYAN)
        telemetry_row("MOUTH", mouth_status, 345, RED if mouth_status == "YAWNING" else GREEN)
        telemetry_row("MAR", f"{mar:.2f}", 385, CYAN)
        telemetry_row("HEAD", head_status, 425, GREEN if head_status == "FORWARD" else RED)

        # Controls
        button(dashboard, 850, 490, 375, 45, "CALIBRATE CAMERA")
        button(dashboard, 850, 550, 375, 45, "MUTE ALARM" if not alarm_muted else "ALARM MUTED", active=alarm_muted, danger=alarm_muted)
        button(dashboard, 850, 610, 375, 45, "NIGHT MODE: ON" if night_mode else "NIGHT MODE: OFF", active=night_mode)

        # System info panel
        panel(dashboard, 850, 675, 375, 80, "SYSTEM")
        draw_text(dashboard, f"BLINK COUNT  {total_blinks:03d}", (870, 725), 0.43, CYAN, 1)
        draw_text(dashboard, "Q = EXIT", (1100, 725), 0.43, MUTED, 1)

        draw_blink_graph(dashboard)

        # Alert banner across bottom of camera area
        if drowsiness_status == "DROWSINESS DETECTED":
            cv2.rectangle(dashboard, (40, 570), (800, 595), RED, -1)
            draw_text(dashboard, "!!! DROWSINESS DETECTED // ALARM ACTIVE !!!", (225, 589), 0.42, WHITE, 1)
        else:
            cv2.rectangle(dashboard, (40, 570), (800, 595), (20, 80, 30), -1)
            draw_text(dashboard, "DRIVER ALERT // CONTINUOUS MONITORING", (250, 589), 0.42, GREEN, 1)

        # Optional night mode: dim the dashboard while keeping status colors readable.
        if night_mode:
            dashboard = (dashboard.astype('float32') * 0.55).astype('uint8')
            # redraw key status indicator so the mode is obvious
            draw_text(dashboard, "NIGHT MODE", (1060, 90), 0.38, CYAN, 1)

        cv2.imshow("NEXUS // Driver Intelligence", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# ============================================================
# CLEANUP
# ============================================================
stop_alarm()
cap.release()
cv2.destroyAllWindows()
print("NEXUS // Driver Intelligence stopped.")
