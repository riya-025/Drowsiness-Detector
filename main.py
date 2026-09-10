import cv2
from Scripts.detection import DrowsinessDetection
from Scripts.alerts import AlertSystem
from Scripts.monitoring import DrowsinessMonitor
from Scripts.logger import EventLogger


# Create objects
detector = DrowsinessDetection()
alert_system = AlertSystem()
monitor = DrowsinessMonitor()
logger = EventLogger()

# Open camera
cap = cv2.VideoCapture(0)

# Keep track of previous drowsiness state
previous_drowsy = False


if not cap.isOpened():
    print("Camera could not be opened.")
    detector.close()
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect face, eyes, EAR and drowsiness
    result = detector.process_frame(frame)

    # Update monitoring
    monitor.update(result["drowsy"])

    # Trigger alert when drowsiness is detected
    if result["drowsy"]:
        alert_system.trigger_alert()

    # Log only when a new drowsiness event starts
    if result["drowsy"] and not previous_drowsy:
        logger.log_event(0)

    # Save current drowsiness state
    previous_drowsy = result["drowsy"]


    # Decide status
    if not result["face_detected"]:
        status = "NO FACE"

    elif result["drowsy"]:
        status = "DROWSY"

    else:
        status = "AWAKE"


    # Display status on screen
    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if result["drowsy"] else (0, 255, 0),
        2
    )


    # Display EAR value
    if result["ear"] is not None:
        cv2.putText(
            frame,
            f"EAR: {result['ear']:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # Show camera window
    cv2.imshow("Drowsiness Detector", frame)


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera and close detector
cap.release()
detector.close()
cv2.destroyAllWindows()
