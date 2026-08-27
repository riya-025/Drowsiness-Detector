import cv2
from Scripts.detection import DrowsinessDetection


detector = DrowsinessDetection()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    detector.close()
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = detector.process_frame(frame)

    if not result["face_detected"]:
        status = "NO FACE"
    elif result["drowsy"]:
        status = "DROWSY"
    else:
        status = "AWAKE"

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if result["drowsy"] else (0, 255, 0),
        2
    )

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

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
detector.close()
cv2.destroyAllWindows()