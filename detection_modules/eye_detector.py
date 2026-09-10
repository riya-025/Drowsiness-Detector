# MediaPipe Face Mesh landmark indexes for the eyes

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def get_eye_points(landmarks, eye_indices, frame_width, frame_height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]

        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        points.append((x, y))

    return points


def get_both_eyes(landmarks, frame_width, frame_height):
    left_eye = get_eye_points(
        landmarks,
        LEFT_EYE,
        frame_width,
        frame_height
    )

    right_eye = get_eye_points(
        landmarks,
        RIGHT_EYE,
        frame_width,
        frame_height
    )

    return left_eye, right_eye