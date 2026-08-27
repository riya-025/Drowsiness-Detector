import math


def euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def eye_aspect_ratio(eye):
    vertical_1 = euclidean_distance(eye[1], eye[5])
    vertical_2 = euclidean_distance(eye[2], eye[4])
    horizontal = euclidean_distance(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear