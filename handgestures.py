import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

x = 320
y = 240
size = 50
color = (0, 255, 0)

previous_x = None
previous_y = None
points = []

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        hand = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand)

        if area > 3000:
            M = cv2.moments(hand)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if previous_x is not None and previous_y is not None:
                    dx = cx - previous_x
                    dy = cy - previous_y

                    if abs(dx) > 8:
                        x += dx

                    if abs(dy) > 8:
                        y += dy

                previous_x = cx
                previous_y = cy

                hull = cv2.convexHull(hand)
                hull_indices = cv2.convexHull(hand, returnPoints=False)

                fingers = 0

                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(hand, hull_indices)

                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]

                            start = tuple(hand[s][0])
                            end = tuple(hand[e][0])
                            far = tuple(hand[f][0])

                            a = math.sqrt(
                                (end[0] - start[0]) ** 2 +
                                (end[1] - start[1]) ** 2
                            )

                            b = math.sqrt(
                                (far[0] - start[0]) ** 2 +
                                (far[1] - start[1]) ** 2
                            )

                            c = math.sqrt(
                                (end[0] - far[0]) ** 2 +
                                (end[1] - far[1]) ** 2
                            )

                            if b * c != 0:
                                angle = math.acos(
                                    max(
                                        -1,
                                        min(
                                            1,
                                            (b * b + c * c - a * a)
                                            / (2 * b * c)
                                        )
                                    )
                                )

                                angle = angle * 180 / math.pi

                                if angle < 90 and d > 10000:
                                    fingers += 1

                if fingers >= 3:
                    color = (255, 0, 0)
                    points.append((cx, cy))

                elif fingers == 0:
                    color = (0, 255, 0)
                    points = []

                else:
                    color = (0, 255, 255)

                size = int(np.interp(area, [3000, 30000], [30, 120]))
                size = max(30, min(size, 120))

                x = max(size, min(frame.shape[1] - size, x))
                y = max(size, min(frame.shape[0] - size, y))

                cv2.drawContours(frame, [hand], -1, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

    cv2.circle(frame, (x, y), size, color, -1)

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, 4)

    cv2.putText(
        frame,
        "Move your hand to control the circle",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Open hand: draw | Closed hand: stop",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Gesture Control", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()