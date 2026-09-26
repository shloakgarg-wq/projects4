import cv2
import numpy as np

image = cv2.imread("images/input.jpg")

red = 255
green = 255
blue = 255

while True:
    result = image.copy()

    result[:, :, 2] = red
    result[:, :, 1] = green
    result[:, :, 0] = blue

    cv2.imshow("Color Filter", result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        red = 255
        green = 0
        blue = 0

    elif key == ord("g"):
        red = 0
        green = 255
        blue = 0

    elif key == ord("b"):
        red = 0
        green = 0
        blue = 255

    elif key == ord("t"):
        red = min(255, red + 25)

    elif key == ord("d"):
        blue = max(0, blue - 25)

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

name = input("Enter a filename: ")
cv2.imwrite("images/" + name + ".jpg", result)