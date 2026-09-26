import cv2
import numpy as np

image = cv2.imread("sample.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

mode = "original"
message = "Press r, g, b, s, or c"

while True:
    if mode == "red":
        result = np.zeros_like(image)
        result[:, :, 2] = image[:, :, 2]

    elif mode == "green":
        result = np.zeros_like(image)
        result[:, :, 1] = image[:, :, 1]

    elif mode == "blue":
        result = np.zeros_like(image)
        result[:, :, 0] = image[:, :, 0]

    elif mode == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobel_x, sobel_y)
        result = cv2.convertScaleAbs(result)

    elif mode == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)

    else:
        result = image.copy()

    cv2.imshow("Image Processing", result)
    print(message)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("r"):
        mode = "red"
        message = "Red tint selected."

    elif key == ord("g"):
        mode = "green"
        message = "Green tint selected."

    elif key == ord("b"):
        mode = "blue"
        message = "Blue tint selected."

    elif key == ord("s"):
        mode = "sobel"
        message = "Sobel edge detection selected."

    elif key == ord("c"):
        mode = "canny"
        message = "Canny edge detection selected."

    elif key == ord("q"):
        break

    else:
        message = "Invalid key. Press r, g, b, s, c, or q."

cv2.destroyAllWindows()