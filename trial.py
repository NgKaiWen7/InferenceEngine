import cv2
import numpy as np

img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

# Ensure BGRA
if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

mask = (s < 20) & (v > 180)

# Replace grey/white pixels with white
img[mask] = [255, 255, 255, 255]

cv2.imwrite("output.png", img)