import cv2
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

img = cv2.imread("test.jpg")
result, _ = engine(img)

for box, text, score in result:
    points = [(int(x), int(y)) for x, y in box]

    for i in range(4):
        cv2.line(img, points[i], points[(i + 1) % 4], (0, 255, 0), 2)

    x, y = points[0]
    cv2.putText(img, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("test_result.jpg", img)