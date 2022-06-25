import os
import cv2
from signature_detector import *


image = cv2.imread("001.png")
signature_detector = YoloSignatureDetector(path_to_model='model.pt')
res = signature_detector.predict(images=[image])
for im in res:
    for signature in im:
        image = cv2.rectangle(image,
                              (int(signature.x_min), int(signature.y_min)),
                              (int(signature.x_max), int(signature.y_max)),
                              (255, 0, 0),
                              2)

cv2.imshow("Window", image)
cv2.waitKey()


