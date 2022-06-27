import os
import cv2
from signature_detector import *


picture = cv2.imread("001.png")
signature_detector = YoloSignatureDetector(path_to_model='model.pt')
predicted = signature_detector.predict(images=[picture])
for image in predicted:
    for signature in image:
        picture = cv2.rectangle(picture,
                                (int(signature.x_min), int(signature.y_min)),
                                (int(signature.x_max), int(signature.y_max)),
                                (255, 0, 0),
                                2)

cv2.imshow("some", picture)
cv2.waitKey()


