import os
import cv2
from signature_detector import *


image = cv2.imread("001.png")
signature_detector = YoloSignatureDetector(path_to_model='model.pt')
predicted = signature_detector.predict(images=[image])


