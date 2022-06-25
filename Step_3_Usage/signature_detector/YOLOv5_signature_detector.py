import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any
from signature_detector.YOLOv5_signature_object import *


class YoloSignatureDetector:
    def __init__(self, path_to_model: str):
        self.__model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_model)

    def predict(self, images: Image or np.ndarray or List[Image or np.ndarray]) -> List[List[YoloObjectClass]]:
        if isinstance(images, Image.Image) or isinstance(images, np.ndarray):
            return [self.__predict(image=images)]
        if isinstance(images, list):
            return [self.__predict(image=image) for image in images]
        raise Exception("Perhaps this type of images is not supported!")

    def __predict(self, image: Image or np.ndarray) -> List[YoloObjectClass]:
        marks = []
        image, scaling = YoloSignatureDetector.__transform(image=image)
        data = self.__model(image).pandas().xyxy[0]
        for i in range(len(data)):
            marks.append(YoloObjectClass(name=data.at[i, 'name'],
                                         yolo_class=data.at[i, 'class'],
                                         confidence=data.at[i, 'confidence'],
                                         x_min=data.at[i, 'xmin'],
                                         y_min=data.at[i, 'ymin'],
                                         x_max=data.at[i, 'xmax'],
                                         y_max=data.at[i, 'ymax'],
                                         scaling=scaling))
        return marks

    @staticmethod
    def __transform(image):
        page_height, page_width = image.shape[:2]
        max_height = 640
        max_width = 480
        if max_height < page_height or max_width < page_width:
            scaling_factor = max_height / float(page_height)
            if max_width / float(page_width) < scaling_factor:
                scaling_factor = max_width / float(page_width)
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            return image, scaling_factor
        return image, 1.0

