import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any


class YoloObjectClass(object):
    def __init__(self, name: str, confidence: float,  yolo_class: int, scaling: float,
                 x_min: float, y_min: float, x_max: float, y_max: float):
        self.__name = name
        self.__confidence = confidence
        self.__yolo_class = yolo_class
        self.__scaling = scaling
        self.__x_min = x_min * (1.0 / scaling)
        self.__y_min = y_min * (1.0 / scaling)
        self.__x_max = x_max * (1.0 / scaling)
        self.__y_max = y_max * (1.0 / scaling)

    def __str__(self) -> str:
        return f"YOLOv5 label: {self.__name}, YOLOv5 class: {self.__yolo_class}, " \
               f"confidence: {round(self.__confidence, 2)}, scaling: {self.__scaling}, coordinates: " \
               f"Top-Left ({round(self.__x_min, 2)}, {round(self.__y_min, 2)}), " \
               f"Top-Right ({round(self.__x_min, 2)}, {round(self.__y_max, 2)}), " \
               f"Bottom-Left ({round(self.__x_max, 2)}, {round(self.__y_min, 2)}), " \
               f"Bottom-Right  ({round(self.__x_max, 2)}, {round(self.__y_max, 2)})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def confidence(self) -> float:
        return self.__confidence

    @property
    def class_id(self) -> int:
        return self.__yolo_class

    @property
    def scaling(self) -> float:
        return self.__scaling

    @property
    def x_min(self) -> float:
        return self.__x_min

    @property
    def y_min(self) -> float:
        return self.__y_min

    @property
    def x_max(self) -> float:
        return self.__x_max

    @property
    def y_max(self) -> float:
        return self.__y_max


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
        print(image.shape)
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

