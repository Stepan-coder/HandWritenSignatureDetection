

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