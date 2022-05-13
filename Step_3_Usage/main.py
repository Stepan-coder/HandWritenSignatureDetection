import os
import torch


# Model
class YoloSignatureDetector:
    def __init__(self, path_to_model: str):
        self.__model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Images

imgs = ['aah97e00-page02_1.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
result = results.pandas().xyxy[0]
print(results.pandas().xyxy[0])
# results.save()  # or .show()
#
# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]  # img1 predictions (pandas)
# #      xmin    ymin    xmax   ymax  confidence  class    name
# # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie