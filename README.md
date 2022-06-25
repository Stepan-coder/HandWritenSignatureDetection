# Deep Learning based Signature Detection (YOLOv5x)
## Introduction

In the modern, rapidly developing world, where all spheres of human life are actively digitized, the problem of interaction with a large number of documents is quite acute. Yes, in the XIX and XX century, the accountant's vacancy was considered fashionable and prestigious, because there were very few qualified specialists, and their work was evaluated quite highly. But times are changing, technologies are developing, and people are tired of doing the same type of work on their own. Therefore, by automating this process, it would be possible to save a significant amount of time and qualified resources. This project is part of a large system for obtaining named entities from documents.

## Theory

Initially, I was thinking about developing my own signature detection model. But since this is only a small part of the project and the quality of detecting the boundaries of an object is not as important to me as the fact of its presence. Then I decided to use a ready-made pre-trained model. Among popular architectures, I considered YOLOv5x to be the simplest and most convenient option. If you pay attention to the graph below, you can see that YOLOv5x shows the best accuracy results compared to YOLOv5l, YOLOv5m, YOLOv5s.

![YOLO_COMPARSION](Images/yolo_comparison.png)

Speaking more seriously, YOLOv5 is a modern object detection algorithm that is widely used both in scientific circles and in industry. This is the latest version of a universal and powerful object detection algorithm called YOLO. It surpasses all other real-time object detection models in the world.
YOLO uses convolutional neural networks instead of the region-based methods employed by alogorithms like R-CNN. The convolutional network Only Look Once, ie it requires only one forward pass through the neural network to make predictions. It makes two predictions, the class of object and the position of the objects present in the image.

![YOLO_MODEL_LIST](Images/yolo_model_list.png)

YOLO devices an image into nine regions and predicts whether a target class is present in each region or not. It also predicts the bounding box coordinates of the target class. Non-max suppression is used to prevent same object being detected multiple times.

![YOLO_WORKING](Images/yolo_working.jpeg)

The original YOLO paper could be accessed [here](https://arxiv.org/abs/1506.02640) and YOLOv5 repo could be found [here](https://github.com/ultralytics/yolov5).

## Workflow

In machine learning , the process of creating a model is usually divided into the following steps:
* [Data preparation](Step_1_Convertiong_Dataset_to_YOLOv5)
  * Data collection / Dataset search
  * Clearing data
  * Data preparation / Data markup
  * Splitting the available data into training, validation and test samples
* [Model](Step_2_Custom_YOLO_training)
  * Selection (implementation) of the model architecture
  * Model Training
  * Model Test
* [Deployment](Step_3_Usage)
  * Preparing the model for further use

### [Data preparation](Step_1_Convertiong_Dataset_to_YOLOv5)
### [Model](Step_2_Custom_YOLO_training)
### Training the model
Use [this notebook](Step_2_Custom_YOLO_training/CustomYOLOv5_using_Tobcco800_dataset.ipynb) to train and test the YOLOv5 model.
  
Clone the official [YOLOv5 repo] and install the requirements using the `requirements.txt` file.  
We need to create a `tobacco_data.yaml` and add the path of training `train:` and validation `valid:` directories, number of classes `nc:` and class names `['DLLogo', 'DLSignature']` and add this file to the `yolov5` directory we cloned.  

Now, we have to select a pre-trained model from the available pre-trained checkpoints. These `model.yaml` files are present inside `yolov5\models`. I have used `yolov5x` for performing my experiments.  
![Yolo models](../../Images/yolo_model_list.png)  
  
**Training arguments**  
`--img 640` is the width of the images.  
`--batch` - batch size>  
`--epochs` - no of epochs  
`--data` - Dataset.yaml (`tobacco_data.yaml`) path  
`--cfg models/model.yaml` is used to set the model we want to train on. I have used yolov5x.yaml, more information could be found [here.](https://github.com/ultralytics/yolov5#pretrained-checkpoints)  
  
**To Train the model**, run the following line.  
> **!python yolov5/train.py --img 640 --batch 16 --epochs 300 --data tobacco_data.yaml --cfg models/yolov5x.yaml --name Tobacco-run**

### [Deployment](Step_3_Usage)


This project is based on these two papers [[1]](https://repositum.tuwien.at/bitstream/20.500.12708/16962/1/Hauri%20Marcel%20Rene%20-%202021%20-%20Detecting%20Signatures%20in%20scanned%20document%20images.pdf) and [[2]](https://arxiv.org/abs/2004.12104).  