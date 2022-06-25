# Training the model(YOLOv5x)
  
Clone the official [YOLOv5 repo](https://github.com/ultralytics/yolov5) and install the requirements using the `requirements.txt` file.  
We need to create a `tobacco_data.yaml` and add the path of training `train:` and validation `valid:` directories, number of classes `nc:` and class names `['DLLogo', 'DLSignature']` and add this file to the `yolov5` directory we cloned.  
 
**Training arguments**  
`--img 640` is the width of the images.  
`--batch` - batch size
`--epochs` - no of epochs  
`--data` - Your path to `tobacco_data.yaml`  
`--cfg models/model.yaml` is used to set the model we want to train on. I have used yolov5x.yaml, more information could be found [here.](https://github.com/ultralytics/yolov5#pretrained-checkpoints)  
`--name` - The folder where the weights of the model will be saved

**To Train the model, run the following line.**  
> **!python yolov5/train.py --img 640 --batch 16 --epochs 300 --data tobacco_data.yaml --cfg models/yolov5x.yaml --name Tobacco-run**

**Testing/ Inference arguments**  
`--hide-labels` is used to hide the labels in the detected images.  
`--hide-conf` is used to hide the confidence scores in the detected images.  
`--classes 0, 1`, etc used to detect only the classes mentioned here. For our use case we need only signature class, so use `--classes 1`.  
`--line-thickness` integer used to set the thickness of bounding box.  
`--save-crop` and `--save-txt` used to save the crops and labels.  
`--project` could be used to specify the results path  
  
**To test/run inference on a directory of images.**  
> **!python yolov5/detect.py --source /images/valid/ --weights 'runs/train/Tobacco-run/weights/model.pt' --hide-labels --hide-conf --classes 1 --line-thickness 2**

**To pedict a single image**  
> **!python yolov5/detect.py --source /images/valid/imagename --weights 'runs/train/Tobacco-run/weights/model.pt' --hide-labels --hide-conf --classes 1 --line-thickness 2**  
   
