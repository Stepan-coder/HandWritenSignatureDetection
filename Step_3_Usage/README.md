# Deployment

To use it, you need to install all the libraries from the `requirements.txt` file, as well as move the `signature_detector` folder to the `root` of your project. An example of a simple use can be seen below:

```Python3
import cv2
from signature_detector import *


image = cv2.imread("001.png")
signature_detector = YoloSignatureDetector(path_to_model='*path to model*.pt')
# predicted = signature_detector.predict(images=cv2.imread("001.png")])  # Single image prediction
predicted = signature_detector.predict(images=[cv2.imread("001.png"), cv2.imread("002.png"), cv2.imread("003.png")])  # Multi image preditcion
```
To find captions on an image (images), pass `images=*your image*` or `images=[*your images*]` as an argument. The result of the method will be `List[List[YoloObjectClass]]`. YoloObjectClass contains the following properties:
* `name` - Name of the class label
* `confidence` - The **confidence** of the model in a particular answer
* `class_id` - Id of the **class** the model is leaning towards
* `scaling` - The **scaling factor** of the original image to the images that YOLOv5x works with [640 x 480]
* `x_min` - The **left border** of the frame around the signature
* `y_min` - The **top border** of the frame around the signature
* `x_max` - The **right border** of the frame around the signature
* `y_max` - The **bottom border** of the frame around the signature

An example of the simplest Python code for highlighting areas where, according to YOLOv5x, signatures may be located.

```Python3
import cv2
from signature_detector import *


picture = cv2.imread("001.png")
signature_detector = YoloSignatureDetector(path_to_model='*path to model*.pt')
predicted = signature_detector.predict(images=[picture])
for image in predicted:  # predicted - a list of lists, where the external list is the images, and the internal list is the signatures found on the pictures.
    for signature in image:
        picture = cv2.rectangle(picture,
                                (int(signature.x_min), int(signature.y_min)),
                                (int(signature.x_max), int(signature.y_max)),
                                (255, 0, 0),
                                2)

cv2.imshow("Window", picture)
cv2.waitKey()
```
