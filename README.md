# Mask-RCNN
Machine vision is one of the booming fields of machine learning. While there are several models and algorithms like RCNN, Faster RCNN, SSD Mobilenet to help implement machine vision, Mask RCNN is the most efficient and accurate model of them all. But why?



Let’s start by understanding **what Mask RCNN is**.


Mask RCNN is a model that helps detect various objects in any given image. To put it in simple words, Mask RCNN is the extended version of Faster RCNN. Mask RCNN does instance segmentation. 




**What is Instance Segmentation?**


Instance segmentation is a combination of object detection and semantic segmentation. Object detection helps detect objects in an image, classify individual objects and localize each object instance using a bounding box. In contrast, Semantic Segmentation detects all the objects present in a picture at the pixel level. Thus, it enables us to detect objects in an image while precisely segmenting a mask for each object instance.
Semantic segmentation groups pixels in a meaningful way. Pixels belonging to a person, road, building, fence, bicycle, cars or any other objects grouped separately. Instance segmentation assigns a label to each pixel of the image.

![instance segmentation](/images/instancesegmentation.jpg)

**How does Mask R-CNN work?**


To put it in simple words, Mask R-CNN model is divided into two parts:
*Region proposal network (RPN) to propose candidate object bounding boxes.
*Binary mask classifier to generate a mask for every class.

![Mask RCNN](/images/maskrcnn.png)

First, the backbone network and Regional proposal network (RPN) run once per image to give a set of region proposals. Region proposals are regions in the feature map which contain the object.
Next, The RoI Align network gives out multiple bounding boxes rather than a single definite one and warps them into a fixed dimension. This RoI pooling layer generates the output of size (7*7*D) (where D =256 for ZF and 512 of VGG-16, i.e. Different values of D for different backbone networks).
The warped features thus obtained are then fed into fully connected layers to make classification using softmax. Then the boundary box prediction is further refined using a regression model.
The features are also fed into Mask classifier, which consists of two CNN’s to generate a binary mask for each Region of Interest. Mask Classifier allows the network to create masks for every class without competing among several classes.

![Mask RCNN2](/images/maskrcnn2.jpg)

Like any other model, Mask RCNN uses anchor boxes to detect multiple objects including overlapping objects in an image. This process increases the speed and efficiency of object detection.  Anchor boxes are a set of predefined bounding boxes of a certain height and width. These boxes are defined to capture the scale and aspect ratio of specific object classes you want to detect.

![Prediction](/images/prediction.png)

Mask R-CNN makes thousands of predictions to predict multiple objects or multiple instances of objects in an image. For a convolution feature map of W * H, we get N = W* H* k anchor boxes. Final object detection is done by removing anchor boxes that belong to the background class and the remaining ones are filtered by their confidence score. Anchor boxes with the greatest confidence score are selected using Non-Max suppression equation:

IoU (Intersection over Union) = (Area of the intersection)/(Area of union)
We find the anchor boxes with IoU greater than 0.5.

![Prediction2](/images/prediction2.png)

**Math behind Mask RCNN:**



Training and Loss function for RPN:


L({pi},{ti})=1/Ncls(∑ipi,pi*))+ Nreg(∑ipi*x Lreg(ti, ti*))


where, 
pi = predicted probability of anchors contains an object or not.



pi* = ground truth value of anchors contains and object or not.


ti = coordinates of predicted anchors.


ti* = ground truth coordinate associated with bounding boxes.


Lcls = Classifier Loss (binary log loss over two classes).



Lreg = Regression Loss (Here, Lreg  = R(ti-ti*) where R is smooth L1 loss)


Ncls = Normalization parameter of mini-batch size (~256).


Nreg = Normalization parameter of regression (equal to number of anchor locations ~2400).


ƛ=10 In order to make n=btoh loss parameter equally weighted right




**Implementing MASK RCNN**


**Step 1: Clone the Mask R-CNN repository**

```
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN$ 
python setup.py install
```


**Step 2: Download the pre-trained weights for COCO model**

```
You can download the model from this website- https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
Place the file in the Mask_RCNN folder with name “mask_rcnn_coco.h5”
```

**Step 3: Install dependencies**

```
numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow>=1.3.0
keras>=2.0.8
opencv-python
h5py
imgaug
IPython
```


**Step 4: Importing required libraries:**


Create a new Python notebook inside the “samples” folder of the cloned Mask_RCNN repository.

```
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("../")

import warnings
warnings.filterwarnings("ignore")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  
import coco

%matplotlib inline
```

**Step 5: Define the path for the pretrained weights and the images**
```
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
```

**Step 6:Create an inference class**
```
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
```

**Step 7: Loading weights**
```
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('mask_rcnn_coco.h5', by_name=True)
```

**Step 8: Define the classes**
```
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
```
**Step 9: Testing the model**
```
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
```
**Visualizing separate masks**
```
for i in range(mask.shape[2]):
    temp = skimage.io.imread('sample.jpg')
    for j in range(temp.shape[2]):
        temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    plt.figure(figsize=(8,8))
    plt.imshow(temp)
```



You can train your own custom dataset and implement Instance segmentation on it as well. 
