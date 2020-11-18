# Mask-RCNN
Machine vision is one of the booming fields of machine learning. While there are several models and algorithms like RCNN, Faster RCNN, SSD Mobilenet to help implement machine vision, Mask RCNN is the most efficient and accurate model of them all. But why?



Let’s start by understanding **what Mask RCNN is**.
Mask RCNN is a model that helps detect various objects in any given image. To put it in simple words, Mask RCNN is the extended version of Faster RCNN. Mask RCNN does instance segmentation. 




**What is Instance Segmentation?**
Instance segmentation is a combination of object detection and semantic segmentation. Object detection helps detect objects in an image, classify individual objects and localize each object instance using a bounding box. In contrast, Semantic Segmentation detects all the objects present in a picture at the pixel level. Thus, it enables us to detect objects in an image while precisely segmenting a mask for each object instance.
Semantic segmentation groups pixels in a meaningful way. Pixels belonging to a person, road, building, fence, bicycle, cars or any other objects grouped separately. Instance segmentation assigns a label to each pixel of the image.

![instance segmentation](/images/instancesegmantation.png)

**How does Mask R-CNN work?**
To put it in simple words, Mask R-CNN model is divided into two parts:
*Region proposal network (RPN) to propose candidate object bounding boxes.
*Binary mask classifier to generate a mask for every class.

![Mask RCNN](/images/maskrcnn.png)

First, the backbone network and Regional proposal network (RPN) run once per image to give a set of region proposals. Region proposals are regions in the feature map which contain the object.
Next, The RoI Align network gives out multiple bounding boxes rather than a single definite one and warps them into a fixed dimension. This RoI pooling layer generates the output of size (7*7*D) (where D =256 for ZF and 512 of VGG-16, i.e. Different values of D for different backbone networks).
The warped features thus obtained are then fed into fully connected layers to make classification using softmax. Then the boundary box prediction is further refined using a regression model.
The features are also fed into Mask classifier, which consists of two CNN’s to generate a binary mask for each Region of Interest. Mask Classifier allows the network to create masks for every class without competing among several classes.

![Mask RCNN2](/images/maskrcnn2.png)

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


