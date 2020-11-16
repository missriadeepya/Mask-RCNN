# Mask-RCNN
What is Mask RCNN? How to implement it?
Mask RCNN is the extended version of Faster RCNN. Mask RCNN does instance segmentation. Instance segmentation is a combination of object detection and semantic segmentation. Object detection helps detect objects in an image, classify individual objects and localize each object instance using a bounding box. In contrast, Semantic Segmentation detects all the objects present in a picture at the pixel level. 
Semantic segmentation groups pixels in a semantically meaningful way. Pixels belonging to a person, road, building, fence, bicycle, cars or any other objects are grouped separately. Instance segmentation assigns a label to each pixel of the image.
How does Mask R-CNN work?
To put it in simple words, Mask R-CNN model is divided into two parts: 
Region proposal network (RPN) to propose candidate object bounding boxes, and 
binary mask classifier to generate a mask for every class.

 
First, the image is run through CNN to generate the feature maps. Then, Region Proposal Network(RPN) uses a convolutional neural network to generate the multiple Region of Interest(RoI) using a lightweight binary classifier. Since it uses a binary classifier, it returns object(yes) or no-object(no) scores. Non-Max suppression is applied to Anchors with a high objectness score.
The RoI Align network gives out multiple bounding boxes rather than a single definite one and warps them into a fixed dimension.
The warped features thus obtained are then fed into fully connected layers to make classification using softmax. Then the boundary box prediction is further refined using a regression model.
The features are also fed into Mask classifier, which consists of two CNN’s to generate a binary mask for each Region of Interest. Mask Classifier allows the network to create masks for every class without competing among several classes.
 
Like any other model, Mask RCNN uses anchor boxes to detect multiple objects including overlapping objects in an image. This process increases the speed and efficiency of object detection.  Anchor boxes are a set of predefined bounding boxes of a certain height and width. These boxes are defined to capture the scale and aspect ratio of specific object classes you want to detect.

 
To predict multiple objects or multiple instances of objects in an image, Mask R-CNN makes thousands of predictions. Final object detection is done by removing anchor boxes that belong to the background class and the remaining ones are filtered by their confidence score. We find the anchor boxes with IoU greater than 0.5. Anchor boxes with the greatest confidence score are selected using Non-Max suppression explained below.

IoU = Area of the intersection / Area of the union. 
IoU computes intersection over the union of the two bounding boxes, the bounding box for the ground truth and the bounding box for the predicted box by algorithm.




