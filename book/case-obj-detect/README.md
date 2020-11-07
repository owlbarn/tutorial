# Case - Instance Segmentation

Computer vision is a field dealing with many different automated tasks whose goal are to give high-level descriptions of images and videos. It has been applied to a wide variety of domains ranging from highly technical ones, such as automatic tagging of satellite images, analysis of medical images etc., to more mundane applications, including categorising pictures in your phone, making your face into an emoji, etc.
This field has seen tremendous progress since 2012, when [A. Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) used the first deep learning approach to computer vision, crushing all their opponents in the [ImageNet challenge](http://image-net.org/challenges/LSVRC/2012/results.html). 
We have already discussed this image recognition task in the previous chapter.

In this chapter, we are going to introduce another classical computer vision task: the *Instance Segmentation*, which is to give different labels for separate instances of objects in an image.
We will discuss its connection with other similar applications, how the deep neural network is constructed in Owl, and how the network, loaded with pre-trained weights, can be used to process users' input images.
We have also included an online demo for this application. 
Let's begin!

## Introduction

In the previous chapter about the image recognition task, we have introduced how the DNN can be applied to classify the one single object in an image.
However, this technique can easily get confused if it is applied on an image with multiple objects, which frequently happens in the real world. 
That's why we need other methods. 

For example, *Object Detection* is another classical computer vision task. Given an image that contains multiple objects, an object detection application aims to classify individual objects and localize each one using a bounding box, as shown in the [@fig:case-obj-detect:example_obj].

![](images/case-obj-detect/example_obj.jpg){#fig:case-obj-detect:example_obj}
*Example of object detection ([src](https://en.wikipedia.org/wiki/File:Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg))*

Similarly,  the *Semantic Segmentation* task requires to classify the pixels on an image into different categories. Each segment is recognised by a "mask" that follows cover the whole object.
This can be illustrated in the [@#fig:case-obj-detect:example_seg].

![](images/case-obj-detect/example_seg.jpg){#fig:case-obj-detect:example_seg}
*Example of semantic segmentation ([src](https://gts.ai/how-do-we-solve-the-challenges-faced-due-to-semantic-segmentation/))*

In 2017, the *Mask R-CNN* (Mask Region-based Convolutional Neural Network) architecture was published. With sufficient training, it can solve these problems at once: it can detect objects on an image, label each of them, and provide a binary mask to tell which pixels belong to the objects.
This task is called the *Instance Segmentation*.
As a preliminary example and visual motivation, the examples below show what this task generates:

![Example: Street view](images/case-obj-detect/example_00.jpg){#fig:case-obj-detect:example_01}

![Example: Sheep](images/case-obj-detect/example_01.jpg){#fig:case-obj-detect:example_02}

In these two examples, normal pictures are processed by the pre-trained Mask R-CNN (MRCNN) network, and the objects (people, sheep, bag, car, bus, umbrella, etc.) are segmented from the picture and recognised with a percentage of confidence, represented by a number between 0 and 1. 

Image segmentation have massive application scenarios in the industry, such as medical imaging (locating tumours, detecting cancer cells ...), traffic control systems, locate objects in satellite images, etc. 
The Mask R-CNN network has been implemented in Owl. 
In the next sections, we will explain how this network is constructed. 

## Mask R-CNN Network

This section will briefly outline the main parts of architecture of Mask R-CNN and how it stands out from its predecessors.
You can get more detailed and technical explanations in the original paper [@he2017mask].
The Owl implementation of the inference mode is available in [this repository](https://github.com/pvdhove/owl-mask-rcnn).
The code was mostly ported from the [Keras/TensorFlow implementation](https://github.com/matterport/Mask_RCNN).
Work in this chapter is conducted by [Pierre Vandenhove](http://math.umons.ac.be/staff/Vandenhove.Pierre/) during his internship in the OCamlLabs.

MRCNN network is extended based on the Faster R-CNN network [@ren2015faster], which itself extends the Fast R-CNN [@girshick2015fast].
In Fast R-CNN, the authors propose a network that accepts input images and regions of interest (RoI). For each region, features are extracted by several fully-connected layers, and the features are fed into a branch.
One output of this branch contains the output classification (together with possibility of that classification) of the object in that region, and the other specifies the rectangle location of the object.

In Faster R-CNN, the authors point out that, there is no need to find RoIs using other methods than the network itself. They propose a Region Proposal Network (RPN) that share the same feature extraction backbone as that in Fast R-CNN.
RPN is a small convolutional network that scans the feature maps quickly, produces a set of rectangular possible object region proposals, each associated with a number that could be called the *objectness* of that region.
The RoI feature extraction part of Fast R-CNN is kept unchanged here.
In this way, a single Faster R-CNN network can be trained and then perform the object detection task without extra help from other region proposal methods.

To perform the task of not just objection detection, but also semantic segmentation, Mask R-CNN keeps the architecture of Faster R-CNN, only adding an extra branch in the final stage of its RoI feature layer.
Where previously the outputs includes object classification and location, now a third branch contains information about the mask of object in the RoI.
Therefore, for any RoI, the Mask R-CNN can retrieve its rectangle bound, classification results, classification possibility, and the mask of that object, all information in one pass.

## Building Mask R-CNN

After a quick introduction of the MRCNN and how it is developed in theory, let's look at the code to understand how it is constructed in Owl, one piece at a time.
Please feel free to jump this part if you are eager to try to run the code and process one of your images.

```
open Owl

module N = Dense.Ndarray.S

open CGraph
open Graph
open AD

module RPN = RegionProposalNetwork
module PL = ProposalLayer
module FPN = FeaturePyramidNetwork
module DL = DetectionLayer
module C = Configuration

let image_shape = C.get_image_shape () in
if image_shape.(0) mod 64 <> 0 || image_shape.(1) mod 64 <> 0 then
invalid_arg "Image height and width must be divisible by 64";

let inputs = inputs
    ~names:[|"input_image"; "input_image_meta"; "input_anchors"|]
    [|image_shape; [|C.image_meta_size|]; [|num_anchors; 4|]|] in
let input_image = inputs.(0)
and input_image_meta = inputs.(1)
and input_anchors = inputs.(2) i
```

The network accepts three inputs, representing images, meta data, and number of anchors (the rectangular regions).
The `Configuration` module contains a list of constants that will be used in building the network.

### Feature Extractor

The picture is first fed to a convolutional neural network in order to extract features of the image. 
The first few layers detect low-level features of an image, such as edges and basic shapes. But as you go deeper into the network, these features are assembled to detect higher level features such as "people", "cars" (which, some argue, works in the same way as the brain). 
Five of these layers (called "feature maps") of various sizes, both high- and low-level, are then passed on to the next parts. This implementation chooses Microsoft’s [ResNet101](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) as a feature extractor.

```
let tdps = C.top_down_pyramid_size in
let str = [|1; 1|] in
let p5 = conv2d [|1; 1; 2048; tdps|] str ~padding:VALID ~name:"fpn_c5p5" c5 in
let p4 =
add ~name:"fpn_p4add"
  [|upsampling2d [|2; 2|] ~name:"fpn_p5upsampled" p5;
    conv2d [|1; 1; 1024; tdps|] str ~padding:VALID ~name:"fpn_c4p4" c4|] in
let p3 =
add ~name:"fpn_p3add"
  [|upsampling2d [|2; 2|] ~name:"fpn_p4upsampled" p4;
    conv2d [|1; 1; 512; tdps|] str ~padding:VALID ~name:"fpn_c3p3" c3|] in
let p2 =
add ~name:"fpn_p2add"
  [|upsampling2d [|2; 2|] ~name:"fpn_p3upsampled" p3;
    conv2d [|1; 1; 256; tdps|] str ~padding:VALID ~name:"fpn_c2p2" c2|] in

let conv_args = [|3; 3; tdps; tdps|] in
let p2 = conv2d conv_args str ~padding:SAME ~name:"fpn_p2" p2 in
let p3 = conv2d conv_args str ~padding:SAME ~name:"fpn_p3" p3 in
let p4 = conv2d conv_args str ~padding:SAME ~name:"fpn_p4" p4 in
let p5 = conv2d conv_args str ~padding:SAME ~name:"fpn_p5" p5 in
let p6 = max_pool2d [|1; 1|] [|2; 2|] ~padding:VALID ~name:"fpn_p6" p5 in

let rpn_feature_maps = [|p2; p3; p4; p5; p6|] in
let mrcnn_feature_maps = [|p2; p3; p4; p5|]
```

The features are extracted combining both ResNet101 and the Feature Pyramid Network.
ResNet extracts features of the image (the first layers extract low-level features, the last layers extract high-level features).  
Feature Pyramid Network creates a second pyramid of feature maps from top to bottom so that every map has access to high and low level features.
This combination proves to achieve excellent gains in both accuracy and speed.

### Proposal Generation

To try to locate the objects, about 250,000 overlapping rectangular regions or anchors are generated.

```
let nb_ratios = Array.length C.rpn_anchor_ratios in
let rpns = Array.init 5 (fun i ->
  RPN.rpn_graph rpn_feature_maps.(i)
  nb_ratios C.rpn_anchor_stride
  ("_p" ^ string_of_int (i + 2))) in
let rpn_class = concatenate 1 ~name:"rpn_class"
                (Array.init 5 (fun i -> rpns.(i).(0))) in
let rpn_bbox = concatenate 1 ~name:"rpn_bbox"
                (Array.init 5 (fun i -> rpns.(i).(1)))
```

Single RPN graphs are applied on different features in `rpn_features_maps`, and the results from these networks are then concatenated together.
For each bounding box on the image, the RPN returns the likelihood that it contains an object and a refinement for the anchor, both are represented by rank-3 ndarrays.

Next, in the proposal layer, the 1000 best anchors are then selected according to their objectness (higher is better). Anchors that overlap too much with each other are eliminated, to avoid detecting the same object multiple times. Each selected anchor is also refined in case it was not perfectly centred around the object.

```
let rpn_rois =
    let prop_f = PL.proposal_layer C.post_nms_rois C.rpn_nms_threshold in
    MrcnnUtil.delay_lambda_array [|C.post_nms_rois; 4|] prop_f ~name:"ROI"
      [|rpn_class; rpn_bbox; input_anchors|] in
```

The proposal layer picks the top anchors from the RPN output, based on non maximum suppression and anchor scores. It then applies the deltas to the anchors.

### Classification

All anchor proposals from the previous layer are resized to a fixed size and fed into a 10-layer neural network.
The network assigns each of them the probability that it belongs to each class. The network is pre-trained on fixed classes; changing the set of classes requires to re-train the whole network.
Note that this step does not take as much time for each anchor as a full-fledged image classifier such as Inception, since it reuses the pre-computed feature maps from the Feature Pyramid Network. 
Therefore there is no need to go back to the original picture.
The class with the highest probability is chosen for each proposal, and, thanks to the class predictions, the anchor proposals are even more refined.
Proposals classified in the background class are deleted. Eventually, only the proposals with an objectness over some threshold are kept, and we have our final detections, each having a bounding box and a label.

```
let mrcnn_class, mrcnn_bbox =
FPN.fpn_classifier_graph rpn_rois mrcnn_feature_maps input_image_meta
  C.pool_size C.num_classes C.fpn_classif_fc_layers_size in

let detections = MrcnnUtil.delay_lambda_array [|C.detection_max_instances; 6|]
                (DL.detection_layer ()) ~name:"mrcnn_detection"
                [|rpn_rois; mrcnn_class; mrcnn_bbox; input_image_meta|] in
let detection_boxes = lambda_array [|C.detection_max_instances; 4|]
                (fun t -> Maths.get_slice [[]; []; [0;3]] t.(0))
                [|detections|]
```

A Feature Pyramid Network classifier associates a class to each proposal and refines the bounding box for that class even more.
The only thing left to do is to generate a binary mask on each object. This is handled by a small convolutional neural network which produces a small square of values between 0 and 1 for each detected bounding box. 
This square is resized to the original size of the bounding box with bilinear interpolation, and pixels with a value over 0.5 are tagged as being part of the object.

```
let mrcnn_mask = FPN.build_fpn_mask_graph detection_boxes mrcnn_feature_maps
    input_image_meta C.mask_pool_size C.num_classes
```

And finally, the output contains detection results and masks from the previous steps. 

```
outputs ~name:C.name [|detections; mrcnn_mask|]
```

## Run the Code

After getting to know the internal mechanism of the MRCNN architecture, we can then try run the code to see how it works.
One example of using the MRCNN code is in [this example script](https://github.com/owlbarn/owl_mask_rcnn/blob/master/examples/evalImage.ml). 
The core part is listed below:

```
open Mrcnn

let src = "image.png" in
let fun_detect = Model.detect () in
let Model.({rois; class_ids; scores; masks}) = fun_detect src in
let img_arr = Image.img_to_ndarray src in
let filename = Filename.basename src in
let format = Images.guess_format src in
let out_loc = out ^ filename in
Visualise.display_masks img_arr rois masks class_ids;
let camlimg = Image.img_of_ndarray img_arr in
Visualise.display_labels camlimg rois class_ids scores;
Image.save out_loc format camlimg;
Visualise.print_results class_ids rois scores
```

The most important step is to apply the `Model.detect` function on the input images, which returns the region of interests, the classification result ID of the object in that region, the classification certainty scores, and a mask that shows the outline of that object in the region.
With this information, the `Visualise` module runs for three passes on the original image: the first for adding bounding boxes and object masks, the second for adding the numbers close to the bounding box, and finally for printing out the resulting images from the previous two steps.  

In this example, the pre-trained weights on 80 classes of common objects are provided, which have been converted from the TensorFlow implementation mentioned above.
As to the execution speed, processing one image with a size of 1024x1024 pixels takes between 10 and 15 seconds on a moderate laptop.
You can try a [demo](http://demo.ocaml.xyz/mrcnn.html) of the network without installing anything.

## Summary

This chapter introduces Instance Segmentation, another powerful computer vision task that is wildly used in various fields. 
It encompasses several different tasks, including image recognition, semantic segmentation, and object detection. 
This task can be performed with the Mask R-CNN network architecture. 
We explain in detail how it is constructed using the Owl code. 
Besides, we also provide example code and online demo so that the readers can try to play with this powerful application conveniently.

## References
