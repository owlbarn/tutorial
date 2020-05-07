# Case - Instance Segmentation

Computer vision is a field dealing with many different automated tasks whose goal is to give high-level descriptions of images and videos. It has been applied to a wide variety of domains ranging from highly technical (automatic tagging of satellite images, analysis of medical images, ...) to more mundane (categorise pictures in your phone, make your face into an emoji, ...).
It has seen tremendous progress since 2012, when [A. Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) used the first deep learning approach to computer vision, crushing all their opponents in the [ImageNet challenge](http://image-net.org/challenges/LSVRC/2012/results.html). It has therefore evolved quite a lot since Inception was first described in 2014 and it was relevant to implement a more recent and involved network with Owl.

In this chapter, we are going to introduce another classical computer vision task: Instance Segmentation, including its connection with other similar applications, how the deep neural network is constructed in Owl, and how the network, loaded with pre-trained weights, can be used to process users' input image.
We have also included an online demo for this application.

## Introduction

In the chapter about the [image classification](https://ocaml.xyz/book/case-image-inception.html), we have introduced how the DNN can be applied to classify the one single object in an image.
It gets easily confused when there are lots of objects.   

*Object Detection* is another classical computer vision task. Given an image that contains multiple objects, an object detection applications aims to classify individual objects and localize each using a bounding box.

![](images/case-obj-detect/example_obj.jpg){#fig:case-obj-detect:example_obj}
*Example of object detection ([src](https://en.wikipedia.org/wiki/File:Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg))*

Similarly,  *Semantic Segmentation* classify the pixels on an image in different categories. Each segment is recognised by a "mask" that follows cover the whole object.

![](images/case-obj-detect/example_seg.jpg){#fig:case-obj-detect:example_seg}
*Example of semantic segmentation ([src](https://gts.ai/how-do-we-solve-the-challenges-faced-due-to-semantic-segmentation/))*

In 2017, the *Mask R-CNN* (Mask Region-based Convolutional Neural Network) architecture was published and with sufficient training, it can solve all these problems at once: it can detect objects on an image, label each of them and provide a binary mask to tell which pixels belong to the objects.
This task is called *Instance Segmentation*.
This network has now been implemented in Owl.
As a preliminary example, this is what it can do:

![Example: Street view](images/case-obj-detect/example_00.jpg){#fig:case-obj-detect:example_01}

![Example: Sheep](images/case-obj-detect/example_01.jpg){#fig:case-obj-detect:example_02}

In these two examples, normal pictures are processed by MRCNN, and the objects (people, sheep, bag, car, bus, umbrella, etc.) are segmented from the picture and recognised with a percentage of confidence, represented by a number between 0 and 1.

Image segmentation have massive application scenarios in the industry, such as medical imaging (locating tumours, detecting cancer cells ...), traffic control systems, locate objects in satellite images, etc.

## Mask R-CNN Network

This section will briefly outline the main parts of architecture of Mask R-CNN and how it stands out from its predecessors.
You can of course get more detailed and technical explanations in the [original paper](https://arxiv.org/abs/1703.06870).
The Owl implementation of the inference mode is available in [this repository](https://github.com/pvdhove/owl-mask-rcnn).
The code was mostly ported from this [Keras/TensorFlow implementation](https://github.com/matterport/Mask_RCNN).
This work in this chapter is conducted by [Pierre Vandenhove](http://math.umons.ac.be/staff/Vandenhove.Pierre/) during his internship in the OCamlLabs.

MRCNN extends [Faster R-CNN](https://arxiv.org/abs/1506.01497), which itself extends [Fast R-CNN](https://arxiv.org/abs/1504.08083).
In Fast R-CNN, the authors propose a network that accepts input images and regions of interest (RoI). For each region, features are extracted by several fully-connected layers, and the features are fed into a branch.
One output of this branch contains the output classification (together with possibility of that classification) of the object in that region, and the other specifies the rectangle location of the object.

In Faster R-CNN, the authors point out that, there is no need to find RoIs using other methods. They propose a Region Proposal Network (RPN) that share the same feature extraction backbone with that in Fast R-CNN.
RPN is a small convolutional network that scans the feature maps quickly, output  a set of rectangular possible object region proposals, each associated with a number that could be called the *objectness* of that region.
The RoI feature extraction part of Fast R-CNN is kept unchanged here.
In this way, a single Faster R-CNN network can be trained and then perform the object detection task without extra help from other region proposal methods.

To perform the task of not just objection detection, but also semantic segmentation Mask R-CNN keeps the architecture of Faster R-CNN, only adding an extra branch in the final stage of its RoI feature layer.
Where previously the outputs includes object classification and location, now a third branch contains information about the mask of object in the RoI.
Therefore, for any RoI, the Mask R-CNN retrieves its rectangle bound, classification results, classification possibility, and the mask of that object, all information at one pass.

## Building Mask R-CNN

After a quick introduction of the MRCNN and how it is developed in theory, let's look at the code to understand how it is constructed in Owl, one piece at a time.
Feel free to jump this part for now if you are just interested in using the network directly.

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

The network accepts three inputs, for images, meta data, and number of anchors.
The `Configuration` module contains a list of constants that will be used in building the network.

### Feature Extractor

The picture is first fed to a convolutional neural network in order to extract features on the image. The first few layers detect low-level features (edges and basic shapes) but as you go deeper into the network, these features are assembled to detect higher level features (people, cars) (which, some argue, works in the same way as the brain). Five of these layers (called feature maps) of various sizes, both high- and low-level, are then passed on to the next parts. This implementation chooses Microsoft’s [ResNet101](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) as a feature extractor.

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

To try to locate the objects, about 250,000 overlapping rectangular regions (anchors) are generated.

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

Single RPN graphs are applied on the different features in `rpn_features_maps`, and the results from these networks are then concatenated together.
For each anchor (or bounding box) on the image, the RPN returns the likelihood that it contains an object and a refinement for the anchor, both are rank-3 ndarrays.


Next, in the proposal layer, the 1000 best anchors are then selected according to their objectness (higher is better). Anchors that overlap too much with each other are eliminated, to avoid detecting the same object multiple times. Each selected anchor is also refined in case it was not perfectly centred around the object.

```
let rpn_rois =
    let prop_f = PL.proposal_layer C.post_nms_rois C.rpn_nms_threshold in
    MrcnnUtil.delay_lambda_array [|C.post_nms_rois; 4|] prop_f ~name:"ROI"
      [|rpn_class; rpn_bbox; input_anchors|] in
```

The proposal layer picks the top anchors from the RPN output, based on non maximum suppression and anchor scores. It applies the deltas to the anchors.

### Classification

All anchor proposals from the previous layer are resized to a fixed size and fed into a 10-layer neural network that assigns to each of them probabilities that it belongs to each class (the network is pre-trained on fixed classes; changing the set of classes requires to re-train the whole network).
Note that this step does not take as much time for each anchor as a full-fledged image classifier (such as Inception) since it reuses the pre-computed feature maps from the Feature Pyramid Network, therefore no need to go back to the original picture.
The class with the highest probability is chosen for each proposal and thanks to the class predictions, the anchor proposals are even more refined.
Proposals classified in the ’background’ class are deleted. Eventually, only the proposals with an objectness over some threshold are kept, and we have our final detections, each coming with a bounding box and a label!

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

The only thing left to do is to generate a binary mask on each object. This is handled by a small convolutional neural network which outputs for each detected bounding box a small square of values between 0 and 1. This square is resized to the original size of the bounding box with bilinear interpolation, and pixels with a value over 0.5 are tagged as being part of the object.

```
let mrcnn_mask = FPN.build_fpn_mask_graph detection_boxes mrcnn_feature_maps
    input_image_meta C.mask_pool_size C.num_classes
```

And finally, the output contains detection results and mask from previous steps.
```
outputs ~name:C.name [|detections; mrcnn_mask|]
```

## Run the Code

One example of using the MRCNN code is in [this example](https://github.com/owlbarn/owl_mask_rcnn/blob/master/examples/evalImage.ml). The core part is list below:

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

The most import step is to apply `Model.detect ()` on the input images, which returns the region of interests (rois), the classification result id of the object in that region, the classification certainty scores, and a mask that shows the outline of that object in the region.
With this information, the `Visualise` module runs for three passes on the original image: the first for adding bounding boxes and object masks, the second for adding the number close to the bounding box, and finally for printing out the resulting images from the previous two steps.  


Pre-trained weights on 80 classes of common objects are provided, which have been converted from the TensorFlow implementation mentioned above.

Processing one image with a size of 1024x1024 pixels takes between 10 and 15 seconds on a moderate laptop.
You can try a [demo](http://demo.ocaml.xyz/mrcnn.html) of the network without installing anything.

## Summary
