# Case - Image Recognition

How can a computer take an image and answer questions like "what is in this picture? A cat, dog, or something else?"
In the last few years the field of machine learning has made tremendous progress on addressing this difficult problem. In particular, Deep Neural Network (DNN) can achieve reasonable performance on visual recognition tasks -- matching or exceeding human performance in some domains.

We have introduced the neural network module in previous chapter. 
In this chapter, we will show one specific example that is built on the neural network module: using the InceptionV3 architecture to perform the image classification task. 

## Background

InceptionV3 is a a widely-used image classification DNN architecture that can attain significant accuracy with small amount of parameters. 
It is not invented out of thin air. The development using DNN to perform image recognition is a stream that dates back to more than 20 years ago.
During this period, the research in this area is pushed forward again and again in various work. 
In this chapter, we first introduce how image classification architectures are developed up until Inception. 
They will be helpful to understand how Inception architectures are built. 

### LeNet

In the regression chapter, we have seen a simple neural network that contains three layer, and use it to recognise the simple handwritten numbers from the MNSIT dataset.
Here, each pixel acts as an input feature. 
Remember that each image can be seen as a ndarray. 
For a black and white image such as the MNSIT image, pixels are interpreted as a matrix. Every pixel has a value between 0 and 255.  For a color image, it can be interpreted as a 3-dimension array with  three channels, each corresponding to the blue, green, and red layer.

However, we cannot rely on adding more fully connected layers to do real world high precision image detections.
One important improvement that the Convolution Neural Network make is that it uses filters in the convolution operation.
As a result, instead of using the whole image as an array of features, the image is divided into a number of tiles. 
They will then serve as the basic feature of the network's prediction.

Explain and visualise "feature" and feature map.

The next building block is the pooling layer. Recall from the neural network chapter that, both average pooling and max pooling can aggregate information from multiple pixels into one and "blur" the input image or feature. 
So why it is so important?
By reducing the size of input, pooling helps to reduce the number of parameters and the amount of computation required. 
Besides, blurring the features is a way to limit over-fitting training data.

At the end, only connect high-level features with fully connection.
This is the structure proposed in [@lecun1998gradient].
This whole process can be shown in [@fig:case-image-inception:workflow].

![Workflow of image classification](images/case-image-inception/cnn_workflow.png "workflow"){width=100% #fig:case-image-inception:workflow}

(Or use the LeCun paper figure)

In general, layers of convolution retrieve information from detailed to more abstracted gradually.
In a DNN, The lower layers of neurons retrieve information about simple shapes such as edges and points.
Going higher, the neurons can capture complex structures, such as the tire of a car, the face of a cat, etc.
Close to the top layers, the neurons can retrieve and abstract complex ideas, such as "car", "cat", etc.
And then finally generates the classification results. 

### AlexNet

Next breakthrough comes from the AlexNet proposed in [@krizhevsky2012imagenet].
The authors introduce better *non-linearity* in the network with the ReLU activation.
Operations such as convolution includes mainly linear operations such as matrix multiplication and `add`.
But that's not how the real world data looks like. Remember that from the previous Regression chapter, though linear regression is basic, but for most of the real world application we need more complex method such as polynomial regression. 
The same can be applied here. We need to increase the non-linearity to accommodate real world data such as image. 

There are multiple activation choice, such as `tanh` and `sigmoid`.
However, the `relu` operation, which set negative values of a feature map to zero, is frequently used.
It makes training faster, and accuracy loss in gradient computation is small. 

Another thing that AlexNet proposes is to use the `dropout` layer.
It is mainly used to solve the over-fitting problem. 
This operation only works at training time. It randomly makes the elements in a ndarray from the last layer to be zero, and thus "deactivate" the knowledge that can be learnt from these points. 
In this way, the network intentionally drops certain part of training examples and avoid the over-fitting problem.
It is similar to the regularisation method we use in the linear regression.

The one more thing that we need to take a note from AlexNet is that by going deeper make the network "longer", we achieve better accuracy. 
So instead of just convolution and pooling, we now build convolution followed by convolution and then pooling, and repeat this process again.... 
A deeper network captures finer features, and this would be a trend that is followed by successive architectures.

### VGG

The VGG network proposed in [@simonyan2014very] is the next step after AlexNet.
The most notable change that introduced by VGG is that it uses small kernel sizes such as `3x3` instead of the `11x11` with a large stride of 4 in AlexNet. 

Using multiple small kernels is much more flexible than only using a large one. 
For example, for an input image, by applying two `3x3` kernels with slide size of 1, that equals to using a `5x5` kernel. 
If stacking three `3x3`, it equals using one `7x7` convolution. 

By replacing large kernels with multiple small kernels, the number of parameter is visibly reduced.
In the previous two examples, replace one `5x5` with two `3x3`, we reduce the parameter by $1 - 2 * 3 * 3 / (5 * 5) = 28\%$. Replace the `7x7` kernel, and we save parameters by $1 - 3 * 3 * 3 / ( 7 * 7) = 45\%$.

Therefore, with this reduction of parameter size, we can now build network with more layers, which tends to yield better performance.
The VGG networks comes with two variates, VGG16 and VGG19, which are the same in structure, and the only difference is that VGG19 is deeper. 
The code to build a VGG16 network with Owl is shown in [@zoo2019vgg16].

One extra benefit is that using small kernels increases non-linearity. 
Image an extreme case where the kernel is as large as the input image, then the whole convolution is just one big matrix multiplication, a totally linear operations. 
As we have just explained in the previous section, we hope to reduce the linearity in training CNN to accommodate more real-world problems.

### ResNet

We keep saying that building deeper network is the trend. 
However, going deeper has its limit.
The deeper you go, the more you will experience the "vanishing gradient" problem. 
This problems is that, in a very deep network, during the back-propagation phase, the repeated multiplication operations will make the gradients very small, and thus the performance affected. 

The ResNet in [@he2016deep] proposes an "identity shortcut connection" that skips one or more layers and combine with predecessor layers. It is called a residual block, as shown in [@fig:case-image-inception:residual] (Src: original paper).

![Residual block in the ResNet](images/case-image-inception/residual-block.png "residual block"){width=60% #fig:case-image-inception:residual}

We can see that there is the element-wise addition that combines the information of the current output and its predecessors two layers ago. 
It solves the gradient problem in stacking layers, since now the the error can be backpropagated through multiple paths.
The authors shows that during training the deeper layers do not produce a error higher than its predecessor in lower layer.

Also note that the residual block aggregating features from different level of layers, instead of purely stacking them. 
This patten proves to be useful and will also be used in the Inception architecture. 

The ResNet can also be constructed by stacking for different layers, so that we have ResNet50, ResNet101, ResNet152, etc.
The code to build a ResNet50 network with Owl is shown in [@zoo2019resnet50].

### SqueezeNet

All these architectures, including Inception, mainly aim to push the detection accuracy forward.
However, at some point we are faced with the tradeoff between accuracy and model size. 
We have seen that sometimes reducing the parameter  size can help the network to go deeper, and thus tends to give better accuracy.
However, with the growing trend od edge computing, there is requirement for extremely small deep neural networks so that it can be easily distributed and deployed on less powerful devices. 
For that, sacrificing a bit of accuracy is acceptable.

There are more and more efforts towards that direction, and the SqueezeNet [@iandola2016squeezenet] is one of them. 
It claims to have the AlexNet level of accuracy, but with 50 times fewer parameters.

There are mainly three design principles for the SqueezeNet.
The first one is what we are already familiar with: using a lot of 1x1 convolutions.
The second one also sounds familiar. 
The 1x1 convolutions are used to reduced the output channels before feeding the result to the more complex 3x3 convolutions. Therefore the 3x3 convolution can have smaller input channels and thus smaller parameter size. 
These two principles are incorporated into a basic building block called "fire module".

Now that we can have a much smaller network, what about the accuracy? We don't really want to discard the accuracy requirement totally.
The third design principle in SqueezeNet is to delay the down-sampling to later in the network. 
Recall that the down-sampling layers such as maxpooling "blur" the information intentionally. 
The intuition is that, if we only use them occasionally and in the deeper layer of the network, 
we can have large activation maps that preserve more information and make the detection result more accurate.
The code to build a SqueezeNet network with Owl is shown in [@zoo2019squeezenet].

### Capsule Network

The research on image detection network structures is still on-going.
Besides the parameter size and detection accuracy, more requirements are proposed.
For example, there is the problem of recognising an object, e.g. a car, from different perspective. 
And the "Picasso problem" in image recognition where some feature in an object is intentionally distorted or misplaced.
These problems shows one deficient in the existing image classification approach: the lack of connection between features. 
It may recognise a "nose" feature, and a "eye" feature, and then the object is recognised as a human face, even though the nose is perhaps above the eyes. 
The "Capsule network" is proposed to address this problem. Instead of using only scalar to represent feature, it uses a vector that includes more information such as orientation and object size etc. 
The "capsule" utilises these information to capture the relative relationship between features. 

There are many more networks that we cannot cover them one by one here, but hopefully you can see that there are some common theme in the development of image recognition architectures. 
Next, we will come to the main topic of this chapter: how InceptionV3 is designed and built based on these previous work. 

## Building InceptionV3 Network

Proppsed by Christian Szegedy et. al., [InceptionV3](https://arxiv.org/abs/1512.00567) is one of Google's latest effort to do image recognition. It is trained for the [ImageNet Large Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/). This is a standard task in computer vision, where models try to classify entire images into 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher", etc. Compared with previous DNN models, InceptionV3 has one of the most complex networks architectures in computer vision.

The design of image recognition networks is about the tradeoff between computation cost, memory usage, and accuracy.
Just increasing model size and computation cost tends to increase the accuracy, but the benefit will decrease soon. 
To solve this problem, compared to previous similar networks, the Inception architecture aims to perform well with strict constraints on memory and computational budget.
This design follows several principles, such as balancing the width and depth of the network, and performing spatial aggregation over lower dimensional embeddings can lead to small loss in representational power of networks. 
The resulting Inception network architectures has high performance and a relatively modest computation cost compared to simpler, more monolithic architectures.

[@fig:case-image-inception:inceptionv3] shows the overall architecture of this network ([src](https://cloud.google.com/tpu/docs/inception-v3-advanced)):

![Network Architecture of InceptionV3](images/case-image-inception/inceptionv3.png "inceptionv3"){width=95% #fig:case-image-inception:inceptionv3}

We can see that the whole network can be divided into several parts, and the inception module A, B, and C are both repeated based on one structure. 
That's where the name "Inception" comes from: like dreams, you can have stack these basic units layer by layer.

### InceptionV1 and InceptionV2

The reason we say "InceptionV3" is because it is developed based on two previous similar architectures.
To understand InceptionV3, we first need to know the characteristics of its predecessors. 

The first version of Inception, GoogLeNet  [@szegedy2015going], proposes to combine convolutions with different filter sizes on the same input, and then concatenate the resulting features together.
Think about an image of a bird. If you sticking with using a normal square filter, then perhaps the features such as "feather" is a bit difficult to capture, but easier to do if you use a "thin" filter with a size of e.g. `1x7`.
By aggregating information from applying different features, we can extract feature from multi-level at each step.

Of course, adding extra filters increase computation complexity. 
To remedy this effect, the Inception network proposes to utilise the `1x1` convolution to reduce the dimensions of feature maps. 
For example, we want to apply a `3x3` filter to input ndarray of size `[|1; 300; 300; 768|]` and the output channel should be `320`.
Instead of applying a convolution layer of `[|3; 3; 768; 320|]` directly, we first reduce the dimension to, say, 192 by using a small convolution layer `[|1; 1; 768; 192|]`, and then apply a `[|3; 3; 192; 320|]` convolution layer to get the final result. 
By reducing input dimension before the more complex computation with large kernels, the computation complexity is reduced.

For those who are confused about the meaning of this array: recall from previous chapter that the format of an image as ndarray is `[|batch; column; row; channel|]`, and that the format of a convolution operation is mainly represented as `[|kernel_column; kernel_row; input_channel; output_channel|]`.
Here we ignore the other parameters such as slides and padding since here we focus only on the change of channels of feature map. 

Then in a updated version of GoogLeNet, the InceptionV2 (or BN-inception), utilises the "Batch Normalisation" layer.
We have seen how normalising input data plays a vital role in improving the efficiency of gradient descent in the Optimisation chapter. 
Batch normalisation follows a similar path, only at it now works between each layer instead of just at the input data.
This layer rescale each mini-batch with the mean and variance of this mini-batch.

Image that we train a network to recognise horse, but most of the training data are actually black or blown horse. Then the network's performance on white horse might not be quite ideal. 
This again leads us back to the over-fitting problem.
The batch normalisation layer adds noise to input by scaling. 
As a result, the content at deeper layer is less sensitive to content in lower layers.  
Overall, the batch normalisation layer greatly improves the efficiency of training. 

### Factorisation 

Now that we understand how the image recognition architectures are developed, finally it's time to see who these factors are utilised into the InceptionV3 structure. 

```ocaml env=incpetionv3
open Owl
open Owl_types
open Neural.S
open Neural.S.Graph

let conv2d_bn ?(padding=SAME) kernel stride nn =
  conv2d ~padding kernel stride nn
  |> normalisation ~training:false ~axis:3
  |> activation Activation.Relu
```

Here the `conv2d_bn` is a basic building block used in this network, consisting of a convolution layer, a normalisation layer, and a relu activation layer.
We have already introduced how these different types of layer work.
You can think of `conv2d_bn` as an enhanced convolution layer.

Based on this basic block, the aim in building Inception network is still to go deeper, but here the authors introduces the three type of *Inception Modules* as a unit of stacking layers. 
Each module factorise large kernels into smaller ones.
Let's look at them one by one. 


```ocaml env=incpetionv3
let mix_typ1 in_shape bp_size nn =
  let branch1x1 = conv2d_bn [|1;1;in_shape;64|] [|1;1|] nn in
  let branch5x5 = nn
    |> conv2d_bn [|1;1;in_shape;48|] [|1;1|]
    |> conv2d_bn [|5;5;48;64|] [|1;1|]
  in
  let branch3x3dbl = nn
    |> conv2d_bn [|1;1;in_shape;64|] [|1;1|]
    |> conv2d_bn [|3;3;64;96|]  [|1;1|]
    |> conv2d_bn [|3;3;96;96|]  [|1;1|]
  in
  let branch_pool = nn
    |> avg_pool2d [|3;3|] [|1;1|]
    |> conv2d_bn [|1;1;in_shape; bp_size |] [|1;1|]
  in
  concatenate 3 [|branch1x1; branch5x5; branch3x3dbl; branch_pool|]
```

The `mix_typ1` structure implement the the first type of inception module.
In `branch3x3dbl` branch, it replace a `5x5` kernel convolution layer with two `3x3` convolution layers.
It follows the design in the VGG network. 
Of course, as we have explained, both branches uses the `1x1` to reduce dimensions before complex convolution computation. 

In some implementation the `3x3` convolutions can also be further factorised into `3x1` and then `1x3` convolutions.
There are more tha one way to do the factorisation.
You might be think that it is also a good idea to replace the `3x3` convolution with two `2x2`s. 
Well, we could do that but it saves only about 11% parameters, compared to the 33% save of current practice. 


```ocaml env=incpetionv3
let mix_typ4 size nn =
  let branch1x1 = conv2d_bn [|1;1;768;192|] [|1;1|] nn in
  let branch7x7 = nn
    |> conv2d_bn [|1;1;768;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;192|] [|1;1|]
  in
  let branch7x7dbl = nn
    |> conv2d_bn [|1;1;768;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;size|] [|1;1|]
    |> conv2d_bn [|7;1;size;size|] [|1;1|]
    |> conv2d_bn [|1;7;size;192|] [|1;1|]
  in
  let branch_pool = nn
    |> avg_pool2d [|3;3|] [|1;1|]
    |> conv2d_bn [|1;1; 768; 192|] [|1;1|]
  in
  concatenate 3 [|branch1x1; branch7x7; branch7x7dbl; branch_pool|]
```

As shown in the code above, `mix_typ4` shows the Type B inception module, another basic unit.
It still separate into three branches and then concatenate them together. 
The special about this this type of branch is that it factorise a `7x7` convolution into the combination of a `7x1` and then a `1x7` convolution. 
Again, this change saves $(49 - 14) / 49 = 71.4\%$ parameters.

If you have doubt about this replacement, you can do a simple experiment:

```ocaml
open Neural.S
open Neural.S.Graph

let network_01 =
  input [|28;28;1|]
  |> conv2d ~padding:VALID [|7;7;1;1|] [|1;1|]
  |> get_network

let network_02 =
  input [|28;28;1|]
  |> conv2d ~padding:VALID [|7;1;1;1|] [|1;1|]
  |> conv2d ~padding:VALID [|1;7;1;1|] [|1;1|]
  |> get_network
```

Checking the output log, you can find out that both network give the same output shape. 
This factorisation is intuitive: convolution of size does not change the output shape at that dimension.
Then the two convolutions actually performs feature extraction along each dimension (height and width) respectively.


```ocaml env=incpetionv3
let mix_typ9 input nn =
  let branch1x1 = conv2d_bn [|1;1;input;320|] [|1;1|] nn in
  let branch3x3 = conv2d_bn [|1;1;input;384|] [|1;1|] nn in
  let branch3x3_1 = branch3x3 |> conv2d_bn [|1;3;384;384|] [|1;1|] in
  let branch3x3_2 = branch3x3 |> conv2d_bn [|3;1;384;384|] [|1;1|] in
  let branch3x3 = concatenate 3 [| branch3x3_1; branch3x3_2 |] in
  let branch3x3dbl = nn |> conv2d_bn [|1;1;input;448|] [|1;1|] |> conv2d_bn [|3;3;448;384|] [|1;1|] in
  let branch3x3dbl_1 = branch3x3dbl |> conv2d_bn [|1;3;384;384|] [|1;1|]  in
  let branch3x3dbl_2 = branch3x3dbl |> conv2d_bn [|3;1;384;384|] [|1;1|]  in
  let branch3x3dbl = concatenate 3 [|branch3x3dbl_1; branch3x3dbl_2|] in
  let branch_pool = nn |> avg_pool2d [|3;3|] [|1;1|] |> conv2d_bn [|1;1;input;192|] [|1;1|] in
  concatenate 3 [|branch1x1; branch3x3; branch3x3dbl; branch_pool|]
```

The final inception module is a bit more complex, but by now you should be able to understand its construction. 
It aggregate information from four branches.
The `1x1` convolutions are used to reduce the dimension and computation complexity.
Note that in both the `branch3x3` and `branch3x3dbl` branches, both `3x1` and `1x3` convolutions are used, not as replacement of `3x3`, but as separate branches. 
This module is for promoting high dimensional representations.

TODO: explain "promoting high dimensional representations".

Together, these three modules makes up the core part of the InceptionV3 architecture.
By applying different techniques and designs, these modules take less memory and less prone to over-fitting problem. 
Thus they can be stacked together to make the whole network go deeper. 

### Grid Size Reduction

For the very beginning of the design of CNN, we need to reduce the size of feature maps as well as increasing the number or channel of the feature map. 
We have explained how it is done using the combination of pooling layer and convolution layer.
However, the reduction solution constructed this way is either too greedy or two computationally expensive. 

Inception architecture proposes a grid size reduction module. 
It put the same input feature map into to set of pipelines, one of them uses the pooling operation, and the other uses only convolution layers.
These two type of layers are then not stacked together but concatenated vertically, as shown in the next part of code.

```ocaml env=incpetionv3
let mix_typ3 nn =
  let branch3x3 = conv2d_bn [|3;3;288;384|] [|2;2|] ~padding:VALID nn in
  let branch3x3dbl = nn
    |> conv2d_bn [|1;1;288;64|] [|1;1|]
    |> conv2d_bn [|3;3;64;96|] [|1;1|]
    |> conv2d_bn [|3;3;96;96|] [|2;2|] ~padding:VALID
  in
  let branch_pool = max_pool2d [|3;3|] [|2;2|] ~padding:VALID nn in
  concatenate 3 [|branch3x3; branch3x3dbl; branch_pool|]
```

`mix_typ3` builds the first grid size reduction module. 
This replacement strategy can perform efficient reduction without losing too much information.
Similarly, an extended version of this grid size reduction module is also included.

```ocaml env=incpetionv3
let mix_typ8 nn =
  let branch3x3 = nn
    |> conv2d_bn [|1;1;768;192|] [|1;1|]
    |> conv2d_bn [|3;3;192;320|] [|2;2|] ~padding:VALID
  in
  let branch7x7x3 = nn
    |> conv2d_bn [|1;1;768;192|] [|1;1|]
    |> conv2d_bn [|1;7;192;192|] [|1;1|]
    |> conv2d_bn [|7;1;192;192|] [|1;1|]
    |> conv2d_bn [|3;3;192;192|] [|2;2|] ~padding:VALID
  in
  let branch_pool = max_pool2d [|3;3|] [|2;2|] ~padding:VALID nn in
  concatenate 3 [|branch3x3; branch7x7x3; branch_pool|]
```

`mix_typ8` is the second grid size reduction module in the deeper part of the network. 
It uses three branches instead of two, and each convolution branch is more complex. The `1x1` convolutions are again used.
But in general it still follows the principle of performing reduction in parallel and then concatenate them together, performing an efficient feature map reduction.  

### InceptionV3 Architecture

After introducing the separate units, finally we can construct them together into the whole network in code.

```ocaml env=incpetionv3
let make_network img_size =
  input [|img_size;img_size;3|]
  |> conv2d_bn [|3;3;3;32|] [|2;2|] ~padding:VALID
  |> conv2d_bn [|3;3;32;32|] [|1;1|] ~padding:VALID
  |> conv2d_bn [|3;3;32;64|] [|1;1|]
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  |> conv2d_bn [|1;1;64;80|] [|1;1|] ~padding:VALID
  |> conv2d_bn [|3;3;80;192|] [|1;1|] ~padding:VALID
  |> max_pool2d [|3;3|] [|2;2|] ~padding:VALID
  |> mix_typ1 192 32 |> mix_typ1 256 64 |> mix_typ1 288 64
  |> mix_typ3
  |> mix_typ4 128 |> mix_typ4 160 |> mix_typ4 160 |> mix_typ4 192
  |> mix_typ8
  |> mix_typ9 1280 |> mix_typ9 2048
  |> global_avg_pool2d
  |> linear 1000 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

There is only one last thing we need to mention: the *global pooling*.
Global Average/Max Pooling calculates the average/max output of each feature map in the previous layer. 
For example, if you have an `[|1;10;10;64|]` feature map, then this operation can make it to be `[|1;1;1;64|]`.
This operation seems very simple, but it works at the very end of a network, and can be used to replace the fully connection layer.
The parameter size of the fully connection layer has always been a problem. 
Now that it is replaced with a non-trainable operation, the parameter size is greatly reduced without the performance. 
Besides, the global pooling layer is more robust to spatial translations in the data and mitigate the over-fitting problem in fully connection, accoding to [@lin2013network].

The full code is listed in [@zoo2019inceptionv3]. 
Even if you are not quite familiar with Owl or OCaml, it must still be quite surprising to see the network that contains 313 neuron nodes can be constructed using only about 150 lines of code. And we are talking about one of the most complex neural networks for computer vision. 

Besides InceptionV3, you can also easily construct other popular image recognition networks, such as [ResNet50](https://gist.github.com/pvdhove/a05bf0dbe62361b9c2aff89d26d09ba1), [VGG16](https://gist.github.com/jzstark/f5409c44d6444921a8ceec00e33c42c4), [SqueezeNet](https://gist.github.com/jzstark/c424e1d1454d58cfb9b0284ba1925a48) etc. with elegant Owl code.  
We have already mentioned most of them in previous sections. 

## Preparing Weights

Only building a network structure is not enough. Another important aspect is proper weights of a neural network.
It can be achieved by training on GBs of image data for days or longer on powerful machine clusters. 

The training is usually done via supervised learning using a large set of labelled images. Although Inception v3 can be trained from many different labelled image sets, ImageNet is a common dataset of choice.
ImageNet has over ten million URLs of labelled images. About a million of the images also have bounding boxes specifying a more precise location for the labelled objects.
For the Inception model, the ImageNet dataset is composed of 1,331,167 images which are split into training and evaluation datasets containing 1,281,167 and 50,000 images, respectively.
([COPY ALERT](https://cloud.google.com/tpu/docs/inception-v3-advanced))
The training of this model can take hundreds of hours of training on multiple high-performance GPUs.

However, not everyone has access to such large resource.
Another option is more viable: importing weights from existed pre-trained TensorFlow models, which are currently widely available in model collections such as [this one](https://github.com/fchollet/deep-learning-models/).

The essence of weights is list of ndarrays, which is implemented using `Bigarray` in OCaml. 
So the basic idea is to find a intermediate representation so that we can exchange the ndarray in NumPy and `Bigarray` in OCaml. 
In our implementation, we choose to use the [HDF5](https://portal.hdfgroup.org/display/HDF5/HDF5) as this intermediate data exchange format.
In Python, we use the [h5py](https://www.h5py.org/) library, and in OCaml we use [hdf5-ocaml](https://github.com/vbrankov/hdf5-ocaml).

The method to save or load hdf5 data files are fixed, but the methods to retrieve data from model files vary depending on the type of original files. 
For example, if we choose to import weight form a TensorFlow model, we do something like this to achieves the weight data of each layer:

```python
...
reader = tf.train.NewCheckpointReader(checkpoint_file)
for key in layer_names:
    data=reader.get_tensor(key).tolist()
...
```

In a keras, it's a bit more straightforward:

```python
...
for layer in model.layers:
    weights = layer.get_weights()
...
```

In the OCaml side, we first create a Hashtable and read all the HDF5 key-value pairs into it. 
Each value is saved as a double precision Owl ndarray. 

```text
...
let h = Hashtbl.create 50 in
let f = H5.open_rdonly h5file  in
for i = 0 to (Array.length layers - 1) do
  let w = H5.read_float_genarray f layers.(i) C_layout in
  Hashtbl.add h layers.(i) (Dense.Ndarray.Generic.cast_d2s w)
done;
...
```

And then we can use the mechanism in the Neural Network model to load these values from the hashtable to networks:

```text
...
let wb = Neuron.mkpar n.neuron in 
Printf.printf "%s\n" n.name; 
wb.(0) <- Neuron.Arr (Hashtbl.find layers n.name);
Neuron.update n.neuron wb
...
```

It is very important to make clear the difference in naming of each layer in different platforms, since the creator of the original model may choose any name for each layer. 
Other differences have also to be taken care of.
For example, the `beta` and `gamma` weights in the batch normalisation layer is represented as two different values in TensorFlow model, but they belong to the same layer in Owl. 
Also, some times the dimensions has to be swapped in a ndarray during this weight conversion.


Note that this is one-off work. 
Once you successfully update the network with weights, the weights can be saved using `Graph.save_weights`, without haveing to repeat all these steps again. 
We have already prepared the weights for the InceptionV3 model and other similar models, and the users don't have to worry about all these trivial model exchanging detail.

## Processing Image

Image processing is challenging, since OCaml does not provide powerful functions to manipulate images.
Though there are image processing libraries such as [CamlImages](http://gallium.inria.fr/camlimages/), but we don't want to add extra liabilities to Owl itself. 

To this end, we choose the non-compressed image format PPM.
A PPM file is a 24-bit color image formatted using a text format. It stores each pixel with a number from 0 to 65536, which specifies the color of the pixel. 
Therefore, we can just use ndarray in Owl and convert that directly to PPM image without using any external libraries.
We only need to take care of header information during this process.

For example, here is the code for converting an 3-dimensional array in Owl `img` into a ppm file. 
We first need to get the content from each of the three color channels using slicing, such as the blue channel:

```
let b = N.get_slice [[];[];[0]] img in 
let b = N.reshape b [|h; w|] in
...
```

Here `h` and `w` are the height and width of the image.
Then we need to merge all these three matrix into a large matrix that of are of size `[|h; 3*w|]`.

```
let img_mat = Dense.Matrix.S.zeros h (3 * w) in
Dense.Matrix.S.set_slice [[];[0;-1;3]] img_mat r;
...
```

Then, after rotate this matrix by 90 degree, we need to rewrite this matrix to a large byte variable.
Note that that though the method is straightforward, you need to be careful about the index of each element during the conversion. 

```
let img_str = Bytes.make (w * h * 3) ' ' in 
let ww = 3 * w in 
for i = 0 to ww - 1 do
  for j = 0 to h - 1 do
    let ch = img_arr.(i).(j) |> int_of_float |> char_of_int in
    Bytes.set img_str ((h - 1 -j) * ww + i) ch;
  done
done;
```

Finally we build another byte string that contains the metadata such as height and width, according to the specification of PPM format.
Concatenating the metadata and data together, we then write the bytes data into a file. 


Similarly, to read an PPM image file into ndarray in Owl, we treat the ppm file line by line with `input_line` function.
The metadata such as version and comments are ignored. 
We get the metadata such as width and heigh from the header.

```text
...
let w_h_line = input_line fp in
let w, h = Scanf.sscanf w_h_line "%d %d" (fun w h -> w, h) in
...
```

Then according these information, we read the rest of data into a large bytes with `Pervasive.really_input`.
Note that under 32bit OCaml, this will only work when reading strings up to about 16 megabytes.

```text
...
let img = Bytes.make (w * h * 3) ' ' in
really_input fp img 0 (w * h * 3);
close_in fp;
...
```

Then we need to re-arrange the data in the bytes into a matrix.

```
let imf_o = Array.make_matrix (w * 3) h 0.0 in
  
let ww = 3 * w  in
for i = 0 to ww - 1 do
  for j = 0 to h - 1 do
    imf_o.(i).(j) <- float_of_int (int_of_char (Bytes.get img ((h - 1 - j ) * ww + i)));
  done
done;
```

This matrix can then be further processed into a ndarray with proper slicing.

```
...
let m = Dense.Matrix.S.of_arrays img in
let m = Dense.Matrix.S.rotate m 270 in
let r = N.get_slice [[];[0;-1;3]] m in
let g = N.get_slice [[];[1;-1;3]] m in
let b = N.get_slice [[];[2;-1;3]] m in
...
```

There are other functions such as reading in ndarray from PPM file. The full image processing code can be viewed in [this gist](https://gist.github.com/jzstark/86a1748bbc898f2e42538839edba00e1).

Of course, most of the time we have to deal with image of other more common format such as PNG and JPEG. 
For the conversion from these format to PPM or the other way around, we use the tool [ImageMagick](https://www.imagemagick.org/).
ImageMagick is a free and open-source tool suite for image related tasks, such as converting, modifying, and editing images. 
It can read and write over 200 image file formats, including the PPM format. 
Therefore, we preprocess the images by converting its format to PPM with the command `convert`.
Also, the computation time of is related with the input size, and we often hope to limit the size of images. That can also be done with the command `convert -resize`.

Another important preprocessing is to normalise the input. 
Instead of processing the input ndarray whose elements ranges from 0 to 255, we need to simply preprocess it so that all the elements fall into the range `[-1, 1]`, as shown in the code below. 

```ocaml
let normalise img = 
  let img = Arr.div_scalar img 255. in 
  let img = Arr.sub_scalar img 0.5  in
  let img = Arr.mul_scalar img 2.   in
  img
```

## Running Inference

Now that we have built the network, and loaded the proper weights, it's time to perform the most exciting and actually the easiest part: inferencing.
Actually the most important part is just one function: `Graph.model`.
In this section we hope to show how to build a service around this function so that a user can perform all the inference steps, from input image to output classification result, with as simple command as possible. 

First let's look at the code:

```
#!/usr/bin/env owl
open Owl

(* Import InceptionV3 Library *)
#zoo "9428a62a31dbea75511882ab8218076f"

let _ = 
  let img = "panda.png" in
  let labels = InceptionV3.infer img in
  let top = 5 in 
  let labels_json   = InceptionV3.to_json ~top labels in
  let labels_tuples = InceptionV3.to_tuples labels in

  Printf.printf "\nTop %d Predictions:\n" top;
  Array.iteri (fun i x -> 
    let cls, prop = x in 
    Printf.printf "Prediction #%d (%.2f%%) : %s\n" i (prop *. 100.) cls;
  ) labels_tuples
```

The code itself is quite simple. 
First, we need to build the whole Inception network structure and loading weight etc., which is what we have already done in [this gist](https://gist.github.com/jzstark/9428a62a31dbea75511882ab8218076f).
All we need is to loading with the zoo module in Owl.
The InceptionV3 module provides three APIs:

- `infer`: Service that performs image recognition tasks over client images. It accept a string that specify the location of a local image. Its return value is a 1x1000 N-dimension array, each element is a float number between 0 and 1, indicating the possibility that the image belongs to one of the 1000 classes from ImageNet.

- `to_json`: Convert the inferred result to a raw JSON string. Parameter `top`: an int value to specify the top-N likeliest labels to return. Default value is 5.

- `to_json`: Convert the inferred result to an array of tuples, each tuple contains label name (“class”, string) and the probability (“prop”, float, between 0 and 1) of target image being in that class.

After loading the inception model with `#zoo` primitive, the use need to designate an absolute path of the local input image. 
Here we use the `extend_zoo_path` utility function to automatically find the "panda.png" image contained in the Gist itself.
But surely a user can have her own choice.  It can be of any common image format (jpg, png, gif, etc.) and size. 
We provide this panda picture just in case you running short of images currently. 

![Panda image that is used for image recognition task](images/case-image-inception/panda.png){width=40% #fig:case-image-inception:panda}

With these work done, we can run inference with the neural network model and the input image by calling the `infer` API from InceptionV3 module, which wraps around the `Graph.model` function.

Then we need to decode the result, getting top-N (N defaults to 5) predictions in human-readable format. 
The output is an array of tuple, each tuple consists of a string for classified type description, and a float number ranging from 0 to 100 to represent the percentage probability of the input image actually being this type.
Finally, if you want, you can pretty-print the result.

Now that the whole script is ready, we can wrap it into a [zoo gist](https://gist.github.com/jzstark/6dfed11c521fb2cd286f2519fb88d3bf).
Then all it takes for the user is just one line:

```
owl -run 6dfed11c521fb2cd286f2519fb88d3bf
```

That’s it. This one-liner is all you need to do to see a image classification example in action. Here is the output (assume using the panda image previously mentioned):

```
Top 5 Predictions:
Prediction #0 (96.20%) : giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca
Prediction #1 (0.12%) : lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
Prediction #2 (0.06%) : space shuttle
Prediction #3 (0.04%) : soccer ball
Prediction #4 (0.03%) : indri, indris, Indri indri, Indri brevicaudatus
```

If you are not interested in installing anything, [here](http://demo.ocaml.xyz/) is a web-based demo of this image classification application powered by Owl. Please feel free to play with it! And the server won’t store your image. Actually, if you are so keen to protect your personal data privacy, then you definitely should try to pull the code here and fast build a local image processing service without worrying your images being seen by anybody else!


## Applications

TODO: rewrite these applications.

Building an image classification application is not the end by itself. It can be used in a wide range of applications.
We list some in this section to show how Owl can be deployed in these scenarios. 

One of the most popular applications of image recognition that we encounter daily is personal photo organization. 
Image recognition is empowering the user experience of photo organization apps. Besides offering a photo storage, apps want to go a step further by giving people better search and discovery functions. They can do that with the automated image organization capabilities provided by machine learning. The image recognition API integrated in the apps categorizes images on the basis of identified patterns and groups them thematically. (COPY ALERT)

Visual recognition on social media is already a fact. Facebook released its facial recognition app Moments, and has been using facial recognition for tagging people on users’ photos for a while.
While face recognition remains a sensitive ground, Facebook hasn’t shied away from integrating it in users’ experience on the social media. Whenever users upload a photo, Facebook is able to recognize objects and scenes in it before people enter a description. The computer vision can distinguish objects, facial expressions, food, natural landscapes and sports, among others. Besides tagging of people on photos, image recognition is used to translate visual content for blind users and to identify inappropriate or offensive images.  ([COPY ALERT](https://imagga.com/blog/the-top-5-uses-of-image-recognition/))

## References
