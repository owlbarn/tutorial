# Case - Neural Style Transfer

What is Neural Style Transfer (NST)? It is a pretty cool application of Deep Neural Networks (DNN),  "the process of using DNN to migrate the semantic content of one image to different styles".

The process is actually very simple, as the title image shows, this application takes two images A and B as input. Let’s say A is "Mona Lisa" of Da Vinci, and B is "The Starry Night" of Vincent van Gogh.

We then specify A as the content image and B as the style image, then what a NST application can produce? Boom! A new Mona Lisa, but with the style of Van Gogh (see the middle of title image)! If you want another style, just replace image B and run the application again. Impressionism, abstractionism, classical art, you name it.

The figure below illustrate this point. You can apply different art styles to the same street view, or apply the same "Starry Sky" style to any pictures.
Isn’t it amazing?

![Example of applying neural style transfer on a street view picture](images/case-nst/nst_example.png){#fig:case-nst:example_01}

## Content and Style

[@gatys2015neural] first propose to use DNN to let programmes to create artistic images of high perceptual quality.
The examples above may look like magic, but surely its not.
In this section, we will first introduce the intuition about how the neural style transfer algorithm works.
For more formal and detailed introduction, please visit the [original paper](https://arxiv.org/abs/1508.06576).

The basic idea is plain: we want to get an image whose content is similar to one image and its artistic style close to the other image.
Of course, to make the algorithm work, first we need to express this sentence in mathematical form so that computers can understand it.
Let's assume for a moment we have already know that, then style transfer can be formalised as an optimisation problem.
Given a content image `c` and a style image `s`, our target is to get an output image `x` so that it minimises:

$$g(x) = \verb|content_distance|(x, c) + \verb|style_distance|(x, s)$$

Here the "distance" between two feature map is calculated by the euclidean distance between the two ndarrays.

You may remember from the regression or neural network chapters that training process is also an optimisation process.
However, do not mistake the optimisation in NST as regression or DNN training.
For the latter one, there is the function $f_w$ that contains parameter $w$ and the training process optimise $w$ to minimise $f_w(x)$.
The optimisation in NST is more like the traditional optimisation problem: we have a function $f$, and we start we an initial input $x_0$ and update it iteratively until we have satisfying $x$ that minimise the target function.

Now we can comeback to the key problem.
While we human beings can kind of feel the style of a paint and visually recognise the contents in a picture, how can we mathematically express the "content" and the "style" of an image?
That's where the convolution network comes to help.
DNNs, especially the ones that are used for computer vision tasks, are found to be an convenient tool to capture the characteristics of an image.
We have demonstrate in the previous chapter how CNNs are good at spotting the "features" in an image layer by layer.
Therefore, in the next two sub-sections, we will explain how it can be used to express the content and style feature of an image.

We have introduced several CNN architectures to perform image detection task in the previous chapter. We choose to use VGG19 since its follows a simple linear stack structure and is proved to have good performance.
We have built the VGG19 network structure in [this gist](https://gist.github.com/jzstark/da5cc7f771bc8d9699cedc491b23f856).
It contains 38 layers in total, and we prepared pre-trained weights for it.

### Content Reconstruction

From the image detection case, we know that the CNN extract features layer by layer until the features are so abstract that it can give an answer such as "this is a car" "this is an apple" etc.
Therefore, we can use the feature map to reconstruct content of an image.

But which layer's output should we use as a suitable indication of the image content?
Let's perform a simplified version of NST: we only care about re-constructing the content of the input image, so our target is to minimise:

$$f(x) = \verb|content_distance|(x, c)$$

As an example, we use [@fig:case-nst:content-example] as the target content.
(This image ["Tourists in Nanzen-Ji Hojo"](https://ccsearch.creativecommons.org/photos/f4024dc8-ce39-4e86-acfd-47532fef824d) by blieusong is licensed under CC BY-SA 2.0.)

![Example content image in neural style transfer](images/case-nst/hojo.png "hojo tourists"){width=40% #fig:case-nst:content-example}


Suppose we choose the output of `idx` layer as the chosen feature map to represent the content.
First, we need to compute the target feature map:

```
let fill_content_targets x net =
  let selected_topo = Array.sub nn.topo 0 (idx + 1) in
  run' selected_topo x
```

The function `fill_content_targets` takes the content image and the VGG network as input, and returns the target feature map as output.
We only need to compute the feature map of the target content image once.

Here the `run'` function is implemented by accumulating the inference result along the selected part of the network, from the network input until the chosen layer, instead of processing the whole network:

```
let run' topo x =
  let last_node_output = ref (F 0.) in
  Array.iteri (fun i n ->
    let input  = if i = 0 then x else !last_node_output in
    let output = run [|input|] n.neuron in
    last_node_output := output;
  ) topo;
  !last_node_output
```

Then we can start optimising the input image `x`.
Let's set the initial `x` to be a "white noise" image that only contains random pixels.
This image has the same shape as content image.

```text
let input_shape = Dense.Ndarray.S.shape content_img in
Dense.Ndarray.S.(gaussian input_shape |> scalar_mul 0.256)
```

The feature map of the input image `x` is still calculated using the same process show in function `fill_content_targets`.
We call the resulting feature map `response`, then the loss value can be calculated with the L2Norm of the difference between two feature maps, and then normalised with the feature map size.

```
let c_loss response target =
  let loss = Maths.((pow (response - target) (F 2.)) |> sum') in
  let _, h, w, feature = get_shape target in
  let c = float_of_int ( feature * h * w ) in
  Maths.(loss / (F c))
```

Once the loss value is calculated, we can apply optimisers.
Here we use the `minimise_fun` from `Optimise` module.
The target function can be described as:

```
let f x =
  fill_losses x;
  c_loss response target
```

All it performs is what we just described: first calculating the feature map `response` of input image at a certain layer, and then compute the distance between it and the target content feature map as loss value.

Finally, we can perform the optimisation iterations:

```
let state, img = Optimise.minimise_fun params g (Arr input_img) in
let x' = ref img in
while Checkpoint.(state.current_batch < state.batches) do
  Checkpoint.(state.stop <- false);
  let a, img = Optimise.minimise_fun ~state params g !x' in
  x' := img
done;
```

We keep updating the image `x` for fixed number of iterations.
Particularly, we use the Adam adaptive learning rate method, for it proves to be quite effective in style transfer optimisation:

```
let params = Params.config
    ~learning_rate:(Learning_Rate.Adam (learning_rate, 0.9, 0.999))
    ~checkpoint:(Checkpoint.Custom chkpt)
    iter
```

Using the process above, we return to the problem of choosing a suitable layer as the indication of the image content.
In this 38-layer VGG network, we choose these layers: 2, 7, 12, 21, 30.
Then we can compare the optimisation result to see the effect of image reconstruction.
Each one is generated using 100 iterations.

![Contents reconstruction](images/case-nst/contents-reconstruction.png "contents-reconstruction"){width=100% #fig:case-nst:contents-rec}

It is shown that, the content information is kept accurate at the lower level.
Along the processing hierarchy of the network, feature map produced by the lower layer cares more about the small features that at the pixel level, while the higher layer gives more abstract information but less details to help with content reconstruction.

### Style Recreation

Then similarly, we explore the other end of this problem. Now we only care about recreating an image with only the style of an input image.
That is to say, we optimise the input image with this target to minimise:

$$h(x) = \verb|style_distance|(x, s)$$

As an example, we will use the famous "The Great Wave of Kanagawa" by Hokusai as our target style image:

![Example style image in neural style transfer](images/case-nst/hokusai.png "hokusai"){width=40% #fig:case-nst:style-example}

The basic approach is the same as before: first compute the style representation of target image using the output from one or more layers, and then compute the style representation of the input image following the same method. The normalised distance between these two ndarrays are used as the optimisation target.

However, the difference is that, unlike the content representation, we cannot directly take one filter map from certain layer as the style representation.
Instead, we need to t computes the correlations between different filters from the output of a layer.
This correlation can be represented by the Gram matrix, which intuitively captures the "distribution of features" of feature maps from a certain layer.
The $(i,j)$-th element of an Gram matrix is computed by element-wisely multiplying the $i$-th and $j$-th channels in the feature maps and summing across both width and height.
This process can be simplified as a matrix multiplication. The result is normalised with the size of the feature map.
The code is shown below.

```
let gram x =
  let _, h, w, feature = get_shape x in
  let new_shape = [|h * w; feature|] in
  let ff = Maths.(reshape x new_shape) in
  let size = F (float_of_int (feature * h * w)) in
  Maths.((transpose ff) *@ ff / size)
```

Now that we have a method to represent the "style" of an image, we can proceed to calculate the loss value during optimisation.
It is very similar to that of content recreation, and the only difference is that we use the distance between Gram matrices instead of the feature map from a certain layer as loss value.

```
let s_loss response_gram target_gram =
  let loss = Maths.((pow (response_gram - target_gram) (F 2.)) |> sum') in
  let s = Algodiff.shape target_gram in
  let c = float_of_int (s.(0) * s.(1)) in
  Maths.(loss / (F c))
```

However, note that for the optimisation, instead of using output from one layer, we usually utilises the loss value from multiple layers and the optimisation target for style reconstruction:

```
let h x =
  fill_losses x;
  Array.fold_left Maths.(+) (F 0.) style_losses
```

Here the `fill_losses` function compute style losses at different layers, and store them into the `style_losses` array.
Then they are added up as the optimisation target.
The rest process is the as in the content reconstruction.

As example, we choose the similar five layers from the VGG19 network: layer 2, 7, 12, 21, and 30.
Then we computing the aggregated loss value fo the first layer, the first two layers, the first three layers, the first four layers, and all layers, as five different optimisation target.
The results are shown in [@fig:case-nst:style-rec].

![Style reconstruction](images/case-nst/style-reconstruction.png "style-reconstruction"){width=100% #fig:case-nst:style-rec}

As the result shows, features from the beginning tends to contain low level information such as pixels, so reconstructing styles according to them results in a fragmented white-noise-like representation, which really does not show any obvious style.
Only by adding more deep layer features can the style be gradually be reconstructed.

### Combining Content and Style

Now that we have seen these two extremes: only recreating content and only recreating style, it's straightforward to understand the theory of style transfer: to synthesised an image that has similar content with one image and style close to the other.
The code would be mostly similar to what we have seen, and the only difference now is simply adding the loss value of content and styles as the final optimisation target.

One thing we need to note during combining contents and style is the proportion of each part, and the choice of layers as representation.
This problem is actually more artistic than technique, so here we only follow the current practice about parameter configuration.
Please refer to the original paper about the effect of parameter tuning.

As suggested by previous experiment results, we use the feature maps from 23rd layer for content recreation, and combines the output of layer 2, 7, 12, 21, and 30 in VGG19 to represent the style feature of an image.
When combining the loss values, we multiply the style loss with a weight number, and then add it to the content loss.
Practice shows that a weigh number of 20 shows good performance.

You might also be wondering: why not choose the 2nd layer if it show the best content reconstruction result?
The intuition is that we don't want the synthesised image to be too close to the content image in content, because that would mean less style.
Therefore we use a layer from the middle of CNN which shows to keep most of the information for content reconstruction.

![Combining content and style reconstruction](images/case-nst/nst_example_01.png "nst_example_01"){width=90% #fig:case-nst:nst_example_01}

Combining all these factors together, [@fig:case-nst:nst_example_01] shows the result about running our code and creating an artistic view based on the original image.

All the code (about 180 lines) is included in [this Gist](https://gist.github.com/jzstark/6f28d54e69d1a19c1819f52c5b16c1a1).
The pre-trained weight file for VGG19 is also included.
As with the image detection applications, it also relies on `ImageMagick` to manipulate image format conversion and resizing.
We only list part of it above, and there are many implementation details such as garbage collection are omitted to focus on the theory of the application itself.
We therefore suggest you to play with the code itself with images or parameters of your choice.

### Running NST

To make the code above more suitable to use, this NST application provides a simple interfaces to use. Here is an example showing how to use it with two lines of code:

```
#zoo "6f28d54e69d1a19c1819f52c5b16c1a1"

Neural_transfer.run
  ~ckpt:50
  ~src:"path/to/content_img.jpg"
  ~style:"path/to/style_img.jpg"
  ~dst:"path/to/output_img.png" 250.;;
```

Similar to the image detection application, the command can be simplified using the Zoo system in owl.
The first line downloads gist files and imported this gist as an OCaml module, and the second line uses the `run` function to produce an output image to your designated path.
Its syntax is quite straightforward, and you may only need to note the final parameter. It specifies how many iterations the optimisation algorithm runs. Normally 100 ~ 500 iterations is good enough.

This module also supports saving the intermediate images to the same directory as output image every N iterations (e.g. `path/to/output_img_N.png`).
`N` is specified by the `ckpt` parameter, and its default value is 50 iterations.
If users are already happy with the intermediate results, they can terminate the program without waiting for the final output image.

That's all. Now you can try the code easily.
If you don't have suitable input images at hand, the gist already contains exemplar content and style images to get you started.
More examples can be seen on our online [demo](http://demo.ocaml.xyz/neuraltrans.html) page.

## Extending NST

The neural style transfer has since attracts a lot of attentions.
It is the core technology to many successful industrial applications, most notably photo rendering applications.
For example, the [Prisma Photo Editor](https://play.google.com/store/apps/details?id=com.neuralprisma&hl=en_GB) features transforming your photos into paintings of hundreds of styles.

There are also many research work that aim to extend this work.
One of these work is the *Deep Photo Style Transfer* proposed in [@luan2017deep].
The idea is simple: instead of using an art image, can I use another normal image as style reference?
For example, we have a normal daylight street view  in New York as content image, then we want to use the night view of London as reference, to synthesise an image of the night view of New York.

The authors identify two key challenges in this problem.
The first is that, unlike in NST, we hope to only  change to colours of the style image, and keep the content un-distorted, so as to create a "real" image as much as possible.
For this challenge, the authors propose to add an regularisation item to our existing optimisation target "content distance + style distance".
This item, depending on only input and outpu images, penalises image distortion and seeks an image transform that is locally affine in color space.
The second challenge is that, we don't want the styles to be applied globally.
For example, we only want to apply the style of an sunset sky to a blue sky, not a building.
For this problem, the authors propose to coarsely segment input images into several parts before apply style transfer separately.
If you are interested to check the original paper, the resulting photos are indeed beautifully and realistically rendered.

Another similar application is the "image-to-image translation".
This computer vision broadly involves translating an input image into certain output image.
The style transfer or image colourisation can be seen as examples of it.
There are also applications that change the lighting/weather in a photo. These can also be counted as examples of image to image translation.

In [@isola2017image] the authors propose to use the Generative Adversarial Networks (GANs) to provide general framework for this task.
In GAN, there are two important component: the generator, and the discriminator.
During training, the generator synthesises images based on existing parameters, and the discriminator tries its best to separate the generated data and true data.
This process is iterated until the discriminator can no longer tell the difference between the generated data and the true data.
[@isola2017image] utilises convolution neural network to construct the generator and discriminator in the GAN.
This approach is successfully applied in mnany applications, such as [Pix2Pix](https://phillipi.github.io/pix2pix/), [face ageing](https://ieeexplore.ieee.org/document/8296650), [increase photo resolution](https://arxiv.org/abs/1609.04802), etc.

Another variant is called the Fast Style Transfer.
Instead of iteratively updating image, it proposes to use one pre-trained feed-forward network to do style transfer, and therefore improving the speed of rendering by orders of magnitude.
That's what we will be talking about in the rest of this chapter.

## Fast Style Transfer

One disadvantage of NST is that it could take a very long time to rendering an image, and if you want to change to another content or style image, then you have to wait a long time for the training again.
If you want to render some of your best (or worst) selfies fast and send to your friends, NST is perhaps not a perfect choice.  

This problem then leads to another application: Fast Neural Style Transfer (FST). FST sacrifice certain degrees of flexibility, which is that you cannot choose style images at will. But as a result, you only need to feed your content image to a DNN, finish an inference pass, and then the output will be the rendered styled image as you expected.
The best part is that, one inference pass is much much faster that keep running a training phase.

### Building FST Network

The Fast Style Transfer network is proposed in [@Johnson2016Perceptual].
The authors propose to build and train an *image transformation network*.
Image transformation is not a totally new idea. It takes some input image and transforms it into a certain output image.
One way to do that is to train a feed-forward CNN.
This method is applied in different applications such as colourising grayscale photo or image segmentation.
In this work the author use the similar approach to solve the style transfer problem.

![System overview of the image transformation network and its training.](images/case-nst/fst.png "fst"){width=100% #fig:case-nst:fst}

[@fig:case-nst:fst] shows a system overview of the image transformation network and its training.
It can be divided into two part.
The first part includes the image transformation network architecture.
To synthesise an image of the same size as input image, it first uses down-sampling layers, and then then up-sampling layers.
One benefit of first down-sampling images is to reduce the computation, which enables building a deeper network.
We have already seen this design principle in the image detection case chapter.

Instead of using the normal pooling or upsampling layer in CNN, here the convolution layers are used for down/up-sampling.
We want to keep the image information as much as possible during the whole transformation process.
Specifically, we use the transpose convolution for the upsampling. It goes the opposite direction of a normal convolution, from small feature size to larger one, and still maintains the connectivity pattern in convolution.

```ocaml env=case-nst:fst
open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
module N = Dense.Ndarray.S

let conv2d_layer ?(relu=true) kernel stride nn  =
  let result =
    conv2d ~padding:SAME kernel stride nn
    |> normalisation ~decay:0. ~training:true ~axis:3
  in
  match relu with
  | true -> (result |> activation Activation.Relu)
  | _    -> result

let conv2d_trans_layer kernel stride nn =
  transpose_conv2d ~padding:SAME kernel stride nn
  |> normalisation ~decay:0. ~training:true ~axis:3
  |> activation Activation.Relu
```

Here, combined with batch normalisation and Relu activation layers, we build two building block `conv2d_layer` and the `conv2d_trans_layer`. Think of them as enhanced convolution and transpose convolution layers.
The benefit of adding these two types of layers is discussed in previous chapter.

Connecting these two parts are multiple residual blocks, which is proposed in the ResNet architecture.
The authors claim that the using residual connections makes it easier to keep the structure between output and input.
It is an especially attractive property for an style transfer neural network.
Specifically, the authors use the residual structure proposed [here](http://torch.ch/blog/2016/02/04/resnets.html).
All the convolution layers use the common 3x3 kernel size.  
This residual block can be implemented with the `conv2d_layer` unit we have built.

```ocaml env=case-nst:fst

let residual_block wh nn =
  let tmp = conv2d_layer [|wh; wh; 128; 128|] [|1;1|] nn
    |> conv2d_layer ~relu:false [|wh; wh; 128; 128|] [|1;1|]
  in
  add [|nn; tmp|]
```

Here in the code the `wh` normally takes a value of 3.
The residual block, as with in the ResNet, is repeatedly stacked for several times.
With these three different part rea
Finally, we can piece them together. Note how the output channel of each convolution increase, stay the same, and then decrease symmetrically.
Before the final output, we use the `tanh` activation layer to ensure all the values are between `[0, 255]` for the output image.

```ocaml env=case-nst:fst
let make_network h w =
  input [|h;w;3|]
  |> conv2d_layer [|9;9;3;32|] [|1;1|]
  |> conv2d_layer [|3;3;32;64|] [|2;2|]
  |> conv2d_layer [|3;3;64;128|] [|2;2|]
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> residual_block 3
  |> conv2d_trans_layer [|3;3;128;64|] [|2;2|]
  |> conv2d_trans_layer [|3;3;64;32|] [|2;2|]
  |> conv2d_layer ~relu:false [|9;9;32;3|] [|1;1|]
  |> lambda (fun x -> Maths.((tanh x) * (F 150.) + (F 127.5)))
  |> get_network
```

After constructing the image transformation network, let's look at the training process.
In previous work, when training a image transformation network, normally the output will be compared with the ground-truth image pixel-wisely as the loss value.
That is not an ideal approach here since we cannot know what is a "correct" style-transferred image in advanced.
Instead, the authors are inspired by the NST work.
They use the same training process with a pre-trained VGG19 network to compute the loss (they call it the *perceptual loss* against the per-pixel loss, since high level perceptual information is contained in this loss).

Therefore, we should be familiar with the training process now.
The output image $x$ from image transformation network is the image to be optimised, the input image itself is content image, and we provide another fixed style image.
We can then proceed to calculated the final loss by computing the distance between image $x$ with the input with regard to content and the distance with the style images measured by gram matrix from multiple layers.
All of these are the same as in the NST.
The only difference is that, where we train for an image before, now we train the weight for image transformation network during back-propagation.
Note that this process means that we can only train one set of weight for only one style.
Considering that the artistic styles are relatively fixed compared to the unlimited number fo content image, and the orders of magnitude of computation speed improved, fixing the styles is an acceptable tradeoff.

Even better, this training phase is one-off. We can train the network for once and the reuse it inthe inference phase again and again.
We refer you to the original paper if you want to know more detail about the training phase.
In our implementation, we directly convert and import weights from a [TensorFlow implementation](https://github.com/lengstrom/fast-style-transfer).
Next we will show how to use it to perform fast style transfer.

### Running FST

Like NST and image classification, we have wrapped all things up in a [Gist](https://gist.github.com/jzstark/f937ce439c8adcaea23d42753f487299), and provide a simple user interface to users.
Here is an example:

```
#zoo "f937ce439c8adcaea23d42753f487299"

FST.list_styles ();; (* show all supported styles *)
FST.run ~style:1 "path/to/content_img.png" "path/to/output_img.jpg"
```

The `run` function mainly takes one content image and output to a new image file, the name of which is designated by the user. The image could be of any popular formats: jpeg, png, etc. This gist contains exemplar content images for you to use.

Note that we did say "given suitable weights". A set of trained weight for the FST DNN represents a unique artistic style. We have already include six different weight files for use, and the users just need to pick one of them and load them into the DNN, without worrying about how to train these weights.

Current we support six art styles:
"[Udnie](https://bit.ly/2nBW0ae)" by Francis Picabia,
"[The Great Wave off Kanagawa](https://bit.ly/2nKk8Hl)" by Hokusai,
"[Rain Princess](https://bit.ly/2KA7FAY)" by Leonid Afremov,
"[La Muse](https://bit.ly/2rS1fWQ)" by Picasso,
"[The Scream](https://bit.ly/1CvJz5d)" by Edvard Munch, and
"[The shipwreck of the Minotaur](https://bit.ly/2wVfizH)" by J. M. W. Turner.
These style images are shown in [@fig:case-nst:fst-styles].

![Artistic Styles used in fast style transfer](images/case-nst/fst-styles.png "fst-styles"){width=85% #fig:case-nst:fst-styles}

Maybe six styles are not enough for you, but think about it, you can now render any of your image to a nice art style fast, maybe about half a minute, or even faster if you are using GPU or other accelerators.
As an example, we use the Willis Tower in Chicago as an input image:

![Example input image: Willis tower of Chicago](images/case-nst/chicago.png){width=40% #fig:case-nst:chicago}

![Fast style transfer examples](images/case-nst/example_fst00.png){width=85% #fig:case-nst:fst-example}

We then apply FST on this input image with the styles shown above. The rendered city view with different styles are shown in [@fig:case-nst:fst-example].

Moreover, based these code, we have built a [demo](http://demo.ocaml.xyz/fst.html) website for the FST application.
You can choose a style, upload an image, get yourself a cup of coffee, and then checkout the rendered image.
To push things even further, we apply FST to some videos frame-by-frame, and put them together to get some artistic videos, as shown in this [Youtube list](https://www.youtube.com/watch?v=cFOM-JnyJv4&list=PLGt9zVony2zVSiHZb8kwwXfcmCuOH2W-H).
You are welcome to try these services with images of your own.

## Summary

## References
