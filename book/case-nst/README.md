# Case - Neural Style Transfer

What is Neural Style Transfer (NST)? It is a pretty cool application of Deep Neural Networks (DNN),  "the process of using DNN to migrate the semantic content of one image to different styles".

The process is actually very simple, as the title image shows, this application takes two images A and B as input. Let’s say A is "Mona Lisa" of Da Vinci, and B is "The Starry Night" of Vincent van Gogh.

We then specify A as the content image and B as the style image, then what a NST application can produce? Boom! A new Mona Lisa, but with the style of Van Gogh (see the middle of title image)! If you want another style, just replace image B and run the application again. Impressionism, abstractionism, classical art, you name it. 

The figure below illustrate this point. You can apply different art styles to the same street view, or apply the same "Starry Sky" style to any pictures.
Isn’t it amazing? 

![Example of applying neural style transfer on a street view picture](images/case-nst/nst_example.png){#fig:case-nst:example_01}

## Content and Style Reconstruction

[@gatys2015neural] first propose to use DNN to let programmes to create artistic images of high perceptual quality.
The examples above may look like magic, but surely its not.
In this section, we will first introduce the intuition about how the neural style transfer algorithm works. 
For more formal and detailed introduction, please visit the [original paper](https://arxiv.org/abs/1508.06576).

The basic idea is plain: we want to get an image whose content is similar to one image and its artistic style close to the other image. 
Of course, to make the algorithm work, first we need to express this sentence in mathematical form so that computers can understand it. 
Let's assume for a moment we have already know that, then style transfer can be formalised as an optimisation problem.
Given a content image `c` and a style image `s`, our target is to get an output image `x` so that it minimises:

$$f(x) = \verb|content_distance|(x, c) + \verb|style_distance|(x, s)$$

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
It contains 38 layers in total.

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

The feature map of the input image `x` is still calcuated using the same process show in function `fill_content_targets`.
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
let g x = 
  let response = fill_losses x in 
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

IMAGE: a 1x5 images 

It is shown that, the content information is kept accurate at the lower level. 
Along the processing hierarchy of the network, feature map produced by the lower layer cares more about the small features that at the pixel level, while the higher layer gives more abstract information but less details to help with content reconstruction.

### Style Recreation

Then similarly, we explore the other end of this problem: we only care about recreating an image with only the style of an input image:

EQUATION: minimise style(I_style) 

We use this image as input style target:

![Example style image in neural style transfer](images/case-nst/hokusai.png "hokusai"){width=50% #fig:case-nst:style-example}

Expressing the style is bit more complex that content, which directly uses the feature map it self.

Some theory: why Gram etc.

We try different combinations of features. 1, 1 + 2, ...., 1 + 2 + 3 + 4 + 5.

Part of the CODE.

The result is shown below:

IMAGE: 1x5 

You can see that, contrary to content, going deeper in CNN gives more information about feature. 


## Building a NST Network

Now that we have seen these two extremes, it's straightforward to understand the theory of style transfer. 

control the proportion with weights. 

The result: 

IMAGE: 1x5, from white noise, by different steps. 

we simply this process...


The details: Loss function, pre-trained weight, optimiser, etc.

I’ve implement an NST application with Owl. All the code (about 180 lines) is included in [this Gist](https://gist.github.com/jzstark/6f28d54e69d1a19c1819f52c5b16c1a1). This application uses the VGG19 network structure to capture the content and style characteristics of images. The pre-trained network file is also included.
It relies on `ImageMagick` to manipulate image format conversion and resizing. Please make sure it is installed before running.

## Running NST

This application provides a simple interfaces to use. Here is an example showing how to use it with two lines of code:

```
#zoo "6f28d54e69d1a19c1819f52c5b16c1a1"

Neural_transfer.run 
  ~ckpt:50 
  ~src:"path/to/content_img.jpg" 
  ~style:"path/to/style_img.jpg" 
  ~dst:"path/to/output_img.png" 250.;;
```

The first line download gist files and imported this gist as an OCaml module, and the second line uses the `run` function to produce an output image to your designated path. It’s syntax is quite straightforward, and you may only need to note the final parameter. It specifies how many iterations the optimisation algorithm runs. Normally 100 ~ 500 iterations is good enough.

This module also supports saving the intermediate images to the same directory as output image every N iterations (e.g. `path/to/output_img_N.png`). `N` is specified by the `ckpt` parameter, and its default value is 50 iterations. If users are already happy with the intermediate results, they can terminate the program without waiting for the final output image.

That’s all it takes! If you don't have suitable input images at hand, the gist already contains exemplar content and style images to get you started. 



More examples can be seen on our [demo](http://demo.ocaml.xyz/neuraltrans.html) page.

## Extending NST

Many variants. Most notably: 

- Deep Photo Style Transfer [@luan2017deep] 
- Image-to-Image Translation [@zhu2017unpaired]

Industry applications: 

One of the variants is the Fast Style Transfer. Suitable for fast rendering with fixed style. We will introduce it next.

## Fast Style Transfer

Paper: [@Johnson2016Perceptual]

One disadvantage of NST is that it could take a very long time to rendering an image, and if you want to change to another content or style image, then you have to wait a long time for the training again. 
If you want to render some of your best (or worst) selfies fast and send to your friends, NST is perhaps not a perfect choice.  

This problem then leads to another application: Fast Neural Style Transfer (FST). FST sacrifice certain degrees of flexibility, which is that you cannot choose style images at will. But as a result, you only need to feed your content image to a DNN, finish an inference pass, and then the output will be the rendered styled image as you expected! The best part is that, one inference pass is much much faster that keep running a training phase. 

### Theory

### Building FST Network

Based on the [TensorFlow implementation](https://github.com/lengstrom/fast-style-transfer), we have implemented a FST application in Owl, and it's not complicated. Here is the network structure:

```ocaml
open Owl
open Neural.S
open Neural.S.Graph
open Neural.S.Algodiff
module N = Dense.Ndarray.S

(** Network Structure *)

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

let residual_block wh nn = 
  let tmp = conv2d_layer [|wh; wh; 128; 128|] [|1;1|] nn
    |> conv2d_layer ~relu:false [|wh; wh; 128; 128|] [|1;1|]
  in 
  add [|nn; tmp|]

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

### Running FST

That's it. Given suitable weights, running an inference pass on this DNN is all it takes to get a styled image.
Like NST, we have wrapped all things up in a [Gist](https://gist.github.com/jzstark/f937ce439c8adcaea23d42753f487299), and provide a simple user interface to users. 
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
"[The shipwreck of the Minotaur](https://bit.ly/2wVfizH)" by J. M. W. Turner


Yes, maybe six styles are not enough for you, but think about it, you can now render any of your image to a nice art style fast, maybe about half a minute, or even faster if you are using GPU or other accelerators. Here is a teaser that renders one city view image to all these amazing art styles. 

![](images/case-nst/example_fst00.png){#fig:case-obj-detect:example_03}

If you are still not persuaded, here is our ultimate solution for you: a [demo] website, where you can choose a style, upload an image, get yourself a cup of coffee, and then checkout the rendered image. 
To push things even further, we apply FST to some videos frame-by-frame, and put them together to get some artistic videos, as shown in this [Youtube list](https://www.youtube.com/watch?v=cFOM-JnyJv4&list=PLGt9zVony2zVSiHZb8kwwXfcmCuOH2W-H).
And all of these are implemented in Owl.  
