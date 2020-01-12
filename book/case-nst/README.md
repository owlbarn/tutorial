# Case - Neural Style Transfer

## Neural Style Transfer

What is Neural Style Transfer (NST)? It is a pretty cool application of Deep Neural Networks (DNN), “the process of using DNN to migrate the semantic content of one image to different styles”.

Well it may sounds a little bit scary, but the idea is very simple, as the title image shows, this application takes two images A and B as input. Let’s say A is "Mona Lisa" of Da Vinci, and B is "The Starry Night" of Vincent van Gogh.

We then specify A as the content image and B as the style image, then what a NST application can produce? Boom! A new Mona Lisa, but with the style of Van Gogh (see the middle of title image)! If you want another style, just replace image B and run the application again. Impressionism, abstractionism, classical art, you name it. 

The figure below illustrate this point ([src](http://genekogan.com/works/style-transfer/)). You can apply different art styles to the same "Mona Lisa", or apply the same "Starry Sky" style to any pictures, even a normal daily street view.
Isn’t it amazing? 

![](images/case-nst/mona_lisa.jpeg)

## A Very Brief Theory of NST

Without going into details, I will briefly introduce the math behind NST, so please feel free to ignore this part. Refer to the [original paper](https://arxiv.org/abs/1508.06576) for more details if you are interested.

The NST can be seen as an optimisation problem: given a content image `c` and a style image `s` , the target is to get an output image `x` so that it minimises:
$$ f(x) = \textrm{content_distance}(x, c) + \textrm{style_distance}(x, s) $$

This equation can be easily translated as: I want to get such an image that its content is close to `c` , but its style similar to `s` .

DNNs, especially the ones that are used for computer vision tasks, are found to be an convenient tool to capture the content and style characteristics of an image (details emitted here for now).
Then the euclidean distance of these characteristics are used to express the `content_distance()` and `style_distance()` functions.
Finally, the optimisation techniques such as gradient descent are applied to f(x) to get a good enough x .

## NST with Owl

I’ve implement an NST application with Owl. All the code (about 180 lines) is included in [this Gist](https://gist.github.com/jzstark/6f28d54e69d1a19c1819f52c5b16c1a1). This application uses the VGG19 network structure to capture the content and style characteristics of images. The pre-trained network file is also included.
It relies on `ImageMagick` to manipulate image format conversion and resizing. Please make sure it is installed before running.

This application provides a simple interfaces to use. Here is an example showing how to use it with two lines of code:

```
#zoo "6f28d54e69d1a19c1819f52c5b16c1a1"

Neural_transfer.run ~ckpt:50 ~src:"path/to/content_img.jpg" ~style:"path/to/style_img.jpg" ~dst:"path/to/output_img.png" 250.;;
```

The first line download gist files and imported this gist as an OCaml module, and the second line uses the `run` function to produce an output image to your designated path. It’s syntax is quite straightforward, and you may only need to note the final parameter. It specifies how many iterations the optimisation algorithm runs. Normally 100 ~ 500 iterations is good enough.

This module also supports saving the intermediate images to the same directory as output image every N iterations (e.g. `path/to/output_img_N.png`). `N` is specified by the `ckpt` parameter, and its default value is 50 iterations. If users are already happy with the intermediate results, they can terminate the program without waiting for the final output image.

That’s all it takes! If you don't have suitable input images at hand, the gist already contains exemplar content and style images to get you started. 
I have to say I had a lot lot of fun playing with it -- please allow me to introduce you one of my work using the exemplar images:

![](images/case-nst/nst_example.png)

Here is a presentation of how the content image change gradually in style:

<p align="center">
  <img src="images/case-nst/example_01.gif">
</p>

## Fast Style Transfer
