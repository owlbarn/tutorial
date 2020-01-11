# Case - Image Recognition


## Image Recognition

How can a computer take an image and answer questions like “what is in this picture? A cat, dog, or something else?” 
In the last few years the field of machine learning has made tremendous progress on addressing this difficult problem. In particular, Deep Neural Network (DNN) can achieve reasonable performance on visual recognition tasks — matching or exceeding human performance in some domains.

[InceptionV3](https://arxiv.org/abs/1512.00567) is one of Google’s latest effort to do image recognition. It is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012. This is a standard task in computer vision, where models try to classify entire images into 1000 classes, like “Zebra”, “Dalmatian”, and “Dishwasher”. Compared with previous DNN models, InceptionV3 has one of the most complex networks architectures in computer vision.

## Image Recognition with Owl

There exist many good deep learning frameworks that can be used to do image classification, such as TensorFlow, Caffe, Torch, etc. But what if your choice of language is Functional Programming Language such as OCaml? It has long been thought that OCaml is not suitable for advanced computation tasks like machine learning. And now we have Owl.

As a prerequisite, please make sure that the tool [ImageMagick](https://www.imagemagick.org/) is installed.
Besides, prepare one image on your computer. It can be of any common image format (jpg, png, gif, etc.) and size. If you’re not sure which image to use, here is one choice we use in the rest of this chapter: 

<p align="center">
  <img src="images/case-image-inception/panda.png">
</p>


## Let's Roll!

Enough of these boring installation steps. Forget any hello-world code. Let’s do the image classification, right here, right now!

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

## Code Detail

This one line of code uses the Zoo system to import the code from [this gist](https://gist.github.com/jzstark/6dfed11c521fb2cd286f2519fb88d3bf). Let's look at the the code in detail:

```
#!/usr/bin/env owl
open Owl

(* Import InceptionV3 Library *)
#zoo "9428a62a31dbea75511882ab8218076f"

let _ = 
  (* Path to your image; here we use the "panda.png" in this gist as example. *)
  let img = Owl_zoo_path.extend_zoo_path "panda.png" in
  (* Image classification *)
  let labels = InceptionV3.infer img in
  (* Get top-5 human-readable output in the format of JSON string, or...*) 
  let top = 5 in 
  let labels_json   = InceptionV3.to_json ~top labels in
  (* an array of tuples. Each tuple contains a category (string) and 
   * its inferred probability (float), ranging from 1 to 100.
   *)
  let labels_tuples = InceptionV3.to_tuples labels in

  (* (Optional) Pretty-print the results *)
  Printf.printf "\nTop %d Predictions:\n" top;
  Array.iteri (fun i x -> 
    let cls, prop = x in 
    Printf.printf "Prediction #%d (%.2f%%) : %s\n" i (prop *. 100.) cls;
  ) labels_tuples
```

You need 5 steps to do image classification with InceptionV3:

1. Import external code/libraries using Zoo in Owl. Using `#zoo "gist-id"` enables you to use code modules defined in other Gists. Here we want to use [InceptionV3](https://gist.github.com/jzstark/9428a62a31dbea75511882ab8218076f) modules. It defines the InceptionV3 network architecture and loads weights of the network. The downloaded code are cached in `$HOME/.owl/zoo` directory.
The InceptionV3 module provides three APIs:

    - `infer`: Service that performs image recognition tasks over client images. It accept a string that specify the location of a local image. Its return value is a 1x1000 N-dimension array, each element is a float number between 0 and 1, indicating the possibility that the image belongs to one of the 1000 classes from ImageNet.

    - `to_json`: Convert the inferred result to a raw JSON string. Parameter `top`: an int value to specify the top-N likeliest labels to return. Default value is 5.

    - `to_json`: Convert the inferred result to an array of tuples, each tuple contains label name (“class”, string) and the probability (“prop”, float, between 0 and 1) of target image being in that class.

2. Load InceptionV3 model with one line of code.

3. Designate an absolute path of your input image. Here we use the `extend_zoo_path` util function to automatically find the "panda.png" image contained in the Gist itself.

4. Run inference with the neural network model and the input image, and then decode the result, getting top-N (N defaults to 5) predictions in human-readable format. The output is an array of tuple, each tuple consists of a string for classified type description, and a float number ranging from 0 to 100 to represent the percentage probability of the input image actually being this type.

5. If you want, you can pretty-print the result on your screen.


## Online Demo

If you are not interested in installing anything, no problem! [Here](http://demo.ocaml.xyz/) is a web-based demo of this image classification application powered by Owl. Please feel free to play with it! And the server won’t store your image. Actually, if you are so keen to protect your personal data privacy, then you definitely should try to pull the code here and fast build a local image processing service without worrying your images being seen by anybody else!

## Want to Know More?

We do have more! I suggest you to read the code that constructing the whole InceptionV3 network from [this gist](https://gist.github.com/jzstark/9428a62a31dbea75511882ab8218076f). Even if you are not quite familiar with Owl or OCaml, it must still be quite surprising to see the network that contains 313 neuron nodes can be constructed using only about 150 lines of code. And we are talking about one of the most complex neural networks for computer vision. 

Besides InceptionV3, you can also easily construct other popular image recognition networks, such as [ResNet50](https://gist.github.com/pvdhove/a05bf0dbe62361b9c2aff89d26d09ba1), [VGG16](https://gist.github.com/jzstark/f5409c44d6444921a8ceec00e33c42c4), [SqueezeNet](https://gist.github.com/jzstark/c424e1d1454d58cfb9b0284ba1925a48) etc. with elegant Owl code.  
As to other smaller tasks, such as the most common hand-written digits recognition task, you can construct a good deep neural network model with only 9 lines of code!

Happy hunting!
