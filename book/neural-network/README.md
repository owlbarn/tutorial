# Neural Networks

**NOTE: many places need fixes, not finished yet.**

I will cover the neural network module in this chapter. My original purpose of introducing neural network module into Owl is two-fold:

* Test the expressiveness of Owl. Neural network is a useful and complex tool for building modern analytical applications so I chose it.

* To validate my research argument on how to structure modern (distributed) analytical libraries. Namely, the high-level analytical functionality (ML, DNN, optimisation, regression, and etc.) should be "glued" to the classic numerical functions via algorithmic differentiation, and the computation should be distributed via a specialised engine providing several well-defined distribution abstractions.

In the end, I only used less than 3k lines of code to implement a quite full-featured neural network module. Now let's go through what `Neural` module offers.


## A Naive Example

```ocaml env=neural_01
open Algodiff.S

type layer = {
  mutable w : t;
  mutable b : t;
  mutable a : t -> t;
}

type network = { layers : layer array }
```


```ocaml env=neural_01
let run_layer x l = Maths.((x *@ l.w) + l.b) |> l.a

let run_network x nn = Array.fold_left run_layer x nn.layers
```


```ocaml env=neural_01
let l0 = {
  w = Maths.(Mat.uniform 784 300 * F 0.15 - F 0.075);
  b = Mat.zeros 1 300;
  a = Maths.tanh;
}

let l1 = {
  w = Maths.(Mat.uniform 300 10 * F 0.15 - F 0.075);
  b = Mat.zeros 1 10;
  a = Maths.softmax ~axis:1;
}

let nn = {layers = [|l0; l1|]}
```


```ocaml env=neural_01
let backprop nn eta x y =
  let t = tag () in
  Array.iter (fun l ->
    l.w <- make_reverse l.w t;
    l.b <- make_reverse l.b t;
  ) nn.layers;
  let loss = Maths.(cross_entropy y (run_network x nn) / (F (Mat.row_num y |> float_of_int))) in
  reverse_prop (F 1.) loss;
  Array.iter (fun l ->
    l.w <- Maths.((primal l.w) - (eta * (adjval l.w))) |> primal;
    l.b <- Maths.((primal l.b) - (eta * (adjval l.b))) |> primal;
  ) nn.layers;
  loss |> unpack_flt
```


```ocaml env=neural_01
let test nn x y =
  Dense.Matrix.S.iter2_rows (fun u v ->
    Dataset.print_mnist_image u;
    let p = run_network (Arr u) nn |> unpack_arr in
    Dense.Matrix.Generic.print p;
    Printf.printf "prediction: %i\n" (let _, i = Dense.Matrix.Generic.max_i p in i.(1))
  ) (unpack_arr x) (unpack_arr y)
```


```ocaml env=neural_01
let _ =
  let x, _, y = Dataset.load_mnist_train_data () in
  for i = 1 to 9 do
    let x', y' = Dataset.draw_samples x y 100 in
    backprop nn (F 0.01) (Arr x') (Arr y')
    |> Owl_log.info "#%03i : loss = %g" i
  done;
  let x, y, _ = Dataset.load_mnist_test_data () in
  let x, y = Dataset.draw_samples x y 10 in
  test nn (Arr x) (Arr y)
```


## Module Structure

The [Owl.Neural](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural.ml) provides two submodules `S` and `D` for both single precision and double precision neural networks. In each submodule, it contains the following modules to allow you to work with the structure of the network and fine-tune the training.

* `Graph` : create and manipulate the neural network structure.
* `Init` : control the initialisation of the weights in the network.
* `Activation` : provide a set of frequently used activation functions.
* `Params` : maintains a set of training parameters.
* `Batch` : the batch parameter of training.
* `Learning_Rate` : the learning rate parameter of training.
* `Loss` : the loss function parameter of training.
* `Gradient` : the gradient method parameter of training.
* `Momentum` : the momentum parameter of training.
* `Regularisation` : the regularisation parameter of training.
* `Clipping` : the gradient clipping parameter of training.
* `Checkpoint` : the checkpoint parameter of training.
* `Parallel` : provide parallel computation capability, need to compose with Actor engine. (Experimental, a research project in progress.)


## Types of Neuron

I have implemented a set of commonly used neurons in [Owl.Neural.Neuron](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_neuron.ml). Each neuron is a standalone module and adding a new type of neuron is much easier than adding a new one in Tensorflow or other framework thanks to Owl's [Algodiff](https://github.com/ryanrhymes/owl/blob/master/lib/owl_algodiff_generic.mli) module.

`Algodiff` is the most powerful part of Owl and offers great benefits to the modules built atop of it. In neural network case, we only need to describe the logic of the forward pass without worrying about the backward propagation at all, because the `Algodiff` figures it out automatically for us thus reduces the potential errors. This explains why a full-featured neural network module only requires less than 3.5k lines of code. Actually, if you are really interested, you can have a look at Owl's [Feedforward Network](https://github.com/ryanrhymes/owl/blob/master/examples/feedforward.ml) which only uses a couple of hundreds lines of code to implement a complete Feedforward network.

In practice, you do not need to use the modules defined in  [Owl.Neural.Neuron](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_neuron.ml) directly. Instead, you should call the functions in [Graph](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_graph.ml) module to create a new neuron and add it to the network. Currently, Graph module contains the following neurons.

* `input`
* `activation`
* `linear`
* `linear_nobias`
* `embedding`
* `recurrent`
* `lstm`
* `gru`
* `conv1d`
* `conv2d`
* `conv3d`
* `max_pool1d`
* `max_pool2d`
* `avg_pool1d`
* `avg_pool2d`
* `global_max_pool1d`
* `global_max_pool2d`
* `global_avg_pool1d`
* `global_avg_pool2d`
* `fully_connected`
* `dropout`
* `gaussian_noise`
* `gaussian_dropout`
* `alpha_dropout`
* `normalisation`
* `reshape`
* `flatten`
* `lambda`
* `add`
* `mul`
* `dot`
* `max`
* `average`
* `concatenate`

These neurons should be sufficient for creating from simple MLP to the most complicated Google's Inception network.


## Model Training

Owl provides a very functional way to construct a neural network. You only need to provide the shape of the date in the first node (often `input` neuron), then Owl will automatically infer the shape for you in the downstream nodes which saves us a lot of efforts and significantly reduces the potential bugs.

Let's use the single precision neural network as an example. To work with single precision networks, you need to use/open the following modules

```ocaml env=neural_00

  open Owl
  open Neural.S
  open Neural.S.Graph
  open Neural.S.Algodiff

```

The code below creates a small convolutional neural network of six layers. Usually, the network definition always starts with `input` neuron and ends with `get_network` function which finalises and returns the constructed network. We can also see the input shape is reserved as a passed in parameter so the shape of the data and the parameters will be inferred later whenever the `input_shape` is determined.

```ocaml env=neural_00

  let make_network input_shape =
    input input_shape
    |> lambda (fun x -> Maths.(x / F 256.))
    |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
    |> max_pool2d [|2;2|] [|2;2|]
    |> dropout 0.1
    |> fully_connected 1024 ~act_typ:Activation.Relu
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network

```

Next, I will show you how the `train` function looks like. The first three lines in the `train` function is for loading the `MNIST` dataset and print out the network structure on the terminal. The rest lines defines a `params` which contains the training parameters such as batch size, learning rate, number of epochs to run. In the end, we call `Graph.train` to kick off the training process.

```ocaml env=neural_00

  let train () =
    let x, _, y = Dataset.load_mnist_train_data_arr () in
    let network = make_network [|28;28;1|] in
    Graph.print network;

    let params = Params.config
      ~batch:(Batch.Mini 100) ~learning_rate:(Learning_Rate.Adagrad 0.005) 2.
    in
    Graph.train ~params network x y |> ignore

```

After the training is finished, you can call `Graph.model` to generate a functional model to perform inference. Moreover, `Graph` module also provides functions such as `save`, `load`, `print`, `to_string` and so on to help you in manipulating the neural network.

```ocaml env=neural_00

  let predict network data =
    let model = Graph.model network in
    let predication = model data in
    predication

```

You can have a look at Owl's [MNIST CNN example](https://github.com/ryanrhymes/owl/blob/master/examples/mnist_cnn.ml) for more details and run the code by yourself.


## Model Inference

TBD


## Examples

In the following, I will present several neural networks defined in Owl. All have been included in Owl's [examples](https://github.com/ryanrhymes/owl/tree/master/examples) and can be run separately. If you are interested in the computation graph Owl generated for these networks, you can also have a look at [this chapter on Algodiff](algodiff.html).


### Multilayer Perceptron (MLP) for MNIST

```ocaml env=neural_00

  let make_network input_shape =
    input input_shape
    |> linear 300 ~act_typ:Activation.Tanh
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network

```

### Convolutional Neural Network for MNIST

```ocaml env=neural_00

  let make_network input_shape =
    input input_shape
    |> lambda (fun x -> Maths.(x / F 256.))
    |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
    |> max_pool2d [|2;2|] [|2;2|]
    |> dropout 0.1
    |> fully_connected 1024 ~act_typ:Activation.Relu
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network

```


### VGG-like Neural Network for CIFAR10

```ocaml env=neural_00

  let make_network input_shape =
    input input_shape
    |> normalisation ~decay:0.9
    |> conv2d [|3;3;3;32|] [|1;1|] ~act_typ:Activation.Relu
    |> conv2d [|3;3;32;32|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    |> dropout 0.1
    |> conv2d [|3;3;32;64|] [|1;1|] ~act_typ:Activation.Relu
    |> conv2d [|3;3;64;64|] [|1;1|] ~act_typ:Activation.Relu ~padding:VALID
    |> max_pool2d [|2;2|] [|2;2|] ~padding:VALID
    |> dropout 0.1
    |> fully_connected 512 ~act_typ:Activation.Relu
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network

```


### LSTM Network for Text Generation

```ocaml env=neural_00

  let make_network wndsz vocabsz =
    input [|wndsz|]
    |> embedding vocabsz 40
    |> lstm 128
    |> linear 512 ~act_typ:Activation.Relu
    |> linear vocabsz ~act_typ:Activation.(Softmax 1)
    |> get_network

```


### Google's Inception for Image Classification

```ocaml env=neural_00

  let conv2d_bn ?(padding=SAME) kernel stride nn =
    conv2d ~padding kernel stride nn
    |> normalisation ~training:false ~axis:3
    |> activation Activation.Relu

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

  let mix_typ3 nn =
    let branch3x3 = conv2d_bn [|3;3;288;384|] [|2;2|] ~padding:VALID nn in
    let branch3x3dbl = nn
      |> conv2d_bn [|1;1;288;64|] [|1;1|]
      |> conv2d_bn [|3;3;64;96|] [|1;1|]
      |> conv2d_bn [|3;3;96;96|] [|2;2|] ~padding:VALID
    in
    let branch_pool = max_pool2d [|3;3|] [|2;2|] ~padding:VALID nn in
    concatenate 3 [|branch3x3; branch3x3dbl; branch_pool|]

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
      |> avg_pool2d [|3;3|] [|1;1|] (* padding = SAME *)
      |> conv2d_bn [|1;1; 768; 192|] [|1;1|]
    in
    concatenate 3 [|branch1x1; branch7x7; branch7x7dbl; branch_pool|]

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

  let _ = make_network 299 |> print

```


There is a great space for optimisation. There are also some new neurons need to be added, e.g., upsampling, transposed convolution, and etc. Anyway, things will get better and better.
