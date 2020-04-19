# Deep Neural Networks

## Introduction

Brain neuron ect.

## Perceptron

The origin; why do we connect at all. [ liang: you can refer to wiki page ]

Threshold function ...

$$
f(x)=
    \begin{cases}
        1 & \text{if } \mathbf{w \cdot x + b > 0}\\
        0 & \text{otherwise}
    \end{cases}  
$$

Non-linearity [ show various step functions ]

An example code, but do no use the code for now.

```
let make_network input_shape =
  input input_shape
  |> linear 300 ~act_typ:Activation.Tanh
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

## Yet Another Regression

To some extend, a deep neural netowrk is nothing but a regression problem in a very high-dimensional space. We need to minimise its cost function by utilising higher-order derivatives. Before looking into the actual `Neural` module, let's build a small neural network from scratch. 

Follow the previous logistic regression.
In this section we use build a simple neural network with a hidden layer, and train its parameters. 
The task is hand-written recognition. 

Basically follow ML004

### Model Representation

In logistic regression we have... now we need to extend it towards multiple classes.
Add an internal layer (why).

![Extend logistic regression to neural network with one hidden layer](images/neural-network/simple_nn.png "simple_nn"){width=100% #fig:neural-network:simple_nn}

The data we will use is from [MNIST dataset](http://yann.lecun.com/exdb/mnist/). You can use `Owl.Dataset.download_all()` to download the dataset. 

```text
let x, _, y = Dataset.load_mnist_train_data_arr () 
```

```text
# let x_shape, y_shape = 
   Dense.Ndarray.S.shape x, Dense.Ndarray.S.shape y

val x_shape : int array = [|60000; 28; 28; 1|]
val y_shape : int array = [|60000; 10|]
```

The label is in the one-hot format:

```text
val y : Owl_dense_matrix.S.mat =

        C0  C1  C2  C3  C4  C5  C6  C7  C8  C9
    R0   0   0   0   0   0   1   0   0   0   0
    R1   1   0   0   0   0   0   0   0   0   0
    R2   0   0   0   0   1   0   0   0   0   0
       ... ... ... ... ... ... ... ... ... ...
```

It shows the first three labels are 5, 0, and 4. 

![Visualise part of MNIST dataset](images/regression/mnist.png "mnist"){width=60% #fig:neural-network:mnist}

### Forward Propagation

Specifically we use a hidden layer of size 25, and the output class is 10. 

Since we will use derivatives in training parameters, we construct all the computation using the Algorithmic Differentiation module. 

The computation is simple. The logistic regression is repeated:

$$h_\Theta(x) = f(f(x~\theta_0)~\theta_1).$$

It can be implemented as:

```ocaml env=neural-network:simple-nn
open Algodiff.D
module N = Dense.Ndarray.D

let input_size = 28 * 28
let hidden_size = 25
let classes = 10

let theta0 = Arr (N.uniform [|input_size; hidden_size|])
let theta1 = Arr (N.uniform [|hidden_size; classes|])

let h theta0 theta1 x = 
  let t = Arr.dot x theta0 |> Maths.sigmoid in 
  Arr.dot t theta1 |> Maths.sigmoid
```

That's it. We can now classify an input `28x28` array into one of the ten classes... except that we can't.
Currently we only use random content as the parameters. 
We need to train the model and find suitable $\theta_0$ and $\theta_1$ parameters. 

### Back propagation

Training a network is essentially a process of minimising the cost function by adjusting the weight of each layer. 
The core of training is the backpropagation algorithm. As its name suggests, backpropagation algorithm propagates the error from the end of a network back to the input layer, in the reverse direction of evaluating the network. Backpropagation algorithm is especially useful for those functions whose input parameters `>>` output parameters.

Backpropagation is the core of all neural networks, actually it is just a special case of reverse mode AD. Therefore, we can write up the backpropagation algorithm from scratch easily with the help of `Algodiff` module.

Recall in the Regression chapter, training parameters is the process is to find the parameters that minimise the cost function of iteratively.
In the case of this neural network, its cost function $J$ is similar to that of logistic regression.
Suppose we have $m$ training data pairs, then it can be expressed as:

$$J(\theta+0, \theta_1) = \frac{1}{m}\sum_{i=1}^m(-y^{(i)}log(h_\Theta(x^{(i)}))-(1 -y^{(i)})log(1-h_\Theta(x^{(i)}))).$$ {#eq:neural-network:costfun}

```ocaml env=neural-network:simple-nn
let j t0 t1 x y =
  let z = h t0 t1 x in 
  Maths.add
  	(Maths.cross_entropy y z)
  	(Maths.cross_entropy Arr.(sub (ones (shape y)) y)
  	   Arr.(sub (ones (shape z)) z))
```

Here the the `cross_entropy y x` means $-h~\log(x)$. 

In the regression chapter, to find the suitable parameters that minimise $J$, we iteratively apply: 

$$ \theta_j \leftarrow \theta_j - \alpha~\frac{\partial}{\partial \theta_j}~J(\theta_0, \theta_1)$$ 

until it converges.
The same also applies here. But the partial derivative is not intuitive to give a analytical solution. 
But actually we don't have to now that we are using the AD module. 
The partial derivatives of both parameters can be correctly calculated. 
We have show in the Algorithmic Differentiation chapter how it can be done in Owl:

```text
let x', y' = Dataset.draw_samples x y 1
let cost = j t0 t1 (Arr x') (Arr y')
let _ = reverse_prop (F 1.) cost
let theta0' =  adjval t0 |> unpack_arr
let theta1' =  adjval t1 |> unpack_arr
```

That's it for one iteration. 
We get $\frac{\partial}{\partial \theta_j}~J(\theta_0, \theta_1)$, and then can iteratively update the $\Theta$ parameters. 

TODO: finish this example with accuracy value.

## Feed Forward Network

In the next step, we revise the previous example, with a bit of more details added. 

First, the previous example mixes all the computation together.
We need to add the abstraction of *layers*
(Explain).
The following code defines the layer and network type, both are OCaml record types. 

Also note that for each layer, besides the matrix multiplication, we also added an extra *bias* 
(Explain). 
Each linear layer performs the following calculation where $a$ is a non-linear activation function.

$$ y = a(x \times w + b) $$

Each layer consists of three components: weight `w`, bias `b`, and activation function `a`. A network is just a collection of layers.

```ocaml env=neural_01
open Algodiff.S

type layer = {
  mutable w : t;
  mutable b : t;
  mutable a : t -> t;
}

type network = { layers : layer array }
```
Despite of the complicated internal structure, we can treat a neural network as a function, which is fed with input data and outputs predictions. The question is how to evaluate a network. Evaluating a network can be decomposed as a sequence of evaluation of layers. 

The output of one layer will feed into the next layer as its input, moving forward until it reaches the end. The following two lines shows how to evaluate a neural network in *forward mode*.

```ocaml env=neural_01
let run_layer x l = Maths.((x *@ l.w) + l.b) |> l.a

let run_network x nn = Array.fold_left run_layer x nn.layers
```

The `run_network` can generate what equals to the $h_\Theta(x)$ function in previous section.

Here we note there is an extra function `a`. It is the activation function.
(Explain activation function and why they are necessary.)
Previously we use the `sigmoid` function, but that's not the only option.
We can also use `tanh` and `softmax`.

In this small example, we will only use two layers, `l0` and `l1`. 
`l0` uses a `784 x 40` matrix as weight, and `tanh` as activation function. 
`l1` is the output layer and `softmax` is the cost function.

```ocaml env=neural_01
let l0 = {
  w = Maths.(Mat.uniform 784 40 * F 0.15 - F 0.075);
  b = Mat.zeros 1 300;
  a = Maths.tanh;
}

let l1 = {
  w = Maths.(Mat.uniform 40 10 * F 0.15 - F 0.075);
  b = Mat.zeros 1 10;
  a = Maths.softmax ~axis:1;
}

let nn = {layers = [|l0; l1|]}
```

This definition is plain, but there is still one thing to say.
*Initialisation*: previously we use a uniformly random array, but choosing a good initial status is important.
Explain.
Here we use...


### Training

The loss function is constructed in the same way. 

```ocaml env=neural_01
let loss_fun nn x y = 
  let t = tag () in
  Array.iter (fun l ->
    l.w <- make_reverse l.w t;
    l.b <- make_reverse l.b t;
  ) nn.layers;
  Maths.(cross_entropy y (run_network x nn) / (F (Mat.row_num y |> float_of_int)))
```

(Explain why we use only one `cross_entropy`)

```ocaml env=neural_01
let backprop nn eta x y =
  let loss = loss_fun nn x y in 
  reverse_prop (F 1.) loss;
  Array.iter (fun l ->
    l.w <- Maths.((primal l.w) - (eta * (adjval l.w))) |> primal;
    l.b <- Maths.((primal l.b) - (eta * (adjval l.b))) |> primal;
  ) nn.layers;
  loss |> unpack_flt
```

The `backprop` also uses the same procedure as previous example. 
The partial derivative is gotten using `adjval`, and the parameter `w` and `b` of each layer are updated accordingly.
It then uses the gradient descent method we introduced in previous example. 
The learning rate `eta` is fixed.


### Test

We need to see how well our trained model works.
The `test` function performs model inference and compares the predictions with the labelled data. By doing so, we can evaluate the accuracy of a neural network.

```ocaml env=neural_01
let test nn x y =
  Dense.Matrix.S.iter2_rows (fun u v ->
    let p = run_network (Arr u) nn |> unpack_arr in
    Dense.Matrix.Generic.print p;
    Printf.printf "prediction: %i\n" (let _, i = Dense.Matrix.Generic.max_i p in i.(1))
  ) (unpack_arr x) (unpack_arr y)
```

Finally, we put all the previous parts together. 
The following code starts the training for 999 iterations.

```ocaml env=neural_01
let main () =
  let x, _, y = Dataset.load_mnist_train_data () in
  for i = 1 to 999 do
    let x', y' = Dataset.draw_samples x y 100 in
    backprop nn (F 0.01) (Arr x') (Arr y')
    |> Owl_log.info "#%03i : loss = %g" i
  done;
  let x, y, _ = Dataset.load_mnist_test_data () in
  let x, y = Dataset.draw_samples x y 10 in
  test nn (Arr x) (Arr y)
```

When the training starts, our application keeps printing the value of loss function in the end of each iteration. From the output, we can see the value of loss function keeps decreasing quickly after training starts.

```text
2019-11-12 01:04:14.632 INFO : #001 : loss = 2.54432
2019-11-12 01:04:14.645 INFO : #002 : loss = 2.48446
2019-11-12 01:04:14.684 INFO : #003 : loss = 2.33889
2019-11-12 01:04:14.696 INFO : #004 : loss = 2.28728
2019-11-12 01:04:14.709 INFO : #005 : loss = 2.23134
2019-11-12 01:04:14.720 INFO : #006 : loss = 2.21974
2019-11-12 01:04:14.730 INFO : #007 : loss = 2.0249
2019-11-12 01:04:14.740 INFO : #008 : loss = 1.96638
```

After training finished, we test the accuracy of the network. Here is one example where we input hand-written 3. The vector below shows the prediction, we see the model says with $90.14%$ chance it is a number 3. Quite accurate!

![Prediction from the model](images/neural-network/plot_01.png "plot_01"){ width=40% #fig:neural-network:plot01 }

TODO: replace with code. 


## Neural Network Module

More layers, but you can find that previous approach is hard to scale. 

The `Neural` module is actually very similar to the naive framework we just built, but with more compete support to various neurons.

Owl is designed as a general-purpose numerical library, and I never planned to make it yet another framework for deep neural networks. The original motivation of including such a neural network module was simply for demo purpose, since in almost every presentation I had been to, there were always the same question from audience: *"can owl do deep neural network by the way?"*

In the end, we became curious about this question myself, although the perspective was slightly different. I was very sure I could implement a proper neural network framework atop of Owl, but I didn't know how easy it is. I think it is an excellent opportunity to test Owl's capability and expressiveness in developing complicated analytical applications.

The outcome is wonderful. It turns out with Owl's architecture and its internal functionality (Algodiff, CGraph, etc.), combined with OCaml's powerful module system, implementing a full featured neural network module only requires approximately 3500 LOC. Yes, you heard me, 3500 LOC, and it beats TensorFlow's performance on CPU (by the time we measured in 2018).

In this section we talk about the deesign of NN module


### Module Structure

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


### Model Definition

I have implemented a set of commonly used neurons in [Owl.Neural.Neuron](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_neuron.ml). Each neuron is a standalone module and adding a new type of neuron is much easier than adding a new one in Tensorflow or other framework thanks to Owl's [Algodiff](https://github.com/ryanrhymes/owl/blob/master/lib/owl_algodiff_generic.mli) module.

`Algodiff` is the most powerful part of Owl and offers great benefits to the modules built atop of it. In neural network case, we only need to describe the logic of the forward pass without worrying about the backward propagation at all, because the `Algodiff` figures it out automatically for us thus reduces the potential errors. This explains why a full-featured neural network module only requires less than 3.5k lines of code. Actually, if you are really interested, you can have a look at Owl's [Feedforward Network](https://github.com/ryanrhymes/owl/blob/master/examples/feedforward.ml) which only uses a couple of hundreds lines of code to implement a complete Feedforward network.

In practice, you do not need to use the modules defined in  [Owl.Neural.Neuron](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_neuron.ml) directly. Instead, you should call the functions in [Graph](https://github.com/ryanrhymes/owl/blob/master/lib/neural/owl_neural_graph.ml) module to create a new neuron and add it to the network. Currently, Graph module contains the following neurons.

`input`, `activation`, `linear`, `linear_nobias`, `embedding`, `recurrent`, `lstm`, `gru`, `conv1d`, `conv2d`, `conv3d`, `max_pool1d`, `max_pool2d`, `avg_pool1d`, `avg_pool2d`, `global_max_pool1d`, `global_max_pool2d`, `global_avg_pool1d`, `global_avg_pool2d`, `fully_connected`, `dropout`, `gaussian_noise`, `gaussian_dropout`, `alpha_dropout`, `normalisation`, `reshape`, `flatten`, `lambda`, `add`, `mul`, `dot`, `max`, `average`, `concatenate`

These neurons should be sufficient for creating from simple MLP to the most complicated Google's Inception network.


### Model Training

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


### Model Inference

TBD


## Convolution Neural Network 

Introduce CNN

More about the structure in NN module/Optimise module

Implement the same MNIST task with CNN. 


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


**Applications**

For more applications, please check the image recognition, NST, and instance segmentation cases.


## Recurrent Neural Network

Introduce vanilla RNN ...

### Gated Recurrent Unit (GRU)

Include GRU, briefly introduce RNN, then jump to 

### Long Short Term Memory (LSTM)

```ocaml env=neural_00

  let make_network wndsz vocabsz =
    input [|wndsz|]
    |> embedding vocabsz 40
    |> lstm 128
    |> linear 512 ~act_typ:Activation.Relu
    |> linear vocabsz ~act_typ:Activation.(Softmax 1)
    |> get_network

```

The generated computation graph is way more complicated due to LSTM's internal recurrent structure. You can download the [PDF file 1](https://raw.githubusercontent.com/wiki/ryanrhymes/owl/image/plot_030.pdf) for better image quality.


## Generative Adversarial Network

Only define the structure, no training ...


## Summary
