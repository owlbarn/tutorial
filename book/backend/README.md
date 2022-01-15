# Compiler Backends

For a numerical library, it is always beneficial and a challenge to extend to multiple execution backends.
We have seen how we support accelerators such as GPU by utilising symbolic representation and computation graph standard such as ONNX.
In this chapter we introduce how Owl can be used on more edge-oriented backends, including JavaScript and MirageOS.
We also introduce the `base` library in Owl, since this pure OCaml library is built to support these backends.

## Base Library

Before we start, we need to understand how Owl enables compiling to multiple backends by providing different implementations.
The Owl framework, as well as many of its external libraries, is actually divided to two parts: a [Base library](https://github.com/owlbarn/owl/tree/master/src/base) and a [Core library](https://github.com/owlbarn/owl/tree/master/src/owl). The base library is implemented with pure OCaml.
For some backends such as JavaScript, we can only use the functions implemented in OCaml.

You may wonder how much we will be limited by the Base. Fortunately, the most advanced functions in Owl are often implemented in pure OCaml and they live in the Base, which includes e.g. algorithmic differentiation, optimisation, even neural networks and many others.
Here is the structure of the core functor stack in Owl:

![Core functor stack in owl](images/backend/base-structure.png "functor"){width=90% #fig:backend:functor}

Ndarray is the core building block in Owl.
As we have described in the previous chapters how we use C code to push forward the performance of Owl computation.
The base library aims to implements all the necessary functions as the core library ndarray module.
The stack is implemented in such way that the user can switch between these two different implementation without the modules of higher layer.
In the Owl functor stack, ndarray is used to support the CGraph module to provide lazy evaluation functionalities.

You might be wondering: where is the ndarray module then?
Here we use the `Owl_base_algodiff_primal_ops` module, which is simply a wrapper around the base ndarray module.
It also includes a small number of Matrix and Linear Algebra functions.
By providing this wrapper instead of using the Ndarray module directly, we can avoid mixing all the function in the ndarray module and makes it a large Goliath.

Next, the Algorithmic Differentiation can build up its computation module based on normal ndarray or its lazy version.
For example, you can have an AD that relies on the normal single precision base ndarray module:

```ocaml
module AD = Owl_algodiff_generic.Make (Owl_base_algodiff_primal_ops.S)
```

Or it can be built on an double precision lazy evaluated core ndarray module:

```ocaml
module CPU_Engine = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.D)
module AD = Owl_algodiff_generic.Make (CPU_Engine)
```

Going even higher, we have the advanced modules Optimisation and Neural Network modules. They are both based on the AD module.
For example, the code below shows how we can build a neural graph module by layers of functors from the base ndarray.

```ocaml
module G = Owl_neural_graph.Make
            (Owl_neural_neuron.Make
              (Owl_optimise_generic.Make
                (Owl_algodiff_generic.Make
                  (Owl_base_algodiff_primal_ops.S))))
```

Normally the users does not have to care about how these modules are constructed layer by layer, but understanding the functor stack and typing is nevertheless beneficial, especially when you are creating new modules that relies on the base ndarray module.

These examples show that once we have built an application with the core Ndarray module, we can then seamlessly switch it to base ndarray module without changing anything else.
That means that all the code and examples we have seen so far can be used directly on different backends that requires pure implementation.

The base library is still an on-going work and there is still a lot to do.
Though the ndarray module is a large part in base library, there are other modules that also need to be re-implemented in OCaml, such as the Linear Algebra module.
We need to add more functions such as the SVD factorisation.
Even for the Ndarray itself we still cannot totally cover the core ndarray yet.
Our strategy is that, we put most of the signature file in base library, and the core library signature file includes its corresponding signature file from the base library, plus functions that are currently unique to core library.
The target is to total coverage so that the core and base library provide exactly the same functions.

As can be expected, the pure OCaml implementation normally performs worse than the C code implemented version.
For example, for the complex convolution, without the help of optimised routines from OpenBLAS ect., we can only provide the naive implementation that includes multiple for-loops.
Its performance is orders of magnitude slower than the C version.
Currently our priority is to implement the functions themselves instead of caring about function optimisation, nor do we intend to out-perform C code with pure OCaml implementation.


## Backend: JavaScript

At first glance, JavaScript has very little to do with high-performance scientific computing. Then why Owl cares about it? One important reason is that browser is arguably the most widely deployed technology on various edge devices, e.g. mobile phones, tablets, laptops, etc. More functionalities are being pushed from data centers to edge for reduced latency, better privacy and security. And JavaScript applications running in a browser are getting more complicated and powerful.
Moreover, JavaScript interpreters are being increasingly optimised, and even relatively complicated computational tasks can run with reasonable performance.

This chapter uses two simple examples to demonstrate how to compile Owl applications into JavaScript code so that you can deploy the analytical code into browsers, using both native OCaml code and Facebook Reason.
It additionally requires the use of `dune`. As you will see, this will make the compilation to JavaScript effortless.

### Use Native OCaml

We rely on the tool `js_of_ocaml` to convert native OCaml code into JavaScript.
[Js_of_ocaml](http://ocsigen.org/js_of_ocaml) is a compiler from OCaml bytecode programs to JavaScript.
The process can thus be divided into two phases: first, compile the OCaml source code into bytecode executables, and then apply the `js_of_ocaml` command to it.
It supports the core `Bigarray` module among most of the OCaml standard libraries.
However, since the `Sys` module is not fully supported, we are careful to not use functions from this module in the base library.

We have described how Algorithm Differentiation plays a core role in the ecosystem of Owl, so now we use an example of AD to demonstrate how we convert a numerical programme into JavaScript code and then get executed.
The example comes from the Optimisation chapter, and is about optimise the mathematical function `sin`.
The first step is writing down our application in OCaml as follows, then save it into a file `demo.ml`.

```ocaml

  module AlgodiffD = Owl_algodiff_generic.Make (Owl_base_algodiff_primal_ops.D)
  open AlgodiffD

  let rec desc ?(eta=F 0.01) ?(eps=1e-6) f x =
    let g = (diff f) x in
    if (unpack_flt g) < eps then x
    else desc ~eta ~eps f Maths.(x - eta * g)

  let _ =
    let f = Maths.sin in
    let y = desc f (F 0.1) in
    Owl_log.info "argmin f(x) = %g" (unpack_flt y)

```

The code is very simple: the `desc` defines a gradient descent algorithm, and then we use `desc` to calculate the minimum value of `Maths.sin` function. In the end, we print out the result using `Owl_log` module's `info` function.
Note that we pass in the base Ndarray module to the AD functor to create a corresponding AD module.

In the second step, we need to create a `dune` file as follows. This file will instruct how the OCaml code will be first compiled into bytecode then converted into JavaScript by calling `js_of_ocaml`.

```shell
(executable
  (name demo)
  (modes byte js)
  (libraries owl-base))
```

With these two files in the same folder, you can then simply run the following command in the terminal.

```shell
dune build demo.bc && js_of_ocaml _build/default/demo.bc
```

Or even better, since `js_of_ocaml` is natively supported by `dune`, we can simply execute:

```shell
dune build
```

The command builds the application and generates a `demo.bc.js` in the `_build/default/` folder.
Finally, we can run the JavaScript using `Node.js` (or loading into a browser using an appropriate html page).

```shell
node _build/default/demo.bc.js
```

As a result, you should be able to see the output result shows a value that minimise the `sin` function, and should be similar to:

```shell
  2019-12-30 18:05:49.760 INFO : argmin f(x) = -1.5708
```

Even though we present a simple example, you should keep in mind that the base library can be used to produce more complex and interactive browser applications.

### Use Facebook Reason

[Facebook Reason](https://reasonml.github.io/) leverages OCaml as a backend to provide type safe JavaScript.
It is gaining its momentum and becoming a popular choice of developing web applications.
It actually uses another tool, [BuckleScript](https://bucklescript.github.io/), to convert the Reason/OCaml code to JavaScript.
Since Reason is basically a syntax layer built on top of OCaml, it is very straightforward to use Owl in Reason to develop advanced numerical applications.

In this example, we use reason code to manipulate multi-dimensional arrays, the core data structure in Owl.
First, we save the following code into a reason file called `demo.re`. Note the suffix is *.re* now.
It includes several basic math and Ndarray operations in Owl.

```reason

  open! Owl_base;

  /* calculate math functions */
  let x = Owl_base_maths.sin(5.);
  Owl_log.info("Result is %f", x);

  /* create random ndarray then print */
  let y = Owl_base_dense_ndarray.D.uniform([|3,4,5|]);
  Owl_base_dense_ndarray.D.set(y,[|1,1,1|],1.);
  Owl_base_dense_ndarray.D.print(y);

  /* take a slice */
  let z = Owl_base_dense_ndarray.D.get_slice([[],[],[0,3]],y);
  Owl_base_dense_ndarray.D.print(z);
```

The code above is simple, just creates a random ndarray, takes a slice, and then prints them out.
Owl library can be seamlessly used in Reason.
Next, instead of using Reason's own translation of this frontend syntax with `bucklescript`, we still turn to `js_of_ocaml` for help.
Let's look at the `dune` file, which turns out to be the same as that in the previous example.

```shell
  (executable
   (name demo)
   (modes js)
   (libraries owl-base))
```

As in the previous example, you can then compile and run the code with following commands.

```shell
  dune build
  node _build/default/demo.bc.js
```

As you can see, except that the code is written in different languages, the rest of the steps are identical in both example thanks to `js_of_ocaml` and `dune`.

## Backend: MirageOS

### MirageOS and Unikernel

Besides JavaScript, another choice of backend we aim to support is the MirageOS.
It is an approach to build *unikernels*.
A unikernel is a specialised, single address space machine image constructed with library operating systems.
Unlike normal virtual machine, it only contains a minimal set of libraries required for one application.
It can run directly on a hypervisor or hardware without relying on operating systems such as Linux and Windows.
The unikernl is thus concise and secure, and extremely efficient for distributed and executed on either cloud or edge devices.

MirageOS is one solution to building unikernels.
It utilises the high-level languages OCaml and a runtime to provide API for operating system functionalities.
In using MirageOS, the users can think of the [Xen hypervisor](https://xenproject.org/) as a stable hardware platform, without worrying about the hardware details such as devices.
Furthermore, since the Xen hypervisor is widely used in platforms such as Amazon EC2 and Rackspace Cloud, MirageOS-built unikernel can be readily deployed on these platforms.
Besides, benefiting from its efficiency and security, MirageOS also aims to form a core piece of the Nymote/MISO tool stack to power the Internet of Things.

### Example: Gradient Descent

Since MirageOS is based around the OCaml language, we can safely integrate the Owl library with it.
To demonstrate how we use MirageOS as backend, we again use the previous Algorithm Differentiation based optimisation example.
Before we start, please make sure to follow the [installation instruction](https://mirage.io/wiki/install).
Let's look at the code:

```ocaml env=backend:mirage
module A = Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.S)
open A

let rec desc ?(eta=F 0.01) ?(eps=1e-6) f x =
  let g = (diff f) x in
  if (unpack_flt (Maths.abs g)) < eps then x
  else desc ~eta ~eps f Maths.(x - eta * g)

let main () =
  let f x = Maths.(pow x (F 3.) - (F 2.) * pow x (F 2.) + (F 2.)) in
  let init = Stats.uniform_rvs ~a:0. ~b:10. in
  let y = desc f (F init) in
  Owl_log.info "argmin f(x) = %g" (unpack_flt y)
```

This part of code is mostly the same as before. By applying the `diff` function of the algorithmic differentiation module, we use the gradient descent method to find the value that minimises the function $x^3 - 2x^2 + 2$.
Then we need to add something different:

```ocaml env=backend:mirage
module GD = struct
  let start = main (); Lwt.return_unit
end
```

Here the `start` is an entry point to the unikernel.
It performs the normal OCaml function `main`, and the return a `Lwt` thread that will be evaluated to `unit`.
Lwt is a concurrent programming library in OCaml. It provides the "promise" data type that can be determined in the future.
Please refer to its [document](https://github.com/ocsigen/lwt) for more information if you are interested.

All the code above is written to a file called `gd_owl.ml`.
To build a unikernel, next we need to define its configuration.
In the same directory, we create a file called `configure.ml`:

```
open Mirage

let main =
  foreign
    ~packages:[package "owl"]
    "Gd_owl.GD" job

let () =
  register "gd_owl" [main]
```

It's not complex.
First we need to open the `Mirage` module.
Then we declare a value `main` (or you can name it any other name).
It calls the `foreign` function to specify the configuration.
First, in the `package` parameter, we declare that this unikernel requires Owl library.
The next string parameter "Gd_owl.GD" specifies the name of the implementation file, and in that file the module `GD` that contains the `start` entry point.
The third parameter `job` declares the type of devices required by a unikernel, such as network interfaces, network stacks, file systems, etc.
Since here we only do the calculation, there is no extra device required, so the third parameter is a `job`.
Finally, we register the unikernel entry file `gd_owl` with the `main` configuration value.

That's all it takes for coding. Now we can take a look at the compiling part.
MirageOS itself supports multiple backends. The crucial choice therefore is to decide which one to use at the beginning by using `mirage configure`.
In the directory that holds the previous two files, you run `mirage configure -t unix`, and it configures to build the unikernel into a Unix ELF binary that can be directly executed.
Or you can use `mirage configure -t xen`, and then the resulting unikernel will use hypervisor backend like Xen or KVM.
Either way, the unikernel runs as a virtual machine after starting up.
In this example we choose to use UNIX as backends. So we run:

```shell
mirage configure -t unix
```

This command generates a `Makefile` based on the configuration information. It includes all the building rules.
Next, to make sure all the dependencies are installed, we need to run:

```shell
make depend
```

Finally, we can build the unikernels by simply running:

```shell
make
```

and it calls the `mirage build` command.
As a result, now your current directory contains the `_build/gd_owl.native` executable, which is the unikernel we want.
Executing it yields a similar result as before:

```
INFO : argmin f(x) = 1.33333
```

### Example: Neural Network

As a more complex example we have also built a simple neural network to perform the MNIST handwritten digits recognition task:

```ocaml
module N  = Owl_base_algodiff_primal_ops.S
module NN = Owl_neural_generic.Make (N)
open NN
open NN.Graph
open NN.Algodiff

let make_network input_shape =
  input input_shape
  |> lambda (fun x -> Maths.(x / F 256.))
  |> fully_connected 25 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

This neural network has two hidden layer, has a small weight size (146KB), and works well in testing (92% accuracy).
We can right the weight into a text file.
This file is named `simple_mnist.ml`, and similar to previous example, we can add a unikernel entry point function in the file:

```
module Main = struct
  let start = infer (); Lwt.return_unit
end
```

Here the `infer` function creates a neural network, loads the weight, and then performs inference on an input image.
We also need a configuration file. Again, it's mostly the same:

```
open Mirage

let main =
  foreign
    ~packages:[package "owl-base"]
   "Simple_mnist.Main" job

let () =
  register "Simple_mnist" [main]
```

All the code is included in [this gist](https://gist.github.com/jzstark/e94917167754963de701e9e9ce750b2e).
Once compiled to MirageOS unikernel with unix backends, the generated binary is 10MB.
You can also try compiling this application to JavaScript.

By these examples we show that the Owl library can be readily deployed into unikernels via MirageOS.
The numerical functionalities can then greatly enhance the express ability of possible OCaml-MirageOS applications.
Of course, we cannot cover all the important topics about MirageOS, please refer to the documentation of [MirageOS](https://mirage.io/) abd [Xen Hypervisor](https://xenproject.org/) for more information.


## Evaluation

In the evaluation section we mainly compare the performance of different backends we use.
Specifically, we observe three representative groups of operations:
(1) `map` and `fold` operations on ndarray;
(2) using gradient descent, a common numerical computing subroutine, to get $argmin$ of a certain function;
(3) conducting inference on complex DNNs, including SqueezeNet and a VGG-like convolution network.
The evaluations are conducted on a ThinkPad T460S laptop with Ubuntu 16.04 operating system. It has an Intel Core i5-6200U CPU and 12GB RAM.

The OCaml compiler can produce two kinds of executables: bytecode and native.
Native executables are compiled for specific architectures and are generally faster, while bytecode executables have the advantage of being portable.

![Performance of map and fold operations on ndarray on laptop and RaspberryPi](images/zoo/map_fold.png "map-fold"){width=100% #fig:backend:map_fold}

For JavaScript, we use the `js_of_ocaml` approach as described in the previous sections.
Note that for convenience we refer to the pure implementation of OCaml and the mix implementation of OCaml and C as `base-lib` and `owl-lib` separately, but they are in fact all included in the Owl library.
For Mirage compilation, we use both libraries.

[@fig:backend:map_fold](a-b) show the performance of map and fold operations on ndarray.
We use simple functions such as plus and multiplication on 1-d (size $< 1,000$) and 2-d arrays.
The `log-log` relationship between total size of ndarray and the time each operation takes keeps linear.
For both operations, `owl-lib` is faster than `base-lib`, and native executables outperform bytecode ones. The performance of Mirage executives is close to that of native code.
Generally JavaScript runs the slowest, but note how the performance gap between JavaScript and the others converges when the ndarray size grows.
For fold operation, JavaScript even runs faster than bytecode when size is sufficiently large.

Note that for the fold operation, there is an obvious increase in time used at around input size of $10^3$ for fold operations, while there is not such change for the map operation.
That is because I change the input from one dimensional ndarray to two dimensional starting that size.
This change does not affect map operation, since it treats an input of any dimension as a one dimensional vector.
On the other hand, the fold operation considers the factor of dimension, and thus its performance is affected by this change.

![Performance of gradient descent on function $f$](images/zoo/gd_x86.png){width=75% #fig:backend:gd}

In [@fig:backend:gd], we want to investigate if the above observations still hold in more complex numerical computation.
We choose to use a Gradient Descent algorithm to find the value that locally minimise a function. We choose the initial value randomly between $[0, 10]$.
For both $sin(x)$ and $x^3 -2x^2 + 2$, we can see that JavaScript runs the slowest, but this time the `base-lib` slightly outperforms `owl-lib`.

We further compare the performance of DNN, which requires large amount of computation.
We compare SqueezeNet and a VGG-like convolution network.
They have different sizes of weight and networks structure complexities.

    Time (ms) VGG                     SqueezeNet
------------- ----------------------- --------------------------
   owl-native 7.96 ($\pm$ 0.93)       196.26($\pm$ 1.12)
     owl-byte 9.87 ($\pm$ 0.74)       218.99($\pm$ 9.05)
  base-native 792.56($\pm$ 19.95)     14470.97 ($\pm$ 368.03)
    base-byte 2783.33($\pm$ 76.08)    50294.93 ($\pm$ 1315.28)
   mirage-owl 8.09($\pm$ 0.08)        190.26($\pm$ 0.89)
  mirage-base 743.18 ($\pm$ 13.29)    13478.53 ($\pm$ 13.29)
   JavaScript 4325.50($\pm$ 447.22)   65545.75 ($\pm$ 629.10)

: Inference Speed of Deep Neural Networks {#tbl:zoo:dnn}


[@tbl:zoo:dnn] shows that, though the performance difference between `owl-lib` and `base-lib` is not obvious, the former is much better. So is the difference between native and bytecode for `base-lib`.
JavaScript is still the slowest. The core computation required for DNN inference is the convolution operation.
Its implementation efficiency is the key to these differences. Current we are working on improving its implementation in `base-lib`.

We have also conducted the same evaluation experiments on RaspberryPi 3 Model B.
[@fig:backend:map_fold](c) shows the performance of fold operation on ndarray. Besides the fact that all backends runs about one order of magnitude slower than that on the laptop, previous observations still hold.
This figure also implies that, on resource-limited devices such as RaspberryPi, the key difference is between native code and bytecode, instead of `owl-lib` and `base-lib` for this operation.

  Size (KB) native   bytecode   Mirage   JavaScript
----------- -------- ---------- -------- ------------
       base 2,437    4,298      4,602    739
     native 14,875   13,102     16,987   \-

: Size of executables generated by backends {#tbl:zoo:size}

Finally, we also briefly compare the size of executables generated by different backends. We take the SqueezeNet for example, and the results are shown in [@tbl:zoo:size].
It can be seen that `owl-lib` executives have larger size compared to `base-lib` ones, and JavaScript code has the smallest file size.
There does not exist a dominant method of deployment for all these backends. It is thus imperative to choose suitable backend according to deployment environment.

## Summary

The base library in Owl was separated from the core module mainly to accommodate multiple possible execution backends.
This chapter introduces how the base module works.
Then we show two possible backends: the JavaScript and the Unikernel virtual machine. Both back ends are helpful to extend the application of Owl to more devices.
Finally, we use several examples to demonstrate how these backends are used and their performances.
