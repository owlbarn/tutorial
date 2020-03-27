# Compiler Backends

This chapter uses two simple examples to demonstrate how to compile Owl applications into JavaScript code so that you can deploy the analytical code into browsers. It additionally requires the use of `dune`. As you will see, this will make the compilation to JavaScript effortless.

## Base Library

Before we start, you need to know some basic things to understand how Owl is able to support JavaScript

The Owl framework, as well as many of its external libraries, is actually divided to two parts: a [Base library](https://github.com/owlbarn/owl/tree/master/src/base) and a [Core library](https://github.com/owlbarn/owl/tree/master/src/owl). The base library is implemented with pure OCaml and, thanks to [`js_of_ocaml`](https://ocsigen.org/js_of_ocaml/), it can be safely compiled into JavaScript. Many functionalities (e.g. Ndarray) in the base library are replaced by high-performance C implementations in the Core library.

As you can see, to enable JavaScript, we can only use the functions implemented in the Base. You may wonder how much we will be limited by the Base. Fortunately, the most advanced functions in Owl are often implemented in pure OCaml and they live in the Base, which includes e.g. algorithmic differentiation, optimisation, even neural networks and many others.

In this following, we will present how to use `Owl_base` in writing JavaScript applications using both native OCaml code and Facebook Reason.

Even though we will present very simple examples, you should keep in mind that Owl_base is fully compatible with `js_of_ocaml` and can be used to produce more complex and interactive browser applications.

## Backend: JavaScript

At first glance, JavaScript has very little to do with high-performance scientific computing. Then why Owl cares about it? One important reason is that browser is arguably the most widely deployed technology on various edge devices, e.g. mobile phones, tablets, laptops, and etc. More functionalities are being pushed from datacenters to edge for reduced latency, better privacy and security. And JavaScript applications running in a browser are getting more complicated and powerful.

Moreover, JavaScript interpreters are being increasingly optimised, and even relatively complicated computational tasks can run with reasonable performance.

### Use Native OCaml

We know that `Owl_algodiff_generic` is the cornerstone of Owl's fast neural network module. The first example uses Algodiff functor to optimise a mathematical function.

The first step is writing down our application in OCaml as follows, then save it into a file `demo.ml`

```ocaml

  (* JavaScript example: use Owl_base to minimise sin *)

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

The code is very simple: the `desc` defines a gradient descent algorithm, then we use `desc` to calculate the minimum value of `Maths.sin` function. In the end, we print out the result using `Owl_log` module's `info` function. You should have noticed, we used `Owl_algodiff_generic` functor to create and include an algorithmic differentiation module by passing the pure implementation of Ndarray in the base library.

In the second step, we need to create a `dune` file as follows. This file will instruct how the OCaml code will be first compiled into bytecode then converted into JavaScript by calling `js_of_ocaml`.


```shell
  (executable
   (name demo)
   (modes js)
   (libraries owl-base))
```

With these two files in the same folder, you can then simply run the following command in the terminal. The command builds the application and generates a `demo.bc.js` in `_build/default/` folder.


```shell

  dune build

```

Finally, we can run the JavaScript using Node.js (or loading into a browser using an appropriate html page).

```shell

  node _build/default/demo.bc.js

```

You should be able to see the output result similar to 

```shell

  2019-12-30 18:05:49.760 INFO : argmin f(x) = -1.5708

```

### Use Facebook Reason

Facebook Reason is gaining its momentum and becoming a popular choice of developing web applications. Because Reason is basically a syntax on top of OCaml, it is very straightforward to use Owl in Reason to develop advanced numerical applications.

In this example, we use reason code to manipulate multi-dimensional arrays, the core data structure in Owl. First, please save the following code into a reason file `demo.re`. Note the the suffix is *.re* now.


```reason

  /* JavaScript example: Ndarray and Maths */

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

The code above is simple, just creates a random ndarray, takes a slice, then prints them out. Now let's look at the `dune` file, which turns out to be exactly the same as that in the previous example.


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

As you can see, except the code is written in different languages, the rest of the steps are identical in both example thanks to the excellent dune.

## Backend: MirageOS

## Evaluation

In the evaluation section we mainly compare the performance of different
backends we use. Specifically, we observe three representative groups of
operations: (1) `map` and `fold` operations on ndarray; (2) using
gradient descent, a common numerical computing subroutine, to get
$argmin$ of a certain function; (3) conducting inference on complex
DNNs, including SqueezeNet and a VGG-like
convolution network. The evaluations are conducted on a ThinkPad T460S
laptop with Ubuntu 16.04 operating system. It has an Intel Core i5-6200U
CPU and 12GB RAM.

The OCaml compiler can produce two kinds of executables: bytecode and
native. Native executables are compiled specifically for an architecture
and are generally faster, while bytecode executables have the advantage
of being portable. A Docker container can adopt both options.

For JavaScript though, since the Owl library contains functions that are
implemented in C, it cannot be directly supported by `js-of-ocaml`, the
tool we use to convert OCaml code into JavaScript. Therefore in the Owl
library, we have implemented a "base" library in pure OCaml that shares
the core functions of the Owl library. Note that for convenience we
refer to the pure implementation of OCaml and the mix implementation of
OCaml and C as `base-lib` and `owl-lib` separately, but they are in fact
all included in the Owl library. For Mirage compilation, we use both
libraries.

![Performance of map and fold operations on ndarray on laptop and RaspberryPi](images/zoo/map_fold.png){#fig:zoo:map_fold}

[@fig:zoo:map_fold](a-b) show the performance of map and fold
operations on ndarray. We use simple functions such as plus and
multiplication on 1-d (size $< 1,000$) and 2-d arrays. The log-log
relationship between total size of ndarray and the time each operation
takes keeps linear. For both operations, `owl-lib` is faster than
`base-lib`, and native executables outperform bytecode ones. The
performance of Mirage executives is close to that of native code.
Generally JavaScript runs the slowest, but note how the performance gap
between JavaScript and the others converges when the ndarray size grows.
For fold operation, JavaScript even runs faster than bytecode when size
is sufficiently large.

![Performance of gradient descent on function $f$](images/zoo/gd_x86.png){#fig:zoo:gd}

In [@fig:zoo:gd], we want to investigate if the above
observations still hold in more complex numerical computation. We choose
to use a Gradient Descent algorithm to find the value that locally
minimise a function. We choose the initial value randomly between
$[0, 10]$. For both $sin(x)$ and $x^3 -2x^2 + 2$, we can see that
JavaScript runs the slowest, but this time the `base-lib` slightly
outperforms `owl-lib`.

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


[@tbl:zoo:dnn] shows that, though the performance difference
between `owl-lib` and `base-lib` is not obvious, the former is much
better. So is the difference between native and bytecode for `base-lib`.
JavaScript is still the slowest. The core computation required for DNN
inference is the convolution operation. Its implementation efficiency is
the key to these differences. Current we are working on improving its
implementation in `base-lib`.

We have also conducted the same evaluation experiments on RaspberryPi 3
Model B.
[@fig:zoo:map_fold](c) shows the performance of fold operation
on ndarray. Besides the fact that all backends runs about one order of
magnitude slower than that on the laptop, previous observations still
hold. This figure also implies that, on resource-limited devices such as
RaspberryPi, the key difference is between native code and bytecode,
instead of `owl-lib` and `base-lib` for this operation. The other
figures are not presented here due to space limited, but the conclusions
are similar.

  Size (KB) native   bytecode   Mirage   JavaScript
----------- -------- ---------- -------- ------------
       base 2,437    4,298      4,602    739
     native 14,875   13,102     16,987   \-

: Size of executables generated by backends {#tbl:zoo:size}

Finally, we also briefly compare the size of executables generated by
different backends. We take the SqueezeNet for example, and the results
are shown in [@tbl:zoo:size].
It can be seen that `owl-lib` executives
have larger size compared to `base-lib` ones, and JavaScript code has
the smallest file size.
It can be seen that there does not exist a dominant method of deployment
for all these backends. It is thus imperative to choose suitable backend
according to deployment environment.
