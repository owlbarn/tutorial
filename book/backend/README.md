# Compiler Backends

This chapter uses two simple examples to demonstrate how to compile Owl applications into JavaScript code so that you can deploy the analytical code into browsers. It additionally requires the use of `dune`. As you will see, this will make the compilation to JavaScript effortless.

## Background

At first glance, JavaScript has very little to do with high-performance scientific computing. Then why Owl cares about it? One important reason is that browser is arguably the most widely deployed technology on various edge devices, e.g. mobile phones, tablets, laptops, and etc. More functionalities are being pushed from datacenters to edge for reduced latency, better privacy and security. And JavaScript applications running in a browser are getting more complicated and powerful.

Moreover, JavaScript interpreters are being increasingly optimised, and even relatively complicated computational tasks can run with reasonable performance.

## Base Library

Before we start, you need to know some basic things to understand how Owl is able to support JavaScript

The Owl framework, as well as many of its external libraries, is actually divided to two parts: a [Base library](https://github.com/owlbarn/owl/tree/master/src/base) and a [Core library](https://github.com/owlbarn/owl/tree/master/src/owl). The base library is implemented with pure OCaml and, thanks to [`js_of_ocaml`](https://ocsigen.org/js_of_ocaml/), it can be safely compiled into JavaScript. Many functionalities (e.g. Ndarray) in the base library are replaced by high-performance C implementations in the Core library.

As you can see, to enable JavaScript, we can only use the functions implemented in the Base. You may wonder how much we will be limited by the Base. Fortunately, the most advanced functions in Owl are often implemented in pure OCaml and they live in the Base, which includes e.g. algorithmic differentiation, optimisation, even neural networks and many others.

In this following, we will present how to use `Owl_base` in writing JavaScript applications using both native OCaml code and Facebook Reason.

Even though we will present very simple examples, you should keep in mind that Owl_base is fully compatible with `js_of_ocaml` and can be used to produce more complex and interactive browser applications.

## Use Native OCaml

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

With these two files in the same folder, you can then simply run the following command in the terminal. The command builds the application and generates a `demo.js` in `_build/default/` folder.


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


## Use Facebook Reason

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
