# Computation Graph

This chapter first gives a bird's-eye-view on the computation graph in Owl. Then we will continue to cover the design and implementation details of the computation graph and how it is fitted into Owl's functor stack, and its implications on the architecture of numerical systems.

## Introduction

### What is a Computation Graph?

As a functional programmer, it is basic knowledge that a function takes an input then produces an output. The input of a function can be the output of another function which then creates dependency. If we view a function as one node in a graph, and its input and output as incoming and outgoing links respectively, as the computation continues, these functions are chained together to form a directed acyclic graph (DAG). Such a DAG is often referred to as a computation graph.

![Computation graph of a simple function: sin(x*y)](images/cgraph/plot_cgraph_01.png "plot_cgraph_01.png"){ width=30% #fig:cgraph:plot_01 }

The [@fig:cgraph:plot_01] shows an example graph for calculating function `sin (x * y)`.
The generated computation graph contains several pieces of information which are essential for debugging the applications. These information include node index, operation type, reference counter, and shapes of data. In the figure above, we can see the row vector `y` of shape [1; 4] is broadcast on the matrix `x` of shape [8; 4] in `Mul` operation.

### From Dynamic to Static

The computation graph can be either implicitly constructed or explicitly declared in the code. Often, implicit construction is done by operator overloading while explicit declaration uses domain specific languages (DSL). The two methods lead to two different kinds of computation graphs -- *dynamic graph* and *static graph*, each has its own pros and cons.

A dynamic graph is constructed during the runtime. Due to operator overloading, its construction can be naturally blended with a language's native constructs such as `if ... else ...` and `for` loops. This renders greatest flexibility and expressiveness. On the other hand, a static graph needs to be declared using a specific DSL (which has a steeper learning curve). Because the structure of a graph is already known during the compilation phase, there is a great space for optimisation. However, it is sometimes very difficult to use static graphs to express conditions and loops when using with native code together.

As we can see, the flexibility of a dynamic graph comes at the price of lower performance. Facebook's PyTorch and Google's TensorFlow are the typical examples of dynamic and static graph respectively.
Many programmers need to make a choice between these two different types.
A common practice is "using PyTorch at home and using TensorFlow in the company", In other words, PyTorch is preferred for prototyping and TensorFlow is ideal for production use.
(The TensorFlow 2.0 uses eager execution by default, which is easier for users to get start with.)

Owl does something slightly different from these two in order to get the best parts of both worlds.
Owl achieves this by converting a dynamic graph into static one in the runtime. The motivation is based on a very important observation: in many cases, a computation graph is continuously re-evaluated after its construction. This is especially true for those iterative optimisation algorithms. We only update some inputs of the graph in each iteration.

If we know that the graph structure remains the same in every iteration, rather than re-constructing it all the time, we can convert it into a static graph before the iterative evaluation. This is exactly what Owl does. By so doing, the programmer can enjoy the flexibility offered by the dynamic graph construction with operator overloading, but still achieve the best performance from static graph.

Comparing to TensorFlow, the time overhead (for graph conversion and optimisation) is shifted to the runtime in Owl. You may worry about the performance: "Is it going to slow down my fancy DNN application?" The fact is, even for large and complex graphs, this Just-in-Time compilation (JIT) and optimisation are often quite fast. In this [lazy_lstm.ml](https://github.com/owlbarn/owl/blob/master/examples/mnist_cnn.ml) example, there are 15,105 nodes and 21,335 edges. Owl is able to compile the graph within 230ms then optimise it within 210ms. The optimised graph contains only 8,224 nodes, 14,444 edges and runs much faster. Remember that you only need to do it once before training. For smaller networks, it often just takes several milliseconds.

Technically, JIT is very straightforward to implement in Owl's architecture. Given a deep neural network, Owl first runs both forward pass and backward pass. Because of the computation graph, the calculation becomes symbolic and we can obtain the complete computation graph to calculate the loss and gradients of a neural network. We can then pass this static graph to the optimisation engine to optimise. The [Neural Compiler](https://github.com/owlbarn/owl/blob/master/src/base/neural/owl_neural_compiler.ml) functor is parameterised by a computation engine then compiles a DNN definition and training configuration into a device-dependent static graph.


### Significance in Computing

Now that you know the basic ideas of computation graph, you may ask why it matters? Well, the computation graph makes many things a lot easier. Here is an incomplete list of potential benefits:

- simulate lazy evaluation in a language with eager evaluation;
- incremental computation (a.k.a Self-Adjusted Computation);
- reduce computation complexity by optimising the structure of a graph;
- reduce memory management overhead by pre-allocating the space;
- reduce memory footprint by reusing allocated memory space;
- natural support for parallel and distributed computing;
- natural support for heterogeneous computing;
- natural support for symbolic maths.

Some of the benefits are very obvious. Memory usage can certainly be optimised if the graph structure is fixed and the input shapes are known. One optimisation is reusing previously allocated memory, which is especially useful for those applications involving large ndarray calculations. In fact, this optimisation can also be performed by a compiler by tracking the reference number of allocated memory, a technique referred to as linear types.

Some may appear less obvious at the first glance. For example, we can decompose a computation graph into multiple independent subgraphs and each can be evaluated in parallel on different cores or even computers. Maintaining the graph structure also improves fault-tolerance, by providing natural support for rollback mechanisms.

The computation graph provides a way to abstract the flow of computations, therefore it is able to bridge the high-level applications and low-level machinery of various hardware devices. This is why we say it has natural support for heterogeneous computing.

The computation graph has more profound implications. Because the memory allocated for each node is mutable, Algodiff becomes more scalable when evaluating large and complex graphs. At the same time, mutable transformation is handled by Owl so programmers can still write safe functional code.

## Examples

Before diving into the details of the design of the computation graph module, let's first shows some examples of using the CGraph modules and how the computation can be transformed into lazy evaluation.

### Example 01: Basic CGraph

Let's start with a simple operation that adds up one ndarray and one scalar.
Normally with Ndarray module what we do is:

```ocaml
module N = Dense.Ndarray.D
let x = N.ones [|2;2|]
let y = 2.
let g = N.add_scalar x y
```

Now, let's make it into a lazy evaluation calculation with CGraph:

```ocaml env=cgraph:example-01
module N = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.D)
```

The computation graph is designed as a functor stack. A CGraph module can be built based on a ndarray module, since in the end a lazy evaluation still requires specific computation at some point.

```ocaml env=cgraph:example-01
let x = N.var_arr ~shape:[|2;2|] "x"
let y = N.var_elt "y"
let g = N.add_scalar x y
```

Next we define two variables, the first `x` is a ndarray, and `y` is a scalar. At this stage, we only define these two as placeholders with no real data.
Then we use the `add_scalar` function to get another lazy evaluated array `g`.

To get the value of the lazy expression `g`, we need to first assign real values to `x` and `y`:

```ocaml env=cgraph:example-01
let x_val = Dense.Ndarray.D.ones [|2;2|]
let y_val = 2.
let _ = N.assign_arr x x_val
let _ = N.assign_elt y y_val
```

The real values are the familiar dense ndarray and float number.
Note the two different assignment method for ndarray and scalar.
Finally, we can evaluate the ndarray `g`:

```ocaml env=cgraph:example-01
# N.eval_arr [|g|]
- : unit = ()
# N.unpack_arr g
- : Owl_algodiff_primal_ops.D.arr =
   C0 C1
R0  3  3
R1  3  3

```

The `eval_arr` returns nothing. To get the value, we need to use the `unpack_arr` or `unpack_elt` function.

### Example 02: CGraph with AD

In real applications, we normally need to deal with CGraphs that are constructed in the Algorithmic Differentiation process.
Here is an example of using the dense ndarray module to compute the gradients of a function:

```ocaml
include Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.D)

let f x y = Maths.((x * sin (x + x) + ((pack_flt 1.) * sqrt x) / (pack_flt 7.)) * (relu y) |> sum')

let x = Dense.Ndarray.D.ones [|2;2|] |> pack_arr
let y = pack_elt 2.
let z = (grad (f x)) y |> unpack_elt
```

Obviously, it is extremely difficult for the users to manually construct the computation graph that computes the gradient of the function `f`.
Instead, we use the computation graph as the base module to build the Algorithmic Differentiation module:

```ocaml env=cgraph:example-02
module G = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.D)
include Owl_algodiff_generic.Make (G)

let f x y = Maths.((x * sin (x + x) + ((pack_flt 1.) * sqrt x) / (pack_flt 7.)) * (relu y) |> sum')

let x = G.var_arr ~shape:[|2;2|] "x" |> pack_arr
let y = G.var_elt "y" |> pack_elt
let z = (grad (f x)) y
```

Note how the CGraph module are treated as equal to the Ndarray module in building the AD module.
They decide if the AD module uses normal or lazy evaluation.
Now we can evaluate `z` with the approach as before. Or we can use another approach: building a graph based on the input and output.

```ocaml env=cgraph:example-02
let inputs  = [| unpack_arr x |> G.arr_to_node; unpack_elt y |> G.elt_to_node |]
let outputs = [| unpack_elt z |> G.elt_to_node |]
let g = G.make_graph inputs outputs "graph"
```

To build a graph, we need to specify the input and output *nodes*.
Here it might be a bit confusing, since there are two layers of packing and unpacking.
Currently the `x`, `y`, and `z` are both AD values of type `AD.t`, therefore we need `AD.unpack_arr` and `AD.unpack_elt` to make them CGraph lazy array and scalar values.
And then, to build the explicit computation graph, we need to use the `G.arr_to_node` and `G.elt_to_node` functions to make them into graph nodes first.
Finally an explicit computation graph can be built with `make_graph` function.

You might be wondering why bother to build the graph if we can directly evaluate the value `z`.
The reason is that evaluation is not always the target. For example, we often need to visualise the generated computation graph.
Backward mode generates and maintains a computation graph in order to back propagate the error. The computation graph is very helpful in both debugging and understanding the characteristic of your numerical functions.

Owl provides the `graph_to_dot` function to facilitate you in generating computation graphs.
It converts the computation graph into a [dot](https://www.graphviz.org/doc/info/lang.html) format string. The dot file can be visualised with professional tools such as [graphviz](https://www.graphviz.org/).

```text
let s = G.graph_to_dot g
let _ = Owl_io.write_file "cgraph.dot" s
```

The generated computation graph looks like below. The Owl source code contains more examples about visualising a computation graph.

![Computation graph of a simple math function](images/algodiff/plot_028.png "plot 028"){ width=60% #fig:algodiff:plot28 }

Come back to the evaluation of graph. After constructing the graph `g`, we can then assign real data values to the computation graph.
The only difference is that, now we need to first unpack the AD value to CGraph value before assignment:

```text
let x_val = Dense.Ndarray.D.ones [|2;2|]
let y_val = 2.
let _ = G.assign_arr (unpack_arr x) x_val
let _ = G.assign_elt (unpack_elt y) y_val
```

Finally, we can evaluate the whole graph with

```text
G.eval_graph g
```

Since the whole graph is evaluated, then surely the output ndarray `z` is also evaluated. We can first unpack it from AD value into normal CGraph ndarray and then get its value by:

```text
# unpack_elt z |> G.unpack_elt

- : float = 4.20861827873129801
```

### Example 03: CGraph with DNN

Since the optimisation and neural network modules are built on Algorithmic Differentiation module, they can also benefit from the power of CGraph.
Suppose we have a network built of CGraph based neural network `nn`, we can then use the `forward` and `backward` function to get the forward inference and backward propagation computation graph from the neural network graph module, with CGraph array variable.

Actually, for ease of access, Owl has provided another functor to build the neural network module based on the CGraph module:

```ocaml env=cgraph:example-03
module CPU_Engine = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.S)
module CGCompiler = Owl_neural_compiler.Make (CPU_Engine)

open CGCompiler.Neural
open CGCompiler.Neural.Graph
open CGCompiler.Neural.Algodiff

let make_network input_shape =
  input input_shape
  |> lambda (fun x -> Maths.(x / pack_flt 256.))
  |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2|]
  |> dropout 0.1
  |> fully_connected 1024 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network ~name:"mnist"
```

The CGraph-built neural network module does not require any change of code in building the CNN except for the headers.
We can then use the training function in `CGCompiler` module.

```ocaml env=cgraph:example-03
let pack x = CGCompiler.Engine.pack_arr x |> Algodiff.pack_arr

let train network =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  let x = pack x in
  let y = pack y in
  CGCompiler.train network x y
```

And similarly the inference can be done with `CGCompiler.model` function.
You can see that to make the existing DNN programme into lazy evaluation version, all you need to do is to update the header and use packing/unpacking properly for the data.


You might be asking: the lazy evaluation version of neural network looks cool and all, but why do I need it?
That brings to the large performance improvement the CGraph module can bring about to computation.
To motivate you to continue to understand more about the design and optimisation of the CGraph module, you can try to run both [mnist_cnn.ml](https://github.com/owlbarn/owl/blob/master/examples/mnist_cnn.ml) and [lazy_mnist.ml](https://github.com/owlbarn/owl/blob/master/examples/lazy_mnist.ml) then compare their performance.
Both Zoo scripts train the same convolution neural network to recognise the handwritten digits using MNIST datasets in 60 iterations.
On a normal laptop, `mnist_cnn.ml` takes 30s to finish and consumes approximate 4GB memory, whilst `lazy_mnist.ml` only takes 5s and consumes about 0.75GB. `lazy_mnist.ml` achieves the state-of-the-art performance which you can obtain by using TensorFlow (with its recent XLA optimisation), actually Owl runs even faster on 3 out of 4 machines we have tested.

If these numbers make you interested in knowing how the magic happens, let's unveil the underlying mechanism of Owl's computation graph in the following sections.

## Design Rationale

How the computation graph is designed? In the older versions, Algodiff module has some partial support of computation graph in order to perform reverse mode algorithmic differentiation (AD). The full support was only introduced in Owl 0.4.0.

Owl implements the computation graph in a very unique and interesting way. Let's first see several principles which we followed:

- Non-intrusive, the original functor stack should work as it was;
- Transparent to the programmers as much as possible;
- Support both eager and lazy evaluation;
- Flexible enough for future extension on other devices.

The computation graph is implemented in a very self-contained stack. I have devised a good way to "inject" it into Owl's original functor stack. If it sounds too abstract, please have a look at the final product in the following figure.

![Computation graph functor stack in Owl](images/cgraph/owl_cgraph_functor_stack.png "owl_cgraph_functor_stack"){ width=90% #fig:cgraph:functor }


The left figure shows part of Owl's original functor stack, and the right one shows how the current one looks like after injection. We know the functor stack plays a central role in Owl's architecture. In the old design, Ndarray implements a set of fundamental n-dimensional array operations, then Algodiff defines abstract mathematical operations for differentiation, finally Optimise engine glues low-level maths with high-level deep neural network applications. The whole stack is parameterised by the number type abstraction in Ndarray.


- `Ndarray`: provides number type abstraction and implements the fundamental numerical operations.
- `Algodiff`: implements algorithmic differentiation.
- `Optimise`: uses the derivative information to build an optimisation engine.
- `Neural_Neuron`: implements various kinds of neuron functions which can be optimised.
- `Neural_Graph`: connects neurons together to form a network so that we can train a useful model.


The functor stack of computation graph is injected between `Ndarray` and `Algodiff`. *The design principle is that the functor stack of a numerical system should be parameterised by both number type and device type.* Number type provides data representation (real or complex, single or double, row-based or column-based layout, etc.) which decides how a maths construct should be built and operated. Device type provides hardware representation (CPU, GPU, FPGA, etc.) which decides how the computation should be performed on a specific device.

The list below summarises the functionality of each functor. The order and naming of these functors can give you a rough understanding about how it is designed.

- `Device`: device abstraction contains device-dependent types and functions.
- `Type`: type definition of various (mathematical) operations.
- `Shape`: provides the shape inference function in the graph.
- `Symbol`: provides various functions to manipulate symbols.
- `Operator`: implements maths operators (`+`, `-`, `*`, `/`, and etc.) which decide how the symbols should  be connected to form a graph.
- `Optimiser`: optimises the structure of a given graph by searching and optimising various patterns.
- `Graph`: manipulates computation graphs at high level, e.g. visualisation, connecting inputs and outputs.
- `Engine`: evaluates a computation graph on a specific device.


Why the magic can happen? Simply put, the injected computation graph stack provides an abstraction layer similar to symbolic maths. The original eager evaluation becomes symbolic operation (or graph construction) therefore they can be lazily evaluated.

The shape inference functionality is able to infer the data shape of every node in a graph from its input. This allows Owl to calculate how much memory is required to evaluate the graph and pre-allocate this space. Owl can further track the reference number of each function node and reuse the allocated memory as much as possible, which reduces both memory footprint and Garbage Collector (GC) overhead, significantly improves the computation speed.

The Optimiser functor searches for various structural patterns in a graph, removes unnecessary computations and fusing computation nodes if possible. All the patterns are defined in [owl_computation_optimiser.ml](https://github.com/owlbarn/owl/blob/master/src/base/compute/owl_computation_optimiser.ml), and it is very straightforward to plug in more patterns to extend Optimiser's capability. Here are some example patterns.

*Constant folding* is a very basic pattern to reduce graph size. We can pre-calculate some subgraphs. For example, the inputs which node `#241` depends on are all constants, so the value of `#241` is already decided. We can fold all the constants to node `#241` before evaluating the whole graph.

![Optimisation techniques in computation graph: constant folding](images/cgraph/owl_cgraph_opt_0.png "owl_cgraph_opt_0"){ width=90% #fig:cgraph:opt_0 }

*Fusing operations* can effectively reduce the round trips to the memory, which saves a lot of time when operating large ndarrys. In the figure below, nodes `#421`, `#463`, and `#464` are fused into one `fma` node (i.e. fused-multiply-add operation), which also improves numerical accuracy. Owl also recognises quite complicated patterns, e.g. pattern formed by nodes `#511` -- `#515` appears a lot in DNN training that uses Adagrad (Adaptive Subgradient Methods), the Optimiser is able to fuse all these operations into one-pass calculation.

![Optimisation techniques in computation graph: fusing operations](images/cgraph/owl_cgraph_opt_1.png "owl_cgraph_opt_1"){ width=90% #fig:cgraph:opt_1 }

In the next example, the *Adding zero* pattern is firstly detected hence `#164` and `#166` are removed and others are folded. Moreover, nodes `#255` for `repeat` operation is also removed because `add` operation already supports broadcasting operation. Removing `#255` can save some runtime memory in the evaluation.

![Optimisation techniques in computation graph: remove zero](images/cgraph/owl_cgraph_opt_2.png "owl_cgraph_opt_2"){ width=90% #fig:cgraph:opt_2}

To understand how effective the Optimiser works, we present both the [original computation graph](images/cgraph/owl_cgraph_mnist_raw.png) and the [optimised graph](images/cgraph/owl_cgraph_mnist_opt.png) taken from [lazy_mnist.ml](https://github.com/owlbarn/owl/blob/master/examples/lazy_mnist.ml). Comparing to the original network which has 201 nodes, 239 edges, the optimised one contains only 103 nodes, 140 edges.


Engine functor sits on top of the stack. This is where a computation graph finally gets executed. Engine functor contains two sub modules, one for initialising the graph and the other for evaluating graph.

Before we finish this section, we can try the following snippet in `utop`. Both snippets generate a module for DNN applications, the difference is that the first one uses the old stack whereas the second one uses the new stack with computation graph.


```text

  module M =
    Owl_neural_generic.Flatten (
      Owl_neural_graph.Make (
        Owl_neural_neuron.Make (
          Owl_optimise_generic.Make (
            Owl_algodiff_generic.Make (
              Dense.Ndarray.S)))));;

```

For the new stack, we can see it is indeed much deeper.


```text

  module M =
    Owl_neural_generic.Flatten (
      Owl_neural_graph.Make (
        Owl_neural_neuron.Make (
          Owl_optimise_generic.Make (
            Owl_algodiff_generic.Make (
              Owl_computation_engine.Flatten (
                Owl_computation_cpu_engine.Make_Nested (
                  Owl_computation_graph.Make (
                    Owl_computation_optimiser.Make (
                      Owl_computation_operator.Make (
                        Owl_computation_symbol.Make (
                          Owl_computation_shape.Make (
                            Owl_computation_type.Make (
                              Owl_computation_cpu_device.Make (
                                Dense.Ndarray.S))))))))))))));;

```

## Optimisation of CGraph

The design of Owl is often driven by real-world applications.
Besides the MNIST example, we find the image segmentation another challenging application for Owl. Seeking to push the performance of this application, we manage to further optimise the design of CGraph module.
This work is done by Pierre Vandenhove, and you can visit his [report](http://math.umons.ac.be/staff/Vandenhove.Pierre/resources/ocamllabs_internship_report.pdf) for more details.
It starts with the MRCNN-based Object Detection application we introduce in the [Case - Object Detection](https://ocaml.xyz/book/case-obj-detect.html) chapter.
Please refer to this chapter for detail explanation of this application.

The first issue after constructing the network in Owl was that the memory usage, in inference mode, was huge. The network has over 400 layers and to avoid reinitialising the network for every picture, it is good to keep its input size fixed and to resize instead all the images to that size --- a larger size takes more time and memory but yields more accurate results. A reasonable input size for this network is a 1024-pixel-wide square. Unfortunately, obtaining detections for one picture with this size required over 11 GB of RAM, which was too much for a laptop. As a comparison, the TensorFlow implementation only uses 1 GB. There was a big room for improvement!

This is where CGraph comes to rescue.
A computation graph is always directed and acyclic. Representing the structure of a program as a computation graph has several advantages, especially for computationally-intensive code dealing with big multi-dimensional arrays.
A really useful one is that prior to evaluating the nodes, you can optimise the structure of the graph: for instance, useless calculations such as adding an array with nothing but zeros can be removed, common patterns can be merged into one node and executed more efficiently, etc.
This helps a bit: thanks to these
optimisations, the number of nodes of Mask R-CNN drops from 4095 to 3765.
Another really important feature in this case is the ability to pre-allocate a memory space to each node, to decrease the overall memory consumption and reduce the garbage collector overhead.

### Optimising memory with pebbles

To describe the problem of allocating memory in a computation graph, it
is interesting to look at the *pebble game*, which was introduced [in 1973](http://perso.ens-lyon.fr/loris.marchal/scheduling/sethi\_complete\_register\_allocation.pdf)
to explain register allocation.

The *pebble game* is played on a directed acyclic graph. Each node can
store at most one pebble. The game begins with no pebble on any node. At
each step, the player can do one of the following moves:

1.  if a vertex $v$ has no predecessor, the player can place a pebble on
    `v`.
2.  if all predecessors of a vertex $v$ are pebbled, the player can
    place a pebble on `v` or `slide` a pebble from one of its
    predecessors to `v`.
3.  the player can remove any pebble from a vertex (and reuse that
    pebble later).

The goal of the game is to place a pebble at least once on some fixed
output vertices of the graph.

Here is an example of an optimal pebbling strategy using the previous
computation graph (gray nodes are pebbled), using moves 1 -> 2 -> 3 -> 1 ->
2 -> 2. We assume that the goal is to pebble node 5:

![Modelling computation graph memory optimisation problem as a pebble game](images/cgraph/owl_vision_pebble.png){#fig:cgraph:pebble}

This relates to the memory allocation of the computation graph if we see
pebbles as memory blocks used to store the output value of a node. We
assume that the values of the inputs are known (move 1). We can only
compute the value of a vertex if all its predecessors are simultaneously
stored in memory (move 2). The *sliding* move means that the memory of a
node can be overwritten by its successor during its computation
(*inplace reuse*). We can always reuse a memory block from any other
node (move 3). Given a graph, the idea is thus to find a strategy to
pebble it using a minimum number of pebbles (in other words, using as
little memory as possible).

We also want to avoid pebbling any node twice (in order the keep the
execution time as low as possible, because that would mean that we
compute the same node twice). Given these constraints, finding a
strategy using the least amount of pebbles is unfortunately
[NP-complete](http://perso.ens-lyon.fr/loris.marchal/scheduling/sethi_complete_register_allocation.pdf).
Since computation graphs can have a few thousand nodes, we will be
looking for a fast heuristic instead of an exact algorithm.

### Allocation Algorithm

The initially implemented strategy to allocate memory to a node $u$ in
Owl's computation graph module was simply to reuse the memory of a
direct predecessor with same output shape as $u$ when that is possible.
This optimisation decreases the memory consumption of Mask
R-CNN from 11 GB to 7 GB --- much better, but still quite far from the 1
GB of the TensorFlow implementation!

We can actually make it much better by sharing memory between
nodes

-  that are not necessarily a parent/child pair;
-  that do not have the same output size (by allocating a large block
   of memory once, without necessarily using all of it all the time).

To do this efficiently, we first have to fix an evaluation order (in
practice, any topological order). Given this order, we can pinpoint the
moment when the memory of a node becomes useless by keeping a counter of
how many times it has been used. When it has been used by all its
children, we can recycle its memory. Then to allocate memory to a node,
we simply check which blocks are available and we select the one with
the closest size (in order not to waste too much memory). If no block is
available, we allocate a new one. This can be executed in
$\mathcal{O}(n * \log(n))$ time, which is negligible compared to the
actual cost of evaluating the graph.


Then we just have to be careful that some operations cannot overwrite
their inputs while they are being computed (the *sliding* move from the
pebble game is forbidden) and that some nodes cannot be overwritten for
practical purposes (typically constant nodes or neural network weights).
Implementing this effectively reduced the memory consumption of Mask
R-CNN from 7 GB to 1 GB for a 1024x1024 picture, making it as efficient
as the TensorFlow implementation! A summary of the changes can be found
in [this pull request](https://github.com/owlbarn/owl/pull/318). Here
are some more statistics illustrating what the computation graph with
this new algorithm achieves:

  -------------- ----------- ----------------------- ------------- -----------
  Architecture   Time        Time with CG (building  Memory        Memory with
                 without CG  + evaluating) (s)       without CG    CG (MB)
                 (s)                                 (MB)

  InceptionV3    0.565       0.107 + 0.228 = 0.335   625.76        230.10

  ResNet50       0.793       0.140 + 0.609 = 0.749   1309.9        397.07

  MNIST          20.422      0.144 + 10.920 = 11.064 3685.3        895.32
  (training)

  Mask R-CNN     11.538      0.363 + 8.379 = 8.742   6483.4        870.48
  -------------- ----------- ----------------------- ------------- -----------
  : Evaluation of the effect of CGraph memory optimisation using different DNN architectures {#tbl:cgraph:perf}

InceptionV3 and ResNet50 networks are tested with a 299x299 image; Mask R-CNN is
tested with a 768x768 image. The MNIST line refers to a small neural
network trained to recognize hand-written digits whose implementation
can be found [in this code
repository](https://github.com/owlbarn/owl/blob/master/examples/lazy_mnist.ml).
The time is the average over 30 evaluations, without reusing
pre-computed nodes when a computation graph is used. The graph building
phase includes graph construction, optimisation and memory
initialisation. The memory is the maximum resident set size of the
program. This was evaluated on a laptop with an Intel i5-6300HQ and 8 GB
of RAM.

For instance, when evaluated in the right order, the following computation graph, which can be used to recognise hand-written digits, needs only two different blocks of memory (each colour corresponds to a memory block, white nodes always need to be kept in memory).
Part of the generated computation graph is shown in [@fig:cgraph:lazy].

![Optimised memory allocation](images/cgraph/owl_vision_lazymnistinf_small.png "allocation"){width=50% #fig:cgraph:lazy}

You can find bigger visualisations of the allocation performed by the
new algorithm in this [link](https://drive.google.com/drive/folders/12KCY9OC6GjuHiH2pRiAjqNi-pz2sNcc1?usp=sharing).
You can also check [this page](http://demo.ocaml.xyz/mrcnn.html) for a demo of this Owl-powered network.
If you want to apply it on videos, large images or experiment a bit more, see the [GitHub repository](https://github.com/pvdhove/owl-mask-rcnn).
Pre-trained weights on 80 classes of common objects are provided, which have been converted from the TensorFlow implementation mentioned above.

## As Intermediate Representations

Programming a GPU is very much like programming a computer cluster. The gain of parallel computing comes with inevitable synchronisation and communication overhead. Therefore GPU computing only makes sense when the computation complexity is high enough to dwarf other overhead.

When offloading the computation to a GPU, we should avoid transmitting data back and forth between the host and the device memory, so eager evaluation is not ideal in this context because the performance will be throttled by copying. This is the gap between CPU computing and a language with eager evaluation. Computation graph essentially fills the gap between Owl and GPU computing simply because the laziness can be simulated now.

From implementation perspective, we only need to write a new engine functor for GPU devices to evaluate a graph; all the others remain the same. I am currently working on the [OpenCL engine](https://github.com/owlbarn/owl/blob/master/src/opencl/compute/owl_computation_opencl_engine.ml). The amount of code for implementing OpenCL engine is surprisingly small, only around 700 ~ 900 LOC. Comparing to the [CPU engine](https://github.com/owlbarn/owl/blob/master/src/base/compute/owl_computation_cpu_engine.ml), the OpenCL engine maintains the memory allocated on both host and device for each node, copying only happens whenever it is necessary, the allocated memory on the device is reused as much as possible.


## Summary

In this chapter, we have introduced the core Computation Graph module in Owl.
We start with the general introduction of the computation graph in numerical computing and why we build that in Owl.
Then we use several examples to demonstrate how the computation graph module is used in Owl.
This is followed by the internal design of this module, most importantly the CGraph stack and its position in the Owl architecture.
The computation graph creates a large optimisation space, and this chapter we present one of them in detail, which is to use the pebble game to optimise the memory allocation in Owl computation.

The computation graph is a hot research topic, and there is still much we can do to improve Owl's performance based on this module.
For example, the Neural Compiler still takes extra time to convert and optimise a graph. Both tasks can actually be moved into compilation phase using MetaOCaml, which will squeeze out some extra performance gain for us.
