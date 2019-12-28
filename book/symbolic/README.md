# Symbolic Maths

## Introduction

The development of `owl_symbolic` library is motivated by multiple factors.
For one thing, scientific computation can be considered as consisting of two broad categories: numerical computation, and symbolic computation. Owl has achieved a solid foundation in the former, but as yet to support the latter one, which is heavily utilised in a lot of fields.
For another, with the development of neural network compilers such as [TVM](https://tvm.apache.org/), it is a growing trend that the definition of computation can be separated out, and the low level compilers to deal with optimisation and code generation etc. to pursue best computation performance.
Besides, tasks such as visualising a computation also require some form or intermediate representation (IR).
Owl has already provided a computation graph layer to separate the definition and execution of computation to improve the performance, but it's not an IR layer to perform these different tasks as mentioned before.
Towards this end, we begin to develop an intermediate symbolic representation of computations and facilitate various tasks based on this symbol representation.


## Design

`owl_symbolic` is divided into two parts: the core symbolic representation that constructs a symbolic graph, and various engines that perform different task based on the graph.

### Core abstraction

The core part is designed to be minimal and contains only necessary information.
Currently it has already covered many common computation types, such as math operations, tensor manipulations, neural network specific operations such as convolution, pooling etc.
Each symbol in the symbolic graph performs a certain operation.
Input to a symbolic graph can be constants such as integer, float number, complex number, and tensor. The input can also be variables with certain shapes. An empty shape indicates a scalar value. The users can then provide values to the variable after the symbolic graph is constructed.

Each operation is implemented as a module. These modules share common attributes such as name, input operation names, output shapes, and then each module contains zero or more attributes of itself.
The graph is implemented using Owl's `Owl_graph` data structure, with a symbol as attribution of a node in `Owl_graph`.

Currently we adopt a global naming scheme, which is to add an incremental index number after each node's type. For example, if we have an `Add` symbol, a `Div` symbol, and then another `Add` symbol in a graph, then each node will be named `add_0`, `div_1`, and `add_1`.
One exception is the variable, where a user has to explicitly name when create a variable. Of course, users can also optionally any node in the graph, but the system will check to make sure the name of each node is unique.

One task the symbolic core needs to perform is shape checking and shape inferencing. The type supported by `owl_symbolic` is listed as follows:
```ocaml
type elem_type =
  | SNT_Noop
  | SNT_Float
  | SNT_Double
  | SNT_Complex32
  | SNT_Complex64
  | SNT_Bool
  | SNT_String
  | SNT_Int8
  | SNT_Int16
  | SNT_Int32
  | SNT_Int64
  | SNT_Uint8
  | SNT_Uint16
  | SNT_Uint32
  | SNT_Uint64
  | SNT_Float16
  | SNT_SEQ of elem_type
```
This list of types covers most number and non-number types. `SNT_SEQ` means the type a list of the basic elements as inputs/outputs.
Type inference happens every time a user uses an operation to construct a symbolic node and connect it with previous nodes. It is assumed that the parents of the current node are already known. The inferenced output shape is saved in each node.
In certain rare cases, the output shape depends on the runtime content of input nodes, not just the shapes of input nodes and attributions of the currents node. In that case, the output shapes is set to `None`.
Once the input shapes contain `None`, the shape inference results hereafter will all be `None`, which means the output shapes cannot be decided at compile time.

The core part provides symbolic operations as user interface.
Each operation constructs a `symbol` and creates a `symbol Owl_graph.node` as output.
Some symbol generates multiple outputs. In that case, an operation returns not a node, but a tuple or, when output numbers are uncertain, an array of nodes.


### Engines

Based on this simple core abstraction, we use different *engines* to provide functionalities: converting to and from other computation expression formats, print out to human-readable format, graph optimisation, etc.
As we have said, the core part is kept minimal. If the engines require information other than what the core provides, each symbol has an `attr` property as extension point.

All engines must follow the signature below:

```text
type t

val of_symbolic : Owl_symbolic_graph.t -> t
val to_symbolic : t -> Owl_symbolic_graph.t
val save : t -> string -> unit
val load : string -> t
```

It means that, each engine has its own core type `t`, be it a string or another format of graph, and it needs to convert `t` to and from the core symbolic graph type, or save/load a type `t` data structure to file. 
An engine can also contain extra functions besides these four.

Now that we have explained the design of `owl_symbolic`, let's look at the details of some engines in the next few sections.

## ONNX Engine

The ONNX Engine is the current focus of development in `owl_symbolic`.
[ONNX](https://onnx.ai) is a widely adopted open neural network exchange format. A neural network model defined in ONNX can be, via suitable converters, can be run on different frameworks and thus hardware accelerators.
The main target of ONNX is to promote the interchangeability of neural network and machine learning models, but it is worthy of noting that the standard covers a lot of basic operations in scientific computation, such as power, logarithms, trigonometric functions, etc.
Therefore, ONNX engines serves as a good starting point for its coverage of operations.

Taking a symbolic graph as input, how would then the ONNX engine produce ONNX model? We use the [ocaml-protoc](https://github.com/mransan/ocaml-protoc), a protobuf compiler for OCaml, as the tool. The ONNX specification is defined in an [onnx.proto](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto) file, and the `ocaml-protoc` can compile this protobuf files into OCaml types along with serialisation functions for a variety of encodings.

For example, the toplevel message type in onnx.proto is `MessageProto`, defined as follows:

```proto
message ModelProto {
  optional int64 ir_version = 1;
  repeated OperatorSetIdProto opset_import = 8;
  optional string producer_name = 2;
  optional string producer_version = 3;
  optional string domain = 4;
  optional int64 model_version = 5;
  optional string doc_string = 6;
  optional GraphProto graph = 7;
  repeated StringStringEntryProto metadata_props = 14;
};
```

And the generated OCaml types and serialisation function are:

```ocaml file=../../examples/code/symbolic/interface_00.mli
open Owl_symbolic_specs.PT

type model_proto =
  { ir_version : int64 option
  ; opset_import : operator_set_id_proto list
  ; producer_name : string option
  ; producer_version : string option
  ; domain : string option
  ; model_version : int64 option
  ; doc_string : string option
  ; graph : graph_proto option
  ; metadata_props : string_string_entry_proto list
  }

val encode_model_proto : Onnx_types.model_proto -> Pbrt.Encoder.t -> unit
```
Besides the meta information such as model version and IR version etc., a model is mainly a graph, which includes input/output information and an array of nodes.
A node specifies operator type, input and output node name, and its own attributions, such as the `axis` attribution in reduction operations.

Therefore, all we need is to build up a `model_proto` data structure gradually from attributions to nodes, graph and model. It can then be serialised using `encode_model_proto` to generate a protobuf format file, and that is the ONNX model we want.

Besides building up the model, one other task to be performed in the engine is type checking and type inferencing. The [operator documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md) lists the type constraints of each operator. For example, the sine function can only accept input of float or double number types, and generate the same type of input as that of input.
Each type of operator has its own rules of type checking and inferencing. Starting from input nodes, which must contain specific type information, this chain if inferencing can thus verify the whole computation meets the type constraints for each node, and then yield the final output types of the whole graph.
The reason that type checking is performed at the engine side instead of the core is that each engine may have different type constraints and type inferencing rules for the operators.

### Example 1: Basic operations

Let's look at a simple example.

```ocaml
open Owl_symbolic
open Op
open Infix

let x = variable "X"
let y = variable "Y"
let z = exp ((sin x ** float 2.) + (cos x ** float 2.)) + (float 10. * (y ** float 2.))
let g = SymGraph.make_graph [| z |] "sym_graph"
let m = ONNX_Engine.of_symbolic g
let _ = ONNX_Engine.save m "test.onnx"
```

After including necessary library component, the first three line of code creates a symbolic representation `z` using the symbolic operators such as `sin`, `pow` and `float`. The `x` and `y` are variables that accept user input. It is then used to create a symbolic graph. This step mainly checks if there is any duplication of node names.
Then the `of_symbolic` function in ONNX engine takes the symbolic graph as input, and generates a `model_proto` data structure, which can be further saved as a model named `test.onnx`.

To use this ONNX model we could use any framework that supports ONNX. Here we use the Python-based [ONNX Runtime](https://github.com/microsoft/onnxruntime) as an example. We prepare a simple Python script as follows:

```python
import numpy as np
import math
import onnxruntime as rt

sess = rt.InferenceSession("test.onnx")
input_name_x = sess.get_inputs()[0].name
input_name_y = sess.get_inputs()[1].name
x = np.asarray(math.pi, dtype="float32")
y = np.asarray(3., dtype="float32")

pred_onx = sess.run(None, {input_name_x: x, input_name_y: y})[0]
print(pred_onx)
```
This script is very simple: it loads the ONNX model we have just created, and then get the two input variables, and assign two values to them in the `sess.run` command. All the user need to know in advance is that there are two input variables in this ONNX model. Note that we could define not only scalar type input but also tensor type variables in `owl_symbolic`, and then assign NumPy array to them when evaluating.


### Example 2: Neural network

The main purpose of the ONNX standard is for expressing neural network models, and we have already cover most of the common operations that are required to construct neural networks.
However, to construct a neural network model directly from existing `owl_symbolic` operations requires a lot of details such as input shapes or creating extra nodes.
To make things easier for the users, we create neural network layer based on existing symbolic operations. This light-weight layer takes only 180 LoC, and yet it provides a Owl-like clean syntax for the users to construct neural networks. For example, we can construct a MNIST-DNN model:

```ocaml
open Owl_symbolic_neural_graph
let nn =
  input [| 100; 3; 32; 32 |]
  |> normalisation
  |> conv2d [| 32; 3; 3; 3 |] [| 1; 1 |]
  |> activation Relu
  |> max_pool2d [| 2; 2 |] [| 2; 2 |] ~padding:VALID
  |> fully_connected 512
  |> activation Relu
  |> fully_connected 10
  |> activation (Softmax 1)
  |> get_network

let _ =
  let onnx_graph = Owl_symbolic_engine_onnx.of_symbolic nn in
  Owl_symbolic_engine_onnx.save onnx_graph "test.onnx"
```

Besides this simple DNN, we have also created the complex architectures such as ResNet, InceptionV3, SqueezeNet, etc.
They are all adapted from existing Owl DNN models with only minor change.
The execution of the generated ONNX model is similar:

```python
import numpy as np
import onnxruntime as rt

sess = rt.InferenceSession("test.onnx")
input_name_x = sess.get_inputs()[0].name
input_name_shape = sess.get_inputs()[0].shape
input_x = np.ones(input_name_shape , dtype="float32")
pred_onx = sess.run(None, {input_name_x: input_x})[0]
```

For simplicity, we generate a dummy input for the execution/inference phase of this model.
Of course, currently in our model the weight data is not trained.
Training of a model should be completed on a framework such as TensorFlow. Combining trained weight data into the ONNX model remains to be a future work.

Furthermore, by using tools such as `js_of_ocaml`, we can convert both examples into JavaScript; executing them can create the ONNX models, which in turn can be executed on the browser using [ONNX.js](https://github.com/microsoft/onnxjs) that utilises WebGL.
In summary, using ONNX as the intermediate format for exchange computation across platforms enables numerous promising directions.

## LaTeX Engine

The LaTeX engine takes a symbolic representation as input, and produce LaTeX strings which can then be visualised using different tools.
For example, we have built a web UI in this Engine that utilises [KaTeX](https://katex.org/), which renders LaTeX string directly on a browser.
Below is an example, where we define an math symbolic graph, convert it into LaTeX string, and show this string on our web UI using the functionality the engine provides.

```ocaml
# open Owl_symbolic
# open Op
# open Infix

# let make_expr0 () =
    let x = variable "x_0" in
    (* construct *)
    let y =
      exp ((sin x ** float 2.) + (cos x ** float 2.))
      + (float 10. * (x ** float 2.))
      + exp (pi () * complex 0. 1.)
    in
    SymGraph.make_graph [| y |] "sym_graph"
val make_expr0 : unit -> Owl_symbolic_graph.t = <fun>
# let () = make_expr0 () 
    |> LaTeX_Engine.of_symbolic 
    |> print_endline
\exp(\sin(x_0) ^ 2 + \cos(x_0) ^ 2) + 10 \times x_0 ^ 2 + \exp(\pi \times 1.00i)
# let () = 
    let exprs = [ make_expr0 () ] in 
    LaTeX_Engine.html ~dot:true ~exprs "example.html"
```

The generated "example.html" webpage is a standalone page that contains all the required scripts. Once opened in a browser, it looks like this:

![](images/symbolic/latex_01.png)

For each expression, the web UI contains its rendered LaTeX form and corresponding computation graph.

## Owl Engine

An Owl Engine enables converting Owl computation graph to or from a symbolic representation. Symbolic graph can thus benefit from the concise syntax and powerful features such as Algorithm Differentiation in Owl.

We can also chain multiple engines together. For example, we can use Owl engine to converge the computation define in Owl to symbolic graph, which can then be converted to ONNX model and get executed on multiple frameworks.
Here is such an example. A simple computation graph created by `make_graph ()` is processed by two chained engines, and generates an ONNX model.

```ocaml
open Owl_symbolic
module G = Owl_computation_cpu_engine.Make (Owl_algodiff_primal_ops.S)
module AD = Owl_algodiff_generic.Make (G)
module OWL_Engine = Owl_symbolic_engine_owl.Make (G)

let make_graph () =
  let x = G.ones [| 2; 3 |] |> AD.pack_arr in
  let y = G.var_elt "y" |> AD.pack_elt in
  let z = AD.Maths.(sin x + y) in
  let input = [| AD.unpack_elt y |> G.elt_to_node |] in
  let output = [| AD.unpack_arr z |> G.arr_to_node |] in
  G.make_graph ~input ~output "graph"

let _ =
  let k = make_graph () |> OWL_Engine.to_symbolic |> ONNX_Engine.of_symbolic in
  ONNX_Engine.save k "test.onnx"
```

## Algebraic Simplification


## Conclusion
