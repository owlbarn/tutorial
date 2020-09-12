# Symbolic Representation


## Introduction

The development of `owl_symbolic` library is motivated by multiple factors.
For one thing, scientific computation can be considered as consisting of two broad categories: numerical computation, and symbolic computation. Owl has achieved a solid foundation in the former, but as yet to support the latter one, which is heavily utilised in a lot of fields.
For another, with the development of neural network compilers such as [TVM](https://tvm.apache.org/), it is a growing trend that the definition of computation can be separated out, and the low level compilers to deal with optimisation and code generation etc. to pursue best computation performance.
Besides, tasks such as visualising a computation also require some form or intermediate representation (IR).
Owl has already provided a computation graph layer to separate the definition and execution of computation to improve the performance, but it's not an IR layer to perform these different tasks as mentioned before.
Towards this end, we begin to develop an intermediate symbolic representation of computations and facilitate various tasks based on this symbol representation.

One thing to note is that do not mistake our symbolic representation as the classic symbolic computation (or Computer Algebra System) that manipulate mathematical expressions in a symbolic way, which is similar to the traditional manual computations.
It is indeed one of our core motivation to pursue the symbolic computation with Owl. 
Currently we provide a symbolic representation layer as the first step towards that target.
More discussion will be added in future versions with the development with the support of symbolic math in Owl. 

## Design

`owl_symbolic` is divided into two parts: the core symbolic representation that constructs a symbolic graph, and various engines that perform different task based on the graph.
The architecture design of this system is shown in [@fig:symbolic:architecture].

![Architecture of the symbolic system](images/symbolic/architecture.png "architecture"){width=90% #fig:symbolic:architecture}

The core abstraction is a independent symbolic representation layer.
Based on this layer, we have various engines that can be translated to and from this symbolic representation.
Currently we support three engines: the ONNX binary format, the computation graph in Owl, and the LaTeX string. 
The CAS engine is currently still an on-going research project, and we envision that, once finished, this engine can be used to pre-process a symbolic representation so that it as an simplified canonical form before being processed by other engines. 


### Core abstraction

The core part is designed to be minimal and contains only necessary information.
Currently it has already covered many common computation types, such as math operations, tensor manipulations, neural network specific operations such as convolution, pooling etc.
Each symbol in the symbolic graph performs a certain operation.
Input to a symbolic graph can be constants such as integer, float number, complex number, and tensor. The input can also be variables with certain shapes. An empty shape indicates a scalar value. The users can then provide values to the variable after the symbolic graph is constructed.

**Symbol**

The symbolic representation is defined mainly as array of `symbol`.
Each `symbol` is a graph node that has an attribution of type ` Owl_symbolic_symbol.t`.
It means that we can traverse through the whole graph by starting with one `symbol`.
Besides symbols, the `name` field is the graph name, and `node_names` contains all the nodes' name contained in this graph.

```ocaml
type symbol = Owl_symbolic_symbol.t Owl_graph.node

type t =
  { mutable sym_nodes : symbol array
  ; mutable name : string
  ; mutable node_names : string array
  }
```

Let's look at `Owl_symbolic_symbol.t`. It defines all the operations contained in the symbolic representation:

```
type t =
  | NOOP
  | Int                   of Int.t
  | Complex               of Complex.t
  | Float                 of Float.t
  | Tensor                of Tensor.t
  | Variable              of Variable.t
  | RandomUniform         of RandomUniform.t
  | Sin                   of Sin.t
  | Cos                   of Cos.t
  | Exp                   of Exp.t
  | ReduceSum             of ReduceSum.t
  | Reshape               of Reshape.t
  | Conv                  of Conv.t
  ....
```

There are totally about 150 operations included in our symbolic representation. 
Each operation is implemented as a module. These modules share common attributes such as name, input operation names, output shapes, and then each module contains zero or more attributes of itself.
For example, the `Sin` operation module is implemented as:


```
module Sin = struct
  type t =
    { mutable name : string
    ; mutable input : string array
    ; mutable out_shape : int array option array
    }

  let op_type = "Sin"

  let create ?name x_name =
    let input = [| x_name |] in
    let name = Owl_symbolic_utils.node_name ?name op_type in
    { name; input; out_shape = [| None |] }
end
```

The module provides properties such as `op_type` and functions such as `create` that returns object of type `Sin.t`.
The `name`, `input` and `out_shape` are common attributes in the operation modules. 

In implementing the supported operations, we follow the category used in ONNX. These operations can be generally divided into these different groups:

- Generators: operations that generate data, taking no input. For example, the `Int`, `Float`, `Tensor`, `Variable`, etc. 
- Logical: logical operations such as `Xor`.
- Math: mathematical operations. This group of operations makes a large part of the total operations supported. 
- Neural Network: neural network related operations such as convolution and pooling. 
- Object detection: also used in neural network, but the operations that are closely related with object detection applications, including `RoiAlign` and `NonMaxSuppression`.
- Reduction: reduction (or folding) math operations such as sum reduce.
- RNN: Recurrent neural network related operations such as LTSM.
- Tensor: Normal tensor operations, like the ones that are included in the Ndarray module, such as `concat`, `reshape`, etc.
- Sequence: take multiple tensor as one single object called `sequence`, and there are different corresponding functions on the sequence type data, such as `SequenceInsert`, `SequenceLength` etc.


Based on these operation modules, we provide several functions on the `Owl_symbolic_symbol.t` type:

- `name`: get the name of operation
- `op_type`: get the operation type string
- `input`: get the input nodes name of an operation
- `set_input`: update the input nodes name 
- `output`: get the output nodes name 
- `set_output`: update the output nodes name 

There are also some functions that only apply to certain types of operations. 
The generator type of operations all need to specify the type of data it supports. Therefore, we use `dtype` function to check their data types.
Another example is the `output` property. For most of the operation, it has only one output, and therefore its name is its output name.
However, for operations such as `MaxPool` that contains multiple output, we need another function: `output`. 

**Type Checking**

The type supported by `owl_symbolic` is listed as follows:
```ocaml
type number_type =
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
  | SNT_SEQ of number_type
```
This list of types covers most number and non-number types. `SNT_SEQ` means the type a list of the basic elements as inputs/outputs.

**Operators**

All these operations are invisible to users. 
What the users really uses is the *operators*. 
To build a graph, we first need to build the required attributes into an operation, and then put it into a graph node. This is what an operator does.
Take the `sin` operator as an example:

```
let sin ?name x =
  let xn = Owl_symbolic_graph.name x in
  let s = Owl_symbolic_ops_math.Sin.create ?name xn in
  make_node (Owl_symbolic_symbol.Sin s) [| x |]
```

Here the `sin` operator takes its parent node `x` as input, get its name as input property, and create a symbol node with the function `make_node`.
This function takes an operation and an array of parent symbols, and then creates one symbol as return.
What it does is mainly creating a child node using the given operation as node attribution, updating the child's input and output shape, and then connecting the child with parents before returning the child node.
The connection is on both direction:

```
connect_ancestors parents [| child |];
let uniq_parents = Owl_utils_array.unique parents in
Array.iter (fun parent -> connect_descendants [| parent |] [| child |]) uniq_parents
```

Therefore, the users can use the operators to build an graph representation, here is an example:

```ocaml
open Owl_symbolic
open Op
open Infix

let x = variable "x_0"
let y = exp ((sin x ** float 2.) + (cos x ** float 2.))
    + (float 10. * (x ** float 2.))
    + exp (pi () * complex 0. 1.)
```

Here we start with the `variable` operator, which creates a placeholder for incoming data later. 
You can specify the shape of the variable with `~shape` parameter. If not specified, then it defaults to a scalar.
You can also choose to initialise this variable with a *tensor* so that even if you don't feed any data to the variable, the default tensor value will be used. 
A tensor in `owl-symbolic` is defined as:

```ocaml
type tensor =
  { mutable dtype : number_type
  ; mutable shape : int array
  ; mutable str_val : string array option
  ; mutable flt_val : float array option
  ; mutable int_val : int array option
  ; mutable raw_val : bytes option
  }
```

A tensor is of a specific type of data, and then it contains the value: string array, float array, integer array, or bytes.
Only one of these fields can be used. 
If initialised with a tensor, a variable takes the same data type and shape as that of the tensor.

**Naming**

Currently we adopt a global naming scheme, which is to add an incremental index number after each node's type. For example, if we have an `Add` symbol, a `Div` symbol, and then another `Add` symbol in a graph, then each node will be named `add_0`, `div_1`, and `add_1`.
One exception is the variable, where a user has to explicitly name when create a variable. Of course, users can also optionally any node in the graph, but the system will check to make sure the name of each node is unique.
The symbolic graph contains the `node_names` field that include all the nodes' names in the graph.

**Shape Inferencing**

One task the symbolic core needs to perform is shape checking and shape inferencing. 
Shape inference is performed in the `make_node` function and therefore happens every time a user uses an operation to construct a symbolic node and connect it with previous nodes. It is assumed that the parents of the current node are already known. 

```
let (in_shapes : int array option array array)= 
  Array.map (fun sym_node -> 
    Owl_graph.attr sym_node |> Owl_symbolic_symbol.out_shape
  ) parents 
  in
let (shape : int array option array) = 
  Owl_symbolic_shape.infer_shape in_shapes sym
...
```

As the code shows, for each node, we first find the output shapes of its parents.
The `in_shape` is of type `int array option array array`. 
You can understand it this way: `int array` is a shape array; `int array option` means this shape could be `None`.Then `int array option array` is one whole input from previous parent, since one parent may contains multiple outputs.
Finally, `int array option array array` includes output from all parents.
The main function `Owl_symbolic_shape.infer_shape` then infer the output shape of current node, and save it to the `out_shape` property of that symbol.

The `infer_shape` function itself check the symbol type and then match with specific implementation. 
For example, a large number of operations actually takes one parent and keep its output shape:

```
let infer_shape input_shapes sym =
  | Sin _ -> infer_shape_01 input_shapes
  | Exp _ -> infer_shape_01 input_shapes
  | Log _ -> infer_shape_01 input_shapes
....

let infer_shape_01 input_shapes =
  match input_shapes.(0).(0) with
  | Some s -> [| Some Array.(copy s) |]
  | None   -> [| None |]
```

This pattern `infer_shape_01` covers these operations. It simply takes the input shape, and returns the same shape. 

There are two possible reasons for the input shape to be `None`.
At first each node will be initialised with `None` output shape. 
During shape inference, in certain cases, the output shape depends on the runtime content of input nodes, not just the shapes of input nodes and attributions of the currents node. 
In that case, the output shapes is set to `None`.
Once the input shapes contain `None`, the shape inference results hereafter will all be `None`, which means the output shapes cannot be decided at compile time.

**Multiple output**

Most of the operators are straightforward to implement, but some of them returns multiple symbols as return. 
In that case, an operation returns not a node, but a tuple or, when output numbers are uncertain, an array of nodes.
For example, the `MaxPool` operation returns two outputs, one is the normal maxpooling result, and the other is the corresponding tensor that contains indices of the selected values during pooling.
Or we have the `Split` operation that splits a tensor into a list of tensors, along the specified axis. It returns an array of symbols. 

### Engines

Based on this simple core abstraction, we use different *engines* to provide functionalities: converting to and from other computation expression formats, print out to human-readable format, graph optimisation, etc.
As we have said, the core part is kept minimal. If the engines require information other than what the core provides, each symbol has an `attr` property as extension point.

All engines must follow the signature below:

```ocaml file=../../examples/code/symbolic/interface_01.mli
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

For example, the toplevel message type in onnx.proto is `ModelProto`, defined as follows:

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


### Example 2: Variable Initialisation

We can initialise the variables with tensor values so that these default values are used even if no data are passed in.
Here is one example:

```ocaml
open Owl_symbolic
open Op

let _ =
  let flt_val = [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t = Type.make_tensor ~flt_val [| 2; 3 |] in
  let x = variable ~init:t "X" in
  let y = sin x in
  let g = SymGraph.make_graph [| y |] "sym_graph" in
  let z = ONNX_Engine.of_symbolic g in
  ONNX_Engine.save z "test.onnx"
```

This computation simply takes an input variable `x` and then apply the `sin` operation.
Let's look at the Python side.

```python
import numpy as np 
import onnxruntime as rt

sess = rt.InferenceSession("test.onnx")
pred_onx = sess.run(None, input_feed={})
print(pred_onx[0])
```

The expected output is:

```python
[[ 0.84147096  0.9092974   0.14112   ]
 [-0.7568025  -0.9589243  -0.2794155 ]]
```

Note how the initializer works without user providing any input in the input feed dictionary.
Of course, the users can still provide their own data to this computation, but the mechanism may be a bit different.
For example, in `onnx_runtime`, using `sess.get_inputs()` gives an empty set this time.
Instead, you should use `get_overridable_initializers()`:

```python
input_x = sess.get_overridable_initializers()[0]
input_name_x = input_x.name 
input_shape_x = input_x.shape
x = np.ones(input_shape_x, dtype="float32")
pred_onx = sess.run(None, {input_name_x: x})
```

### Example 3: Neural network

The main purpose of the ONNX standard is for expressing neural network models, and we have already cover most of the common operations that are required to construct neural networks.
However, to construct a neural network model directly from existing `owl_symbolic` operations requires a lot of details such as input shapes or creating extra nodes.
For example, if you want to build a neural network with operators directly, you need to write something like:

```
let dnn =
  let x = variable ~shape:[| 100; 3; 32; 32 |] "X" in
  let t_conv0 = conv ~padding:Type.SAME_UPPER x
      (random_uniform ~low:(-0.138) ~high:0.138 [| 32; 3; 3; 3 |]) in
  let t_zero0 =
    let flt_val = Array.make 32 0. in
    let t = Type.make_tensor ~flt_val [| 32 |] in
    tensor t
  in
  let t_relu0 = relu (t_conv0 + t_zero0) in
  let t_maxpool0, _ = maxpool t_relu0 ~padding:VALID ~strides:[| 2; 2 |] [| 2; 2 |] in
  let t_reshape0 = reshape [| 100; 8192 |] t_maxpool0 in
  let t_rand0 = random_uniform ~low:(-0.0011) ~high:0.0011 [| 8192; 512 |] in
  ....
```

Apparently that's too much information for the users to handle.
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
It's design is simple, mainly about matching symbol type and project it to correct implementation.
Again, let's look at an example that builds up a symbolic representation of a calculation $\exp(\sin(x_0) ^ 2 + \cos(x_0) ^ 2) + 10 \times x_0 ^ 2 + \exp(\pi~i)$

```ocaml env=symbolic:latex-engine
open Owl_symbolic
open Op
open Infix

let make_expr0 () =
  let x = variable "x_0" in
  let y =
    exp ((sin x ** float 2.) + (cos x ** float 2.))
    + (float 10. * (x ** float 2.))
    + exp (pi () * complex 0. 1.)
  in
  SymGraph.make_graph [| y |] "sym_graph"
```

This expression can be converted into a corresponding LaTeX string:

```ocaml env=symbolic:latex-engine
# let () = make_expr0 () 
    |> LaTeX_Engine.of_symbolic 
    |> print_endline
\exp(\sin(x_0) ^ 2 + \cos(x_0) ^ 2) + 10 \times x_0 ^ 2 + \exp(\pi \times 1.00i)
```

Simply putting it in the raw string form is not very helpful for visualisation.
We have built a web UI in this Engine that utilises [KaTeX](https://katex.org/), which renders LaTeX string directly on a browser.
Below we use the `html` function provided by the engine to show this string on our web UI using the functionality the engine provides.

```ocaml env=symbolic:latex-engine
# let () = 
    let exprs = [ make_expr0 () ] in 
    LaTeX_Engine.html ~dot:true ~exprs "example.html"
```

The generated "example.html" webpage is a standalone page that contains all the required scripts. Once opened in a browser, it looks like this:

![UI of LaTeX engine](images/symbolic/latex_01.png){ width=90% #fig:symbolic:ui}

For each expression, the web UI contains its rendered LaTeX form and corresponding computation graph.

## Owl Engine

An Owl Engine enables converting Owl computation graph to or from a symbolic representation. Symbolic graph can thus benefit from the concise syntax and powerful features such as Algorithm Differentiation in Owl.

The conversion between Owl CGraph and the symbolic representation is straightforward, since both are graph structures.
We only need to focus on make the operation projection between these two system correct.

```
let cnode_attr = Owl_graph.attr node in
match cnode_attr.op with
| Sin -> Owl_symbolic_operator.sin ~name sym_inputs.(0)
| Sub -> Owl_symbolic_operator.sub ~name sym_inputs.(0) sym_inputs.(1)
| SubScalar -> Owl_symbolic_operator.sub ~name sym_inputs.(0) sym_inputs.(1)
| Conv2d (padding, strides) ->
    let pad =
      if padding = SAME then Owl_symbolic_types.SAME_UPPER else Owl_symbolic_types.VALID
    in
    Owl_symbolic_operator.conv ~name ~padding:pad ~strides sym_inputs.(0) sym_inputs.(1)
```

The basic logic is simple: find the type of symbol and its input node in CGraph, and then do the projection to symbolic representation.
For most of the math operators such as `sin`, the projection is one-to-one, but that's not all the cases.
For some operations such as subtraction, we have `Sub`, `SubScalar` and `ScalarSub` etc. depending on the type of input, but they can all be projected to the `sub` operator in symbolic representation.
Or for the convolution operation, we need to first convert the parameters in suitable way before the projection.

Let's see an example of using the Owl engine:

```ocaml env=symbolic:owl-engine
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

let g = make_graph () |> OWL_Engine.to_symbolic 
```

Here we build a simple computation graph with the algorithmic differentiation module in Owl.
Then we perform the conversion by calling `OWL_Engine.to_symbolic`.

We can also chain multiple engines together. For example, we can use Owl engine to converge the computation define in Owl to symbolic graph, which can then be converted to ONNX model and get executed on multiple frameworks.
Here is such an example. A simple computation graph created by `make_graph ()` is processed by two chained engines, and generates an ONNX model.


```ocaml env=symbolic:owl-engine
let _ =
  let k = make_graph () |> OWL_Engine.to_symbolic |> ONNX_Engine.of_symbolic in
  ONNX_Engine.save k "test.onnx"
```

And this `test.onnx` file can further be processed with Python code as introduced in the previous section.

## Summary
