# Symbolic Maths

## Introduction

- The background
  * Owl, numerical computing, OCaml ecosystem. Its powerful. de facto etc.
  * Neural compilers
- The inefficiency
  * The other side of the world: symbolic computation.
  * Also, symbolic manipulation is not limited to this.
  * We have computation graph, but limited to ...
- My work and solution
  * A pure-ocaml impl. based on owl-base. Symbolic maniputation, and multiple engines for executing the core symbolic layer.
- The result
  - This would enable a series of computation in Owl.

It starts with tf-graph.
Works fine. [reference]
We find this approach is limited.

The we find a standaline generic symbolic representation could be used



## Design

- Core
  - operations classification. Each module contains these common attributes, and also extra attributes such as axis.
    - input ops: int, float, tensor, variable, random, PI
  - naming
  - data structure: graph
  - shape checking and inferencing
  - types
  - The difference of `owl_graph` input and input names in each module.
  - multiple outputs

- Engines
  Based on this simple core abstraction, we use different *engines* to provide functinalities: ONNX, Owl, LaTeX, pprint, etc.
  We provide each module an `attr` as extension points. Defined as follow.


  The unified interfaces of each engine: of/to_symbolic.

## ONNX Engine

The ONNX Engine is the current focus of development in `owl_symbolic`.
[ONNX](https://onnx.ai) is a widely adopted open neural network exchange format. A neural network model defined in ONNX can be, via suitable converters, can be run on different frameworks and thus hardware accelerators.
The main target of ONNX is to promote the interchangeability of neural network and machine learning models, but it is worthy of noting that the standard covers a lot of basic operations in scientific computation, such as power, logarithms, trigonometric functions, etc.
Therefore, ONNX engines serves as a good starting point for its coverage of operations.

Taking a symbolic graph as input, how would then the ONNX engine produce ONNX model? We use the [ocaml-protoc](https://github.com/mransan/ocaml-protoc), a protobuf compiler for OCaml, as the tool. The ONNX specification is defined in a [onnx.proto](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto) file, and the `ocaml-protoc` can compile this protobuf files into OCaml types along with serialisation functions for a variety of encodings.

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

```ocaml

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
Besides the meta information such as model version and IR version etc., a model is mainly a graph, which include input/output information and an array of nodes.
A node specifies operator type, input and output node name, and its own attributions, such as the `axis` attribution in reduction operations.

Therefore, all we need is to build up a `model_proto` data structure gradually from attributions to nodes, graph and model. It can then be serialised using `encode_model_proto` to generate a protobuf format file, and that is the ONNX model we want.

Besides building up the model, one other task to be performed in the engine is type checking and type inferencing. The [operator documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md) list the type constraints of each operator. For example, the sine function can only accept input of float or double number types, and generate the same type of input as that of input.
Each type of operator has its own rules of type checking and inferencing. Starting from input nodes, which must contain specific type information, this chain if inferencing can thus verify the whole computation meets the type constraints for each node, and then yield the final output types of the whole graph.
The reason that type checking is performed at the engine side instead of the core is that each engine may have different type constraints and type inferencing rules for the operators.

### Example 1: Basic operations

Let's look at an simple example.

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

After including necessary library component, the first three line of code creates a symbolic representation `z` using the symbolic operators such as `sin`, `pow` and `float`. The `x` and `y` are variables that accept user input. It is then used to be create a symbolic graph. This step mainly check if there is any duplication of node names.
Then the `of_symbolic` function in ONNX engine takes the symbolic graph as input, and generate a `model_proto` data structure, which can be further saved as a model named `test.onnx`.

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
This script is very simple: it loads the ONNX model we have just created, and then get the two input variables, and assign two values to them in the `sess.run` command. All the user need to know in advance is that there are two input variables in this ONNX model. Note that not only we could define scalar type input, but also tensor type variables in `owl_symbolic`, and then assign NumPy array to them when evaluating.


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

Besides this simple DNN, we have also created the complex artchitectures such as ResNet, InceptionV3, SqueezeNet, etc.
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

For simplicity, we generate an dummy input for the execution/inference phase of this model.
Of course, currently in our model the weight data is not trained.
The training of a model is completed on a framework such as TensorFlow, and combining trained weight data into the ONNX model remains to be a future work.

Furthermore, by using tools such as `js_of_ocaml`, we can convert both examples into JavaScript; executing them can create the ONNX models, which in turn can be executed on the browser using [ONNX.js](https://github.com/microsoft/onnxjs) that utilises WebGL.
In summary, using ONNX as the intermediate format for exchange computation across platforms enables numerous promising directions.

## LaTeX Engine


## Owl Engine


## Conclusion
