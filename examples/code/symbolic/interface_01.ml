open Owl_symbolic 

type t = Onnx_types.model_proto

let of_symbolic = ONNX_Engine.of_symbolic
let to_symbolic = ONNX_Engine.to_symbolic
let save = ONNX_Engine.save
let load = ONNX_Engine.load