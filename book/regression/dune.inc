(alias
 (name runtest)
 (deps
  (:x README.md)
  (:y0 ../../examples/code/linear-algebra/example_00.ml)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml --direction=to-md %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y0} %{y0}.corrected))))
