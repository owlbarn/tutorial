(alias
 (name runtest)
 (deps
  (:x README.md)
  (:y2 ../../examples/code/dataframe/example_00.ml)
  (:y1 ../../examples/code/dataframe/example_01.ml)
  (:y0 ../../examples/code/dataframe/example_02.ml)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml --direction=to-md %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y2} %{y2}.corrected)
   (diff? %{y1} %{y1}.corrected)
   (diff? %{y0} %{y0}.corrected))))
