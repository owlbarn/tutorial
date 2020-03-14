(alias
 (name runtest)
 (deps
  (:x README.md)
  (:y1 ../../examples/code/symbolic/interface_00.mli)
  (:y0 ../../examples/code/symbolic/interface_01.mli)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y1} %{y1}.corrected)
   (diff? %{y0} %{y0}.corrected))))
