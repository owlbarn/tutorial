(alias
 (name runtest)
 (deps
  (:x README.md)
  (:y0 ../../examples/code/aeos/interface_00.mli)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y0} %{y0}.corrected))))
