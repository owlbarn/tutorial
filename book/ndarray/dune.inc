(alias
 (name   runtest)
 (deps   (:x README.md)
         (:y0 ../../examples/code/ndarray/interface_00.mli)
         prelude.ml)
 (action (progn
           (run ocaml-mdx test --prelude=prelude.ml --direction=to-md %{x})
           (diff? %{x} %{x}.corrected)
           (diff? %{y0} %{y0}.corrected))))
(alias
 (name   runtest-all)
 (deps   (:x README.md)
         (:y0 ../../examples/code/ndarray/interface_00.mli)
         prelude.ml)
 (action (progn
           (run ocaml-mdx test --prelude=prelude.ml --direction=to-md --non-deterministic %{x})
           (diff? %{x} %{x}.corrected)
           (diff? %{y0} %{y0}.corrected))))
