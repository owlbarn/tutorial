(alias
 (name runtest)
 (deps
  (:x README.md)
  (:y4 ../../examples/code/ndarray/interface_00.mli)
  (:y3 ../../examples/code/ndarray/interface_01.mli)
  (:y2 ../../examples/code/ndarray/interface_02.mli)
  (:y1 ../../examples/code/ndarray/interface_03.mli)
  (:y0 ../../examples/code/ndarray/interface_04.mli)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y4} %{y4}.corrected)
   (diff? %{y3} %{y3}.corrected)
   (diff? %{y2} %{y2}.corrected)
   (diff? %{y1} %{y1}.corrected)
   (diff? %{y0} %{y0}.corrected))))

(alias
 (name runtest-all)
 (deps
  (:x README.md)
  (:y4 ../../examples/code/ndarray/interface_00.mli)
  (:y3 ../../examples/code/ndarray/interface_01.mli)
  (:y2 ../../examples/code/ndarray/interface_02.mli)
  (:y1 ../../examples/code/ndarray/interface_03.mli)
  (:y0 ../../examples/code/ndarray/interface_04.mli)
  prelude.ml)
 (action
  (progn
   (run ocaml-mdx test --prelude=prelude.ml --non-deterministic %{x})
   (diff? %{x} %{x}.corrected)
   (diff? %{y4} %{y4}.corrected)
   (diff? %{y3} %{y3}.corrected)
   (diff? %{y2} %{y2}.corrected)
   (diff? %{y1} %{y1}.corrected)
   (diff? %{y0} %{y0}.corrected))))
