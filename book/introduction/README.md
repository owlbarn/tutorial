Introduction
===========================================================

Currently, this page is used to illustrate how to embed OCaml code into tutorial book.

The following code snippet embeds and evaluates some shell code.

```sh
$ echo "Test shell command"
Test shell command
```

The following code snippet embeds and evaluates OCaml code.

```ocaml
# print_endline "hello world"
hello world
- : unit = ()

# Mat.uniform 4 4;;
- : Mat.mat =

         C0       C1        C2       C3 
R0 0.378545 0.861025  0.712662 0.563556 
R1 0.964339 0.582878  0.834786 0.722758 
R2 0.265025 0.712912 0.0894476  0.13984 
R3 0.475555 0.616536  0.202631 0.983487 

```

The following code snippet embeds the code from a file.

```ocaml file=../../examples/code/introduction/hello.ml
let hello () = print_endline "test ..."
```

Similarly, the following snippet also loads the code from a file.

```ocaml file=../../examples/code/introduction/world.ml
let test () = print_endline "again ..."
let test2 () = print_endline "test 02 ..."
```

Finished.
