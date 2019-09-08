Introduction
-----------------------------------------------------------

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
