# Owl Online Tutorial

The main purpose of the online tutorial is for teaching how to use Owl software. The tutorial also convers many useful numerical and system techniques in numerical computing. Furthermore, the tutorial includes in-depth discussion on the lessons learnt by building a large and complex software system to enable scientific computing in OCaml.

The tooling and template reuse those in [Real World OCaml](https://realworldocaml.org/) under original authors' permission, with minor modidications.


## Compile

- `make` uses docker container to build the book.
- `make test` synchronises with Owl API by evaluating the code snippet in the tutorial book.
- `make compile` generates html files.
- Edit `book/toc.scm` to add more chapters. Do not edit `book/tune` and `static/dune` directly.
- Refer to [RWO](https://github.com/realworldocaml/book/blob/master/README.md) for details.

Note that tooling is not finished at the moment. Structure and tools are copied mostly from RWO book.


## Contribute

Currently contribution to the book is mainly in the form of Pull Request. 
Normally you only need to change one of the `README.md` files in `book/` directory, though adding inline scripts requires some special care.
Please make sure that your local changes compile without any error, and include both the change of markdown file in `docs/` directory in a PR. 


## Tooling

The following tools are used in the project, please refer to their documentation.

- [Pandoc](https://pandoc.org/MANUAL.html)
- [Pandoc-crossref](http://lierdakil.github.io/pandoc-crossref/)