# Introduction


## Installation

Owl requires OCaml `>=4.06.0`. Please make sure you have a working OCaml environment before you start installing Owl. Here is a guide on [Install OCaml](https://ocaml.org/docs/install.html).

Owl's installation is rather trivial. There are four possible ways as shown below, from the most straightforward one to the least one.


### Option 1: Install from OPAM

Thanks to the folks in [OCaml Labs](http://ocamllabs.io/), OPAM makes package management in OCaml much easier than before. You can simply type the following in the command line to start.

```shell

  opam depext owl
  opam install owl

```

If you want to try the newest development features, I recommend the other ways to install Owl, as below.

In case of linking issues, known to happen on `ubuntu`-based distribution. You will need to compile `openblas` by hand, and use the appropriate environment variables to point at your newly compiled library. You can use `[owl's docker file](https://github.com/owlbarn/owl/blob/master/docker/Dockerfile.ubuntu) as a reference for this.


### Option 2: Pull from Docker Hub

[Owl's docker image](https://hub.docker.com/r/ryanrhymes/owl/) is perfectly synced with master branch. The image is always automatically built whenever there are new commits. You can check the building history on [Docker Hub](https://hub.docker.com/r/ryanrhymes/owl/builds/).

You only need to pull the image then start a container.

```shell

  docker pull ryanrhymes/owl
  docker run -t -i ryanrhymes/owl

```

Besides the complete Owl system, the docker image also contains an enhanced OCaml toplevel - `utop`. You can start `utop` in the container and try out some examples.

The source code of Owl is stored in `/root/owl` directory. You can modify the source code and rebuild the system directly in the started container.


### Option 3: Pin the Dev-Repo

`opam pin` allows you to pin the local code to Owl's development repository on Github. The first command `opam depext` installs all the dependencies Owl needs.

```shell

  opam depext owl
  opam pin add owl --dev-repo

```


### Option 4: Compile from Source

This is my favourite option. First, you need to clone the repository.

```shell

  git clone git@github.com:ryanrhymes/owl.git

```

Second, you need to figure out the missing dependencies and install them.

```shell

  jbuilder external-lib-deps --missing @install

```

Last, this is perhaps the most classic step.

```shell

  make && make install

```

If your OPAM is older than `V2 beta4`, you need one extra steps. This is due to a bug in OPAM which copies the compiled library into `/.opam/4.06.0/lib/stubslibs` rather than `/.opam/4.06.0/lib/stublibs`. If you don't upgrade OPAM, then you need to manually move `dllowl_stubs.so` file from `stubslib` to `stublib` folder, then everything should work.


### CBLAS/LAPACKE Dependency

The most important dependency is [OpenBLAS](https://github.com/xianyi/OpenBLAS). Linking to the correct OpenBLAS is the key to achieve the best performance. Depending on the specific platform, you can use `yum`, `apt-get`, `brew` to install the binary format. For example on my Mac OSX, the installation looks like this:

```shell

  brew install homebrew/science/openblas

```

However, installing from OpenBLAS source code leads to way better performance in my own experiment. OpenBLAS already contains an implementation of LAPACKE, as long as you have a Fortran complier installed on your computer, the LAPACKE will be compiled and included in the installation automatically.

Another benefit of installing from OpenBLAS source is: some systems' native package management tool installs very old version of OpenBLAS which misses some functions Owl requires.


## Interacting with Owl

Owl is well integrated with `utop`. You can use `utop` to try out the experiments in our tutorials. If you want `utop` to automatically load Owl for you, you can also edit `.ocamlinit` file in your home folder by adding the following lines. (Note that the library name is `owl` with lowercase `o`.)

```ocaml
# #require "owl_top"
```

The `owl_top` is the toplevel library of Owl, it automatically loads `owl` core library and installs the corresponding pretty printers of various data types.

**(duplicate)**

There are several ways to interact with Owl system. The most classic one is to write an OCaml application, compile the code, then run it natively on a computer. You can also skip the compilation step, and use Zoo system to run the code as a script.

However, the easiest way for a beginner to try out Owl is REPL (Read–Eval–Print Loop), or an interactive toplevel. The toplevel offers a convenient way to play with small code snippets. The code run in the toplevel is compiled into bytecode rather than native code. Bytecode often runs much slower than native code. However, this has very little impact on Owl's performance because all its performance-critical functions are implemented in C language.

In the following, I will introduce two options to set up an interactive environment for Owl.



### Using Toplevel

OCaml language has bundled with a simple toplevel, but I recommend to use *utop* as a more advance replacement. Installing *utop* is straightforward in OPAM, simply run the following command in the system shell.


```shell

  opam install utop

```

After installation, you can load Owl in *utop* with the following commands. `owl-toplevel` is Owl's toplevel library which will automatically load several dependent libraries (including `owl-zoo`, `owl-base`, and `owl` core library) to set up a complete numerical environment.

```ocaml
#  #require "owl-top"
#  open Owl
```

If you do not want to type these commands every time you start *toplevel*, you can add them to `.ocamlinit` file. The toplevel reads `.ocamlinit` file when it starts and uses it to initialise the environment. This file is often stored in the home directory on your computer.



### Using Notebook

Jupyter Notebook is a popular way to mix presentation with interactive code execution. It originates in Python world but is widely supported by various languages. One attractive feature of notebook is that it uses client/server architecture and runs in a browser.

If you want to know how to use a notebook and its technical details, please read [Jupyter Documentation](http://jupyter.org/documentation). Here let me show you how to set up a notebook to run Owl step by step.

Run the following commands in the shell will install all the dependency for you. This includes Jupyter Notebook and its [OCaml language extension](https://github.com/akabe/ocaml-jupyter).


```shell

  pip install jupyter
  opam install jupyter
  jupyter kernelspec install --name ocaml-jupyter "$(opam config var share)/jupyter"

```

To start a Jupyter notebook, you can run this command. The command starts a local server running on `http://127.0.0.1:8888/`, then opens a tab in your browser as the client.


```shell

  jupyter notebook

```

If you want to run a notebook server remotely, please refer to ["Running a notebook server"](http://jupyter-notebook.readthedocs.io/en/stable/public_server.html). If you want to set up a server for multiple users, please refer to [JupyterHub](https://jupyterhub.readthedocs.io/en/latest/) system.

When everything is up and running, you can start a new notebook in the web interface. In the new notebook, you must run the following OCaml code in the first input field to load Owl environment.


```ocaml
# #use "topfind"
# #require "owl-top, jupyter.notebook"
```

At this point, a complete Owl environment is set up in the Jupyter Notebook, and you are free to go with any experiments you like.

For example, you can simply copy & paste the whole [lazy_mnist.ml](https://github.com/owlbarn/owl/blob/master/examples/lazy_mnist.ml) to train a convolutional neural network in the notebook. But here, let us just use the following code.


```ocaml env=intro_00
# #use "topfind"
# #require "owl-top, jupyter.notebook"

# open Owl
# open Neural.S
# open Neural.S.Graph
# open Neural.S.Algodiff

# let make_network input_shape =
    input input_shape
    |> lambda (fun x -> Maths.(x / F 256.))
    |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
    |> max_pool2d [|2;2|] [|2;2|]
    |> dropout 0.1
    |> fully_connected 1024 ~act_typ:Activation.Relu
    |> linear 10 ~act_typ:Activation.(Softmax 1)
    |> get_network
val make_network : int array -> network = <fun>
```

Calling the function ...

```ocaml env=intro_00
# make_network [|28;28;1|]
- : network =
18839

[ Node input_0 ]:
    Input : in/out:[*,28,28,1]
    prev:[] next:[lambda_1]

[ Node lambda_1 ]:
    Lambda       : in:[*,28,28,1] out:[*,28,28,1]
    customised f : t -> t
    prev:[input_0] next:[conv2d_2]

[ Node conv2d_2 ]:
    Conv2D : tensor in:[*;28,28,1] out:[*,28,28,32]
    init   : tanh
    params : 832
    kernel : 5 x 5 x 1 x 32
    b      : 32
    stride : [1; 1]
    prev:[lambda_1] next:[activation_3]

[ Node activation_3 ]:
    Activation : relu in/out:[*,28,28,32]
    prev:[conv2d_2] next:[maxpool2d_4]

[ Node maxpool2d_4 ]:
    MaxPool2D : tensor in:[*,28,28,32] out:[*,14,14,32]
    padding   : SAME
    kernel    : [2; 2]
    stride    : [2; 2]
    prev:[activation_3] next:[dropout_5]

[ Node dropout_5 ]:
    Dropout : in:[*,14,14,32] out:[*,14,14,32]
    rate    : 0.1
    prev:[maxpool2d_4] next:[fullyconnected_6]

[ Node fullyconnected_6 ]:
    FullyConnected : tensor in:[*,14,14,32] matrix out:(*,1024)
    init           : standard
    params         : 6423552
    w              : 6272 x 1024
    b              : 1 x 1024
    prev:[dropout_5] next:[activation_7]

[ Node activation_7 ]:
    Activation : relu in/out:[*,1024]
    prev:[fullyconnected_6] next:[linear_8]

[ Node linear_8 ]:
    Linear : matrix in:(*,1024) out:(*,10) 
    init   : standard
    params : 10250
    w      : 1024 x 10
    b      : 1 x 10
    prev:[activation_7] next:[activation_9]

[ Node activation_9 ]:
    Activation : softmax 1 in/out:[*,10]
    prev:[linear_8] next:[]


```

Jupyter notebook should nicely print out the structure of the neural network.


.. figure:: ../figure/jupyter_example_01.png
   :scale: 50 %
   :align: center
   :alt: jupyter example 01


Second example demonstrates how to plot figures in notebook. Because Owl's Plot module does not support in-memory plotting, the figure needs to be written into a file first then passed to `Jupyter_notebook.display_file` to render.


```text

  #use "topfind";;
  #require "owl-top, jupyter.notebook";;
  open Owl;;

  (* Plot a normal figure using Plot *)

  let f x = Maths.sin x /. x in
  let h = Plot.create "plot_003.png" in
  Plot.set_foreground_color h 0 0 0;
  Plot.set_background_color h 255 255 255;
  Plot.set_title h "Function: f(x) = sine x / x";
  Plot.set_xlabel h "x-axis";
  Plot.set_ylabel h "y-axis";
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.plot_fun ~h f 1. 15.;
  Plot.output h;;

  (* Load into memory and display in Jupyter *)

  Jupyter_notebook.display_file ~base64:true "image/png" "plot_003.png"

```

Then we can see the plot is correctly rendered in the notebook running in your browser. Plotting capability greatly enriches the content of an interactive presentation.


.. figure:: ../figure/jupyter_example_02.png
   :scale: 50 %
   :align: center
   :alt: jupyter example 02


### Using owl-jupyter

There is a convenient library `owl-jupyter` specifically for running Owl in a notebook. The library is a thin wrapper of `owl-top`. The biggest difference is that it overwrites `Plot.output` function so the figure is automatically rendered in the notebook without calling `Jupyter_notebook.display_file`.

This means that all the plotting code can be directly used in the notebook without any modifications. Please check the following example and compare it with the previous plotting example, we can see `display_file` call is saved.


.. code-block:: ocaml

  #use "topfind";;
  #require "owl-jupyter";;
  open Owl_jupyter;;

  let f x = Maths.sin x /. x in
  let g x = Maths.cos x /. x in
  let h = Plot.create "" in
  Plot.set_foreground_color h 0 0 0;
  Plot.set_background_color h 255 255 255;
  Plot.set_pen_size h 3.;
  Plot.plot_fun ~h f 1. 15.;
  Plot.plot_fun ~h g 1. 15.;
  Plot.output h;;


One thing worth noting is that, if you pass in empty string in `Plot.create` function, the figure is only rendered in the browser. If you pass in non-empty string, then the figure is both rendered in the browser and saved into the file you specified. This is to guarantee `output` function has the consistent behaviour when used in or out of a notebook.


.. figure:: ../figure/jupyter_example_03.png
   :scale: 50 %
   :align: center
   :alt: jupyter example 03
