# Internal Utility Modules

During development of Owl, we find some utility modules are immensely handy. 
In this chapter, we share some of them. 
These are not the main feature of Owl, and perhaps you can implement your own version very quickly. 
But we hope to present how these features are used in Owl. 

## Dataset Module

The dataset modules provides easy access to various datasets to be used in Owl, mainly the MNSIT and CIFAR10 datasets.
You can get all these data in Owl by executing: `Dataset.download_all ()`.
The data are downloaded in the home directory, for example,  `~/.owl/dataset` on Linux.

### MNIST

The [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits has a training set of 60,000 examples,
and a test set of 10,000 examples. Each example is of size 28 x 28.
It is a good starting point for deep neural network related tasks. 

You can get MNIST data via these Owl functions:

- `Dataset.load_mnist_train_data ()`: returns a triplet `x, y, y'`.
  + `x` is a [60000, 784] ndarray (`Owl_dense_ndarray.S.mat`) where each row represents a [28, 28] image.
  + `y` is a [60000, 1] label ndarray. Each row is an integer ranging from 0 to 9,
  indicating the digit on each image.
  + `y'` is a [60000, 10] label ndarray. Each one-hot row vector corresponds to
  one label.

- `Dataset.load_mnist_test_data ()`: returns a triplet.
  Similar to `load_mnist_train_data`, only that it returns the test set, so
  the example size is 10,000 instead of 60,000.

- `Dataset.load_mnist_train_data_arr ()`: similar to `load_mnist_train_data`,   but returns `x` as [60000,28,28,1] ndarray

- `Dataset.load_mnist_test_data_arr ()`: similar to
  `load_mnist_train_data_arr`, but it returns the test set, so the example size
  is 10, 000 instead of 60, 000.

- `Dataset.draw_samples x y n` draws `n` random examples from images ndarray `x` and label ndarray `y`.  

Here is what the dataset looks like when loaded into Owl:

```
# let x, _, y = Dataset.load_mnist_train_data_arr ();;
val x : Owl_dense_ndarray.S.arr =

                C0
      R[0,0,0]   0
      R[0,0,1]   0
      R[0,0,2]   0
      R[0,0,3]   0
      R[0,0,4]   0
               ...
R[59999,27,23]   0
R[59999,27,24]   0
R[59999,27,25]   0
R[59999,27,26]   0
R[59999,27,27]   0

val y : Owl_dense_matrix.S.mat =

        C0  C1  C2  C3  C4  C5  C6  C7  C8  C9
    R0   0   0   0   0   0   1   0   0   0   0
    R1   1   0   0   0   0   0   0   0   0   0
    R2   0   0   0   0   1   0   0   0   0   0
    R3   0   1   0   0   0   0   0   0   0   0
    R4   0   0   0   0   0   0   0   0   0   1
       ... ... ... ... ... ... ... ... ... ...
R59995   0   0   0   0   0   0   0   0   1   0
R59996   0   0   0   1   0   0   0   0   0   0
R59997   0   0   0   0   0   1   0   0   0   0
R59998   0   0   0   0   0   0   1   0   0   0
R59999   0   0   0   0   0   0   0   0   1   0
```

You can find the MNIST dataset used in training and testing a DNN in Owl:

```
let train network =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  Graph.train network x y |> ignore;
  network

let test network =
  let imgs, _, labels = Dataset.load_mnist_test_data () in
  let m = Dense.Matrix.S.row_num imgs in
  let imgs = Dense.Ndarray.S.reshape imgs [|m;28;28;1|] in

  let mat2num x = Dense.Matrix.S.of_array (
      x |> Dense.Matrix.Generic.max_rows
        |> Array.map (fun (_,_,num) -> float_of_int num)
    ) 1 m
  in

  let pred = mat2num (Graph.model network imgs) in
  let fact = mat2num labels in
  let accu = Dense.Matrix.S.(elt_equal pred fact |> sum') in
  Owl_log.info "Accuracy on test set: %f" (accu /. (float_of_int m))
```

### CIFAR-10

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) include small scale color images for more realistic complex image classification tasks.
It includes 10 classes of images: aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
It consists of 60,000 32 x 32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Due to the limit of file size on Github, the training set is cut into 5 smaller batch. You can get CIFAR-10 data using `Owl`:

- `Dataset.load_cifar_train_data batch`: returns a triplet `x, y, y'`.
  + The input paramter `batch` can range from 1 to 5, indicating which training set batch to choose.
  + `x` is an [10000, 32, 32, 3] ndarray (`Owl_dense_ndarray.S.arr`). The last
  dimension indicates color channels (first Red, then Green, finally Blue).
  + `y` is an [10000, 1] label ndarray, each number representing
  an image class.
  + `y'` is the corresponding [10000, 10] one-hot label ndarray.  

- `Dataset.load_cifar_test_data ()`: similar to `load_cifar_train_data`, only
  that it loads test data.

- `Dataset.draw_samples_cifar x y n` draws `n` random examples from images ndarray `x` and label ndarray `y`.

Note that all elements in the loaded matrices and ndarrays are of `float32` format.

The CIFAR10 dataset is similar to MNIST in training DNN:

```
let train network =
  let x, _, y = Dataset.load_cifar_train_data 1 in
  Graph.train network x y
```

### Text Datasets


## Graph Module

The Graph module in Owl provides a general data structure to manipulate graphs. It is defined as:

```ocaml
type 'a node =
  { mutable id : int
  ; (* unique identifier *)
    mutable name : string
  ; (* name of the node *)
    mutable prev : 'a node array
  ; (* parents of the node *)
    mutable next : 'a node array
  ; (* children of the node *)
    mutable attr : 'a (* indicate the validity *)
  }
```

The attribution here is generic so that you can define your own graph where each node contains an integer, a string, or any data type you define. 
This make the graph module extremely flexible. 

Graph module provides a rich set of APIs.
First, you can build a Graph using these methods:

- `node ~id ~name ~prev ~next attr` creates a node with given id and name
string. The created node is also connected to parents in ``prev`` and children
in ``next``. The ``attr`` will be saved in ``attr`` field.

- `connect parents children` connects a set of parents to a set of children.
The created links are the Cartesian product of parents and children. In other
words, they are bidirectional links between parents and children.
Note that this function does not eliminate any duplicates in the array.

- `connect_descendants parents children` connects parents to their children.
This function creates unidirectional links from parents to children. In other
words, this function save `children` to `parent.next` field.

- `connect_ancestors parents children` connects children to their parents.
This function creates unidirectional links from children to parents. In other
words, this function save `parents` to `child.prev` field.

- `remove_node x` removes node `x` from the graph by disconnecting itself
from all its parent nodes and child nodes.

- `remove_edge src dst` removes a link `src -> dst` from the graph. Namely,
the corresponding entry of `dst` in `src.next` and `src` in `dst.prev`
will be removed. Note that it does not remove [dst -> src] if there exists one.

- `replace_child x y` replaces `x` with `y` in `x` parents. Namely, `x`
parents now make link to `y` rather than `x` in `next` field.
Note that the function does note make link from `y` to `x` children. Namely,
the `next` field of `y` remains intact.


Then, to obtain and update properties of a graph:

```
val id : 'a node -> int
(** ``id x`` returns the id of node ``x``. *)

val name : 'a node -> string
(** ``name x`` returns the name string of node ``x``. *)

val set_name : 'a node -> string -> unit
(** ``set_name x s`` sets the name string of node ``x`` to ``s``. *)

val parents : 'a node -> 'a node array
(** ``parents x`` returns the parents of node ``x``. *)

val set_parents : 'a node -> 'a node array -> unit
(** ``set_parents x parents`` set ``x`` parents to ``parents``. *)

val children : 'a node -> 'a node array
(** ``children x`` returns the children of node ``x``. *)

val set_children : 'a node -> 'a node array -> unit
(** ``set_children x children`` sets ``x`` children to ``children``. *)

val attr : 'a node -> 'a
(** ``attr x`` returns the ``attr`` field of node ``x``. *)

val set_attr : 'a node -> 'a -> unit
(** ``set_attr x`` sets the ``attr`` field of node ``x``. *)
```

Similarly, you can get other properties of a graph use the other functions: 

- `indegree x` returns the in-degree of node 
- `outdegree x` returns the out-degree of node 
- `degree x` returns the total number of links of `x`
- `num_ancestor x` returns the number of ancestors of `x`
- `num_descendant x` returns the number of descendants of `x`
- `length x` returns the total number of ancestors and descendants of `x`

Finally, we provide functions for traversing the graph in either Breadth-First order or Depth-First order. 
You can also choose to iterate the descendants or ancestors of a given node. 

```
val iter_ancestors
  :  ?order:order
  -> ?traversal:traversal
  -> ('a node -> unit)
  -> 'a node array
  -> unit
(** Iterate the ancestors of a given node. *)

val iter_descendants
  :  ?order:order
  -> ?traversal:traversal
  -> ('a node -> unit)
  -> 'a node array
  -> unit
(** Iterate the descendants of a given node. *)

val iter_in_edges : ?order:order -> ('a node -> 'a node -> unit) -> 'a node array -> unit
(** Iterate all the in-edges of a given node. *)

val iter_out_edges : ?order:order -> ('a node -> 'a node -> unit) -> 'a node array -> unit
(** Iterate all the out-edges of a given node. *)

val topo_sort : 'a node array -> 'a node array
(** Topological sort of a given graph using a DFS order. Assumes that the graph is acyclic.*)
```

You can also use functions: `filter_ancestors`, `filter_descendants`, `fold_ancestors`, `fold_descendants`, `fold_in_edges`, and `fold_out_edges` to perform fold or filter operations when iterating the graph.

Within Owl, the Graph module is heavily use to facilitate the Computation Graph module. 

TODO: Explain how it is used in CGraph.

TODO: Use examples and text, not just code.

## Stack and Heap Modules

Both *Stack* and *Heap* are two common abstract data types for collection of elements.
They are also used in Owl code.
Similar to graph, they use generic types so that any data type can be plugged in. 
Here is the definition of a stack:

```
type 'a t =
  { mutable used : int
  ; mutable size : int
  ; mutable data : 'a array
  }
```

The stack and heap modules support four standards operations:

- `push`: push element into a stack/heap
- `pop`: pops the top element in stack/heap. It returns None if the container is empty
- `peek`: returns the value of top element in stack/heap but it does not remove the element from the stack. `None` is returned if the container is empty.
- `is_empty`: returns true if the stack/heap is empty

The stack data structure is used in many places in Owl:
the `filteri` operation in ndarray module,
the topological sort in graph,
the reading file IO operation for keeping the content from all the lines,
the data frame...
The heap structure is used in key functions such as `search_top_elements` in the Ndarray module, which searches the indices of top values in input ndarray according to the comparison function.

## Count-Min Sketch

