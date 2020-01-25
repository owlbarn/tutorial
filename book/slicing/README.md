# Slicing and Broadcasting

TODO: rewrite with visual illustration.

Indexing and slicing is arguably the most important and fundamental functions in any numerical library. The flexible design can significantly simplify the code and enables us to write concise algorithms. In this chapter, I will present how to use slicing function in Owl.

Before we start, let's clarify some things.

* Slicing refers to the operation that extracts part of the data from an ndarrays or a matrix according to the well-defined **slice definition**.

* Slicing can be applied to all the dense data structures, i.e. both ndarrays and matrice.

* Slice definition is an `index list` which clarifies **what indices** should be accessed and in **what order** for each dimension of the passed in variable.

* There are two types of slicing in Owl: `basic slicing` and `fancy slicing`. The difference between the two is how the slice is define.



## Basic Slicing

For basic slicing, each dimension in the slice definition must be defined in the format of **[start:stop:step]**. Owl provides two functions `get_slice` and `set_slice` to retrieve and assign slice values respectively.

```text

  val get_slice : int list list -> ('a, 'b) t -> ('a, 'b) t

  val set_slice : int list list -> ('a, 'b) t -> ('a, 'b) t -> unit

```

Both functions accept `int list list` as its slice definition. Every `list` element in the `int list list` is assumed to be a range. E.g., `[ []; [2]; [-1;3] ]` is equivalent to its full slice definition `[ R []; R [2]; R [-1;3] ]`, as we will introduce below in fancy slicing.


## Fancy Slicing

Fancy slicing is more powerful than the basic one thanks to its slice definition. With fancy slicing, we can pass in a list of arbitrary indices which may not be possible to specify with aforementioned `[start;stop;step]` format.

```ocaml

  type index =
    | I of int       (* single index *)
    | L of int list  (* list of indices *)
    | R of int list  (* index range *)

```

Fancy slice is defined by an `index list` where you can use three type constructors to specify:

* an individual index (using `I` constructor);
* a list of indices (using `L` constructor);
* a range of indices (using `R` constructor).


There are two functions to handle fancy slicing operations.

```text

  val get_fancy : index list -> ('a, 'b) t -> ('a, 'b) t

  val set_fancy : index list -> ('a, 'b) t -> ('a, 'b) t -> unit

```

`get_fancy s x` retrieves a slice of `x` defined by `s`; whereas `set_fancy s x y` assigns the slice of `x` defined by `s` according to values in `y`. Note that `y` must have the same shape as that defined by `s`.

Basic slicing is a special case of fancy slicing where only type constructor `R` is used in the definition. For example, the following two definitions are equivalent.

```ocaml env=slicing_env0

  let x = Arr.sequential [|10; 10; 10|];;

  Arr.get_slice [ []; [0;8]; [3;9;2] ] x;;

  Arr.get_fancy [ R[]; R[0;8]; R[3;9;2] ] x;;

```

Note that both `get_basic` and `get_fancy` return a copy (rather than a view as that in Numpy); whilst `set_basic` and `set_fancy` modifies the original data in place.



## Conventions in Definition

Essentially, Owl's slicing functions are very similar to those in Numpy. So if you already know how to slice n-dimensional arrays in Numpy, you should find this chapter very easy.

The following conventions require our attentions in order to write correct slice definition. These conventions can be equally applied to both basic and fancy slicing.

* Slice definition is a `index list`. Each element within the `index list` corresponds one dimension in the passed in data, and it defines how the indices along this dimension should be accessed. Owl provides three constructors `I`, `L`, and `R` to let you specify single index, a list of indices, or a range of indices.

* Constructor `I` is trivial, it specifies a specific index. E.g., `[ I 2; I 5 ]` returns the element at position `(2, 5)` in a matrix.

* Constructor `L` is used to specify a list of indices. E.g., `[ I 2; L [5;3] ]` returns a `1 x 2` matrix consists of the elements at `(2, 5)` and `(2, 3)` in the original matrix.

* Constructor `R` is for specifying a range of indices. It has more conventions but by no means complicated. The following text is dedicated for range conventions.

* The format of the range definition follows **R [ start; stop; step ]**. Obviously, `start` specifies the starting index; `stop` specifies the stopping index (inclusive); and `step` specifies the step size. You do not have to specifies all three variables in the definition, please see the following rules.

* All three variables `start`, `stop`, and `step` can take both positive and negative values, but `step` is not allowed to take `0` value. Positive step indicates that indices will be visited in increasing order from `start` to `stop`; and vice versa.

* For `start` and `stop` variables, positive value refers to a specific index; whereas negative value `a` will be translated into `n + a` where `n` is the total number of indices. E.g., `[ -1; 0 ]` means from the last index to the first one.

* If you pass in an empty list `R []`, this will be expanded into `[ 0; n - 1; 1 ]` which means all the indices will be visited in increasing order with step size `1`.

* If you only specify one variable such as `[ start ]`, then `get_slice` function assumes that you will take one specific index by automatically extending it into `[ start; start; 1 ]`. As we can see, `start` and `stop` are the same, with step size 1.

* If you only specify two variables then `slice` function assumes they are `[ start; stop ]` which defines the range of indices. However, how `get_slice` will expand this slice definition depends, as we can see in the below, `slice` will visit the indices in different orders.

  - if `start <= stop`, then it will be expanded to `[ start; stop; 1 ]`;
  - if `start > stop`, then it will be expanded to `[ start; stop; -1 ]`;

- It is not necessary to specify all the definitions for all the dimensions, `get_slice` function will also expand it by assuming you will take all the data in higher dimensions. E.g., `x` has the shape `[ 2; 3; 4 ]`, if we define the slice as `[ [0] ]` then `get_slice` will expand the definition into `[ [0]; []; [] ]`

OK, that's all. Please make sure you understand it well before you start, but it is also fine you just learn by doing.



## Extended Operators

The operators for indexing and slicing are built atop of the extended indexing operators introduced in OCaml 4.06. Three are used in Owl as follows. All of them are defined in the functors in  `Owl_operator` module.

* `.%{ }`   : `get`
* `.%{ }<-` : `set`
* `.${ }`   : `get_slice`
* `.${ }<-` : `set_slice`
* `.!{ }`   : `get_fancy`
* `.!{ }<-` : `set_fancy`

Here are some examples to show how to use them.

**.%{ }** for indexing, as follows.

.. code-block:: ocaml

  open Arr;;

  let x = sequential [|10; 10; 10|];;
  let a = x.%{ [|2; 3; 4|] };;         (* i.e. Arr.get *)
  x.%{ [|2; 3; 4|] } <- 111.;;         (* i.e. Arr.set *)


**.${ }** for basic slicing, as follows.

```ocaml env=slicing_env1

  open Arr;;

  let x = sequential [|10; 10; 10|] in
  let a = x.${ [[0;4]; [6;-1]; [-1;0]] } in  (* i.e. Arr.get_slice *)
  let b = zeros (shape a) in
  x.${ [[0;4]; [6;-1]; [-1;0]] } <- b;;     (* i.e. Arr.set_slice *)

```

**.!{ }** for fancy slicing, as follows.

```ocaml env=slicing_env1

  open Arr;;

  let x = sequential [|10; 10; 10|] in
  let a = x.!{ [ L[2;2;1]; R[6;-1]; I 5] } in  (* i.e. Arr.get_fancy *)
  let b = zeros (shape a) in
  x.!{ [L[2;2;1]; R[6;-1]; I 5] } <- b;;      (* i.e. Arr.set_fancy *)

```


## Slicing Examples

I always believe that nothing is better than concrete examples. I will use the basic slicing to demonstrate some examples in the following. Note that all the following examples can be equally applied to ndarray. OK, here they are.

Let's first define a sequential matrix as the input data for the following examples.

```ocaml env=slicing_env2

  let x = Mat.sequential 5 7;;

```

You should be able to see the following output in `utop`.

```text

     C0 C1 C2 C3 C4 C5 C6
  R0  0  1  2  3  4  5  6
  R1  7  8  9 10 11 12 13
  R2 14 15 16 17 18 19 20
  R3 21 22 23 24 25 26 27
  R4 28 29 30 31 32 33 34

  val x : Mat.mat =

```

Now, we can finally start our experiment. One benefit of running code in `utop` is that you can observe the output immediately to understand better how `slice` function works.

```ocaml env=slicing_env2

  let x = Arr.sequential [|10; 10; 10|];;

  (* simply take all the elements *)
  let s = [ ] in
  Mat.get_slice s x;;

  (* take row 2 *)
  let s = [ [2]; [] ] in
  Mat.get_slice s x;;

  (* same as above, take row 2, but only specify low dimension slice definition *)
  let s = [ [2] ] in
  Mat.get_slice s x;;

  (* take from row 1 to 3 *)
  let s = [ [1;3] ] in
  Mat.get_slice s x;;

  (* take from row 3 to 1, same as the example above but in reverse order *)
  let s = [ [3;1] ] in
  Mat.get_slice s x;;

```

Let' see some more complicated examples.

```ocaml env=slicing_env2

  (* take from row 1 to 3 and column 3 to 5, so a sub-matrix of x *)
  let s = [ [1;3]; [3;5] ] in
  Mat.get_slice s x;;

  (* take from row 1 to the last row *)
  let s = [ [1;-1]; [] ] in
  Mat.get_slice s x;;

  (* take the rows of even number indices, i.e., 0;2;4 *)
  let s = [ [0;-1;2] ] in
  Mat.get_slice s x;;

  (* take the column of odd number indices, i.e.,1;3;5 ... *)
  let s = [ []; [1;-1;2] ] in
  Mat.get_slice s x;;

  (* reverse all the rows of x *)
  let s = [ [-1;0] ] in
  Mat.get_slice s x;;

  (* reverse all the elements of x, same as applying reverse function *)
  let s = [ [-1;0]; [-1;0] ] in
  Mat.get_slice s x;;

  (* take the second last row, from the first column to the last, with step size 3 *)
  let s = [ [-2]; [0;-1;3] ] in
  Mat.get_slice s x;;

```


## Advanced Usage

The following are some more advanced examples to show how to use slicing to achieve quite complicated operations. Let's use a `5 x 5` sequential matrix for illustration.

```ocaml env=slicing_env2
# let x = Mat.sequential 5 5
val x : Mat.mat =

   C0 C1 C2 C3 C4
R0  0  1  2  3  4
R1  5  6  7  8  9
R2 10 11 12 13 14
R3 15 16 17 18 19
R4 20 21 22 23 24

```

The first function `flip` a matrix upside down, i.e. flip vertically.

```ocaml env=slicing_env2
# let flip x = Mat.get_slice [ [-1; 0]; [ ] ] x in
  flip x
- : Mat.mat =

   C0 C1 C2 C3 C4
R0 20 21 22 23 24
R1 15 16 17 18 19
R2 10 11 12 13 14
R3  5  6  7  8  9
R4  0  1  2  3  4

```

The second `reverse` function treats a matrix as one-dimensional vector and rerverse the elements. This operation is equivalent to flipping in both vertical and horizontal directions.

```ocaml env=slicing_env2
# let reverse x = Mat.get_slice [ [-1; 0]; [-1; 0] ] x in
  reverse x
- : Mat.mat =

   C0 C1 C2 C3 C4
R0 24 23 22 21 20
R1 19 18 17 16 15
R2 14 13 12 11 10
R3  9  8  7  6  5
R4  4  3  2  1  0

```

The third function rotates a matrix 90 degrees in clockwise direction. As we see, slicing function leads to very consicise code.

```ocaml env=slicing_env2
# let rotate90 x = Mat.(transpose x |> get_slice [ []; [-1;0] ]) in
  rotate90 x
- : Mat.mat =

   C0 C1 C2 C3 C4
R0 20 15 10  5  0
R1 21 16 11  6  1
R2 22 17 12  7  2
R3 23 18 13  8  3
R4 24 19 14  9  4

```

The last function `cshift` performs right circular shift along the columns of a matrix.

```ocaml env=slicing_env2
let cshift x n =
  let c = Mat.col_num x in
  let h = Utils.Array.(range (c - n) (c - 1)) |> Array.to_list in
  let t = Utils.Array.(range 0 (c - n -1)) |> Array.to_list in
  Mat.get_fancy [ R []; L (h @ t) ] x
```

Applying to the previous `x`, the outcome should look like this.

```ocaml env=slicing_env2
# cshift x 2
- : Mat.mat =

   C0 C1 C2 C3 C4
R0  3  4  0  1  2
R1  8  9  5  6  7
R2 13 14 10 11 12
R3 18 19 15 16 17
R4 23 24 20 21 22

```

Slicing and indexing is an important topic in Owl, make sure you understand it well before proceeding to other chapters.


## Broadcasting Operation

Indexing, slicing, and broadcasting are three fundamental functions in [Ndarray module](https://github.com/ryanrhymes/owl/blob/master/src/owl/dense/owl_dense_ndarray_generic.mli). This chapter introduces the broadcasting operation in Owl. For indexing and slicing, please refer to [this Chapter](slicing.html).



### What Is Broadcasting?

There are many binary (mathematical) operators take two ndarrays as inputs, e.g. `add`, `sub`, and etc. In the trivial case, the inputs have exactly the same shape. However, in many real-world applications, we need to operate on two ndarrays whose shapes do not match, then how to apply the smaller one to the bigger one is referred to as `broadcasting`.

Broadcasting can save unnecessary memory allocation. E.g., assume we have a `1000 x 500` matrix `x` containing 1000 samples, and each sample has 500 features. Now we want to add a bias value for each feature, i.e. a bias vector `v` of shape `1 x 500`.

Because the shape of `x` and `v` do not match, we need to tile `v` so that it has the same shape as that of `x`.

```ocaml

  let x = Mat.uniform 1000 500;;  (* generate random samples *)
  let v = Mat.uniform 1 500;;     (* generate random bias *)
  let u = Mat.tile v [|1000;1|];; (* align the shape by tiling *)
  Mat.(x + u);;

```

The code above certainly works, but it is obvious that the solution uses much more memory. High memory consumption is not desirable for many applications, especially for those running on resource-constrained devices. Therefore we need the broadcasting operation come to rescue.

```ocaml

  let x = Mat.uniform 1000 500;;  (* generate random samples *)
  let v = Mat.uniform 1 500;;     (* generate random bias *)
  Mat.(x + u);;                   (* returns 1000 x 500 ndarray *)

```


### Shape Constraints

In broadcasting, the shapes of two inputs cannot be arbitrarily different, they must be subject to some constraints.

The convention used in broadcasting operation is much simpler than slicing. Given two matrices/ndarrays of the same dimensionality, for each dimension, one of the following two conditions must be met:

* both are equal.
* either is one.

Here are some **valid** shapes where broadcasting can be applied between `x` and `y`.

```text

  x : [| 2; 1; 3 |]    y : [| 1; 1; 1 |]
  x : [| 2; 1; 3 |]    y : [| 2; 1; 1 |]
  x : [| 2; 1; 3 |]    y : [| 2; 3; 1 |]
  x : [| 2; 1; 3 |]    y : [| 2; 3; 3 |]
  x : [| 2; 1; 3 |]    y : [| 1; 1; 3 |]
  ...

```

Here are some **invalid** shapes that violate the aforementioned constraints so that the broadcasting cannot be applied.

```text

  x : [| 2; 1; 3 |]    y : [| 1; 1; 2 |]
  x : [| 2; 1; 3 |]    y : [| 3; 1; 1 |]
  x : [| 2; 1; 3 |]    y : [| 3; 1; 1 |]
  ...

```


What if `y` has less dimensionality than `x`? E.g., `x` has the shape `[|2;3;4;5|]` wheras `y` has the shape `[|4;5|]`. In this case, Owl first calls `Ndarray.expand` function to increase `y`'s dimensionality to the same number as `x`'s. Technically, two ndarrays are aligned along the highest dimension. In other words, this is done by appending `1` s to lower dimension of `y`, so the new shape of `y` becomes `[|1;1;4;5|]`.

You can try `expand` by yourself, as below.

```ocaml

  let y = Arr.sequential [|4;5|];;
  let y' = Arr.expand y 4;;
  Arr.shape y';;    (* returns [|1;1;4;5|] *)

```


### Supported Operations

The broadcasting operation is transparent to programmers, which means it will be automatically applied if the shapes of two operators do not match (given the constraints are met of course). Currently, the following operations in Owl support broadcasting:

```text

==========================    ===========
Function Name                 Operators
==========================    ===========
`add`                       `+`
`sub`                       `-`
`mul`                       `*`
`div`                       `/`
`pow`                       `**`
`min2`
`max2`
`atan2`
`hypot`
`fmod`
`elt_equal`                 `=.`
`elt_not_equal`             `!=.` `<>.`
`elt_less`                  `<.`
`elt_greater`               `>.`
`elt_less_equal`            `<=.`
`elt_greater_equal`         `>=.`
==========================    ===========

```
