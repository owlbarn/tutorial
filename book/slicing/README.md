# Slicing and Broadcasting

Indexing, slicing, and broadcasting are three fundamental functions to manipulate multidimensional arrays.
They are so basic and are used in practically every application, therefore understanding the nuts and bolts is very important.
In this chapter we will introduce how to use these functions in Owl.

## Slicing

Indexing and slicing is arguably the most important function in any numerical library. A flexible design is able to significantly simplify the code and allow us to write concise algorithms.

Before we start, let's clarify some things.

* Slicing refers to the operation that extracts part of the data from an ndarray or a matrix according to a well-defined *slice definition*.

* Slicing can be applied to all the dense data structures, i.e. both ndarrays and matrices.

* Slice definition is an `index list` which clarifies *what indices* should be accessed and in *what order* for each dimension of the passed in variable.

* There are two types of slicing in Owl: *basic slicing* and *fancy slicing*. The difference between the two is how the slice is defined.

### Basic Slicing

For basic slicing, each dimension in the slice definition must be defined in the format of **[start:stop:step]**. Owl provides two functions `get_slice` and `set_slice` to retrieve and assign slice values respectively.

```text
val get_slice : int list list -> ('a, 'b) t -> ('a, 'b) t

val set_slice : int list list -> ('a, 'b) t -> ('a, 'b) t -> unit
```

Both functions accept `int list list` as its slice definition. Every `list` element in the `int list list` is assumed to be a range. E.g., `[ []; [2]; [-1;3] ]` is equivalent to its full slice definition `[ R []; R [2]; R [-1;3] ]`, as we will introduce below in fancy slicing.


### Fancy Slicing

Fancy slicing is more powerful than the basic one thanks to its slice definition. With fancy slicing, we can pass in a list of arbitrarily ordered indices which may not be possible to specify with aforementioned `[start;stop;step]` format.

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


### Conventions in Definition

Essentially, Owl's slicing functions are very similar to those in NumPy. So if you already know how to slice n-dimensional arrays in NumPy, you should find this chapter very easy.

The core building block is the slice definition. Slice definition is a `index list`. Each element within the `index list` corresponds one dimension in the passed in data, and it defines how the indices along this dimension should be accessed. Owl provides three constructors `I`, `L`, and `R` to let you specify single index, a list of indices, or a range of indices.

* Constructor `I` is trivial, it specifies a specific index. E.g., `[ I 2; I 5 ]` returns the element at position `(2, 5)` in a matrix.

* Constructor `L` is used to specify a list of indices. E.g., `[ I 2; L [5;3] ]` returns a `1 x 2` matrix consists of the elements at `(2, 5)` and `(2, 3)` in the original matrix.

* Constructor `R` is for specifying a range of indices. It has more conventions but by no means complicated. The following text is dedicated for range conventions.

The following conventions require our attentions in order to write correct slice definition. These conventions can be equally applied to both basic and fancy slicing.

**Rule #1**: The format of the range definition follows **R [ start; stop; step ]**. Obviously, `start` specifies the starting index; `stop` specifies the stopping index (inclusive); and `step` specifies the step size. You do not have to specifies all three variables in the definition, please see the following rules.

**Rule #2**: All three variables `start`, `stop`, and `step` can take both positive and negative values, but `step` is not allowed to take `0` value. Positive step indicates that indices will be visited in increasing order from `start` to `stop`; and vice versa.

**Rule #3**: For `start` and `stop` variables, positive value refers to a specific index; whereas negative value `a` will be translated into `n + a` where `n` is the total number of indices. E.g., `[ -1; 0 ]` means from the last index to the first one.

**Rule #4**: If you pass in an empty list `R []`, this will be expanded into `[ 0; n - 1; 1 ]` which means all the indices will be visited in increasing order with step size `1`.

**Rule #5**: If you only specify one variable such as `[ start ]`, then `get_slice` function assumes that you will take one specific index by automatically extending it into `[ start; start; 1 ]`. As we can see, `start` and `stop` are the same, with step size 1.

**Rule #6**: If you only specify two variables then `slice` function assumes they are `[ start; stop ]` which defines the range of indices. However, how `get_slice` will expand this slice definition depends, as we can see in the below, `slice` will visit the indices in different orders.

  * if `start <= stop`, then it will be expanded to `[ start; stop; 1 ]`;
  * if `start > stop`, then it will be expanded to `[ start; stop; -1 ]`;

**Rule #7**: It is not necessary to specify all the definitions for all the dimensions, `get_slice` function will also expand it by assuming you will take all the data in higher dimensions. E.g., `x` has the shape `[ 2; 3; 4 ]`, if we define the slice as `[ [0] ]` then `get_slice` will expand the definition into `[ [0]; []; [] ]`

OK, that's all. Please make sure you understand it well before you start, but it is also fine you just learn by doing.

Here is some illustrated examples that can get you started with some of these rules.
These examples are based on a `8x8` matrix.

```ocaml env=slicing_example_00
let x = Arr.sequential [|8; 8|]
```

![Illustrated Examples of Slicing](images/slicing/example_slice_01.png "slicing example 01"){width=95% #fig:slicing:example_slice_01}

The first example as shown in [@fig:slicing:example_slice_01](a)is to take one column of this matrix. It can be achieved by using both basic and fancy slicing:

```ocaml env=slicing_example_00
# Arr.get_fancy [ R[]; I 2 ] x;;
- : Arr.arr =

   C0
R0  2
R1 10
R2 18
R3 26
R4 34
R5 42
R6 50
R7 58

```

```ocaml env=slicing_example_00
# Arr.get_slice [ []; [2]  ] x;;
- : Arr.arr =

   C0
R0  2
R1 10
R2 18
R3 26
R4 34
R5 42
R6 50
R7 58

```

The second example in in [@fig:slicing:example_slice_01](b)is similar, but part of a row. Still, this can be gotten using both methods.

```ocaml env=slicing_example_00
# Arr.get_fancy [ I 2; R [4; 6] ] x
- : Arr.arr =
   C0 C1 C2
R0 20 21 22

```

```ocaml env=slicing_example_00
# Arr.get_slice [ [2]; [4; 6] ] x
- : Arr.arr =
   C0 C1 C2
R0 20 21 22

```

![Illustrated Examples of Slicing (Cont.)](images/slicing/example_slice_02.png "slicing example 02"){width=95% #fig:slicing:example_slice_02}


The next example in [@fig:slicing:example_slice_02](a) is a bit more complex. It chooses certain rows, and then choose the columns by a fixed step 2. We can use the fancy slicing in this way:

```ocaml env=slicing_example_00
# Arr.get_fancy [ L [3; 5]; R [1; 7; 2] ] x
- : Arr.arr =
   C0 C1 C2 C3
R0 25 27 29 31
R1 41 43 45 47

```

Finally, the last example concerns taking a sub-matrix. We can do it in the similar way as to the example 1 and 2.
Or, since this sub matrix is close to the end of both dimension, we can use the negative integers as indices.

```ocaml env=slicing_example_00
# Arr.get_fancy [ L [-2; -1]; R [-3; -2] ] x
- : Arr.arr =
   C0 C1
R0 53 54
R1 61 62

```

### Extended Operators

The operators for indexing and slicing are built atop of the extended indexing operators introduced in OCaml 4.06. Three are used in Owl as follows. All of them are defined in the functors in  `Owl_operator` module.

* `.%{ }`   : `get`
* `.%{ }<-` : `set`
* `.${ }`   : `get_slice`
* `.${ }<-` : `set_slice`
* `.!{ }`   : `get_fancy`
* `.!{ }<-` : `set_fancy`

Here are some examples to show how to use them.

**.%{ }** for indexing, as follows.

```ocaml env=slicing_env1
  open Arr;;

  let x = sequential [|10; 10; 10|];;
  let a = x.%{2; 3; 4};;         (* i.e. Arr.get *)
  x.%{2; 3; 4} <- 111.;;         (* i.e. Arr.set *)
```

**.${ }** for basic slicing, as follows.

```ocaml env=slicing_env1

  open Arr;;

  let x = sequential [|10; 10; 10|] in
  let a = x.${[0;4]; [6;-1]; [-1;0]} in  (* i.e. Arr.get_slice *)
  let b = zeros (shape a) in
  x.${[0;4]; [6;-1]; [-1;0]} <- b;;      (* i.e. Arr.set_slice *)

```

**.!{ }** for fancy slicing, as follows.

```ocaml env=slicing_env1

  open Arr;;

  let x = sequential [|10; 10; 10|] in
  let a = x.!{L [2;2;1]; R [6;-1]; I 5} in  (* i.e. Arr.get_fancy *)
  let b = zeros (shape a) in
  x.!{L [2;2;1]; R [6;-1]; I 5} <- b;;      (* i.e. Arr.set_fancy *)

```


### Advanced Usage

We believe that nothing is better than concrete examples.
We will first use the basic slicing to demonstrate some examples in the following. Note that all the following examples can be equally applied to ndarray.

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

The second `reverse` function treats a matrix as one-dimensional vector and reverse the elements. This operation is equivalent to flipping in both vertical and horizontal directions.

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

The third function rotates a matrix 90 degrees in clockwise direction. As we see, slicing function leads to very concise code.

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


## Broadcasting

Following indexing and slicing introduced in previous section, this section introduces the broadcasting operation in Owl. In contrast to indexing and slicing which are explicitly called, broadcasting are often implicitly called when certain conditions are met. This automatic behaviour on one hand is able to simplify the code, it can also potentially introduce bugs and make the debugging really difficult.


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


What if `y` has less dimensionality than `x`? E.g., `x` has the shape `[|2;3;4;5|]` whereas `y` has the shape `[|4;5|]`. In this case, Owl first calls `Ndarray.expand` function to increase `y`'s dimensionality to the same number as `x`'s. Technically, two ndarrays are aligned along the highest dimension. In other words, this is done by appending `1` s to lower dimension of `y`, so the new shape of `y` becomes `[|1;1;4;5|]`.

You can try `expand` by yourself, as below.

```ocaml env=broadcasting_example00
let y = Arr.sequential [|4;5|];;
let y' = Arr.expand y 4;;
```

```ocaml env=broadcasting_example00
# Arr.shape y'
- : int array = [|1; 1; 4; 5|]
```

If these seem too abstract, here are three concrete 2D examples for you to better understand how the shapes are extended in the broadcasting.
The first example is vector multiplied by scalar.

![Illustrated example of shape extension in broadcasting](images/slicing/example_broadcast_01.png "example broadcast 01"){width=90% #fig:slicing:broadcast_01}

```ocaml env=broadcasting_example01
let a = Arr.sequential [|1;3|]
```

```ocaml env=broadcasting_example01
# Arr.add_scalar a 3.
- : Arr.arr =
   C0 C1 C2
R0  3  4  5

```

The second example is matrix plus vector.

![Illustrated example of shape extension in broadcasting (cont.)](images/slicing/example_broadcast_02.png "example broadcast 02"){width=90% #fig:slicing:broadcast_02}

```ocaml env=broadcasting_example01
let b0 = Arr.sequential [|3;3|]
let b1 = Arr.sequential ~a:1. [|1;3|]
```

```ocaml env=broadcasting_example01
# Arr.mul b0 b1
- : Arr.arr =
   C0 C1 C2
R0  0  2  6
R1  3  8 15
R2  6 14 24

```

The third example is column vector plus row vector.

![Illustrated example of shape extension in broadcasting (cont.)](images/slicing/example_broadcast_03.png "example broadcast 03"){width=90% #fig:slicing:broadcast_03}


```ocaml env=broadcasting_example01
let c0 = Arr.sequential [|3;1|]
let c1 = Arr.copy b1
```

```ocaml env=broadcasting_example01
# Arr.mul c0 c1
- : Arr.arr =
   C0 C1 C2
R0  0  0  0
R1  1  2  3
R2  2  4  6

```

### Supported Operations

The broadcasting operation is transparent to programmers, which means it will be automatically applied if the shapes of two operators do not match (given the constraints are met of course). Currently, the operations in Owl support broadcasting are listed below:

- basic computation: `add`, `sub`, `mul`, `div`, `pow`
- comparison operations: `elt_equal`, `elt_not_equal`, `elt_less`, `elt_greater`, `elt_less_equal`, `elt_greater_equal`
- other operations: `min2`, `max2`. `atan2`, `hypot`, `fmod`

## Internal Mechanism

TODO: A short history of the development of NumPy and Julia.

The indexing and slicing functions are fundamental in all the multi-dimensional array implementations in various other languages.
For example, the examples in [@fig:slicing:example_slice_01] and [@fig:slicing:example_slice_02] can be implemented using NumPy.

```python
>> import numpy as np
>> x = np.arange(64).reshape([8,8])

>> x[:, 2]
array([ 2, 10, 18, 26, 34, 42, 50, 58])

>> x[2, 4:7]
array([20, 21, 22])

>> x[[3,5], 1:8:2]
array([[25, 27, 29, 31],
       [41, 43, 45, 47]])

>> x[[-2,-1], -3:-1]
array([[53, 54],
       [61, 62]])
```

You can see that the basic indexing syntax are similar, only that Python is not strong-typed, so the users can mix single index, list of indices, or index range.
Note that index range in NumPy is different than that in Owl.

Also, in Julia it can be done with:

```julia
> x = transpose(reshape([0:1:63;],8 ,8))

> x[:, 3]
8-element Array{Int64,1}:
  2
 10
 18
 26
 34
 42
 50
 58


> x[3, 5:7]
3-element Array{Int64,1}:
 20
 21
 22

> x[[4,6], 2:2:8]
2x4 Array{Int64,2}:
 25  27  29  31
 41  43  45  47

> x[7:8, [6,7]]
2x2 Array{Int64,2}:
 53  54
 61  62
```

The Julia interface is similar to that of NumPy. However, there are some crucial differences as shown these examples.
First, the array in Julia uses column-major order, so we need to use the `transpose` function to achieve the same example.
The other obvious difference is that, the indexing of Julia array starts from 1, not 0.
Besides, the negative indexing is not supported in Julia.

However, one important thing to notice in slicing is the difference between *copy* and *view*.
For example, in Owl we can make a slice:

```ocaml env=slicing_example_01
# let x = Arr.sequential [|3;3|]
val x : Arr.arr =
   C0 C1 C2
R0  0  1  2
R1  3  4  5
R2  6  7  8

```
```ocaml env=slicing_example_01
# let y = Arr.get_slice [[0]; []] x
val y : Arr.arr =
   C0 C1 C2
R0  0  1  2

```
```ocaml env=slicing_example_01
# Arr.set y [|0; 2|] 200.;;
- : unit = ()
```
```ocaml env=slicing_example_01
# y
- : Arr.arr =
   C0 C1  C2
R0  0  1 200

```
```ocaml env=slicing_example_01
# x
- : Arr.arr =
   C0 C1 C2
R0  0  1  2
R1  3  4  5
R2  6  7  8

```

We can see that in Owl, changing the local content of the slice `y` does not change that of the original ndarray `x`.
That's because in Owl each slice makes a different copy.

As a contrast, in NumPy the default indexing only makes a "view" of the original array, and any change on the view will also change the original array.

```python
>> x = np.arange(9).reshape([3,3])
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])

>> y = x[0, :]
array([0, 1, 2])

>> y[2] = 200
>> y
array([  0,   1, 200])

>> x
array([[  0,   1, 200],
       [  3,   4,   5],
       [  6,   7,   8]])
```


For performance, slicing is implemented in C.
The basic algorithm of slicing is simple. We need to copy part of the source array `x` to the target array `y`.
So you can imagine two cursors move one step at a time, only that for each dimension, the cursor start at different position and offset compared to the starting point, and at different increments.
At each step, we simply copy the content from `x` to `y`.
To do this, we define a structure `slice_pair` for slicing operations.

```c
struct slice_pair {
  int64_t dim;          // number of dimensions, x and y must be the same
  int64_t dep;          // the depth of current recursion.
  intnat *n;            // number of iteration in each dimension, i.e. y's shape
  void *x;              // x, source if operation is get, destination if set.
  int64_t posx;         // current offest of x.
  int64_t *ofsx;        // offset of x in each dimension.
  int64_t *incx;        // stride size of x in each dimension.
  void *y;              // y, destination if operation is get, source if set.
  int64_t posy;         // current offest of y.
  int64_t *ofsy;        // offset of y in each dimension.
  int64_t *incy;        // stride size of y in each dimension.
};
```

Taking a 2-dimensional slicing as example, here is the core step:

```c
    for (int64_t i0 = 0; i0 < n0; i0++) {
      posx1 = posx0 + ofsx1;
      posy1 = posy0 + ofsy1;

      for (int64_t i1 = 0; i1 < n1; i1++) {
        MAPFUN (*(x + posx1), *(y + posy1));
        posx1 += incx1;
        posy1 += incy1;
      }

      posx0 += incx0;
      posy0 += incy0;
    }
```

So this algorithm basically says that for each row, we calculate its starting points in `x` and `y`, and then for each column, copy the element, and them move the cursors forward until the current row is finished.
And then move the rows forward.
If it becomes multiple dimension, we implement it with recursive algorithm.

## Summary
