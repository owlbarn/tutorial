---
layout: page
---

# N-Dimensional Arrays


N-dimensional array (a.k.a ndarray) is the building block of Owl library. Ndarray to Owl is like NumPy to SciPy. It serves as the core dense data structure and many advanced numerical functions are built atop of it. For example, `Algodiff`, `Optimise`, `Neural`, and `Lazy`... all these functors take Ndarray module as the module input.

Due to its importance, Owl has implemented a comprehensive set of operations on Ndarray, all of which are defined in the file  [owl_dense_ndarray_generic.mli](https://github.com/owlbarn/owl/blob/master/src/owl/dense/owl_dense_ndarray_generic.mli). Many of these functions (especially the critical ones) in Owl's core library have corresponding C-stub code to guarantee the best performance. If you take a look at the Ndarray's `mli` file, you probably can see hundreds of them. But do not get scared by the number, since many of them are similar and can be grouped together. In this chapter, we will explain these functions in details regarding these several groups.


## Ndarray Types

The very first thing to understand is the types used in Ndarray. Owl's Ndarray module is built directly on top of OCaml's native `Bigarray`. More specifically, it is `Bigarray.Genarray`. Ndarray has the same type as that of `Genarray`, therefore exchanging data between Owl and other libraries relying on Bigarray is trivial.

OCaml's Bigarray uses `kind` GADT to specify the number type, precision, and memory layout. Owl only keeps the first two but fixes the last one because Owl only uses `C-layout`, or `Row-based layout` in its implementation. The same design decisions can also be seen in ONNX. See the type definition in Ndarray module.

```ocaml
  type ('a, 'b) t = ('a, 'b, c_layout) Genarray.t
```

Technically, `C-layout` indicates the memory address is continuous at the highest dimensions, comparing to the `Fortran-layout` whose continuous memory address is at the lowest dimensions. The reasons why we made this decision are as follows.

* Mixing two layouts together opens a can of worms and is the source of bugs. Especially, indexing in FORTRAN starts from 1 whereas indexing in C starts form 0. Many native OCaml data structures such as `Array` and `List` all start indexing from 0, so using `C-layout` avoids many potential troubles in using the library.

* Supporting both layouts adds a significant amount of complexity in implementing underlying Ndarray functions. Due to the difference in memory layout, code performs well on one layout may not does well on another. Many functions may require different implementations given different layout. This will add too much complexity and increase the code base significantly with marginal benefits.

* Owl has rather different design principles comparing to OCaml's Bigarray. The Bigarray serves as a basic tool to operate on a chunk of memory living outside OCaml's heap, facilitating exchanging data between different libraries (including FORTRAN ones). Owl focuses on providing high-level numerical functions allowing programmers to write concise analytical code. The simple design and small code base outweighs the benefits of supporting both layouts.

Because of Bigarray's mechanism, Owl's Ndarray is also subject to maximum 16 dimensions limits. Moreover, matrix is just a special case of n-dimensional array, and in fact many functions in the `Matrix` module simply calls the same functions in Ndarray. But the module does provide more matrix-specific functions such as iterating rows or columns, etc.

## Creation Functions

The first group of functions we would like to introduce is the ndarray creation functions. They generate dense data structures for you to work on further. The most frequently used ones are probably these four:

```ocaml
open Owl.Dense.Ndarray.Generic

val empty : ('a, 'b) kind -> int array -> ('a, 'b) t

val create : ('a, 'b) kind -> int array -> 'a -> ('a, 'b) t

val zeros : ('a, 'b) kind -> int array -> ('a, 'b) t

val ones : ('a, 'b) kind -> int array -> ('a, 'b) t
```

These functions return ndarrays of specified shape, number type, and precision.
The `empty` function is different from the other three. It does not really allocate any memory until you access it. Therefore, calling `empty` function is very fast.
The other three functions are self-explained. The `zeros` and `ones` fill the allocated memory with zeros and one respectively, whereas `create` function fills the memory with the specified value.

If you need random numbers, you can use another three creation functions that return an ndarray where the elements follow certain distributions.

```ocaml 
open Owl.Dense.Ndarray.Generic

val uniform : ('a, 'b) kind -> ?a:'a -> ?b:'a -> int array -> ('a, 'b) t

val gaussian : ('a, 'b) kind -> ?mu:'a -> ?sigma:'a -> int array -> ('a, 'b) t

val bernoulli : ('a, 'b) kind -> ?p:float -> int array -> ('a, 'b) t
```

Sometimes, we want to generate numbers with equal distance between two consecutive elements. These ndarrays are useful in generating intervals and plotting figures.

```ocaml 
open Owl.Dense.Ndarray.Generic

val sequential : ('a, 'b) kind -> ?a:'a -> ?step:'a -> int array -> ('a, 'b) t

val linspace : ('a, 'b) kind -> 'a -> 'a -> int -> ('a, 'b) t

val logspace : ('a, 'b) kind -> ?base:float -> 'a -> 'a -> int -> ('a, 'b) t
```

If these functions cannot satisfy your need, `Ndarray` provides a more flexible mechanism allowing you to have more control over the initialisation of an ndarray.

```ocaml 
open Owl.Dense.Ndarray.Generic

val init : ('a, 'b) kind -> int array -> (int -> 'a) -> ('a, 'b) t

val init_nd : ('a, 'b) kind -> int array -> (int array -> 'a) -> ('a, 'b) t
```

The difference between the two group is: `init` passes 1-d indices to the user-defined function, whereas `init_nd` passes n-dimensional indices. As a result, `init` is much faster than `init_nd`.
As an example, the following code creates an ndarray where all the elements are even numbers.

```ocaml

# let x = Arr.init [|6;8|] (fun i -> 2. *. (float_of_int i));;
val x : Arr.arr =

   C0 C1 C2 C3 C4 C5 C6 C7
R0  0  2  4  6  8 10 12 14
R1 16 18 20 22 24 26 28 30
R2 32 34 36 38 40 42 44 46
R3 48 50 52 54 56 58 60 62
R4 64 66 68 70 72 74 76 78
R5 80 82 84 86 88 90 92 94

```


## Properties Functions

After an ndarray is created, you can use various functions in the module to obtain its properties. For example, the following functions are commonly used ones.

```ocaml 
open Owl.Dense.Ndarray.Generic

val shape : ('a, 'b) t -> int array
(** [shape x] returns the shape of ndarray [x]. *)

val num_dims : ('a, 'b) t -> int
(** [num_dims x] returns the number of dimensions of ndarray [x]. *)

val nth_dim : ('a, 'b) t -> int -> int
(** [nth_dim x] returns the size of the nth dimension of [x]. *)

val numel : ('a, 'b) t -> int
(** [numel x] returns the number of elements in [x]. *)

val nnz : ('a, 'b) t -> int
(** [nnz x] returns the number of non-zero elements in [x]. *)

val density : ('a, 'b) t -> float
(** [density x] returns the percentage of non-zero elements in [x]. *)

val size_in_bytes : ('a, 'b) t -> int
(** [size_in_bytes x] returns the size of [x] in bytes in memory. *)

val same_shape : ('a, 'b) t -> ('a, 'b) t -> bool
(** [same_shape x y] checks whether [x] and [y] has the same shape or not. *)

val kind : ('a, 'b) t -> ('a, 'b) kind
(** [kind x] returns the type of ndarray [x]. *)
```
Property functions are easy to understand.
Note that `nnz` and `density` need to traverse through all the elements in an ndarray, but because the implementation is in C so even for a very large ndarray the performance is still good.
In the following, we focus on three typical operations on n-dimensional array worth your special attention : the `map`, `fold`, and `scan`.


## Map Functions

The `map` function transforms one ndarray to another according to a given function, which is often done by applying the transformation function to every element in the original ndarray.
The `map` function in Owl is pure and always generates a fresh new data structure rather than modifying the original one.
For example, the following code creates a three-dimensional ndarray, and then adds 1 to every element in `x`.

```ocaml
# let x = Arr.uniform [|3;4;5|];;
val x : Arr.arr =

              C0        C1       C2       C3         C4
R[0,0]  0.378545  0.861025 0.712662 0.563556   0.964339
R[0,1]  0.582878  0.834786 0.722758 0.265025   0.712912
R[0,2] 0.0894476   0.13984 0.475555 0.616536   0.202631
R[0,3]  0.983487 0.0167333  0.25018 0.483741   0.736418
R[1,0] 0.0757294  0.662478 0.460645 0.203446   0.725446
             ...       ...      ...      ...        ...
R[1,3]   0.83694  0.897979 0.912516 0.833211     0.4145
R[2,0]  0.903692  0.883623 0.809134 0.859235   0.188514
R[2,1]  0.236758  0.566636 0.613932 0.215875 0.00911335
R[2,2]  0.859797  0.708086 0.518328 0.974299   0.472426
R[2,3]  0.126273  0.946126  0.42223 0.955181   0.422184

```


```ocaml
# let y = Arr.map (fun a -> a +. 1.) x;;
val y : Arr.arr =

            C0      C1      C2      C3      C4
R[0,0] 1.37854 1.86103 1.71266 1.56356 1.96434
R[0,1] 1.58288 1.83479 1.72276 1.26503 1.71291
R[0,2] 1.08945 1.13984 1.47556 1.61654 1.20263
R[0,3] 1.98349 1.01673 1.25018 1.48374 1.73642
R[1,0] 1.07573 1.66248 1.46065 1.20345 1.72545
           ...     ...     ...     ...     ...
R[1,3] 1.83694 1.89798 1.91252 1.83321  1.4145
R[2,0] 1.90369 1.88362 1.80913 1.85923 1.18851
R[2,1] 1.23676 1.56664 1.61393 1.21588 1.00911
R[2,2]  1.8598 1.70809 1.51833  1.9743 1.47243
R[2,3] 1.12627 1.94613 1.42223 1.95518 1.42218

```

The `map` function can be very useful in implementing vectorised math functions. Many functions in Ndarray can be categorised into this group, such as `sin`, `cos`, `neg`, etc. Here are some examples to show how to make your own vectorised functions.

```ocaml

  let vec_sin x = Arr.map sin x;;

  let vec_cos x = Arr.map cos x;;

  let vec_log x = Arr.map log x;;

```

If you need indices in the transformation function, you can use the `mapi` function which accepts the 1-d index of the element being accessed.

```ocaml

  val mapi : (int -> 'a -> 'a) -> ('a, 'b) t -> ('a, 'b) t

```


## Fold Functions

The `fold` function is often referred to as "reduction" in other programming languages. It has a named parameter called `axis`, with which you can specify along what axis you want to fold a given ndarray.

```ocaml

  val fold : ?axis:int -> ('a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> ('a, 'b) t

```

The `axis` parameter is optional. If you do not specify one, the ndarray will be flattened first folding happens along the zero dimension. In other words, the all the elements will be folded into a one-element one-dimensional ndarray. The `fold` function in Ndarray is actually folding from left, and you can also specify an initial value of the folding.
The code below demonstrates how to implement your own `sum'` function.

```ocaml

  let sum' ?axis x = Arr.fold ?axis ( +. ) 0. x;;

```

The functions `sum`, `sum'`, `prod`, `prod'`, `min`, `min'`, `mean`, and `mean'` all belong to this group. The difference between the functions with and without prime ending is that the former one returns an ndarray, while the latter one returns a number.

Similarly, if you need indices in folding function, you can use `foldi` which passes in 1-d indices.

```ocaml

  val foldi : ?axis:int -> (int -> 'a -> 'a -> 'a) -> 'a -> ('a, 'b) t -> ('a, 'b) t

```


## Scan Functions

To some extent, the `scan` function is like the combination of `map` and `fold`. It accumulates the value along the specified axis but it does not change the shape of the input. Think about how we generate a cumulative distribution function (CDF) from a probability density/mass function (PDF/PMF).
The type signature of `scan` looks like this in Ndarray.

```ocaml

  val scan : ?axis:int -> ('a -> 'a -> 'a) -> ('a, 'b) t -> ('a, 'b) t

```

Several functions belong to this group, such as `cumsum`, `cumprod`, `cummin`, `cummax`, etc. To implement one `cumsum` for yourself, you can write in the following way.

```ocaml

  let cumsum ?axis x = Arr.scan ?axis ( +. ) x;;

```

Again, you can use the `scani` to obtain the indices in the passed in cumulative functions.

## Comparison Functions

The comparison functions themselves can be divided into several groups. The first group compares two ndarrays then returns a boolean value.

```ocaml

  val equal : ('a, 'b) t -> ('a, 'b) t -> bool

  val not_equal : ('a, 'b) t -> ('a, 'b) t -> bool

  val less : ('a, 'b) t -> ('a, 'b) t -> bool

  val greater : ('a, 'b) t -> ('a, 'b) t -> bool

  ...
```

The second group compares two ndarrays but returns an 0-1 ndarray of the same shape. The elements where the predicate is satisfied have value 1 otherwise 0.

```ocaml

  val elt_equal : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  val elt_not_equal : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  val elt_less : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  val elt_greater : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t

  ...
```


The third group is similar to the first one but compares an ndarray with a scalar value, the return is a Boolean value.

```ocaml

  val equal_scalar : ('a, 'b) t -> 'a -> bool

  val not_equal_scalar : ('a, 'b) t -> 'a -> bool

  val less_scalar : ('a, 'b) t -> 'a -> bool

  val greater_scalar : ('a, 'b) t -> 'a -> bool

  ...
```


The fourth group is similar to the second one but compares an ndarray with a scalar value, and the returned value is a 0-1 ndarray.

```ocaml

  val elt_equal_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t

  val elt_not_equal_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t

  val elt_less_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t

  val elt_greater_scalar : ('a, 'b) t -> 'a -> ('a, 'b) t

  ...
```


You probably have noticed the pattern in naming these functions.
In general, we recommend using operators rather than calling these function name directly, since it leads to more concise code. Please refer to the chapter about [Conventions](convention.html).

The comparison functions can do a lot of useful things for us. As an example, the following code shows how to keep the elements greater than `0.5` as they are but set the rest to zeros in an ndarray.

```ocaml
let x = Arr.uniform [|10; 10|];;

(* the first solution *)
let y = Arr.map (fun a -> if a > 0.5 then a else 0.) x;;

(* the second solution *)
let z = Arr.((x >.$ 0.5) * x);;
```

As you can see, comparison function combined with operators can lead to more concise code. Moreover, it sometimes outperforms the first solution at the price of higher memory consumption, because the loop is done in C rather than in OCaml.


## Vectorised Functions

Many common operations on ndarrays can be decomposed as a series of `map`, `fold`, and `scan` operations. There is even a specific programming paradigm built atop of this which is called `Map-Reduce`.
It was hyped several years ago in many data processing frameworks. Nowadays, map-reduce is one dominant data-parallel processing paradigm.

The ndarray module has included a very comprehensive set of mathematical functions and all have been vectorised. This means you can apply them directly on an ndarray and the function will be automatically applied to every element in the ndarray.

For binary math operators, there are `add`, `sub`, `mul`, etc. For unary operators, there are `sin`, `cos`, `abs`, etc. You can obtain the complete list of functions in [owl_dense_ndarray_generic.mli](https://github.com/owlbarn/owl/blob/master/src/owl/dense/owl_dense_ndarray_generic.mli>).

Conceptually, Owl can implement all these functions using the aforementioned `map`, `fold`, and `scan`. In reality, these vectorised math is done in C code to guarantee the best performance. Accessing the elements in a bigarray is way faster in C than in OCaml.


## Iteration Functions

Like native OCaml array, Owl also provides `iter` and `iteri` functions with which you can iterate over all the elements in an ndarray.

```ocaml

  val iteri :(int -> 'a -> unit) -> ('a, 'b) t -> unit

  val iter : ('a -> unit) -> ('a, 'b) t -> unit

```

One common use case is iterating all the elements and checking if one (or several) predicate is satisfied.
There is a special set of iteration functions to help you finish this task.

```ocaml

  val is_zero : ('a, 'b) t -> bool

  val is_positive : ('a, 'b) t -> bool

  val is_negative : ('a, 'b) t -> bool

  val is_nonpositive : ('a, 'b) t -> bool

  val is_nonnegative : ('a, 'b) t -> bool

  val is_normal : ('a, 'b) t -> bool

```

The predicates can be very complicated sometimes. In that case you can use the following three functions to pass in arbitrarily complicated functions to check them.

```ocaml

  val exists : ('a -> bool) -> ('a, 'b) t -> bool

  val not_exists : ('a -> bool) -> ('a, 'b) t -> bool

  val for_all : ('a -> bool) -> ('a, 'b) t -> bool

```

All aforementioned functions only tell us whether the predicates are met or not. They cannot tell which elements satisfy the predicate. The following `filter` function can return the 1-d indices of those elements satisfying the predicates.

```ocaml

  val filteri : (int -> 'a -> bool) -> ('a, 'b) t -> int array

  val filter : ('a -> bool) -> ('a, 'b) t -> int array

```

We have mentioned that 1-d indices are passed in. The reason is passing in 1-d indices is way faster than passing in n-d indices. However, if you do need n-dimensional indices, you can use the following two functions to convert between 1-d and 2-d indices, both are defined in the `Owl.Utils` module.

```ocaml

  val ind : ('a, 'b) t -> int -> int array
  (* 1-d to n-d index conversion *)

  val i1d : ('a, 'b) t -> int array -> int
  (* n-d to 1-d index conversion *)

```

Note that you need to pass in the original ndarray because the shape information is required for calculating index conversion.

## Manipulation Functions

Ndarray module contains many useful functions to manipulate ndarrays. For example, you can tile and repeat an ndarray along a specified axis.
Let's first create a sequential ndarray.

```ocaml

# let x = Arr.sequential [|3;4|];;
val x : Arr.arr =

   C0 C1 C2 C3
R0  0  1  2  3
R1  4  5  6  7
R2  8  9 10 11

```

The code below tiles `x` once on both dimensions.

```ocaml
# let y = Arr.tile x [|2;2|];;
val y : Arr.arr =

   C0 C1 C2 C3 C4 C5 C6 C7
R0  0  1  2  3  0  1  2  3
R1  4  5  6  7  4  5  6  7
R2  8  9 10 11  8  9 10 11
R3  0  1  2  3  0  1  2  3
R4  4  5  6  7  4  5  6  7
R5  8  9 10 11  8  9 10 11

```

Comparing to `tile`, the `repeat` function replicates each element in their adjacent places along specified dimension.

```ocaml
# let z = Arr.repeat x [|2;1|];;
val z : Arr.arr =

   C0 C1 C2 C3
R0  0  1  2  3
R1  0  1  2  3
R2  4  5  6  7
R3  4  5  6  7
R4  8  9 10 11
R5  8  9 10 11

```


You can also expand the dimensionality of an ndarray, or squeeze out those dimensions having only one element, or even padding elements to an existing ndarray.

```ocaml

  val expand : ('a, 'b) t -> int -> ('a, 'b) t

  val squeeze : ?axis:int array -> ('a, 'b) t -> ('a, 'b) t

  val pad : ?v:'a -> int list list -> ('a, 'b) t -> ('a, 'b) t

```

Another two useful functions are `concatenate` and `split`.
The `concatenate` allows us to concatenate an array of ndarrays along the specified axis. The constraint on the shapes is that, except for the dimension of concatenation, the rest dimensions must be equal.
For matrices, there are two operators associated with concatenation: `@||` for concatenating horizontally (i.e. along axis 1); `@=` for concatenating vertically (i.e. along axis 0).
The `split` is simply the inverse operation of concatenation.

```ocaml

  val concatenate : ?axis:int -> ('a, 'b) t array -> ('a, 'b) t

  val split : ?axis:int -> int array -> ('a, 'b) t -> ('a, 'b) t array

```

You can also sort an ndarray but note that modification will happen in place.

```ocaml

  val sort : ('a, 'b) t -> unit

```

Converting between ndarrays and OCaml native arrays can be efficiently done with these conversion functions:

```ocaml

  val of_array : ('a, 'b) kind -> 'a array -> int array -> ('a, 'b) t

  val to_array : ('a, 'b) t -> 'a array

```

Again, there also exist the `to_arrays` and `of_arrays` two functions for the special case of matrix module.


## Serialisation

Serialisation and de-serialisation are simply done with the `save` and `load` functions.

```ocaml

  val save : out:string -> ('a, 'b) t -> unit

  val load : ('a, 'b) kind -> string -> ('a, 'b) t

```

Note that you need to pass in type information in the `load` function, otherwise Owl cannot figure out what is contained in the chunk of binary file. Alternatively, you can use the corresponding `load` functions in the `S/D/C/Z` modules to save the type information.

```ocaml
# let x = Mat.uniform 8 8 in
  Mat.save "data.mat" x;
  let y = Mat.load "data.mat" in
  Mat.(x = y);;
Line 2, characters 3-11:
Warning 6 [labels-omitted]: label out was omitted in the application of this function.
- : bool = true
```

The `save` and `load` currently use the `Marshall` module which is brittle since it depends on specific OCaml versions. In the future, these two functions will be improved.

With the help of [npy-ocaml](https://github.com/LaurentMazare/npy-ocaml), we can save and load files in the format of npy file.
Proposed by NumPy, [NPY](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html) is a standard binary file format for persisting a single arbitrary ndarray on disk.
The format stores all of the shape and data type information necessary to reconstruct the array correctly even on another machine with a different architecture.
NPY is a widely used serialisation format.
Owl can thus easily interact with the Python-world data by using this format.

Using NPY files are the same as that of normal serialisation methods. Here is a simple example:

```ocaml
# let x = Arr.uniform [|3; 3|] in
  Arr.save_npy ~out:"data.npy" x;
  let y = Arr.load_npy "data.npy" in
  Arr.(x = y);;
- : bool = true
```

There are way more functions contained in the `Ndarray` module than the ones we have introduced here. Please refer to the API documentation for the full list.

## Tensors

In the last part of this chapter, we will briefly introduce the idea of *tensor*.
If you look at some articles online the tensor is often defined as an n-dimensional array.
However, mathematically, there are differences between these two.
In a n-dimension space, a tensor that contains $$m$$ indices is a mathematical object that obeys certain transformation rules.
For example, in a three dimension space, we have a value `A = [0, 1, 2]` that indicate a vector in this space.
We can find each element in this vector by a single index $$i$$, e.g. $$A_1 = 1$$.
This vector is an object in this space, and it stays the same even if we change the standard cartesian coordinate system into other systems.
But if we do so, then the content in $$A$$ needs to be updated accordingly.
Therefore we say that, a tensor can normally be expressed in the form of an ndarray, but it is not an ndarray.
That's why we keep using the term "ndarray" in this chapter and through out the book.

The basic idea about tensor is that, since the object stays the same, if we change the coordinate towards one direction, the component of the vector needs to be changed to another direction.
Considering a single vector $$v$$ in a coordinate system with basis $e$.
We can change the coordinate base to $$\tilde{e}$$ with linear transformation: $$\tilde{e} = Ae$$ where A is a matrix. For any vector in this space using $e$ as base, its content will be transformed as: $$\tilde{v} = A^{-1}v$$, or we can write it as:

$$\tilde{v}^i = \sum_j~B_j^i~v^j.$$

Here $$B=A^{-1}$$.
We call a vector *contravector* because it changes in the opposite way to the basis.
Note we use the superscript to denote the element in contravectors.

As a comparison, think about a matrix multiplication $$\alpha~v$$. The $$\alpha$$ itself forms a different vector space, the basis of which is related to the basis of $$v$$'s vector space.
It turns out that the direction of change of $$\alpha$$ is the same as that of $$e$$. When $$v$$ uses new $$\tilde{e} = Ae$$, its component changes in the same way:

$$\tilde{\alpha}_j = \sum_i~A_j^i~\alpha_i.$$

It is called a *covector*, denoted with subscript.
We can further extend it to matrix. Think about a linear mapping $$L$$. It can be represented as a matrix so that we can apply it to any vector using matrix dot multiplication.
With the change of the coordinate system, it can be proved that the content of the linear map $$L$$ itself is updated to:

$$\tilde{L_j^i} = \sum_{kl}~B_k^i~L_l^k~A_j^l.$$

Again, note we use both superscript and subscript for the linear map $$L$$, since it contains one covariant component and one contravariant component.
Further more, we can extend this process and define the tensor.
A tensor $T$ is an object that is invariant under a change of coordinates, and with a change of coordinates its component changes in a special way.
The way is that:

$$\tilde{T_{xyz~\ldots}^{abc~\ldots}} = \sum_{ijk\ldots~rst\ldots}~B_i^aB_j^bB_k^c\ldots~T_{rst~\ldots}^{ijk~\ldots}~A_x^rA_y^sA_z^t\ldots$$

Here the $ijk\ldots$ are indices of the contravariant part of the tensor and the $$rst\ldots$$ are that of the covariant part.

One of the important operations of tensor is the *tensor contraction*. We are familiar with the matrix multiplication:
$$C_j^i = \sum_{k}A_k^iB_j^k.$$ 
The *contraction* operations extends this process to multiple dimension space.
It sums the products of the two ndarrays' elements over specified axes.
For example, we can perform the matrix multiplication with contraction:

```ocaml:matmul
let x = Mat.uniform 3 4
let y = Mat.uniform 4 5

let z1 = Mat.dot x y
let z2 = Arr.contract2 [|(1,0)|] x y
```

We can see that the matrix multiplication is a special case of contraction operation and can be implemented with it.

Next, let's extend the two dimension case to multiple dimensions.
Let's say we have two three-dimensional array A and B. We hope to compute the matrix C so that:

$$C_j^i = \sum_{hk}~A_{hk}^i~B_j^{kh}$$

We can use the `contract2` function in the `Ndarray` module. It takes an array of `int * int` tuples to specifies the pair of indices in the two input ndarrays. Here is the code:

```ocaml:contraction
let x = Arr.sequential [|3;4;5|]
let y = Arr.sequential [|4;3;2|]

let z1 = Arr.contract2 [|(0, 1); (1, 0)|] x y
```

The indices mean that, in the contraction, the 0th dimension of `x` corresponds with the 1st dimension of `y`, an the 1st dimension of `x` corresponds with the 0th dimension of `y`.
We can verify the result with the naive way of implementation:

```ocaml:contraction
let z2 = Arr.zeros [|5;2|]

let _ =
  for h = 0 to 2 do
    for k = 0 to 3 do
      for i = 0 to 4 do
        for j = 0 to 1 do
          let r = (Arr.get x [|h;k;i|]) *. (Arr.get y [|k;h;j|]) in
          Arr.set z2 [|i;j|] ((Arr.get z2 [|i;j|]) +. r)
        done
      done
    done
  done
```

Then we can check if the two results agree:

```ocaml:contraction
# Arr.equal z1 z2;;
- : bool = true
```

The contraction can also be applied on one single ndarray to perform the reduction operation using the `contract1` function.

```ocaml:contraction-01
# let x = Arr.sequential [|2;2;3|];;
val x : Arr.arr =

       C0 C1 C2
R[0,0]  0  1  2
R[0,1]  3  4  5
R[1,0]  6  7  8
R[1,1]  9 10 11

```

```ocaml:contraction-01
# let y = Arr.contract1 [|(0,1)|] x;;
val y : Arr.arr =
  C0 C1 C2
R  9 11 13

```

We can surely perform the matrix multiplication with contraction.
High-performance implementation of the contraction operation has been a research topic.
Actually, many tensor operations involve summation over particular indices.
Therefore in using tensors in applications such as linear algebra and physics, the *Einstein notation* is used to simplified notations.
It removes the common summation notation, and also, any twice-repeated index in a term is summed up (no index is allowed to occur three times or more in a term).
For example, the matrix multiplication notation $$C_{ij} = \sum_{k}A_{ik}B_{kj}$$ can be simplified as C = $$A_{ik}B_{kj}$$.

The tensor calculus is of important use in disciplines such as geometry and physics.
More details about the tensor calculation is beyond the scope of this book. 

## Summary

N-dimensional array is the fundamental data type in Owl, as well as in many other numerical libraries such as NumPy.
This chapter explain in detail the Ndarray module, including its creation, properties, manipulation, serialization, etc.
Besides, we also discuss the subtle difference between tensor and ndarray in this chapter.
This chapter is easy to follow, and can serve as a reference whenever users need a quick check of functions they need.

