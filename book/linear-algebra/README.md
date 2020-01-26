# Linear Algebra

Linear Algebra: important. 
It is beyond the scope of this bool. Please refer to [...] for this subject.

This chapter briefly covers the linear algebra modules in Owl. 

There are two levels abstraction in Owl's `Linalg` module:
* low-level raw interface to CBLAS and LAPACKE;
* high-level wrapper functions in `Linalg` module;
The example in this chapter mostly use the high level wrapper. Please refer to the last section for details on CBLAS.


## Matrices

Before diving into the linear algebra, first we need to understand how matrix in Owl works. 
It is based on ndarray, but provides unique functions. 

Owl supports eight kinds of matrices as below, all the elements in a matrix are (real/complex) numbers.

* `Dense.Matrix.S` : Dense matrices of single precision float numbers.
* `Dense.Matrix.D` : Dense matrices of double precision float numbers.
* `Dense.Matrix.C` : Dense matrices of single precision complex numbers.
* `Dense.Matrix.Z` : Dense matrices of double precision complex numbers.
* `Sparse.Matrix.S` : Sparse matrices of single precision float numbers.
* `Sparse.Matrix.D` : Sparse matrices of double precision float numbers.
* `Sparse.Matrix.C` : Sparse matrices of single precision complex numbers.
* `Sparse.Matrix.Z` : Sparse matrices of double precision complex numbers.

There are many common functions shared by these eight modules, therefore I will use `Mat` module (which is an alias of `Dense.Matrix.D` module) in the following examples. These examples should be able to applied to other modules without too much changes, but note some modules do have its own specific functions such as `Dense.Matrix.Z.re`.

### Create Matrices

There are multiple functions to help you in creating an initial matrix to start with.

```ocaml

  Mat.empty 5 5;;        (* create a 5 x 5 matrix with initialising elements *)
  Mat.create 5 5 2.;;    (* create a 5 x 5 matrix and initialise all to 2. *)
  Mat.zeros 5 5;;        (* create a 5 x 5 matrix of all zeros *)
  Mat.ones 5 5;;         (* create a 5 x 5 matrix of all ones *)
  Mat.eye 5;;            (* create a 5 x 5 identity matrix *)
  Mat.uniform 5 5;       (* create a 5 x 5 random matrix of uniform distribution *)
  Mat.uniform_int 5 5;;  (* create a 5 x 5 random integer matrix *)
  Mat.sequential 5 5;;   (* create a 5 x 5 matrix of sequential integers *)
  Mat.semidef 5;;        (* create a 5 x 5 random semi-definite matrix *)
  Mat.gaussian 5 5;;     (* create a 5 x 5 random Gaussian matrix *)

```

As you noticed, the last example is to create a random matrix where the elements follow a Gaussian distribution. What about creating another matrix where the element follow another distribution, e.g., `t-distribution`? Easy!

```ocaml

# Mat.(empty 5 5 |> map (fun _ -> Stats.t_rvs ~df:1. ~loc:0. ~scale:1.))
- : Mat.mat =

          C0        C1        C2          C3        C4
R0   1.44768 -0.349538 -0.600692    -15.6261   2.07554
R1  -1.25034   1.20008  -1.76243   -0.719415 -0.580605
R2  -1.71484   10.1152 -0.138612    0.276529 -0.355326
R3   0.83227  -6.36336   1.48695 -0.00277443 -0.791397
R4 -0.336031   -1.7789 -0.113224     4.15084   -2.1577

```

So, what we did is first creating an empty `5 x 5` matrix, then mapping each element to a random number following `t-distribution`. The example utilises Owl's `Stats` module which I will introduce in another tutorial.

Alternatively, you can use `uniform` function to generate the input values, as below.

```ocaml

# Mat.(uniform 5 5 |> map Maths.sin)
- : Mat.mat =

         C0        C1       C2       C3       C4
R0  0.11068 0.0997998 0.185571 0.521833 0.583662
R1 0.818164  0.426204 0.524001 0.395543 0.590104
R2 0.420941  0.496159 0.084013 0.425077 0.443924
R3 0.694034  0.147498 0.430752 0.302604 0.128698
R4 0.840643  0.163237 0.658268 0.457176 0.175289

```

Owl can create some special matrices with specific properties. For example, a *magic square* is a `n x n` matrix (where n is the number of cells on each side) filled with distinct positive integers in the range $1,2,...,n^{2}$ such that each cell contains a different integer and the sum of the integers in each row, column and diagonal is equal.

```ocaml env=matrix_env2
# let x = Mat.magic 5
val x : Mat.mat =

   C0 C1 C2 C3 C4
R0 17 24  1  8 15
R1 23  5  7 14 16
R2  4  6 13 20 22
R3 10 12 19 21  3
R4 11 18 25  2  9

```

We can validate this property with the following code. The summation of all the elements on each column is 65.

```ocaml env=matrix_env2
# Mat.sum_rows x
- : Mat.mat =
   C0 C1 C2 C3 C4
R0 65 65 65 65 65

```

The summation of all the elements on each row is 65.

```ocaml env=matrix_env2
# Mat.sum_cols x
- : Mat.mat =
   C0
R0 65
R1 65
R2 65
R3 65
R4 65

```

The summation of all the diagonal elements is also 65.

```ocaml env=matrix_env2
# Mat.trace x
- : float = 65.
```

The last example creates three matrices where the elements follow Bernoulli distribution of different parameters. We then use `Plot.spy` function to visualise how the non-zero elements are distributed in the matrices.

```ocaml
let x = Mat.bernoulli ~p:0.1 40 40 in
let y = Mat.bernoulli ~p:0.2 40 40 in
let z = Mat.bernoulli ~p:0.3 40 40 in

let h = Plot.create ~m:1 ~n:3 "plot_00.png" in
Plot.subplot h 0 0;
Plot.spy ~h x;
Plot.subplot h 0 1;
Plot.spy ~h y;
Plot.subplot h 0 2;
Plot.spy ~h z;
Plot.output h;;
```

<img src="images/matrix/plot_00.png" alt="plot 00" title="matrix example 00" width="700px" />


### Access Elements

All four matrix modules support `set` and `get` to access and modify matrix elements.

```ocaml env=matrix_env0

let x = Mat.uniform 5 5;;
Mat.set x 1 2 0.;;             (* set the element at (1,2) to 0. *)
Mat.get x 0 3;;                (* get the value of the element at (0,3) *)

```

For dense matrices, i.e., `Dense.Matrix.*`, you can also use shorthand `.%{[|i,j|]}` to access elements.

```ocaml env=matrix_env0
# open Mat
# x.%{[|1;2|]} <- 0.;;         (* set the element at (1,2) to 0. *)
- : unit = ()
# let a = x.%{[|0;3|]};;       (* get the value of the element at (0,3) *)
val a : float = 0.181287028128281236
```


The modifications to a matrix using `set` are in-place. This is always true for dense matrices. For sparse matrices, the thing can be complicated because of performance issues. I will discuss about sparse matrices separately in a separate post.


### Iterate Elements, Rows, Columns

In reality, a matrix usually represents a collections of measurements (or points). We often need to go through these data again and again for various reasons. Owl provides very convenient functions to help you to iterate these elements. There is one thing I want to emphasise: Owl uses row-major matrix for storage format in the memory, which means accessing rows are much faster than those column operations.

Let's first create a `4 x 6` matrix of sequential numbers as below.

```ocaml env=matrix_env1

let x = Mat.sequential 4 6;;

```

You should be able to see the following output in your `utop`.

```text

     C0 C1 C2 C3 C4 C5
  R0  1  2  3  4  5  6
  R1  7  8  9 10 11 12
  R2 13 14 15 16 17 18
  R3 19 20 21 22 23 24

```

Iterating all the elements can be done by using `iteri` function. The following example prints out all the elements on the screen.

```ocaml env=matrix_env1

Mat.iteri_2d (fun i j a -> Printf.printf "(%i,%i) %.1f\n" i j a) x;;

```

If you want to create a new matrix out of the existing one, you need `mapi` and `map` function. E.g., we create a new matrix by adding one to each element in `x`.

```ocaml env=matrix_env1

# Mat.map ((+.) 1.) x
- : Mat.mat =

   C0 C1 C2 C3 C4 C5
R0  1  2  3  4  5  6
R1  7  8  9 10 11 12
R2 13 14 15 16 17 18
R3 19 20 21 22 23 24

```

We can take some rows out of `x` by calling `rows` function. The selected rows will be used to assemble a new matrix.

```ocaml env=matrix_env1

# Mat.rows x [|0;2|]
- : Mat.mat =

   C0 C1 C2 C3 C4 C5
R0  0  1  2  3  4  5
R1 12 13 14 15 16 17

```

Similarly, we can also select some columns as below.

```ocaml env=matrix_env1

# Mat.cols x [|3;2;1|]
- : Mat.mat =

   C0 C1 C2
R0  3  2  1
R1  9  8  7
R2 15 14 13
R3 21 20 19

```

Iterating rows and columns are similar to iterating elements, by using `iteri_rows`, `mapi_rows`, and etc. The following example prints the sum of each row.

```ocaml env=matrix_env1

  Mat.iteri_rows (fun i r ->
    Printf.printf "row %i: %.1f\n" i (Mat.sum' r)
  ) x;;

```

You can also fold elements, rows, and columns. Let's first calculate the summation of all elements.

```ocaml env=matrix_env1

  Mat.fold (+.) 0. x;;

```

Now, we calculate the summation of all column vectors by using `fold_cols` fucntion.

```ocaml env=matrix_env1

  let v = Mat.(zeros (row_num x) 1) in
  Mat.(fold_cols add v x);;

```

It is also possible to change a specific row or column. E.g., we make a new matrix out of `x` by setting row `2` to zero vector.

```ocaml env=matrix_env1

  Mat.map_at_row (fun _ -> 0.) x 2;;

```

### Filter Elements, Rows, Columns

To continue use the previous sequential matrix, I will make some examples to show how to examine and filter elements in a matrix. The first one is to filter out the elements in `x` greater than `20`.

```ocaml env=matrix_env1
# Mat.filter ((<) 20.) x
- : int array = [|21; 22; 23|]
```

You can compare the next example which filters out the two-dimensional indices.

```ocaml env=matrix_env1
# Mat.filteri_2d (fun i j a -> a > 20.) x
- : (int * int) array = [|(3, 3); (3, 4); (3, 5)|]
```

The second example is to filter out the rows whose summation is less than `22`.

```ocaml env=matrix_env1
# Mat.filter_rows (fun r -> Mat.sum' r < 22.) x
- : int array = [|0|]
```

If we want to check whether there is one or (or all) element in `x` satisfying some condition, then

```ocaml env=matrix_env1

  Mat.exists ((>) 5.) x;;      (* is there someone smaller than 5. *)
  Mat.not_exists ((>) 5.) x;;  (* is no one smaller than 5. *)
  Mat.for_all ((>) 5.) x;;     (* is everyone smaller than 5. *)

```


### Compare Two Matrices

Comparing two matrices is just so easy by using module infix `=@`, `<>@`, `>@`, and etc. Let's first create another matrix `y` by multiplying two to every elements in `x`.

```ocaml env=matrix_env1

  let y = Mat.map (( *. ) 2.) x;;

```

Then we can compare the relationship of `x` and `y` as below. Note, the relationship is derived by checking every elements in both matrices. E.g., `x` is equal to `y` means every element in `x` is equal the corresponding element in `y`.

```ocaml env=matrix_env1

  Mat.(x = y);;    (* is x equal to y *)
  Mat.(x <> y);;   (* is x unequal to y *)
  Mat.(x > y);;    (* is x greater to y *)
  Mat.(x < y);;    (* is x smaller to y *)
  Mat.(x >= y);;   (* is x not smaller to y *)
  Mat.(x <= y);;   (* is x not greater to y *)

```

All aforementioned infix have their corresponding functions in the module, e.g., `=@` has `Mat.is_equal`. Please refer to the documentation.


### Matrix Arithmetics

The arithmetic operation also heavily uses infix. Similar to matrix comparison, each infix has its corresponding function in the module.

```ocaml env=matrix_env1

  Mat.(x + y);;    (* add two matrices *)
  Mat.(x - y);;    (* subtract y from x *)
  Mat.(x * y);;    (* element-wise multiplication *)
  Mat.(x / y);;    (* element-wise division *)
  Mat.(x *@ y);;    (* dot product of x and y *)

```

If you do match between a matrix and a scalar value, you need to be careful about their order. Please see the examples below. In the following examples, `x` is a matrix as we used before, and `a` is a `float` scalar value.

```ocaml env=matrix_env1
  let a = 2.5;;

  Mat.(x +$ a);;    (* add a to every element in x *)
  Mat.(a $+ x);;    (* add a to every element in x *)

```

Similarly, we have the following examples for other math operations.

```ocaml env=matrix_env1

  Mat.(x -$ a);;    (* sub a from every element in x *)
  Mat.(a $- x);;
  Mat.(x *$ a);;    (* mul a with every element in x *)
  Mat.(a $* x);;
  Mat.(x /$ a);;    (* div a to every element in x *)
  Mat.(a $/ x);;
  Mat.(x **$ a);;   (* power of every element in x *)

```

There are some ready-made functions to ease your life when operating matrices.

```ocaml env=matrix_env1

  Mat.log10 x;;     (* logarithm of every element in x *)
  Mat.abs x;;       (* absolute value of every element in x *)
  Mat.neg x;;       (* negation of every element in x *)

```

For more advanced operations such as `svd` and `qr` operations, you need to use `Linalg` module. Currently, `Linalg` only works on dense matrices of real numbers. I will provide more supports for other types of matrices in future.

```ocaml env=matrix_env1

  Linalg.D.svd x;;  (* svd of x *)
  Linalg.D.qr x;;   (* QR decomposition of x *)
  Linalg.D.inv x;;  (* inverse of x *)
  Linalg.D.det x;;  (* determinant of x *)

```


### Save & Load Matrices

All matrices can be serialised to storage by using `save`. Later, you can load a matrix using `load` function.

```ocaml env=matrix_env1

  Mat.save "m0.mat" x;;    (* save x to m0.mat file *)
  Mat.load "m0.mat";;      (* load m0.mat back to the memory *)

```

I also made corresponding `save_txt` and `load_txt` functions for a simple tab-delimited, human-readable format. Note the performance is much worse than the corresponding `save` and `load`.


### Other Operations

I will use another set of examples to finish this tutorial. I must say this tutorial has not presented all the operations provided by Owl's matrix modules. There are much more operations you can explore by reading its documents.

```ocaml env=matrix_env1

  Mat.(x @= y);;    (* concatenate x and y vertically *)
  Mat.(x @|| y);;   (* concatenate x and y horizontally *)

```

```ocaml
  let x = Sparse.Matrix.D.uniform 10 10;; (* create a sparse matrix with uniform rvs *)
  Sparse.Matrix.D.map_nz ((+.) 1.) x;;    (* add one to non-zero elements in a sparse matrix *)
  Sparse.Matrix.D.density x;;             (* show the density of a sparse matrix *)

```


## Linear Algebra Basic

Now that we are familiar with matrices in Owl, let's look at the linear algebra module. 

The `Linalg` has the following module structure:

- [Owl.Linalg.Generic](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli): generic functions for four number types `S/D/C/Z`.

- [Owl.Linalg.S](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_s.mli): only for `float32` type.

- [Owl.Linalg.D](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_d.mli): only for `float64` type.

- [Owl.Linalg.C](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_c.mli): only for `complex32` type.

- [Owl.Linalg.Z](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_z.mli): only for `complex64` type.

`Generic` actually can do everything that `S/D/C/Z` can but needs some extra type information. The functions in `Linalg` module are divided into the following groups.

Here we will use a simple example to demonstrate the basic functions in LinAlg.
This example is a matrix that represents a graph structure... (example detail. This example has to be carefully chosen so that the operations are meaningful.)

Below is a full list of basic functions provided by LA module.

```text

  val inv : ('a, 'b) t -> ('a, 'b) t
  (* inverse of a square matrix *)

  val pinv : ?tol:float -> ('a, 'b) t -> ('a, 'b) t
  (* Moore-Penrose pseudo-inverse of a matrix *)

  val det : ('a, 'b) t -> 'a
  (* determinant of a square matrix  *)

  val logdet : ('a, 'b) t -> 'a
  (* log of the determinant of a square matrix *)

  val rank : ?tol:float -> ('a, 'b) t -> int
  (* rank of a rectangular matrix *)

  val norm : ?p:float -> ('a, 'b) t -> float
  (* p-norm of a matrix *)

  val cond : ?p:float -> ('a, 'b) t -> float
  (* p-norm condition number of a matrix *)

  val rcond : ('a, 'b) t -> float
  (* estimate for the reciprocal condition of a matrix in 1-norm *)

  val is_triu : ('a, 'b) t -> bool
  (* check if a matrix is upper triangular *)

  val is_tril : ('a, 'b) t -> bool
  (* check if a matrix is lower triangular *)

  val is_symmetric : ('a, 'b) t -> bool
  (* check if a matrix is symmetric *)

  val is_hermitian : ('a, 'b) t -> bool
  (* check if a matrix is hermitian *)

  val is_diag : ('a, 'b) t -> bool
  (* check if a matrix is diagonal *)

  val is_posdef : ('a, 'b) t -> bool
  (* check if a matrix is positive semi-definite *)

```

The following code calculates the inverse of a square matrix `x`.

```ocaml

  let x = Mat.semidef 8;;    (* generate a random semidef matrix *)
  let y = Linalg.D.inv x;;   (* calculate the matrix inverse *)
  Mat.(x *@ y =~ eye 8);;    (* check the approx equality *)

```

## Solve A Linear Equation System

One of the problems encountered most frequently in scientific computation is the solution of systems of simultaneous linear equations.
(Such as? find very concrete real world applications).

A simple example, see reference book Matlab for Section 2.2

### LR Factorisation

A classical method: Gaussian -- LR

```text
  val lu : ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t
```

### What Owl Provides

As user, we don't have to choose or use this tedious approach. Owl has already provides:

```text

  val null : ('a, 'b) t -> ('a, 'b) t
  (* an orthonormal basis for the null space of a matrix *)

  val linsolve : ?trans:bool -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (* solves `A * x = b` linear equation system. *)

  val linreg : ('a, 'b) t -> ('a, 'b) t -> 'a * 'a
  (* simple linear regression using OLS. *)

```

**Example**

The code snippet below first generates some random data, then using `linreg` function to perform a simple linear regression and plots the data as well as the regression line.

```ocaml file=../../examples/code/linear-algebra/example_00.ml
let generate_data () =
  let x = Mat.uniform 500 1 in
  let p = Mat.uniform 1 1 in
  let y = Mat.(x *@ p + gaussian ~sigma:0.05 500 1) in
  x, y

let t1_sol () =
  let x, y = generate_data () in
  let h = Plot.create "plot_00.png" in
  let a, b = Linalg.D.linreg x y in
  let y' = Mat.(x *$ b +$ a) in
  Plot.scatter ~h x y;
  Plot.plot ~h ~spec:[ RGB (0,255,0) ] x y';
  Plot.output h
```

<img src="images/linear-algebra/plot_00.png" alt="plot 00" title="Linalg example 00" width="600px" />



Some discussion about the implementation details of `linsolve`.


## Singular Value Decomposition

We have talked about LR Factorisation. Another important factorisation is SVD.
The singular value decomposition (SVD) is among the most important matrix factorizations
of the computational era.
The SVD provides a numerically stable matrix decomposition that can be used for
a variety of purposes and is guaranteed to exist. 

Refer to: [Data Driven Science and Engineering](https://www.cambridge.org/core/books/datadriven-science-and-engineering/77D52B171B60A496EAFE4DB662ADC36E), chapter 2.

What we provide:

```text
  val svd : ?thin:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * ('a, 'b) t
  (* singular value decomposition *)

  val svdvals : ('a, 'b) t -> ('a, 'b) t
  (* only singular values of SVD *)

  val gsvd : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * ('a, 'b) t * ('a, 'b) t * ('a, 'b) t * ('a, 'b) t
  (* generalised singular value decomposition *)

  val gsvdvals : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
  (* only singular values of generalised SVD *)

```

The following code performs an SVD on a random matrix then check the equality.

```ocaml

  let x = Mat.uniform 8 16;;        (* generate a random matrix *)
  let u, s, vt = Linalg.D.svd x;;   (* perform lq decomposition *)
  let s = Mat.diagm s;;             (* exapand to diagonal matrix *)
  Mat.(u *@ s *@ vt =~ x);;         (* check the approx equality *)

```

## Other Factorisations

Besides LR and SVD, other Factorisation methods are also supported:

```text

  val lq : ?thin:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (* LQ factorisation *)

  val qr : ?thin:bool -> ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t
  (* QR factorisation  *)

  val chol : ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (* Cholesky factorisation *)

  
  val schur : otyp:('c, 'd) kind -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * ('c, 'd) t
  (* Schur factorisation *)

  val hess : ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (* Hessenberg form of a given matrix *)

```

Explain their importance and some math background.

The following code performs an LQ decomposition on a random square matrix. Note that in the last step we used `=~` rather than `=` to check the equality due to float number precision. You can check the difference with `Mat.(l *@ q - x)`.

```ocaml

  let x = Mat.uniform 8 8;;    (* generate a random matrix *)
  let l, q = Linalg.D.lq x;;   (* perform lq decomposition *)
  Mat.(l *@ q =~ x);;          (* check the approx equality *)

```

And more examples.


## Eigenvalues & eigenvectors

Refer to book NR, Chapter 11 for theory details. 

```text

  val eig : ?permute:bool -> ?scale:bool -> otyp:('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t * ('a, 'b) t
  (* right eigenvectors and eigenvalues of an arbitrary square matrix *)

  val eigvals : ?permute:bool -> ?scale:bool -> otyp:('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t
  (* only computes the eigenvalues of an arbitrary square matrix *)

```

Example:
The following code calculates the right eigenvalues and eigenvectors of a positive-definite matrix `x`.

```ocaml

  let x = Mat.semidef 8;;                                  (* generate a random matrix *)
  let v, w = Linalg.D.eig ~permute:false ~scale:false x;;  (* calculate eigenvalues and vectors *)
  let v = Dense.Matrix.Z.re v;;                            (* only real part since [x] is semidef *)
  let w = Dense.Matrix.Z.re w;;                            (* only real part since [x] is semidef *)
  Mat.((x *@ v) =~ (w * v));;                              (* check the approx equality *)

```

## CBLAS & LAPACKE

This section is for those of you who are eager for more low level information.

The Background: BLAS, a brief history. How we include that into Owl.

### Low-level Interface to CBLAS & LAPACKE

Owl has implemented the full interface to CBLAS and LAPACKE. Comparing to Julia which chooses to interface to BLAS/LAPACK, you might notice the extra `C` in `CBLAS` and `E` in `LAPACKE` because they are the corresponding C-interface of Fortran implementations. It is often believed that C-interface may introduce some extra overhead. However, it turns out that we cannot really notice any difference at all in practice when dealing with medium or large problems.

- [Owl_cblas module](https://github.com/ryanrhymes/owl/blob/master/src/owl/cblas/owl_cblas.mli) provides the raw interface to CBLAS functions, from level-1 to level-3. The interfaced functions have the same names as those in CBLAS.

- [Owl_lapacke_generated module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) provides the raw interface to LAPACKE functions (over 1,000) which also have the same names defined in [lapacke.h](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/lapacke.h).

- [Owl_lapacke module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke.ml) is a very thin layer of interface between [Owl_lapacke_generated module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) and [Linalg module](https://github.com/ryanrhymes/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli). The purpose is to provide a unified function to make generic functions over different number types.


### High-level Wrappers in Linalg Module

The functions in [Owl_cblas](https://github.com/ryanrhymes/owl/blob/master/src/owl/cblas/owl_cblas.mli) and [Owl_lapacke_generated](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) are very low-level, e.g., you need to deal with calculating parameters, allocating workspace, post-processing results, and many other tedious details. You do not really want to use them directly unless you have enough background in numerical analysis and chase after the performance. In practice, you should use [Linalg](https://github.com/ryanrhymes/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli) module which gives you a high-level wrapper for frequently used functions.


### Low-level factorisation and Helper functions

```text

  val lufact : ('a, 'b) t -> ('a, 'b) t * (int32, int32_elt) t

  val qrfact : ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t

  val bkfact : ?upper:bool -> ?symmetric:bool -> ?rook:bool -> ('a, 'b) t -> ('a, 'b) t * (int32, int32_elt) t

  val peakflops : ?n:int -> unit -> float
  (* peak number of float point operations using [Owl_cblas.dgemm] function. *)

```
How these low level functions are used in Owl Code. 


