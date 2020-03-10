# Linear Algebra

Linear Algebra: important. 
It is beyond the scope of this book. Please refer to [@strang2006linear] for this subject.
This chapter also follows the basic structure of this book.

This chapter briefly covers the linear algebra modules in Owl. 

There are two levels abstraction in Owl's `Linalg` module:
* low-level raw interface to CBLAS and LAPACKE;
* high-level wrapper functions in `Linalg` module;
The example in this chapter mostly use the high level wrapper. Please refer to the last section for details on CBLAS.

The `Linalg` has the following module structure:

- [Owl.Linalg.Generic](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli): generic functions for four number types `S/D/C/Z`.

- [Owl.Linalg.S](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_s.mli): only for `float32` type.

- [Owl.Linalg.D](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_d.mli): only for `float64` type.

- [Owl.Linalg.C](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_c.mli): only for `complex32` type.

- [Owl.Linalg.Z](https://github.com/owlbarn/owl/blob/master/src/owl/linalg/owl_linalg_z.mli): only for `complex64` type.

`Generic` actually can do everything that `S/D/C/Z` can but needs some extra type information. The functions in `Linalg` module are divided into the following groups.


## Vectors and Matrices

The fundamental problem of linear algebra: solving linear equations.
This is more efficiently expressed with vectors and matrices.
We need to get familiar with these basic structures in Owl.

Owl supports eight kinds of matrices as below, all the elements in a matrix are (real/complex) numbers.

* `Dense.Matrix.S` : Dense matrices of single precision float numbers.
* `Dense.Matrix.D` : Dense matrices of double precision float numbers.
* `Dense.Matrix.C` : Dense matrices of single precision complex numbers.
* `Dense.Matrix.Z` : Dense matrices of double precision complex numbers.

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

![Visualise the distribution of non-zero elements in matrices.](images/matrix/plot_00.png "plot 00"){ width=90% #fig:linear-algebra:mat_plot_00 }


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


## Gaussian Elimination

Solving linear equations systems is the core problem in Linear Algebra and is frequently used in scientific computation.
*Gaussian Elimination* is a classic method to do that.
Here is a simple example. 

EXAMPLE: a three-variable simple example. 
Step 1, 2, 3.

This process can be more concisely expressed with vector:

EQUATION: column vector

or matrix:

Equation: matrix format of the same problem.

### LU Factorisation

Let's check the gaussian elimination example again. 

EQUATION: the final form of elimination

We call it an *upper triangular* matrix.

Then we consider the three steps that changes matrix `A` to `U`.
Each can be represented with an matrix called *elementary matrix*.

Equation: P1*P2*P3*A = U

If we take only the elementary matrices, they leads to another matrix:

Equation: P1*P2*P3 = L

This is called a *Lower Triangular* matrix.

In other word, `A=LU`. This is called an LU factorisation, a very important idea.
We can use the `is_triu` and `is_tril` to verify if a matrix is triangular.

EXAMPLE

The importance of dividing a matrix is that with triangular matrices, it becomes very easy to solve the linear equations. 
`triangular_solve`.


We can use the `lu` function to do it.

EXAMPLE

The input is a matrix.
The first two returned results are the L and U. 

The third is about pivoting.

Explain pivoting. Refer to Matlab book Sec 2.5-2.6.

### Inverse and Transpose 

Zero and one matrices $I$. 
One matrix is a special form of *Diagonal Matrix*, which is a square matrix that only contains non-zero element along its diagonal. 
You can check if a matrix is diagonal with `is_diag` function.

```
CODE: I is diag
```

The inverse of a nxn square matrix: $AA^{-1} = I$

Not all square matrix has inverse.  
One important thing: a n by n matrix A is invertible if and only if it has n pivots. 

We use function `inv` to do the inverse operation. It's straightforward and easy to verify:

```ocaml
  let x = Mat.semidef 8;;    (* generate a random semidef matrix *)
  let y = Linalg.D.inv x;;   (* calculate the matrix inverse *)
  Mat.(x *@ y =~ eye 8);;    (* check the approx equality *)
```

Next is the *Transpose Matrix*. Denoted by $A^T$, its $i$th row is taken from the $i$-th column of the original matrix A.
It has properties such as $(AB)^T=B^T~A^T$. 
We can check this property using the matrix function `Mat.transpose`.

```
CODE: transpose
```

A related special matrix is the *Symmetric Matrix*, which equals to its own transpose. This simple test can be done with the `is_symmetric` function.


## Vector Spaces

EXPLAIN the concepts of vector space and subspace.

EXPLAIN what is column space and null space, and the other two. 

### Rank and Basis

We have seen using LU factorisation to solve Ax=b, but it won't work every time. 
For one thing, $A$ may not be square matrix. 
Or the given information is not enough (multiple solutions)

Example


Besides, some of the equations might be "useless".
One of these equation provide no new information. 

Example: a linear-dependent square matrix. 

Understanding a bit of these theories will be helpful to using functions, instead blindly believing the function can give you solution every time. 
(In this section I need to show some "fail" examples) 


Introduce: rank.
The definition of linear independence.
The definition and implication of rank.

A Example using rank:

```
val rank : ?tol:float -> ('a, 'b) t -> int
  (* rank of a rectangular matrix *)
```

One application of rank is in a crucial Linalg idea: basis. linear independent (Ax=0), and spanning the space (Ax=b).

Dimension of a vector space. 

Suppose in a n-dimension space.
The dimension of column space is $r$; the dimension of nullspace N(A) = n - r. 

### Orthogonality

Basis is like coordinates in  the space. 
But in a 2 or 3 dimensional  cartesian space, we often use the orthogonal axis as basis. The same can be applied here. 

Orthogonal vectors and subspaces.

Orthonormal basis: orthogonal among each other, and then normalised  to unit vector.

Orthogonal matrix: columns ar orthogonal 

For example, the basis for the null space:
```
val null : ('a, 'b) t -> ('a, 'b) t
(* an orthonormal basis for the null space of a matrix *)
```


The method to construct orthogonal basis in a subspace is called the Gram-Schmidt orthogonalisation.

QR factorisation.

```
val qr : ?thin:bool -> ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t
  (* QR factorisation  *)
```

### Solving Ax = b

We can now discuss the general solution to $Ax=b$.

Several theorems:
There exist non-zero solutions to $Ax=0$ if and only if rank(A) < column of A.

Example

Suppose A is mxn matrix, and b is mx1 matrix, then there exist solution if and only if r(A) = r([A, b]). 
If and only if r(A) also equals to n, there exist one solution.

`linsolve a b -> x` solves a linear system of equations `a * x = b` in the following form. By default, `typ=n` and the function use LU factorisation with partial pivoting when `a` is square and QR factorisation with column pivoting otherwise. The number
of rows of `a` must equal the number of rows of `b`.
If `a` is a upper(lower) triangular matrix, the function calls the ``solve_triangular`` function.

```
val linsolve : ?trans:bool -> ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
```

Example: only using the function does not work; you have to do some comparison of rank.


## Determinants

For a square matrix. the definition of *determinants*:

EQUATION

There are  many techniques to simplify this calculation. But we use the `det` function here to calculate the determinants of a matrix. 

There is a similar function `logdet`. 
It computes the logarithm of the determinant, but it avoids the possible overflow or underflow problems in computing determinant of large matrices.

Why is this concept important? 
We use the determinant to calculate the solution to Ax=b.

Cramer's rule.

Example: using `solve` and `det` to solve a Ax=b.

Another important application is to use determinant to decide if a square matrix A is invertible/singular. 
A is invertible if and only if $|A|$ does not equal to 0. 

This theorem is widely used in finding *eigenvalues*. As will be shown in the next section.


## Eigenvalues and Eigenvectors

Now we change from $Ax=b$ to $Ax=\lambda~x$.

We use ODE system as an example. (Following the Textbook)

It can be written as:

$(A - \lambda~I)x = 0$

If there exist number $\lambda$ and non-zero column vector $x$ to satisfy this equation, then $\lambda$ is called *eigenvalue*, and $x$ is called the *eigenvector* of this matrix A.

Continue the previous example to solve it manually.

According to the theory of polynomials, equation xx has and only has n roots in the complex space. 

And we have function `eig` and `eigbvals` to do that:


CODE: using these functions to repeat the previous example.
```text

  val eig : ?permute:bool -> ?scale:bool -> otyp:('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t * ('a, 'b) t
  (* right eigenvectors and eigenvalues of an arbitrary square matrix *)

  val eigvals : ?permute:bool -> ?scale:bool -> otyp:('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t
  (* only computes the eigenvalues of an arbitrary square matrix *)
```

Once we get the eigenvalue, we can return to the solution to ODE:
A linear combination.

### Diagonalisation

Diagonal matrix is the easiest to deal with.
Several properties....

Diagonalisation

Use one example to demonstrate how to solve the ODE system with diagonalisation. 

That leads to a brief discussion of stability with eigenvalues.
Explain the intuition why Eigenvalue is important in different fields.

### Complex Matrices

Eigenvalue and vector in the complex space. 

Hermitian: extend the transpose to complex space. 
Matrix that equal their conjugate transpose. 

```
  val is_hermitian : ('a, 'b) t -> bool
  (* check if a matrix is hermitian *)
```

Definition of Unitary matrices: $UU^H=I$

Example of using eig on complex matrices.


### Similarity Transformation

Definition: Similar matrix.
The point is to make clear its intuition: change basis, linear transformation. 
And transforming to diagonal is the easiest. 

A property: if $A = A^H$, every eigenvalue is real.
And that a real symmetric matrix can be factored into $A=Q\lambda~Q^T$.

```
  val schur : otyp:('c, 'd) kind -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * ('c, 'd) t
  (* Schur factorisation *)
```

Example. 
If possible, shows how similar matrix simplify the problem.

Jordan form: very brief explain.

## Positive Definite Matrices

```
val chol : ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (* Cholesky factorisation *)
```

### Positive Definiteness


```
val is_posdef : ('a, 'b) t -> bool
  (* check if a matrix is positive semi-definite *)
```


### Singular Value Decomposition

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

```
val pinv : ?tol:float -> ('a, 'b) t -> ('a, 'b) t
  (* Moore-Penrose pseudo-inverse of a matrix *)
```

The Moore-Penrose inverse is a direct application of the SVD.

## Computations with Matrices

### Matrix Norm and Condition Number 

```text

  val norm : ?p:float -> ('a, 'b) t -> float
  (* p-norm of a matrix *)

  val cond : ?p:float -> ('a, 'b) t -> float
  (* p-norm condition number of a matrix *)

  val rcond : ('a, 'b) t -> float
  (* estimate for the reciprocal condition of a matrix in 1-norm *)
```

**Other Factorisations**

```
val hess : ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (* Hessenberg form of a given matrix *)
```


## Linear Programming 

TODO: placeholder for future implementation. Or in the optimisation chapter. 
Understand the method used such as interior  point, and then make the decision.

## Internal: CBLAS and LAPACKE

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


## Sparse Matrices

Very brief. 
Focusing on introducing the data structure (CSC, CSR, etc), no the method.

Mention the [owl_suitesparse](https://github.com/owlbarn/owl_suitesparse)

* `Sparse.Matrix.S` : Sparse matrices of single precision float numbers.
* `Sparse.Matrix.D` : Sparse matrices of double precision float numbers.
* `Sparse.Matrix.C` : Sparse matrices of single precision complex numbers.
* `Sparse.Matrix.Z` : Sparse matrices of double precision complex numbers.

TODO: Introduce the sparse data structure in owl, and introduce CSR, CSC, tuples, and other formats.


**extra material on LQ**

```text
  val lq : ?thin:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t
  (* LQ factorisation *)
```

The following code performs an LQ decomposition on a random square matrix. Note that in the last step we used `=~` rather than `=` to check the equality due to float number precision. You can check the difference with `Mat.(l *@ q - x)`.

```ocaml

  let x = Mat.uniform 8 8;;    (* generate a random matrix *)
  let l, q = Linalg.D.lq x;;   (* perform lq decomposition *)
  Mat.(l *@ q =~ x);;          (* check the approx equality *)

```
