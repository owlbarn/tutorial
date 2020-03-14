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
*Gaussian Elimination* is a classic method to do that. With a bit of techniques, elimination works surprisingly well in modern numerical libraries as one way of implementation.
Here is a simple example. 

$$2x_1 + 2x_2 + 2x_3 = 4$$
$$2x_1 + 2x_2 + 3x_3 = 5$$ {#eq:linear-algebra:gauss01}
$$3w_1 + 4x_2 + 5x_3 = 7$$

Divide the first equation by 2:

$$x_1 + x_2 + x_3 = 2$$
$$2x_1 + 2x_2 + 3x_3 = 5$$ {#eq:linear-algebra:gauss02}
$$3w_1 + 4x_2 + 5x_3 = 7$$

Multiply the first equation by `-2`, then add it to the second one. 
Also, multiply the first equation by `-3`, then add it to the third one. We have:

$$x_1 + x_2 + x_3 = 2$$
$$x_3 = 1$$ {#eq:linear-algebra:gauss03}
$$x_2 + 2x_3 = 1$$

Finally, swap the second and third line:

$$x_1 + x_2 + x_3 = 2$$
$$x_2 + 2x_3 = 1$$  {#eq:linear-algebra:gauss04}
$$x_3 = 1$$

Here $x_3 = 1$, and we can put it back in the second equation and get $x_2 = -1$. 
Put both back to the first equation and we have $x_1 = 2$

This process demonstrate the basic process of elimination: eliminate unknown variables until this group of linear equations is easy to solve, and then do the back-substitution.
There are three kinds of basic operations we can use: multiplication, adding one line to another, and swap two lines. 

The starting [@eq:linear-algebra:gauss01] can be more concisely expressed with vector:

$$x_1\left[\begin{matrix}2\\2\\3\end{matrix} \right] + x_2\left[\begin{matrix}2\\2\\4\end{matrix} \right] + x_3\left[\begin{matrix}2\\3\\5\end{matrix} \right] = \left[\begin{matrix}4\\5\\7\end{matrix} \right]$$

or it can be expressed as $Ax=b$ using matrix notation.

$$\left[\begin{matrix}2 & 2 & 2\\2 & 2 & 3\\3 & 4 & 5\end{matrix} \right] \left[\begin{matrix}x_1\\x_2\\x_3\end{matrix} \right] = \left[\begin{matrix}4\\5\\7\end{matrix} \right]$$

Here A is a matrix, b is a column vector, and x is the unknown vector.
The matrix notation is often used to describe the linear equation systems as a concise way. 

### LU Factorisation

Let's check the gaussian elimination example again. The final form in [@eq:linear-algebra:gauss04] can be expressed with the matrix notation as:

$$\left[\begin{matrix}1 & 1 & 1\\0 & 1 & 2\\0 & 0 & 1\end{matrix} \right]$$

Here all the elements below the diagonal of this square matrix is zero. 
Such matrix is called an *upper triangular matrix*, usually denoted by $U$.
Similarly, a square matrix that all the elements below the diagonal of this square matrix is zero is called *lower triangular matrix*, denoted by $L$.
We can use the `is_triu` and `is_tril` to verify if a matrix is triangular.

The diagonal elements of $U$ are called pivots. The i-th pivot is the coefficient of the i-th variable in the i-th equation at the i-th step during the elimination.

In general, a square matrix can often be factorised into the dot product of a lower and a upper triangular matrices: $A = LU$.
It is called the *LU factorisation*. 
It embodies the process of Gauss elimination.
Back to the initial problem of solving the linear equation $Ax=b$.
One reason the LU Factorisation is important is that if the matrix A in $Ax=b$ is triangular, then solving it would be straightforward, as we have seen in the previous example. 
Actually, we can use `triangular_solve` to efficiently solve the linear equations if we already know that the matrix is triangular.

For a normal square matrix that can be factorised into $LU$, we can change $Ax=b$ to $LUx=b$.
First we can find column vector $c$ so that $Lc=b$, then we can find $x$ so that $Ux=c$.
Both triangular equations are easy to solve.

We use the `lu` function to perform the LU factorisation. Let's use the previous example. 

```ocaml env=linear-algebra-lu
# let a = [|2.;2.;2.;2.;2.;3.;3.;4.;5.|]
val a : float array = [|2.; 2.; 2.; 2.; 2.; 3.; 3.; 4.; 5.|]
# let a = Arr.of_array a [|3; 3|]
val a : Arr.arr =
   C0 C1 C2
R0  2  2  2
R1  2  2  3
R2  3  4  5

```

```ocaml env=linear-algebra-lu
# let l, u, p = Linalg.D.lu a
val l : Owl_dense_matrix_d.mat =

         C0 C1 C2
R0        1  0  0
R1 0.666667  1  0
R2 0.666667  1  1

val u : Owl_dense_matrix_d.mat =

   C0        C1        C2
R0  3         4         5
R1  0 -0.666667 -0.333333
R2  0         0        -1

val p : Linalg.D.int32_mat =
   C0 C1 C2
R0  3  2  3

```

The first two returned matrix are the lower and upper triangular matrices.
However, if we try to check the correctness of this factorisation with dot product, the result does not fit:

```ocaml env=linear-algebra-lu
# let a' = Mat.dot l u 
val a' : Mat.mat =
   C0 C1 C2
R0  3  4  5
R1  2  2  3
R2  2  2  2

```

```ocaml env=linear-algebra-lu
# a' = a
- : bool = false
```

It turns out that we need to some extra row exchange to get the right answer. 
That's because the row exchange is required in certain cases, such as when the number we want to use as the pivot could be zero. 
This process is called *pivoting*. It is closely related to the numerical computation stability. Choosing the improper pivots can lead to wrong linear system solution.
It can be expressed with a permutation matrix that has the same rows as the identity matrix, each row and column has exactly one "1" element. 
The full LU factorisation can be expressed as:

$$PA = LU.$$

```ocaml env=linear-algebra-lu
# let p = Mat.of_array  [|0.;0.;1.;0.;1.;0.;1.;0.;0.|] 3 3
val p : Mat.mat =
   C0 C1 C2
R0  0  0  1
R1  0  1  0
R2  1  0  0

```

```ocaml env=linear-algebra-lu
# Mat.dot p a = Mat.dot l u 
- : bool = true
```

How do we translate the third output, the permutation vector, to the required permutation matrix? 
Each element $p_i$ in the vector represents a updated identity matrix. 
On this identity matrix, we set (i, i) and ($p_i$, $p_i$) to zero, and then (i, $p_i$) and  ($p_i$, i) to one. 
Multiply these $n$ matrices, we can get the permutation matrix $P$.
Here is a brief implementation of this process in OCaml:

```ocaml
let perm_vec_to_mat vec =
    let n = Array.length vec in
    let mat = ref (Mat.eye n) in
    for i = n - 1 downto 0 do
      let j = vec.(i) in
      let a = Mat.eye n in
      Arr.set a [| i; i |] 0.;
      Arr.set a [| j; j |] 0.;
      Arr.set a [| i; j |] 1.;
      Arr.set a [| j; i |] 1.;
      mat := Arr.dot a !mat
    done;
    !mat
```

Note that there is more than one way to do the LU factorisation. For example, for the same matrix, we can have:

$$\left[\begin{matrix}1 & 0 & 0\\0 & 0 & 1\\0 & 1 & 0\end{matrix} \right] \left[\begin{matrix}2 & 2 & 2\\2 & 2 & 3\\3 & 4 & 5\end{matrix} \right] = \left[\begin{matrix}1 & 0 & 0\\1.5 & 1 & 0\\1 & 0 & 1\end{matrix} \right] \left[\begin{matrix}2 & 2 & 2\\0 & 1 & 2\\0 & 0 & 1\end{matrix} \right]$$

### Inverse and Transpose 

The concept of inverse matrix is related with the identity matrix, which can be built with $Mat.eye n$, where n is the size of the square matrix.
The identity matrix is a special form of *Diagonal Matrix*, which is a square matrix that only contains non-zero element along its diagonal. 
You can check if a matrix is diagonal with `is_diag` function.

```ocaml
Mat.eye 5 |> Linalg.D.is_diag
```

The inverse of a $n$ by $n$ square matrix $A$ is denoted by $A^{-1}, so that $: $AA^{-1} = I_n$.
Note that not all square matrix has inverse.  
There are many sufficient and necessary conditions to decide if $A$ is invertible, one of them is that A has $n$ pivots.

We use function `inv` to do the inverse operation. It's straightforward and easy to verify according to the definition.
Here we use the `semidef` function to produce a matrix that is certainly invertible.

```ocaml env=linear-algebra:inverse
# let x = Mat.semidef 5
val x : Mat.mat =

         C0       C1       C2       C3       C4
R0  1.38671 0.865127  1.58151  1.49422 0.469741
R1 0.865127 0.708478  1.06377  1.05908 0.284205
R2  1.58151  1.06377   1.9197  1.75276 0.725455
R3  1.49422  1.05908  1.75276  2.09053 0.674717
R4 0.469741 0.284205 0.725455 0.674717 0.825211

```

```ocaml env=linear-algebra:inverse
# let y = Linalg.D.inv x
val y : Owl_dense_matrix_d.mat =

         C0       C1       C2       C3       C4
R0   55.544  34.8501 -61.4899 -12.4567  20.6214
R1  34.8501  34.8324 -44.6476 -10.2098  15.7638
R2 -61.4899 -44.6476  73.3611   13.064 -24.7952
R3 -12.4567 -10.2098   13.064  5.27679  -5.1921
R4  20.6214  15.7638 -24.7952  -5.1921  10.0873

```

```ocaml env=linear-algebra:inverse
# Mat.(x *@ y =~ eye 5)
- : bool = true
```

The next frequently used special matrix is the *Transpose Matrix*. Denoted by $A^T$, its $i$th row is taken from the $i$-th column of the original matrix A.
It has properties such as $(AB)^T=B^T~A^T$. 
We can check this property using the matrix function `Mat.transpose`. Note that this function is deemed basic ndarray operations and is not included in the `Linalg` module.

```ocaml
# let flag = 
    let a = Mat.uniform 4 4 in 
    let b = Mat.uniform 4 4 in 
    let m1 = Mat.(dot a b |> transpose) in
    let m2 = Mat.(dot (transpose b) (transpose a)) in 
    Mat.(m1 =~ m2)
val flag : bool = true
```

A related special matrix is the *Symmetric Matrix*, which equals to its own transpose. This simple test can be done with the `is_symmetric` function.

## Vector Spaces

We have talked about solving the $Ax=b$ linear equations with elimination, and A is a square matrix. 
Now we need to further discuss, how do we know if there exists one or maybe more than one solution. 
To answer such question, we need to be familiar with the concepts of *vector space*. 

A vector space, denoted by $R^n$, contains all the vectors that has $n$ elements.
In this vector space we have the `add` and `multiplication` operation. Applying them to the vectors is called *linear combination*.
Then a *subspace* in a vector space is a non-empty set that linear combination of the vectors in this subspace still stays in the same subspace.

There are four fundamental subspaces concerning solving linear systems $Ax=b, where $A$ is a $m$ by $n$ matrix.
The *column space* consists of all the linear combinations of the columns of A. It is a subspace of $R^m$.
Similarly, the *row space* consists of all the linear combinations of the rows of  A. 
The *nullspace* contains all the vectors $x$ so that $Ax=0$, denoted by $N(A)$. It is a subspace of $R^n$.
The *left nullspace* is similar. It is the nullspace of $A^T$.

### Rank and Basis

In the Gaussian Elimination section, we assume an ideal situation: the matrix A is $n\times~n$ square, and we assume that there exists one solution. 
But that does not happen every time. 
In many cases $A$ is not an square matrix. 
It is possible that these $m$ equations are not enough to solve a $n$-variable linear system when $m < n$. Or there might not exist a solution when $m > n$.
Besides, even it is a square matrix, the information provided by two of the equations are actually repeated. For example, one equation is simply a multiplication of the other. 

For example, if we try to apply LU factorisation to such a matrix:

```ocaml env=linear-algebra:rank_00
# let x = Mat.of_array [|1.; 2.; 3.; 0.; 0.; 1.; 0.; 0.; 2.|] 3 3
val x : Mat.mat =
   C0 C1 C2
R0  1  2  3
R1  0  0  1
R2  0  0  2

```
```ocaml env=linear-algebra:rank_00
# Linalg.D.lu x
Exception: Failure "LAPACKE: 2".
```

Obviously, we cannot have pivot in the second column, and therefore this matrix is singular and cannot be factorised into $LU$.
As can be seen in this example, we cannot expect the linear algebra functions to be a magic lamb and do our bidding every time.  Understanding the theory of linear algebra helps to better understand how these functions work.

To decide the general solutions to $Ax=b$, we need to understand the concept of *rank*.
The rank of a matrix is the number of pivots in the elimination process.
To get a more intuitive understanding of rank, we need to know the concept of *linear independent. 
In a linear combination $\sum_{i=1}^nc_iv_i$ where $v_i$ are vectors and $c_i$ are numbers, if $\sum_{i=1}^nc_iv_i = 0$ only happens when $c_i = 0$ for all the $i$'s, then the vectors $v_1, v_2, \ldots, v_n$ are linearly independent.
Then the rank of a matrix is the number of independent rows in the matrix.
We can understand rank as the number of "effective" rows in the matrix.

As an example, we can check the rank of the previous matrix.


```ocaml env=linear-algebra:rank_00
Linalg.D.rank x
```

As can be example, the rank is 2, which means only two effective rows, and thus cannot be factorised to find the only solution.

One application of rank is in a crucial linear algebra idea: basis. 
A sequence of vectors is the *basis* of a space or subspace if:
1) these vectors are linear independent and 
2) all the the vectors in the space can be represented as the linear combination of vectors in the basis.

A space can have infinitely different bases, but the number of vectors in these bases are the same. This number is called the *dimension* of this vector space.
For example, a $m$ by $n$ matrix A has rank of $r$, then the dimension of its null space is $n-r$, and the dimension of its column space is $r$.
Think about a full-rank matrix where $r=n$, then the dimension of column matrix is $n$, which means all its columns can be a basis of the column space, and that the null space dimension is zero so that the only solution of $Ax=0$ is a zero vector.

### Orthogonality

We can think of the basis of a vector space as the Cartesian coordinate system in a three-dimensional space, where every vector in the space can be represented with the three vectors ni the space: the x, y and z axis.
Actually, we can use many three vectors system as the coordinate bases, but the x. y, z axis is used is because they are orthogonal to each other. 
An orthogonal basis can greatly reduce the complexity of problems.
The same can be applied in the basis of vector spaces.

Orthogonality is not limited to vectors.
Two vectors $a$ and $b$ are orthogonal are orthogonal if $a^Tb = 0$. 
Two subspaces A and B are orthogonal if every vector in A is orthogonal to every vector in B.
For example, the nullspace and row space of a matrix are perpendicular  to each other.

Among the bases of a subspace, if every vector is perpendicular to each other, it is called an orthogonal matrix. 
Moreover, if the length of each vector is normalised to one unit, it becomes the *orthonormal basis*.


For example, we can use the `null` function to find an orthonormal basis vector $x$ or the null space of a matrix, i.e. $Ax=0$.

```ocaml env=linear-algebra:ortho-null
# let a = Mat.magic 4
val a : Mat.mat =

   C0 C1 C2 C3
R0  1 15 14  4
R1 12  6  7  9
R2  8 10 11  5
R3 13  3  2 16

```
```ocaml env=linear-algebra:ortho-null
# let x = Linalg.D.null a
val x : Owl_dense_matrix_d.mat =

          C0
R0 -0.223607
R1  -0.67082
R2   0.67082
R3  0.223607

```
```ocaml env=linear-algebra:ortho-null
# Mat.dot a x |> Mat.l2norm' 
- : float = 2.87802701599908967e-15
```

(Question: but this example does not really show the orthogonal part.)

Now that we know what is orthogonal basis, the next question is, how to build one? 
The method to construct orthogonal basis in a subspace is called the *Gram-Schmidt orthogonalisation*.

TODO: Explain Gram-Schmidt and QR.

```
val qr : ?thin:bool -> ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t
```

`qr x` calculates QR decomposition for an `m` by `n` matrix `x`.
The function returns a 3-tuple, the first two are `Q` and `R`, and the third is the permutation vector of columns. 
The default value of parameter `pivot` is `false`, setting pivot  to true lets `qr` performs pivoted factorisation. Note that
the returned indices are not adjusted to 0-based C layout.
By default, `qr` performs a reduced QR factorisation, full factorisation can be enabled by setting `thin` parameter to `false`.

```ocaml env=linear-algebra:qr
# let a = Mat.of_array [|12.; -51.; 4.; 6.; 167.; -68.; -4.; 24.; -41.|] 3 3 
val a : Mat.mat =

   C0  C1  C2
R0 12 -51   4
R1  6 167 -68
R2 -4  24 -41

```
```ocaml env=linear-algebra:qr
# let q, r, _ = Linalg.D.qr a
val q : Owl_dense_matrix_d.mat =

          C0        C1         C2
R0 -0.857143  0.394286   0.331429
R1 -0.428571 -0.902857 -0.0342857
R2  0.285714 -0.171429   0.942857

val r : Owl_dense_matrix_d.mat =

    C0   C1  C2
R0 -14  -21  14
R1   0 -175  70
R2   0    0 -35

```

### Solving Ax = b

We can now discuss the general solution structure to $Ax=0$ and $Ax=b$.
Again, here $A$ is a $m\times~n$ matrix.
The theorems declare that, there exists non-zeor solution(s) to $Ax=0$ if and only if $\textrm{rank}(a) <= n$.
If $r(A) < n$, then the nullspace of $A$ is of dimension $n - r$ and the $n-r$ orthogonal basis can be found with `null` function.
Here is an example.

```ocaml env=linear-algebra:solve_00
# let a = Mat.of_array [|1.;5.;-1.;-1.;1.;-2.;1.;3.;3.;8.;-1.;1.;1.;-9.;3.;7.|] 4 4
val a : Mat.mat =

   C0 C1 C2 C3
R0  1  5 -1 -1
R1  1 -2  1  3
R2  3  8 -1  1
R3  1 -9  3  7

```
```ocaml env=linear-algebra:solve_00
# Linalg.D.rank a
- : int = 2
```
This a rank 2 matrix, so the nullspace contains 4 - 2 = 2 vectors:

```ocaml env=linear-algebra:solve_00
# Linalg.D.null a
- : Owl_dense_matrix_d.mat =

          C0        C1
R0 -0.851419 0.0136382
R1  0.273706  0.143885
R2 0.0762491  0.962526
R3   0.44086 -0.229465

```

These two vectors are called the *fundamental system of solutions* of $Ax=0$.
All the solutions of $Ax=0$ can then be expressed using the fundamental system:

$$c_1\left[\begin{matrix}-0.85 \\ 0.27 \\ 0.07 \\0.44\end{matrix} \right] + c_2\left[\begin{matrix}0.013\\ 0.14 \\ 0.95 \\-0.23\end{matrix} \right]$$

Here $c_1$ and $c_2$ can be any constant numbers.

For solving the general form $Ax=b$ where b is $m\times~1$ vector, there exist only one solution if and only if $\textrm{rank}(A) = \textrm{rank}([A, b]) = n$. Here $[A, b]$ means concatenating $A$ and $b$ along the column.
If $\textrm{rank}(A) = \textrm{rank}([A, b]) < n$, $Ax=b$ has infinite number of solutions. 
These solutions has a general form:


$$x_0 + c_1x_1 + c2x2 + \ldots +c_kx_k$$

Here $x_0$ is a particular solution to $Ax=b$, and $x_1, x_2, \ldots, x_k$ are the fundamental solution system of $Ax=0$.

We can use `linsolve` function to find one particular solution.
In the Linear Algebra, the function `linsolve a b -> x` solves a linear system of equations `a * x = b`. 
By default, the function uses LU factorisation with partial pivoting when `a` is square and QR factorisation with column pivoting otherwise. 
The number of rows of `a` must be equal to the number of rows of `b`.
If `a` is a upper or lower triangular matrix, the function calls the `solve_triangular` function.

Here is an example.

```text
# let a = Mat.of_array [|2.;3.;1.;1.;-2.;4.;3.;8.;-2.;4.;-1.;9.|] 4 3
val a : Mat.mat =

   C0 C1 C2
R0  2  3  1
R1  1 -2  4
R2  3  8 -2
R3  4 -1  9

# let b = Mat.of_array [|4.;-5.;13.;-6.|] 4 1
val b : Mat.mat =
   C0
R0  4
R1 -5
R2 13
R3 -6

# let x0 = Linalg.D.linsolve a b
val x0 : Owl_dense_matrix_d.mat =
   C0
R0 -5
R1  4
R2  2

```

Then we use `null` to find the fundamental solution system. You can verify that matrix `a` is of rank 2, so that the solution system for $ax=0$ should contain only 3 - 2 = 1 vector.

```text
# let x1 = Linalg.D.null a
val x1 : Owl_dense_matrix_d.mat =

          C0
R0 -0.816497
R1  0.408248
R2  0.408248

```

So the solutions to $Ax=b$ can be expressed as:

$$\left[\begin{matrix}-1 \\ 2 \\ 0 \end{matrix} \right] + c_1\left[\begin{matrix}-0.8\\ 0.4 \\ 0.4 \end{matrix} \right]$$

So the takeaway from this chapter is that the using these linear algebra functions often requires solid background knowledge. 
Blindly using them could leads to wrong or misleading answers.

## Determinants

Other than pivots, another basic quantity in linear algebra is the *determinants*.
For a square matrix A:

$$\left[\begin{matrix}a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ldots & \vdots \\ a_{n1} & a_{n2} & \ldots & a_{nn} \end{matrix} \right]$$

its determinants `det(A)` is defined as:

$$\sum_{j_1~j_2~\ldots~j_n}(-1)^{\tau(j_1~j_2~\ldots~j_3)}a_{1{j_1}}a_{2j_2}\ldots~a_{nj_n}.$$

Here $\tau(j_1~j_2~\ldots~j_n) = i_1 + i_2 + \ldots + i_{n-1}$, 
where $i_k$ is the number of $j_p$ that is smaller than $j_k$ for $p \in [k+1, n]$.

Mathematically, there are many techniques that can be used to simplify this calculation.
But as far as this book is concerned, it is sufficient for us to use the `det` function to calculate the determinants of a matrix. 

Why is the concept of determinant important? 
Its most important application is to using determinant to decide if a square matrix A is invertible or singular.
The determinant $\textrm{det}(A) \neq 0$ if and only if $\textrm{A} = n$. 
Also it can be expressed as $\textrm{det}(A) \neq 0$ if and only if matrix A is invertible. 

We can also use it to understand the solution of $Ax=b$: if $\textrm{det}(A) \neq 0$, then $Ax=b$ has one and only one solution. 
This theorem is part of the *Cramer's rule*.
These properties are widely used in finding *eigenvalues*. As will be shown in the next section.

Since sometimes we only care about if the determinant is zero or not, instead of the value itself, we can also use a similar function `logdet`. 
It computes the logarithm of the determinant, but it avoids the possible overflow or underflow problems in computing determinant of large matrices.

```text
# let x = Mat.magic 5
val x : Mat.mat =

   C0 C1 C2 C3 C4
R0 17 24  1  8 15
R1 23  5  7 14 16
R2  4  6 13 20 22
R3 10 12 19 21  3
R4 11 18 25  2  9

# Linalg.D.det x
- : float = 5070000.

# Linalg.D.logdet x
- : float = 15.4388513755673671

```

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

### Positive Definiteness

In this section we introduce *Positive Definite Matrix*, which unifies the three most basic ideas in linear algebra: pivots, determinants, and eigenvalues. 

The definition of a Positive Definite Matrix: symmetric, $x^TAx > 0$ for all non-zero vectors $x$.
There are several necessary and sufficient condition for testing if a symmetric matrix A is positive definite:

1. $x^TAx>0$ for all non-zero real vectors x
1. $\lambda_i >0$ for all eigenvalues $\lambda_i$ of A
1. all the upper left matrices have positive determinants 
1. all the pivots without row exchange satisfy $d >0$
1. there exists invertible matrix B so that A=B^TB

For the last condition, we can use *Cholesky decomposition* to find B:
```
val chol : ?upper:bool -> ('a, 'b) t -> ('a, 'b) t
  (* Cholesky factorisation *)
```

Example 

In the Linear Algebra module, we use `is_posdef` to do this test. 

```
val is_posdef : ('a, 'b) t -> bool
  (* check if a matrix is positive semi-definite *)
```
It's implementation uses ...

Similar, the definition of semi-positive definite.

The positive definite matrices are frequently used in different fields. 
The pattern $Ax=\lambda~Mx$ exists in many engineering analysis problems.
If $A$ and $M$ are positive definite, this pattern is parallel to the $Ax=\lambda~x$ where $\lambda > 0$.

One such application is the stability of motion. 
Definition of stable system. 

In a linear system:

$$y' = Ax$$

(extend this equation)

A theorem declares that this system is stable if and only if there exists positive and definite matrix $V$ so that $-(VA+A^TV)$ is semi-positive definite. 

The pendulum example

### Singular Value Decomposition

The singular value decomposition (SVD) is among the most important matrix factorizations of the computational era.
The SVD provides a numerically stable matrix decomposition that can be used for a variety of purposes and is guaranteed to exist. 

Definition of SVD

It's close related with eigenvector factorisation of a positive definite matrix.
Detail.

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


The intuition of SVD.

Applications.
Many applications; ....

The Moore-Penrose inverse is a direct application of the SVD.
```
val pinv : ?tol:float -> ('a, 'b) t -> ('a, 'b) t
  (* Moore-Penrose pseudo-inverse of a matrix *)
```

It is also related to the least square.


We will also come back to SVD in NLP.

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

**Hessenberg Factorisations:**

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
