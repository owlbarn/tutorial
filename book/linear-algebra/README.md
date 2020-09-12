# Linear Algebra

Linear Algebra is a key mathematics field behind computer science and numerical computating.
A thorough coverage of this topic is apparently beyond the scope of this book. Please refer to [@strang2006linear] for this subject.
In this chapter we will follow the basic structure of this book, first giving you a overall picture, then focussing on how to use the functions provided in Owl to solve problems and better understand some basic linear algebra concepts.

The high level APIs of Linear Algebra are provided in the `Linalg` module.
The module provides four types of number types: single precision, double precision, complex single precision, and complex double precision.
They are included in `Linalg.S`, `Linalg.D`, `Linalg.C` and `Linalg.Z` modules respectively.
Besides, the `Linalg.Generic` can do everything that `S/D/C/Z` can but needs some extra type information.


## Vectors and Matrices

The fundamental problem of linear algebra: solving linear equations.
This is more efficiently expressed with vectors and matrices.
Therefore, we need to first get familiar with these basic structures in Owl.

Similar to the `Linalg` module, all the matrix functions can be accessed from the `Dense.Matrix` module, and support four different type of modules.
The `Mat` module is an alias of `Dense.Matrix.D`.
Except for some functions such as `re`, most functions are shared by these four submodules.
Note that that matrix module is actually built on the `Ndarray` module, and thus the supported functions are quite similar, and matrices and ndarrays can interoperate with each other.
The vectors are expressed using Matrix in Owl.

### Creating Matrices

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
  Mat.bernoulli 5 5      (* create a 5 x 5 random Bernoulli  matrix *)
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

You can try the similar `sum_cols`.
The summation of all the diagonal elements is also 65.

```ocaml env=matrix_env2
# Mat.trace x
- : float = 65.
```

### Accessing Elements

Similar to ndarray, the matrix module support `set` and `get` to access and modify matrix elements.
The only difference is that instead of accessing according to an array, an element in matrix is accessed using two integers.

```ocaml env=matrix_env0
let x = Mat.uniform 5 5;;
Mat.set x 1 2 0.;;             (* set the element at (1,2) to 0. *)
Mat.get x 0 3;;                (* get the value of the element at (0,3) *)
```

For dense matrices, i.e., `Dense.Matrix.*`, you can also use shorthand `.%{i; j}` to access elements.

```ocaml env=matrix_env0
# open Mat
# x.%{1;2} <- 0.;;         (* set the element at (1,2) to 0. *)
- : unit = ()
# let a = x.%{0;3};;       (* get the value of the element at (0,3) *)
val a : float = 0.563556290231645107
```

The modifications to a matrix using `set` are in-place. This is always true for dense matrices. For sparse matrices, the thing can be complicated because of performance issues.

We can take some rows out of `x` by calling `rows` function. The selected rows will be used to assemble a new matrix.
Similarly, we can also select some columns using `cols`.


### Iterate, Map, Fold, and Filter

In reality, a matrix usually represents a collections of measurements (or points). We often need to go through these data over and over again for various reasons. Owl provides very convenient functions to help you to iterate these elements. There is one thing I want to emphasise: Owl uses row-major matrix for storage format in the memory, which means accessing rows are much faster than those column operations.

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

Iterating rows and columns are similar to iterating elements, by using `iteri_rows`, `mapi_rows`, and etc. The following example prints the sum of each row.

```ocaml env=matrix_env1

  Mat.iteri_rows (fun i r ->
    Printf.printf "row %i: %.1f\n" i (Mat.sum' r)
  ) x;;

```

You can also fold elements, rows, and columns.
We can calculate the summation of all column vectors by using `fold_cols` function.

```ocaml env=matrix_env1

  let v = Mat.(zeros (row_num x) 1) in
  Mat.(fold_cols add v x);;

```

It is also possible to change a specific row or column. E.g., we make a new matrix out of `x` by setting row `2` to zero vector.

```ocaml env=matrix_env1

  Mat.map_at_row (fun _ -> 0.) x 2;;

```

The filter functions is also commonly used in manipulating matrix.
Here are some examples.
The first one is to filter out the elements in `x` greater than `20`.

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

### Math Operations

The math operations can be generally categorised into several groups.

**Comparison**
Suppose we have two matrices:

```ocaml env=matrix_env1
let x = Mat.uniform 2 2;;
let y = Mat.uniform 2 2;;
```

We can compare the relationship of `x` and `y` element-wisely as below.


```ocaml env=matrix_env1

  Mat.(x = y);;    (* is x equal to y *)
  Mat.(x <> y);;   (* is x unequal to y *)
  Mat.(x > y);;    (* is x greater to y *)
  Mat.(x < y);;    (* is x smaller to y *)
  Mat.(x >= y);;   (* is x not smaller to y *)
  Mat.(x <= y);;   (* is x not greater to y *)

```

All aforementioned infix have their corresponding functions in the module, e.g., `=@` has `Mat.is_equal`.


**Matrix Arithmetic**

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

There are some ready-made math functions such as `Mat.log` and `Mat.abs` ect. to ease your life when operating matrices. These math functions apply to every element in the matrix.

There are other functions such as concatenation:

```ocaml env=matrix_env1
  Mat.(x @= y);;    (* concatenate x and y vertically *)
  Mat.(x @|| y);;   (* concatenate x and y horizontally *)
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

$$
\left[\begin{matrix}2 & 2 & 2\\2 & 2 & 3\\3 & 4 & 5\end{matrix} \right] \left[\begin{matrix}x_1\\x_2\\x_3\end{matrix} \right] = \left[\begin{matrix}4\\5\\7\end{matrix} \right]
\Longrightarrow
\left[\begin{matrix}x_1\\x_2\\x_3\end{matrix} \right] = \left[\begin{matrix}2\\-1\\1\end{matrix} \right]
$$

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

        C0       C1       C2      C3       C4
R0 2.56816   1.0088  1.57793  2.6335  2.06612
R1  1.0088 0.441613 0.574465 1.02067 0.751004
R2 1.57793 0.574465  2.32838 2.41251  2.13926
R3  2.6335  1.02067  2.41251 3.30477  2.64877
R4 2.06612 0.751004  2.13926 2.64877  2.31124

```

```ocaml env=linear-algebra:inverse
# let y = Linalg.D.inv x
val y : Owl_dense_matrix_d.mat =

         C0       C1       C2       C3       C4
R0   12.229  -15.606  6.12229 -1.90254 -9.34742
R1  -15.606  33.2823 -4.01361 -4.96414  12.5403
R2  6.12229 -4.01361  7.06372 -2.62899 -7.69399
R3 -1.90254 -4.96414 -2.62899   8.1607 -3.60533
R4 -9.34742  12.5403 -7.69399 -3.60533  15.9673

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

There are four fundamental subspaces concerning solving linear systems $Ax=b$, where $A$ is a $m$ by $n$ matrix.
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
The theorems declare that, there exists non-zero solution(s) to $Ax=0$ if and only if $\textrm{rank}(a) <= n$.
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

```ocaml
# let a = Mat.of_array [|2.;3.;1.;1.;-2.;4.;3.;8.;-2.;4.;-1.;9.|] 4 3
val a : Mat.mat =

   C0 C1 C2
R0  2  3  1
R1  1 -2  4
R2  3  8 -2
R3  4 -1  9

```

```ocaml
# let b = Mat.of_array [|4.;-5.;13.;-6.|] 4 1
val b : Mat.mat =
   C0
R0  4
R1 -5
R2 13
R3 -6

```

```ocaml
# let x0 = Linalg.D.linsolve a b
val x0 : Owl_dense_matrix_d.mat =
   C0
R0 -5
R1  4
R2  2

```

Then we use `null` to find the fundamental solution system. You can verify that matrix `a` is of rank 2, so that the solution system for $ax=0$ should contain only 3 - 2 = 1 vector.

```ocaml
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

### Matrix Sensitivity

The *sensitivity* of a matrix is perhaps not the most important issue in the traditional linear algebra, but is crucial in the numerical computation related problems.
It answers this question: in $Ax=b$, if we change the $A$ and $b$ slightly, how much will the $x$ be affected?
The *Condition Number* is a measurement of the sensitivity of a square matrix.

First, we need to understand the *Norm* of a matrix.
The norm, or 2-norm of a matrix  $\|A\|$ is calculated as square root of the maximum eigenvalue of $A^HA$.
The norm of a matrix is a upper limit so that for any $x$ we can be certain that $\|Ax\| \leq \|A\|\|x\|$.
Here $\|Ax\|$ and $\|x\|$ are the L2-Norm for vectors.
The $\|A\|\$ bounds the how large the $A$ can amplify the input $x$.
We can calculate the norm with `norm` in the linear algebra module.

The most frequently used condition number is that represent the sensitivity of inverse matrix.
With the definition of norm, the *condition number for inversion* of a matrix can be expressed as $\|A\|\|A^{-1}\|$.
We can calculate it using the `cond` function.

Let's look at an example:

```ocaml env=linalg_20
# let a = Mat.of_array [|4.1; 2.8; 9.7; 6.6 |] 2 2;;
val a : Mat.mat =
    C0  C1
R0 4.1 2.8
R1 9.7 6.6

```

```ocaml env=linalg_20
# let c = Linalg.D.cond a
val c : float = 1622.99938385651058
```

Its condition number for inversion is much larger than one. Therefore, a small change in $A$ should leads to a large change of $A^{-1}$.

```ocaml env=linalg_20
# let a' = Linalg.D.inv a
val a' : Owl_dense_matrix_d.mat =
    C0  C1
R0 -66  28
R1  97 -41

```

```ocaml env=linalg_20
# let a2 = Mat.of_array [|4.1; 2.8; 9.67; 6.607 |] 2 2
val a2 : Mat.mat =
     C0    C1
R0  4.1   2.8
R1 9.67 6.607

```

```ocaml env=linalg_20
# let a2' = Linalg.D.inv a2
val a2' : Owl_dense_matrix_d.mat =

         C0       C1
R0  520.236 -220.472
R1 -761.417  322.835

```

We can see that by changing the matrix by only a tiny bit, the inverse of $A$ changes dramatically, and so is the resulting solution vector $x$.


## Determinants

TODO: extend this section, add sub-sections.

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

```ocaml env=linalg_30
# let x = Mat.magic 5
val x : Mat.mat =

   C0 C1 C2 C3 C4
R0 17 24  1  8 15
R1 23  5  7 14 16
R2  4  6 13 20 22
R3 10 12 19 21  3
R4 11 18 25  2  9

```

```ocaml env=linalg_30
# Linalg.D.det x
- : float = 5070000.00000000093

# Linalg.D.logdet x
- : float = 15.4388513755673653
```


## Eigenvalues and Eigenvectors

### Solving $Ax=\lambda~x$

Now we need to change the topic from $Ax=b$ to $Ax=\lambda~x$.
For an $n\times~n$ square matrix, if there exist number $\lambda$ and non-zero column vector $x$ to satisfy:

$$(\lambda~I - A)x = 0,$${#eq:linear-algebra:eigen}

then $\lambda$ is called *eigenvalue*, and $x$ is called the *eigenvector* of $A$.

To find the eigenvalues of A, we need to find the roots of the determinant of $\lambda~I - A$.
$\textrm{det}(\lambda~I - A) = 0$ is called the *characteristic equation*.
For example, for

$$A = \left[\begin{matrix}3 & 1 & 0 \\ -4 & -1 & 0 \\ 4 & -8 & 2 \end{matrix} \right]$$

Its characteristic matrix $\lambda~I - A$ is:

$$\left[\begin{matrix}\lambda-3 & 1 & 0 \\ -4 & \lambda+1 & 0 \\ 4 & -8 & \lambda-2 \end{matrix} \right]$$

According to the definition of determinant,

$$\textrm{det}(\lambda~I - A) = (\lambda-1)^2(\lambda-2) = 0.$$

According to the theory of polynomials, this characteristic polynomials has and only has $n$ roots in the complex space.
Specifically, here we have three eigenvalues: $\lambda_1=1, \lambda_2 = 1, \lambda=2$.

Put $\lambda_1$ back to characteristic equation, we have: $(I - A)x = 0$. Therefore, we can find the fundamental solution system of $I - A$ with:

```ocaml
# let basis =
    let ia = Mat.((eye 3) - (of_array [|3.;1.;0.;-4.;-1.;0.;4.;-8.;2.|] 3 3)) in
    Linalg.D.null ia
val basis : Owl_dense_matrix_d.mat =

           C0
R0 -0.0496904
R1  0.0993808
R2   0.993808

```

We have a fundamental solution $x_0 = [-0.05, 0.1, 1]^T$. Therefore all the $k_0x_0$ are the corresponding eigenvector of the eigenvalue $1$.
Similarly, we can calculate that eigenvectors for the eigenvalue $2$ are $k_1[0, 0, 1]^T$.

We can use `eig` to find the eigenvectors and eigenvalues of a matrix.
`eig x -> v, w` computes the right eigenvectors `v` and eigenvalues `w` of an arbitrary square matrix `x`. The eigenvectors are column vectors in `v`, their corresponding eigenvalues have the same order in `w` as that in `v`.

```ocaml
# let eigvec, eigval =
    let a = Mat.of_array [|3.;1.;0.;-4.;-1.;0.;4.;-8.;2.|] 3 3 in
    Linalg.D.eig a
val eigvec : Owl_dense_matrix_z.mat =

        C0               C1               C2
R0 (0, 0i)  (0.0496904, 0i)  (0.0496904, 0i)
R1 (0, 0i) (-0.0993808, 0i) (-0.0993808, 0i)
R2 (1, 0i)  (-0.993808, 0i)  (-0.993808, 0i)

val eigval : Owl_dense_matrix_z.mat =

        C0      C1      C2
R0 (2, 0i) (1, 0i) (1, 0i)

```

Note that the result are expressed as complex numbers.
If we only want the eigenvalues, we can use the `eigvals` function.
TODO: Explain the `permute` and `scale` parameters.
QUESTION: what if the nullspace is more than one dimension? what would be the resulting eigenvalue matrix look like?

One reason that eigenvalue and eigenvector are important is that the pattern $Ax=\lambda~x$ frequently appears in scientific and engineering analysis to describe the change of dynamic system over time.
TODO: more detail and perhaps example.

### Complex Matrices

As can be seen in the previous example, complex matrices are frequently used in eigenvalues in eigenvectors.
In this section we re-introduce some previous concepts in the complex space.

We have seen the Symmetric Matrix.
It can be extended to the complex numbers, called *Hermitian Matrix*, denoted by $A^H$.
Instead of requiring it to be the same as its transpose, a hermitian matrix equals to its conjugate transpose.
A conjugate transpose means that during transposing, each element $a+bi$ changes to its conjugate $a-bi$.
Hermitian is thus a generalisation of the symmetric matrix.
We can use the `is_hermitian` function to check if a  matrix is hermitian, as can be shown in the next example.

```ocaml env=linalg_35
# let a = Dense.Matrix.Z.of_array [|{re=1.; im=0.}; {re=2.; im=(-1.)}; {re=2.; im=1.}; {re=3.; im=0.}|] 2 2
val a : Dense.Matrix.Z.mat =

        C0       C1
R0 (1, 0i) (2, -1i)
R1 (2, 1i)  (3, 0i)

```

```ocaml env=linalg_35
# Linalg.Generic.is_hermitian a
- : bool = true
```

We can use the `conj` function of a complex matrix to perform the conjugate transpose:

```ocaml env=linalg_35
# Dense.Matrix.Z.(conj a |> transpose)
- : Dense.Matrix.Z.mat =

         C0       C1
R0 (1, -0i) (2, -1i)
R1  (2, 1i) (3, -0i)

```

A theorem declares that if a matrix is hermitian, then for all complex vectors $x$, $x^HAx$ is real, and every eigenvalue is real.

```ocaml env=linalg_35
# Linalg.Z.eigvals a
- : Owl_dense_matrix_z.mat =

                         C0                      C1
R0 (-0.44949, 1.50231E-17i) (4.44949, 2.07021E-16i)

```

A related concept is the *Unitary Matrix*.
A matrix $U$ is unitary if $U^HU=I$. The inverse and conjugate transpose of $U$ are the same.
It can be compared to the orthogonal vectors in the real space.


### Similarity Transformation and Diagonalisation

For a $nxn$ matrix A, and any invertible $nxn$ matrix M, the matrix $B = M^{-1}AM$ is *similar* to A.
One important property is that similar matrices share the same eigenvalues.
Changing from A to B actually changes the linear transformation using one set of basis to another.

TODO: more thorough explanation of the intuition of similar matrices.

In a three dimensional space, if we can change using three random vectors as the basis of linear transformation to using the standard basis $[1, 0, 0]$, $[0, 1, 0]$,  $[0, 0, 1]$, the related problem can be greatly simplified.
Finding the suitable similar matrix is thus important in simplifying the calculation in many scientific and engineering problems.

One possible kind of simplification is to find a triangular matrix as similar.
The *Schur's Lemma* declares that A can be decomposed into $UTU^{-1}$ where $U$ is a unitary function, and T is an upper triangular matrix.

```ocaml env=linear-algebra:schur
# let a = Dense.Matrix.Z.of_array [|{re=1.; im=0.}; {re=1.; im=0.}; {re=(-2.); im=0.}; {re=3.; im=0.}|] 2 2
val a : Dense.Matrix.Z.mat =

         C0      C1
R0  (1, 0i) (1, 0i)
R1 (-2, 0i) (3, 0i)

```

```ocaml env=linear-algebra:schur
# let t, u, eigvals = Linalg.Z.schur a
val t : Owl_dense_matrix_z.mat =

        C0                    C1
R0 (2, 1i) (2.10381, -0.757614i)
R1 (0, 0i)              (2, -1i)

val u : Owl_dense_matrix_z.mat =

                       C0                      C1
R0 (-0.408248, 0.408248i)  (0.563384, -0.590987i)
R1        (-0.816497, 0i) (-0.577185, 0.0138014i)

val eigvals : Owl_dense_matrix_z.mat =

        C0       C1
R0 (2, 1i) (2, -1i)

```

The returned result `t` is apparent a upper triangular matrix, and the `u` can be verified to be a unitary matrix:

```ocaml env=linear-algebra:schur
# Dense.Matrix.Z.(dot u (conj u |> transpose))
- : Dense.Matrix.Z.mat =

                             C0                          C1
R0                      (1, 0i) (7.97973E-17, 5.81132E-17i)
R1 (7.97973E-17, -5.81132E-17i)                     (1, 0i)

```

Another very important similar transformation is *diagonalisation*.
Suppose A has $n$ linear-independent eigenvectors, and make them the columns of a matrix Q, then $Q^{-1}AQ$ is a diagonal matrix $\Lambda$, and the eigenvalues of A are the diagonal elements of $\Lambda$.
It's inverse $A = Q\Lambda~Q^{-1}$ is called *Eigendecomposition*.
Analysing A's diagonal similar matrix $\Lambda$ instead of A itself can greatly simplify the problem.

TODO: Give an example

Not every matrix can be diagonalised.
If any two of the $n$ eigenvalues of A are not the same, then its $n$ eigenvectors are linear-independent ana thus A can be  diagonalised.
Specifically, every real symmetric matrix can be diagonalised by an orthogonal matrix.
Or put into the complex space, every hermitian matrix can be diagonalised by a unitary matrix.


## Positive Definite Matrices

### Positive Definiteness

In this section we introduce the concept of *Positive Definite Matrix*, which unifies the three most basic ideas in linear algebra: pivots, determinants, and eigenvalues.

A matrix is called *Positive Definite* if it is symmetric and that $x^TAx > 0$ for all non-zero vectors $x$.
There are several necessary and sufficient condition for testing if a symmetric matrix A is positive definite:

1. $x^TAx>0$ for all non-zero real vectors x
1. $\lambda_i >0$ for all eigenvalues $\lambda_i$ of A
1. all the upper left matrices have positive determinants
1. all the pivots without row exchange satisfy $d > 0$
1. there exists invertible matrix B so that A=B^TB

For the last condition, we can use the *Cholesky Decomposition* to find the matrix B.
It decompose a Hermitian positive definite matrix into the product of a lower triangular matrix and its conjugate transpose $LL^H$:

```ocaml env=linalg_40
# let a = Mat.of_array [|4.;12.;-16.;12.;37.;-43.;-16.;-43.;98.|] 3 3
val a : Mat.mat =

    C0  C1  C2
R0   4  12 -16
R1  12  37 -43
R2 -16 -43  98

```

```ocaml env=linalg_40
# let l = Linalg.D.chol a
val l : Owl_dense_matrix_d.mat =

   C0 C1 C2
R0  2  6 -8
R1  0  1  5
R2  0  0  3

```

```ocaml env=linalg_40
# Mat.(dot (transpose l) l)
- : Mat.mat =

    C0  C1  C2
R0   4  12 -16
R1  12  37 -43
R2 -16 -43  98

```

If in $Ax=b$ we know that $A$ is hermitian and positive definite, then we can instead solve $L^Lx=b$. As we have seen previously, solving linear system that expressed with triangular  matrices is easy.
The Cholesky decomposition is more efficient than the LU decomposition.

In the Linear Algebra module, we use `is_posdef` function to do this test.
If you look at the code in Owl, it is implemented by checking if the Cholesky decomposition can be performed on the input matrix.

```ocaml
# let is_pos =
    let a = Mat.of_array [|4.;12.;-16.;12.;37.;-43.;-16.;-43.;98.|] 3 3 in
    Linalg.D.is_posdef a
val is_pos : bool = true
```

The definition of *semi-positive definite* is similar, only that it allows the "equals to zero" part. For example,  $x^TAx \leq 0$ for all non-zero real vectors x.

The pattern $Ax=\lambda~Mx$ exists in many engineering analysis problems.
If $A$ and $M$ are positive definite, this pattern is parallel to the $Ax=\lambda~x$ where $\lambda > 0$.
For example, a linear system $y'=Ax$ where $x = [x_1, x_2, \ldots, x_n]$ and $y' = [\frac{dx_1}{dt}, \frac{dx_2}{dt}, \ldots, \frac{dx_n}{dt}]$.
We will see such an example in the Ordinary Differential Equation chapter.
In a linearised differential equations the matrix A is the Jacobian matrix.
The eigenvalues decides if the system is stable or not.
A theorem declares that this system is stable if and only if there exists positive and definite matrix $V$ so that $-(VA+A^TV)$ is semi-positive definite.

### Singular Value Decomposition

The singular value decomposition (SVD) is among the most important matrix factorizations of the computational era.
The SVD provides a numerically stable matrix decomposition that can be used for a variety of purposes and is guaranteed to exist.

Any m by n matrix can be factorised in the form:

$$A=U\Sigma~V^T$$ {#eq:linear-algebra:svg}

Here $U$ is is a $m\times~m$ matrix. Its columns are the eigenvectors of $AA^T$.
Similarly, $V$ is a $n\times~n$ matrix, and the columns of V are eigenvectors of $A^TA$.
The $r$ (rank of A) singular value on the diagonal of the $m\times~n$ diagonal matrix $\Sigma$ are the square roots of the nonzero eigenvalues of both $AA^T$ and $A^TA$.

It's close related with eigenvector factorisation of a positive definite matrix.
For a positive definite matrix, the SVD factorisation is the same as the $Q\Lambda~Q^T$.

We can use the `svd` function to perform this factorisation.
Let's use the positive definite matrix as an example:

```ocaml env=linear-algebra:svd
# let a = Mat.of_array [|4.;12.;-16.;12.;37.;-43.;-16.;-43.;98.|] 3 3
val a : Mat.mat =

    C0  C1  C2
R0   4  12 -16
R1  12  37 -43
R2 -16 -43  98

```
```ocaml env=linear-algebra:svd
# let u, s, vt = Linalg.D.svd ~thin:false a
val u : Owl_dense_matrix_d.mat =

          C0        C1        C2
R0 -0.163007 -0.212727  0.963419
R1 -0.457324 -0.848952  -0.26483
R2  0.874233 -0.483764 0.0410998

val s : Owl_dense_matrix_d.mat =

        C0     C1       C2
R0 123.477 15.504 0.018805

val vt : Owl_dense_matrix_d.mat =

          C0        C1        C2
R0 -0.163007 -0.457324  0.874233
R1 -0.212727 -0.848952 -0.483764
R2  0.963419  -0.26483 0.0410998

```
Note that the diagonal matrix `s` is represented as a vector. We can extend it with

```ocaml env=linear-algebra:svd
# let s = Mat.diagm s
val s : Mat.mat =

        C0     C1       C2
R0 123.477      0        0
R1       0 15.504        0
R2       0      0 0.018805

```
However, it is only possible when we know that the original diagonal matrix is square, otherwise the vector contains the $min(m, n)$ diagonal elements.

Also, we can find to the eigenvectors of $AA^T$ to verify that it equals to the eigenvector factorisation.

```ocaml env=linear-algebra:svd
# Linalg.D.eig Mat.(dot a (transpose a))
- : Owl_dense_matrix_z.mat * Owl_dense_matrix_z.mat =
(
                C0              C1             C2
R0  (0.163007, 0i)  (0.963419, 0i) (0.212727, 0i)
R1  (0.457324, 0i)  (-0.26483, 0i) (0.848952, 0i)
R2 (-0.874233, 0i) (0.0410998, 0i) (0.483764, 0i)
,

              C0                C1            C2
R0 (15246.6, 0i) (0.000353627, 0i) (240.373, 0i)
)
```

In this example we ues the `thin` parameter. By default, the `svd` function performs a reduced SVD, where $\Sigma$ is a $m\times~m$ matrix and $V^T$ is a m by n matrix.

Besides, `svd`, we also provide `svdvals` that only returns the singular values, i.e. the vector of diagonal elements.
The function `gsvd` performs a generalised SVD.
`gsvd x y -> (u, v, q, d1, d2, r)` computes the generalised SVD of a pair of general rectangular matrices `x` and `y`.
`d1` and `d2` contain the generalised singular value pairs of `x` and `y`.
The shape of `x` is `m x n` and the shape of `y` is `p x n`.
Here is an example:

```ocaml env=linalg_50
# let x = Mat.uniform 5 5
val x : Mat.mat =

         C0       C1       C2       C3        C4
R0 0.548998 0.623231  0.95821 0.440292  0.551542
R1 0.406659 0.631188 0.434482 0.519169 0.0841121
R2 0.439047 0.459974 0.767078 0.148038  0.445326
R3 0.307424 0.129056 0.998469 0.163971  0.718515
R4 0.474817 0.176199 0.316661 0.476701  0.138534

```

```ocaml env=linalg_50
# let y = Mat.uniform 2 5
val y : Mat.mat =

         C0       C1       C2       C3         C4
R0 0.523882 0.150938 0.718397   0.1573 0.00542669
R1 0.714052 0.874704 0.436799 0.198898   0.406196

```

```ocaml env=linalg_50
# let u, v, q, d1, d2, r = Linalg.D.gsvd x y
val u : Owl_dense_matrix_d.mat =

          C0        C1        C2        C3        C4
R0 -0.385416 -0.294725 -0.398047 0.0383079 -0.777614
R1   0.18222 -0.404037 -0.754063 -0.206208  0.438653
R2 -0.380469 0.0913876 -0.199462  0.847599  0.297795
R3 -0.807427 -0.147819  0.194202 -0.418909  0.336172
R4  0.146816 -0.848345  0.442095  0.249201 0.0347409

val v : Owl_dense_matrix_d.mat =

         C0        C1
R0 0.558969  0.829189
R1 0.829189 -0.558969

val q : Owl_dense_matrix_d.mat =

          C0        C1        C2        C3        C4
R0 -0.436432 -0.169817  0.642272 -0.603428 0.0636394
R1 -0.124923  0.407939 -0.376937 -0.494889 -0.656494
R2  0.400859  0.207482 -0.268507 -0.567199  0.634391
R3 -0.283012 -0.758558 -0.559553 -0.173745 0.0347457
R4  0.743733 -0.431612  0.245375 -0.197629  -0.40163

val d1 : Owl_dense_matrix_d.mat =

   C0 C1 C2       C3        C4
R0  1  0  0        0         0
R1  0  1  0        0         0
R2  0  0  1        0         0
R3  0  0  0 0.319964         0
R4  0  0  0        0 0.0583879

val d2 : Owl_dense_matrix_d.mat =

   C0 C1 C2      C3       C4
R0  0  0  0 0.94743        0
R1  0  0  0       0 0.998294

val r : Owl_dense_matrix_d.mat =

         C0       C1        C2       C3         C4
R0 -0.91393 0.196148 0.0738038  1.45659  -0.268024
R1        0 0.463548  0.286501  1.38499 -0.0595374
R2        0        0  0.346057 0.954629   0.167467
R3        0        0         0 -1.56104  -0.124984
R4        0        0         0        0   0.555067

```

```ocaml env=linalg_50
# Mat.(u *@ d1 *@ r *@ transpose q =~ x)
- : bool = true
# Mat.(v *@ d2 *@ r *@ transpose q =~ y)
- : bool = true
```

TODO: The intuition of SVD.

The SVD is not only important linear algebra concept, but also has a wide and growing applications.
For example, the [Moore-Penrose pseudo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) that works for non-invertible matrix can be implemented efficiently using SVD (we provide `pinv` function in the linear algebra module for the pseudo inverse).
In the Natural Language Processing chapter we will see how SVD plays a crucial role in the language processing field.


## Linear Programming

TODO: placeholder for future implementation. Or in the optimisation chapter.
Understand the method used such as interior  point, and then make the decision.


## Internal: CBLAS and LAPACKE

This section is for those of you who are eager for more low level information.
The BLAS (Basic Linear Algebara Subprogramms) is a specification that describes a set of low-level routines for common linear algebra operation.
The LAPACKE contains more linear algebra routines, such as solving linear systems and matrix factorisations, etc.
Efficient implementation of these function has been practices for a long time in many softwares.
Interfacing to them can provide easy access to high performance routines.

### Low-level Interface to CBLAS & LAPACKE

Owl has implemented the full interface to CBLAS and LAPACKE. Comparing to Julia which chooses to interface to BLAS/LAPACK, you might notice the extra `C` in `CBLAS` and `E` in `LAPACKE` because they are the corresponding C-interface of Fortran implementations. It is often believed that C-interface may introduce some extra overhead. However, it turns out that we cannot really notice any difference at all in practice when dealing with medium or large problems.

- [Owl_cblas module](https://github.com/ryanrhymes/owl/blob/master/src/owl/cblas/owl_cblas.mli) provides the raw interface to CBLAS functions, from level-1 to level-3. The interfaced functions have the same names as those in CBLAS.

- [Owl_lapacke_generated module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) provides the raw interface to LAPACKE functions (over 1,000) which also have the same names defined in [lapacke.h](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/lapacke.h).

- [Owl_lapacke module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke.ml) is a very thin layer of interface between [Owl_lapacke_generated module](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) and [Linalg module](https://github.com/ryanrhymes/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli). The purpose is to provide a unified function to make generic functions over different number types.


### High-level Wrappers in Linalg Module

The functions in [Owl_cblas](https://github.com/ryanrhymes/owl/blob/master/src/owl/cblas/owl_cblas.mli) and [Owl_lapacke_generated](https://github.com/ryanrhymes/owl/blob/master/src/owl/lapacke/owl_lapacke_generated.mli) are very low-level, e.g., you need to deal with calculating parameters, allocating workspace, post-processing results, and many other tedious details. You do not really want to use them directly unless you have enough background in numerical analysis and chase after the performance. In practice, you should use [Linalg](https://github.com/ryanrhymes/owl/blob/master/src/owl/linalg/owl_linalg_generic.mli) module which gives you a high-level wrapper for frequently used functions.

TODO: Examples

### Low-level factorisation and Helper functions

```text

  val lufact : ('a, 'b) t -> ('a, 'b) t * (int32, int32_elt) t

  val qrfact : ?pivot:bool -> ('a, 'b) t -> ('a, 'b) t * ('a, 'b) t * (int32, int32_elt) t

  val bkfact : ?upper:bool -> ?symmetric:bool -> ?rook:bool -> ('a, 'b) t -> ('a, 'b) t * (int32, int32_elt) t

  val peakflops : ?n:int -> unit -> float
  (* peak number of float point operations using [Owl_cblas.dgemm] function. *)

```

TODO: How these low level functions are used in Owl Code.


## Sparse Matrices

What we have mentioned so far are dense matrix. But when the elements are sparsely distributed in the matrix, such as the identity matrix, the *sparse* structure might be more efficient.
The sparse matrix is proivded in the `Sparse.Matrix` module, and also support the four types of number in the `S`, `D`, `C`, and `Z` submodules.

(Perhaps these contents are better to discuss in Ndarray module.)

Very brief.
Focusing on introducing the data structure (CSC, CSR, etc), no the method.
Mention the [owl_suitesparse](https://github.com/owlbarn/owl_suitesparse)
TODO: Introduce the sparse data structure in owl, and introduce CSR, CSC, tuples, and other formats.


## Summary


## References
