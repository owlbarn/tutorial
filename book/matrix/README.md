# Vectors and Matrices

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

The following examples can be run directly in `utop`. I assume that you already loaded `owl` library and opened `Owl` module in `utop` by executing the following commands.

```ocaml env=matrix_env0
# #require "owl"
# open Owl
```

## Create Matrices

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


## Access Elements

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
val a : float = 0.52388195972889662
```


The modifications to a matrix using `set` are in-place. This is always true for dense matrices. For sparse matrices, the thing can be complicated because of performance issues. I will discuss about sparse matrices separately in a separate post.


## Iterate Elements, Rows, Columns

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

## Filter Elements, Rows, Columns

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


## Compare Two Matrices

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


## Matrix Arithmetics

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


## Save & Load Matrices

All matrices can be serialised to storage by using `save`. Later, you can load a matrix using `load` function.

```ocaml env=matrix_env1

  Mat.save x "m0.mat";;    (* save x to m0.mat file *)
  Mat.load "m0.mat";;      (* load m0.mat back to the memory *)

```

I also made corresponding `save_txt` and `load_txt` functions for a simple tab-delimited, human-readable format. Note the performance is much worse than the corresponding `save` and `load`.


## Other Operations

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

Enjoy Owl and happy coding!
