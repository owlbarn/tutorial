# Conventions

Every software system has its own rules and conventions which require the developers to comply with. Owl is not an exception, for example the [rules on broadcasting operation](broadcast.html) and the [conventions on slice definition](slicing.html). In this chapter, I will cover the function naming and various conventions in Owl.


## OCaml In a Nutshell

TBD


## Pure vs. Impure

`Ndarray` module contains many functions to manipulate and perform mathematical operations over multi-dimensional arrays. The **pure functions** (a.k.a immutable functions) refer to those which do not modify the passed in variables but always return a new one as result. In contrast, **impure functions** (a.k.a mutable functions) refer to those which modifies the passed-in variables in place.

The arguments between pure and impure functions will never end. Functional programming in general promotes the use of immutable data structures. Using impure functions makes it difficult to reason the correctness of the code, therefore you need to think carefully when you decide to use them. On the other hand, generating a fresh `1000 x 1000` matrix every time simply because you modify one element does not seem very practical either.

The introduction of impure functions into Owl is under many careful and practical considerations. One primary motivation of using in-place modification is to avoid expensive memory allocation and deallocation operations, this can significantly improve the runtime performance of a numerical application especially when large ndarrays and matrices involved.

Can we have the best parts of both world, i.e. writing functional code and being memory efficient at the same time? As you learn more about Owl, you will realise that this can be achieved by lazily evaluating a mathematical expression using [computation graph](cgraph.html). The programmer focusses on the functional code, Owl's computation graph module takes care of the "dangerous task" -- allocating and managing the memory efficiently.

Many pure functions in Ndarray module have their corresponding impure version, the difference is that impure version has an extra underscore "`_`" at the end of function names. For example, the following functions are the pure functions in `Arr` module.

```ocaml

  Arr.sin;;
  Arr.cos;;
  Arr.log;;
  Arr.abs;;
  Arr.add;;
  Arr.mul;;

```

Their corresponding impure functions are as follows.

```ocaml

  Arr.sin_;;
  Arr.cos_;;
  Arr.log_;;
  Arr.abs_;;
  Arr.add_;;
  Arr.mul_;;

```

For unary operators such as `Arr.sin x`, the situation is rather straightforward, `x` will be modified in place. However, for binary operates such as `Arr.add_scalar_ x a` and `Arr.add_ x y`, the situation needs some clarifications. For `Arr.add_scalar x a`, `x` will be modified in place and stores the final result, this is trivial because `a` is a scalar.

For `Arr.add_ x y`, the question is where to store the final result when both inputs are ndarray. Let's look at the type of `Arr.add_` function.

```text
val Arr.add_ : ?out:Arr.arr -> Arr.arr -> Arr.arr -> unit
```

As we can see from the function type, the output can be specified by an optional `out` parameter. If `out` is missing in the inputs, then Owl will try to use first operand (i.e. `x`) to store the final result. Because the binary operators in Owl support broadcasting operations by default, this further indicates when using impure functions every dimension of the first argument `x` must not be smaller than that of the second argument `y`. In other words, impure function only allows broadcasting smaller `y` onto `x` which is big enough to accommodate the result.

Most binary math functions in Owl are associated with a shorthand operator, such as `+`, `-`, `*`, and `/`. The impure versions also have their own operators. For example, corresponding to `Arr.(x + y)` which returns the result in a new ndarray, you can write `Arr.(x += y)` which adds up `x` and `y` and saves the result into `x`.

--------------    --------------    --------------
Function Name     Pure              Impure
--------------    --------------    --------------
add               `+`               `+=`

sub               `-`               `-=`

mul               `*`               `*=`

div               `/`               `/=`

add_scalar        `+$`              `+$=`

sub_scalar        `-$`              `-$=`

mul_scalar        `*$`              `*$=`

div_scalar        `/$`              `/$=`
--------------    --------------    --------------
: Alias of pure and impure binary math functions  {#tbl:convention:pure}


##  Ndarray vs. Scalar

There are three types of ndarray operations: *map*, *scan*, and *reduce*. Many functions can be categorised as reduction operations, such as `Arr.sum`, `Arr.prod`, `Arr.min`, `Arr.mean`, `Arr.std`, and etc. All the reduction functions in Owl has a name parameter called `axis`. When you apply these reduction operations on a multi-dimensional array, there are two possible cases:

* if axis is explicitly specified, then Owl reduces along the specified axis;
* if axis is not specified, then Owl flattens the ndarray into a vector first and reduce all the elements along the axis 0.

If the passed in ndarray is already one-dimensional, then two cases are equivalent. In the following code snippet, `a` has shape `[|3;1;3|]` whereas `b` has shape `[|1|]` since it only contains one element.

```ocaml

  let x = Arr.sequential [|3;3;3|];;
  let a = Arr.sum ~axis:1 x;;
  let b = Arr.sum x;;

```

If you want to add the result in `b` with another float number, you need to retrieve the value by calling `get` function.

```ocaml

  let c = Arr.get b [|0|] in
  c +. 10.;;

```

This does not look very convenient if we always need to extract a scalar value from the return of reduction operations. This is not a problem for the languages like Python and Julia since the return type is dynamically determined. However, for OCaml, this turns out to be challenging: we either use a unified type; or we implement another set of functions. In the end, we picked the latter in Owl's design. Every reduction operation has two versions:

* one allows you to reduce along the specified axis, or reduce all the elements, but always returns an ndarray;
* one only reduces all the elements and always returns a scalar value.

The difference between the two is that the functions returning a scalar ends up with an extra prime "`'`" character in their names. For example, for the first type of functions that return an ndarray, their function names look like these.

```ocaml

  Arr.sum;;
  Arr.min;;
  Arr.prod;;
  Arr.mean;;
  Arr.std;;

```

For the second type of functions that return a scalar, their name looks like these.

```ocaml

  Arr.sum';;
  Arr.min';;
  Arr.prod';;
  Arr.mean';;
  Arr.std';;

```

Technically, `Arr.sum'` is equivalent to the following code.

```ocaml

  let sum' x =
    let y = Arr.sum x in
    Arr.get y [|0|]

```

Let's extend the previous code snippet, and test it in OCaml's toplevel. Then you will understand the difference immediately.

```ocaml

  let x = Arr.sequential [|3;3;3|];;
  let a = Arr.sum ~axis:1 x;;
  let b = Arr.sum x;;
  let c = Arr.sum' x;;

```

Rules and conventions often reveals the tradeoffs in a design. By clarifying the restrictions, we hope the programmers can choose the right functions to use in a specific scenario.


## Infix Operators

The operators in Owl are implemented in the functors defined in the `Owl_operator` module. These operators are categorised into `Basic`, `Extend`, `Matrix`, and `Ndarray` four module type signatures, because some operations are only meaningful for certain data structures. E.g., matrix multiplication `*@` is only defined in `Matrix` signature.

As long as a module implements all the functions defined in the module signature, you can use these functors to generate corresponding operators. In most cases, you do not need to work with these functors directly in Owl since I have done the generation part for you already.

The operators have been included in each `Ndarray` and `Matrix` module. The following table summarises the operators currently implemented. In the table, both `x` and `y` represent either a matrix or an ndarray while `a` represents a scalar value.


------------  ------------  ------------------------  ------------  -----------------
Operator      Example       Operation                 Dense/Sparse  Ndarray/Matrix
------------  ------------  ------------------------  ------------  -----------------
`+`           `x + y`       element-wise add          both          both

`-`           `x - y`       element-wise sub          both          both

`*`           `x * y`       element-wise mul          both          both

`/`           `x / y`       element-wise div          both          both

`+$`          `x +$ a`      add scalar                both          both

`-$`          `x -$ a`      sub scalar                both          both

`*$`          `x *$ a`      mul scalar                both          both

`/$`          `x /$ a`      div scalar                both          both

`$+`          `a $+ x`      scalar add                both          both

`$-`          `a $- x`      scalar sub                both          both

`$*`          `a $* x`      scalar mul                both          both

`$/`          `a $/ x`      scalar div                both          both

`=`           `x = y`       comparison                both          both

`!=`          `x != y`      comparison                both          both

`<>`          `x <> y`      same as `!=`              both          both

`>`           `x > y`       comparison                both          both

`<`           `x < y`       comparison                both          both

`>=`          `x >= y`      comparison                both          both

`<=`          `x <= y`      comparison                both          both

`=.`          `x =. y`      element-wise cmp          Dense         both

`!=.`         `x !=. y`     element-wise cmp          Dense         both

`<>.`         `x <>. y`     same as `!=.`             Dense         both

`>.`          `x >. y`      element-wise cmp          Dense         both

`<.`          `x <. y`      element-wise cmp          Dense         both

`>=.`         `x >=. y`     element-wise cmp          Dense         both

`<=.`         `x <=. y`     element-wise cmp          Dense         both

`=$`          `x =$ y`      comp to scalar            Dense         both

`!=$`         `x !=$ y`     comp to scalar            Dense         both

`<>$  `       `x <>$ y`     same as `!=`              Dense         both

`>$`          `x >$ y`      compare to scalar         Dense         both

`<$`          `x <$ y`      compare to scalar         Dense         both

`>=$`         `x >=$ y`     compare to scalar         Dense         both

`<=$`         `x <=$ y`     compare to scalar         Dense         both

`=.$`         `x =.$ y`     element-wise cmp          Dense         both

`!=.$`        `x !=.$ y`    element-wise cmp          Dense         both

`<>.$`        `x <>.$ y`    same as `!=.$`            Dense         both

`>.$`         `x >.$ y`     element-wise cmp          Dense         both

`<.$`         `x <.$ y`     element-wise cmp          Dense         both

`>=.$`        `x >=.$ y`    element-wise cmp          Dense         both

`<=.$`        `x <=.$ y`    element-wise cmp          Dense         both

`=~`          `x =~ y`      approx `=`                Dense         both

`=~$`         `x =~$ y`     approx `=$`               Dense         both

`=~.`         `x =~. y`     approx `=.`               Dense         both

`=~.$`        `x =~.$ y`    approx `=.$`              Dense         both

`%`           `x % y`       mod divide                Dense         both

`%$`          `x %$ a`      mod divide scalar         Dense         both

`**`          `x ** y`      power function            Dense         both

`*@`          `x *@ y`      matrix multiply           both          Matrix

`/@`          `x /@ y`      solve linear system       both          Matrix

`**@`         `x **@ a`     matrix power              both          Matrix

`min2`        `min2 x y`    element-wise min          both          both

`max2`        `max2 x y`    element-wise max          both          both

`@=`          `x @= y`      concatenate vertically    Dense         both

`@||`         `x @|| y`     concatenate horizontally  Dense         both
------------  ------------  ------------------------  ------------  -----------------
: Infix operators in ndarray and matrix modules {#tbl:convention:infix}


There is a list of things worth your attention as below.

- `*` is for element-wise multiplication; `*@` is for matrix multiplication. You can easily understand the reason if you read the source code of [Algodiff](https://github.com/ryanrhymes/owl/blob/master/src/owl/optimise/owl_algodiff_generic.ml) module. Using `*` for element-wise multiplication (for matrices) leads to the consistent implementation of algorithmic differentiation.

- `+$` has its corresponding operator `$+` if we flip the order of parameters. However, be very careful about the operator precedence since OCaml determines the precedence based on the first character of an infix. `+$` preserves the precedence whereas `$+` does not. Therefore, I recommend using `$+` with great care. Please always use parentheses to explicitly specify the precedence. The same also applies to `$-`, `$*`, and `$/`.

- For comparison operators, e.g. both `=` and `=.` compare all the elements in two variables `x` and `y`. The difference is that `=` returns a boolean value whereas `=.` returns a matrix or ndarray of the same shape and same type as `x` and `y`. In the returned result, the value in a given position is `1` if the values of the corresponding position in `x` and `y` satisfy the predicate, otherwise it is `0`.

- For the comparison operators ended with `$`, they are used to compare a matrix/ndarray to a scalar value.

Operators are easy to use, here are some examples.

```ocaml env=convention_00
let x = Mat.uniform 5 5;;
let y = Mat.uniform 5 5;;

Mat.(x + y);;
Mat.(x * y);;
Mat.(x ** y);;
Mat.(x *@ y);;

(* compare the returns of the following two *)

Mat.(x > y);;
Mat.(x >. y);;
```

Here is the return of the first example.

```ocaml env=convention_00
# Mat.(x > y)
- : bool = false
```

Here is the return of the second example.

```ocaml env=convention_00
# Mat.(x >. y)
- : (float, float64_elt) Owl_dense_matrix_generic.t =

   C0 C1 C2 C3 C4
R0  0  1  1  0  1
R1  0  1  1  0  1
R2  0  0  0  0  0
R3  1  0  0  0  1
R4  0  1  0  0  1

```

Now I am sure you can understand the difference between `>` and `>.`, and the same applies to other binary comparison operators.

Note that the extending indexing and slicing operators are not included in the table above, but you can find the detailed explanation in [Indexing and Slicing Chapter](slicing.html).


## Operator Extension

As you can see, the operators above do not allow interoperation on different number types (which may not be bad thing in my opinion actually). E.g., you cannot add a `float32` matrix to `float64` matrix unless you explicitly call the `cast` functions in `Generic` module.

Some people just like Pythonic way of working, `Owl.Ext` module is specifically designed for this purpose, to make prototyping faster and easier. Once you open the module, `Ext` immediately provides a set of operators to allow you to interoperate on different number types, as below. It automatically casts types for you if necessary.

-------------    -------------     --------------------------
Operator         Example           Operation
-------------    -------------     --------------------------
`+`              `x + y`           add

`-`              `x - y`           sub

`*`              `x * y`           mul

`/`              `x / y`           div

`=`              `x = y`           comparison, return bool

`!=`             `x != y`          comparison, return bool

`<>`             `x <> y`          same as `!=`

`>`              `x > y`           comparison, return bool

`<`              `x < y`           comparison, return bool

`>=`             `x >= y`          comparison, return bool

`<=`             `x <= y`          comparison, return bool

`=.`             `x =. y`          element_wise comparison

`!=.`            `x !=. y`         element_wise comparison

`<>.`            `x <>. y`         same as `!=.`

`>.`             `x >. y`          element_wise comparison

`<.`             `x <. y`          element_wise comparison

`>=.`            `x >=. y`         element_wise comparison

`<=.`            `x <=. y`         element_wise comparison

`%`              `x % y`           element_wise mod divide

`**`             `x ** y`          power function

`*@`             `x *@ y`          matrix multiply

`min2`           `min2 x y`        element-wise min

`max2`           `max2 x y`        element-wise max
-------------    -------------     --------------------------
: Operator extensions {#tbl:convention:ext}

You may have noticed, the operators ended with `$` (e.g., `+$`, `-$` ...) disappeared from the table, which is simply because we can add/sub/mul/div a scalar with a matrix directly and we do not need these operators any more. Similar for comparison operators, because we can use the same `>` operator to compare a matrix to another matrix, or compare a matrix to a scalar, we do not need `>$` any longer. Allowing interoperation makes the operator table much shorter.

Currently, the operators in `Ext` only support interoperation on dense structures. Besides binary operators, `Ext` also implements most of the common math functions which can be applied to float numbers, complex numbers, matrices, and ndarray. These functions are:

`im`; `re`; `conj`, `abs`, `abs2`, `neg`, `reci`, `signum`, `sqr`, `sqrt`, `cbrt`, `exp`, `exp2`, `expm1`, `log`, `log10`, `log2`, `log1p`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `floor`, `ceil`, `round`, `trunc`, `erf`, `erfc`, `logistic`, `relu`, `softplus`, `softsign`, `softmax`, `sigmoid`, `log_sum_exp`, `l1norm`, `l2norm`, `l2norm_sqr`, `inv`, `trace`, `sum`, `prod`, `min`, `max`, `minmax`, `min_i`, `max_i`, `minmax_i`.


Note that `Ext` contains its own `Ext.Dense` module which further contains the following submodules.

- `Ext.Dense.Ndarray.S`
- `Ext.Dense.Ndarray.D`
- `Ext.Dense.Ndarray.C`
- `Ext.Dense.Ndarray.Z`
- `Ext.Dense.Matrix.S`
- `Ext.Dense.Matrix.D`
- `Ext.Dense.Matrix.C`
- `Ext.Dense.Matrix.Z`

These modules are simply the wrappers of the original modules in `Owl.Dense` module so they provide most of the APIs already implemented. The extra thing these wrapper modules does is to pack and unpack the raw number types for you automatically. However, you can certainly use the raw data types then use the constructors defined in `Owl_ext_types` to wrap them up by yourself. The constructors are defined as below.

```text

  type ext_typ =
    F   of float
    C   of Complex.t
    DMS of dms
    DMD of dmd
    DMC of dmc
    DMZ of dmz
    DAS of das
    DAD of dad
    DAC of dac
    DAZ of daz
    SMS of sms
    SMD of smd
    SMC of sms
    SMZ of smd
    SAS of sas
    SAD of sad
    SAC of sac
    SAZ of saz

```

There are also corresponding `packing` and `unpacking` functions you can use, please read `owl_ext_types.ml <https://github.com/owlbarn/owl/blob/master/src/owl/ext/owl_ext_types.ml>`_ for more details.

Let's see some examples to understand how convenient it is to use `Ext` module.

```ocaml
open Owl.Ext;;

let x = Dense.Matrix.S.uniform 5 5;;
let y = Dense.Matrix.C.uniform 5 5;;
let z = Dense.Matrix.D.uniform 5 5;;

x + F 5.;;
x * C Complex.({re = 2.; im = 3.});;
x - y;;
x / y;;
x *@ y;;

(** ... *)

x > z;;
x >. z;;
(x >. z) * x;;
(x >. F 0.5) * x;;
(F 10. * x) + y *@ z;;

(** ... *)

round (F 10. * (x *@ z));;
sin (F 5.) * cos (x + z);;
tanh (x * F 10. - z);;

(** ... *)
```

Before we finish this chapter, I want to point out the caveat. `Ext` tries to mimic the dynamic languages like Python by with unified types. This prevents OCaml compiler from doing type checking in compilation phase and introduces extra overhead in calling functions. Therefore, besides fast experimenting in toplevel, I do not recommend to use `Ext` module in the production code.


## Module Structures

In Owl, `Dense` module contains the modules of dense data structures. For example, `Dense.Matrix` supports the operations of dense matrices. Similarly, `Sparse` module contains the modules of sparse data structures.

```text
Dense.Ndarray;;   (* dense ndarray *)
Dense.Matrix;;    (* dense matrix *)

Sparse.Ndarray;;  (* sparse ndarray *)
Sparse.Matrix;;   (* sparse ndarray *)
```

All these four modules consists of five submodules to handle different types of numbers.

* `S` module supports single precision float numbers `float32`;
* `D` module supports double precision float numbers `float64`;
* `C` module supports single precision complex numbers `complex32`;
* `Z` module supports double precision complex numbers `complex64`;
* `Generic` module supports all aforementioned number types via GADT.

With `Dense.Ndarray`, you can create a dense n-dimensional array of no more than 16 dimensions. This constraint originates from the underlying `Bigarray.Genarray` module. In practice, this constraint makes sense since the space requirement will explode as the dimension increases. If you need anything higher than 16 dimensions, you need to use `Sparse.Ndarray` to create a sparse data structure.


## Number and Precision

After deciding the suitable data structure (either dense or sparse), you can create a ndarray/matrix using creation function in the modules, using e.g., `empty`, `create`, `zeros`, `ones` ... The type of numbers (real or complex) and its precision (single or double) needs to be passed to the creations functions as the parameters.

```ocaml env=convention_01
# Dense.Ndarray.Generic.zeros Float64 [|5;5|]
- : (float, float64_elt) Dense.Ndarray.Generic.t =

   C0 C1 C2 C3 C4
R0  0  0  0  0  0
R1  0  0  0  0  0
R2  0  0  0  0  0
R3  0  0  0  0  0
R4  0  0  0  0  0

```

With `zeros` function, all the elements in the created data structure will be initialised to zeros.

Technically, `S`, `D`, `C`, and `Z` are the wrappers of `Generic` module with explicit type information provided. Therefore you can save the type constructor which was passed into the `Generic` module if you use these submodules directly.

```ocaml
Dense.Ndarray.S.zeros [|5;5|];;    (* single precision real ndarray *)
Dense.Ndarray.D.zeros [|5;5|];;    (* double precision real ndarray *)
Dense.Ndarray.C.zeros [|5;5|];;    (* single precision complex ndarray *)
Dense.Ndarray.Z.zeros [|5;5|];;    (* double precision complex ndarray *)
```

The following examples are for dense matrices.

```ocaml
Dense.Matrix.S.zeros 5 5;;     (* single precision real matrix *)
Dense.Matrix.D.zeros 5 5;;     (* double precision real matrix *)
Dense.Matrix.C.zeros 5 5;;     (* single precision complex matrix *)
Dense.Matrix.Z.zeros 5 5;;     (* double precision complex matrix *)
```

The following examples are for sparse ndarrays.

```ocaml
Sparse.Ndarray.S.zeros [|5;5|];;    (* single precision real ndarray *)
Sparse.Ndarray.D.zeros [|5;5|];;    (* double precision real ndarray *)
Sparse.Ndarray.C.zeros [|5;5|];;    (* single precision complex ndarray *)
Sparse.Ndarray.Z.zeros [|5;5|];;    (* double precision complex ndarray *)
```

The following examples are for sparse matrices.

```ocaml
Sparse.Matrix.S.zeros 5 5;;     (* single precision real matrix *)
Sparse.Matrix.D.zeros 5 5;;     (* double precision real matrix *)
Sparse.Matrix.C.zeros 5 5;;     (* single precision complex matrix *)
Sparse.Matrix.Z.zeros 5 5;;     (* double precision complex matrix *)
```

In short, `Generic` module can do everything that submodules can, but for some functions (e.g. creation functions) you need to explicitly pass in the type information.


## Polymorphic Functions

Polymorphism is achieved by pattern matching and GADT in `Generic` module. This means many functions in `Generic` module can handle aforementioned four different number types.

In the following, I use the `sum` function in `Dense.Matrix.Generic` module as an example. `sum` function returns the summation of all the elements in a matrix.

```ocaml
open Owl;;

let x = Dense.Matrix.S.eye 5 in
  Dense.Matrix.Generic.sum x;;

let x = Dense.Matrix.D.eye 5 in
  Dense.Matrix.Generic.sum x;;

let x = Dense.Matrix.C.eye 5 in
  Dense.Matrix.Generic.sum x;;

let x = Dense.Matrix.Z.eye 5 in
  Dense.Matrix.Generic.sum x;;
```

As we can see, no matter what kind of numbers are held in an identity matrix, we always pass it to `Dense.Matrix.Generic.sum` function. Similarly, we can do the same thing for other modules (`Dense.Ndarray`, `Sparse.Matrix`, and etc.) and other functions (`add`, `mul`, `neg`, and etc.).

Meanwhile, each submodule also contains the same set of functions, e.g, as below,

```ocaml
Dense.Matrix.S.(eye 5 |> sum);;
```


## Module Shortcuts

In reality, we often work with double precision numbers, therefore Owl provides some shortcuts to the data structures of double precision float numbers:

* `Arr` is equivalent to double precision real `Dense.Ndarray.D`;
* `Mat` is equivalent to double precision real `Dense.Matrix.D`;

With these shortcut modules, you are no longer required to pass in type information. Here are some examples.

```ocaml
Arr.zeros [|5|];;        (* same as Dense.Ndarray.D.zeros [|5|] *)
Mat.zeros 5 5;;          (* same as Dense.Matrix.D.zeros 5 5 *)
```


More examples besides creation functions are as follows.

```text
Mat.load "data.mat";;    (* same as Dense.Matrix.D.load "data.mat" *)
Mat.of_array 5 5 x;;     (* same as Dense.Matrix.D.of_array 5 5 x *)
Mat.linspace 0. 9. 10;;  (* same as Dense.Matrix.D.linspace 0. 9. 10 *)
```


If you actually work more often with other number types like Complex, you can certainly make your own alias to corresponding `S`, `D`, `C`, and `Z` module if you like.



## Type Casting

As I mentioned before, there are four basic number types. You can therefore cast one value from one type to another one by using the `cast_*` functions in `Generic` module.

* `Generic.cast_s2d`: cast from `float32` to `float64`;
* `Generic.cast_d2s`: cast from `float64` to `float32`;
* `Generic.cast_c2z`: cast from `complex32` to `complex64`;
* `Generic.cast_z2c`: cast from `complex64` to `complex32`;
* `Generic.cast_s2c`: cast from `float32` to `complex32`;
* `Generic.cast_d2z`: cast from `float64` to `complex64`;
* `Generic.cast_s2z`: cast from `float32` to `complex64`;
* `Generic.cast_d2c`: cast from `float64` to `complex32`;

In fact, all these function rely on the following `cast` function.

```text

  val cast : ('a, 'b) kind -> ('c, 'd) t -> ('a, 'b) t

```

The first parameter specifies the cast type. If the source type and the cast type are the same, `cast` function simply makes a copy of the passed in value.

```ocaml env=convention_02
# let x = Arr.uniform [|5;5|]                     (* created in float64 *)
val x : Arr.arr =

         C0        C1        C2       C3       C4
R0 0.648406  0.616945  0.828173 0.579604 0.212017
R1 0.960002 0.0563993  0.219521 0.855164 0.414024
R2 0.526179  0.532062 0.0640247 0.786426 0.956565
R3 0.810557  0.476031  0.516506  0.11439 0.964041
R4 0.981665  0.446936  0.276383 0.414747 0.174775

```

Now let's cast `x` from float64 to complex32.

```ocaml env=convention_02
# let y = Dense.Ndarray.Generic.cast Complex32 x  (* cast to complex32 *)
val y : (Complex.t, complex32_elt) Dense.Ndarray.Generic.t =

               C0              C1              C2             C3             C4
R0 (0.648406, 0i)  (0.616945, 0i)  (0.828173, 0i) (0.579604, 0i) (0.212017, 0i)
R1 (0.960002, 0i) (0.0563993, 0i)  (0.219521, 0i) (0.855164, 0i) (0.414024, 0i)
R2 (0.526179, 0i)  (0.532062, 0i) (0.0640247, 0i) (0.786426, 0i) (0.956565, 0i)
R3 (0.810557, 0i)  (0.476031, 0i)  (0.516506, 0i)  (0.11439, 0i) (0.964041, 0i)
R4 (0.981664, 0i)  (0.446936, 0i)  (0.276383, 0i) (0.414747, 0i) (0.174775, 0i)

```

To know more about the functions provided in each module, please read the corresponding interface file of `Generic` module. The `Generic` module contains the documentation.

* [Dense.Ndarray.Generic](https://github.com/ryanrhymes/owl/blob/master/src/owl/dense/owl_dense_ndarray_generic.mli)
* [Dense.Matrix.Generic](https://github.com/ryanrhymes/owl/blob/master/src/owl/dense/owl_dense_matrix_generic.mli)
* [Sparse.Ndarray.Generic](https://github.com/ryanrhymes/owl/blob/master/src/owl/sparse/owl_sparse_ndarray_generic.mli)
* [Sparse.Matrix.Generic](https://github.com/ryanrhymes/owl/blob/master/src/owl/sparse/owl_sparse_matrix_generic.mli)
