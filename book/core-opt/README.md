# Core Optimisation

In this chapter, we introduce how the core operations are implemented in C language for performance, and use some examples to show the techniques we use to optimise the C code.

TODO: update evaluations

TODO: logic is not very clear; paragraphs are fragmented.

## Background

First, we briefly introduce some background information about numerical libraries and related optimisation.

### Numerical Libraries

There are two widely used specifications of low level linear algebra routines.
Basic Linear Algebra Subprograms (BLAS) consists of three levels of routines, from vector to matrix-vector and then to matrix-matrix operations.
The other one, Linear Algebra Package (LAPACK), specifies routines for advanced numerical linear algebra, including solving systems of linear equations, linear least squares, eigenvalue problems, SVD, etc.

The implementations of these specifications vary in different libraries, e.g. OpenBLAS~\cite{xianyi2012model} and Math Kernel Library (MKL).
OpenBLAS is a popular open source optimised BLAS library.
MKL is a proprietary library, and provides highly optimised mathematical functions on Intel processors. It implements not only BLAS and LAPACK but also FFT and other computationally intensive mathematical functions.
Another implementation is Eigen, a C++ template linear algebra library.
The CPU implementation of many kernels in TensorFlow uses the Eigen Tensor class.
The Automatically Tuned Linear Algebra Software (ATLAS) is another BLAS implementation, featuring automatically-tuned routines on specific hardware.

These basic libraries focus on optimising the performance of operations in different hardware and software environment, but they don't provide APIs that are easy to use for end users. That requires libraries such as NumPy, Julia, Matlab, and Owl.
NumPy is the fundamental package for scientific computing with Python. It contains a powerful N-dimensional array abstraction. Julia is a high-level, high-performance dynamic programming language for numerical computing. Both are widely used and considered state of the art in numerical computing.
Both NumPy and Julia rely on OpenBLAS or MKL for linear algebra backends.
Matlab, the numerical computing library that has millions of uses worldwide, also belongs to this category.

Deep learning libraries such as TensorFlow, PyTorch, and MxNet are popular.
Keras is a user-friendly neural networks API that can run on top of TensorFlow.
Instead of the wide range of numerical functionalities that NumPy etc. provide, these libraries focus on building machine learning applications for both research and production.
Owl library provides its own neural network module.

### Optimisation of Numerical Computation

To achieve optimal performance has always been the target of numerical libraries.
However, the complexity of current computation platforms is growing fast, and the "free" performance boost that benefits from hardware upgrade also stagnates.
These factors have made it difficult to achieve the optimal performance.
Below list some of the techniques that we use to optimise operations in Owl.

One method to utilise the parallelism of a computation platform is to use the Single Instruction Multiple Data (SIMD) instruction sets.
They exploit data level parallelism by executing the same instruction on a set of data simultaneously, instead of repeating it multiple times on a single scalar value.
One easy way to use SIMD is to rely on the automatic vectorisation capabilities of modern compilers,
but in many cases developers have to manually vectorise their code with SIMD intrinsic functions.
The Intel Advanced Vector Extensions (AVX) instruction set is offered on Intel and AMD processors, and the ARM processors provide the Advanced SIMD (NEON) extension.

Another form of parallelism is to execute instructions on multiple cores.
OpenMP is a C/C++/FORTRAN compiler extension that allows shared memory multiprocessing programming.
It is widely supported on compilers such as GCC and Clang, on different hardware platforms.
It is important for a numerical library to porting existing code to the OpenMP standard.

To achieve optimal performance often requires choosing the most suitable system parameters on different machines or for different inputs.
Aiming at providing fast matrix multiplication routines, the ATLAS library runs a set of micro-benchmarks to decide hardware specifications, and then search for the most suitable parameters such as block size in a wide tuning space.

One general algorithm cannot always achieve optimal performance.
One of the most important techniques the Julia uses is ``multiple dispatch'', which means that the library provides different specific implementations according to the type of inputs.

Besides these techniques, the practical experience from others always worth learning during development.
These principles still hold true in the development of modern numerical libraries.
An optimised routine can perform orders of magnitude faster than a naive implementation.

## Interfacing to C Code

Despite the efficiency of OCaml, we rely on C implementation to deliver high performance for core functions.
In the previous chapters in the Part I of this book, we have seen that how some of Owl modules, such as FFT and Linear Algebra, interface to existing C libraries.
Optimising operations in these fields has been the classic topic of high performance computation for years, and thus there is no need to re-invent the wheels.
We can directly interface to these libraries to provide good performance.

### Ndarray Operations

Interfacing to high performance language is not uncommon practice among numerical libraries.
If you look at the source code of [NumPy](https://github.com/numpy/numpy), more than 50% is C code.
In [SciPy](https://github.com/scipy/scipy), the Fortran and C code takes up more than 40%.
Even in [Julia](https://github.com/JuliaLang/julia), about 26% of its code is in C or C++, most of them in the core source code.

Besides interfacing to existing libraries, we focus on implementing the core operations in the Ndarray modules with C code.
As we have seen in the N-Dimensional Arrays chapter, the n-dimensional array module lies in the heart of Owl, and many other libraries.
NumPy library itself focus solely on providing a powerful ndarray module to the Python world.

A ndarray is a container of items of the same type.
It consists of a contiguous block of memory, combined with an indexing scheme that maps N integers into the location of an item in the block.
A stride indexing scheme can then be applied on this block of memory to access elements.
Once converted properly to the C world, a ndarray can be effectively manipulated with normal C code.

Here we list the categories of operations that are optimised with C in Owl.
Many operations are first implemented in OCaml but then updated to C driven by our practical experience and applications.

- mathematics operations, which are divided into map function, fold functions, and comparison functions.
- convolution and pooling operations, since they took up most of the computation resources in DNN-related application
- slicing, the basic operation for n-dimensional array
- matrix operations, including transpose, swapping, and check functions such as `is_hermitian`, `is_symmetric` etc.
- sorting operation
- other functions, including contraction, sliding, and repeat.

### From OCaml to C

TODO: need more detail to show the layer-by-layer structure. A callgraph would be good.

Let's use examples to see exactly how we implement core operations wih C and interface them to OCaml.

In Owl, ndarray is built on OCaml's native `Bigarray.Genarray`.
The Bigarray module implements multi-dimensional numerical arrays of integers and floating-point numbers, and `Genarray` is the type of `Bigarrays` with variable numbers of dimensions.

Genarray is of type `('a, 'b, 't) t`.
It has three parameters: OCaml type for accessing array elements (`'a`),  the actual type of array elements (`'b`), and indexing scheme (`'t`).
The initial design of Owl supports both col-major and row-major indexing, but this choice leads to a lot of confusion, since the FORTRAN way of indexing starts from index 1, while the row-major starts from 0.
Owl sticks with the row-major scheme now, and therefore in the core library the owl ndarray is define as:

```ocaml
open Bigarray
type ('a, 'b) owl_arr = ('a, 'b, c_layout) Genarray.t
```

Now, let's look at the `'a` and `'b`.
In the GADT type `('a, 'b) kind`, an OCaml type `'a` is for values read or written in the Bigarray, such as `int` or `float`, and `'b` represents the actual contents of the Bigarray, such as the `float32_elt` that contains 32-bit single precision floats.
Owl supports four basic types of element: float, double, float complex,
and double complex number. And we use the definition of type `('a, 'b) kind` in the `BigArray` module.

```ocaml
open Bigarray

type ('a, 'b) kind =
|	Float32 : (float, float32_elt) kind
|	Float64 : (float, float64_elt) kind
|	Complex32 : (Complex.t, complex32_elt) kind
|	Complex64 : (Complex.t, complex64_elt) kind
```

Suppose we want to implement the sine math function, which maps the `sin` function on every elements in the . We we need to implement four different versions, each for one of these four number types.
The basic code looks like this:

```text
let _owl_sin : type a b. (a, b) kind -> int -> ('a, 'b) owl_arr -> ('a, 'b) owl_arr -> unit =
  fun k l x y ->
  match k with
  | Float32   -> owl_float32_sin l x y
  | Float64   -> owl_float64_sin l x y
  | Complex32 -> owl_complex32_sin l x y
  | Complex64 -> owl_complex64_sin l x y
  | _         -> failwith "_owl_sin: unsupported operation"
```

The `_owl_sin` implementation takes four input parameter, the first is the number type `kind`, the second is the total number of elements `l` to apply the `sin` function, the third one `x` is the source ndarray, and the final one `y` is the target ndarray.
This function apply the `sin` function on the first `l` elements from `x` and then put the results in `y`.
Therefore we can simply add a simple layer of wrapper around this function in the `Dense` module:

```text
let sin x =
  let y = copy x in
  _owl_sin (kind x) (numel y) x y;
  y
```

But wait, what are the `owl_float32_sin` and `owl_float64_sin` etc. in `_owl_sin` function? How are they implemented? Let's take a look:

```
external owl_float32_sin
  : int
  -> ('a, 'b) owl_arr
  -> ('a, 'b) owl_arr
  -> unit
  = "float32_sin"
```

OCaml provides mechanism for interfacing with C using the `external` keyword: `external ocaml-function-name : type = c-function-name`.
This defines the value name `ocaml-function-name` as a function with type `type` that executes by calling the given C function `c-function-name`.
Here we already have a C function that is called "float32_sin", and `owl_float32_sin` calls that function.

Now, finally, we venture into the world of C. We first need to include the necessary header files provided by OCaml:

```c
#include <caml/mlvalues.h> // definition of the value type, and conversion macros
#include <caml/alloc.h> //allocation functions to create structured ocaml objects
#include <caml/memory.h> // miscellaneous memory-related functions and macros
#include <caml/fail.h> //functions for raising exceptions
#include <caml/callback.h> // callback from C to ocaml
#include <caml/threads.h> //operations for interfacing in the presence of multiple threads
```

In the C file, the outlines of function `float32_sin` is:

```c
CAMLprim value float32_sin(value vN, value vX, value vY) {
  ...
}
```

To define a C primitive to interface with OCaml, we use the `CAMLprim` macro.
Unlike normal C functions, all the input types and output type are defined as `value` instead of `int`, `void` etc.
It is represents OCaml values and encodes objects of several base types such as integers, strings, floats, etc. as well as OCaml data structures.
The specific type of input will be passed in when the functions is called at runtime.

Now let's look at the content within this function. First, the input is of type `value` and we have to change them into the normal types for further processing.

```c
CAMLparam3(vN, vX, vY);
int N = Long_val(vN);
struct caml_ba_array *X = Caml_ba_array_val(vX);
float *X_data = (float*) X->data;
struct caml_ba_array *Y = Caml_ba_array_val(vY);
float *Y_data = (float *) Y->data;
```

These "value" type parameters or local variables must be processed with one of the CAMLparam macros.
Here we use the `CAMLparam3` macro since there are three parameters.
There are six CAMLparam macros from CAMLparam0 to CAMLparam5, taking zero to five parameters. For more than five parameters, you can first call `CAMLparam5`, and then use one or more `CAMLxparam1` to `CAMLxparam5` functions after that.

The next step we convert the `value` type inputs into normal types.
The `Long_val` macro convert the value into long int type.
Similarly, there are also `Double_val`, `Int32_val` etc.
We convert the Bigarray to a structure to the structure of type `caml_ba_array`. The function `Caml_ba_array_val` returns a pointer to this structure.
Its member `data` is a pointer to the data part of the array.
Besides, the information of ndarray dimension is also included. The member `num_dims` of `caml_ba_array` is the number of dimensions, and `dim[i]` is the i-th dimension.

One more thing to do before the "real" coding.
If the computation is complex, we don't want all the OCaml threads to be stuck. Therefore, we need to call the `caml_release_runtime_system` function to release the master lock and other OCaml resources, so as to allow other threads to run.

```c
caml_release_runtime_system();
```

Finally, we can do the real computation, and now that we have finished converting the input data to the familiar types, the code itself is straight forward;

```c
float *start_x, *stop_x;
float *start_y;

start_x = X_data;
stop_x = start_x + N;
start_y = Y_data;

while (start_x != stop_x) {
    float x = *start_x;
    *start_y = sinf(X);
    start_x += 1;
    start_y += 1;
};
```

That's all, we move the pointers forward and apply the `sinf` function from the C standard library one by one.

As you can expect, when all the computation is finished, we need to end the multiple threading.

```c
caml_acquire_runtime_system();
```

And finally, we need to return the result with `CAMLreturn` macro -- not normal type, but the `value` type. In this function we don't need to return anything, so we use the `Val_unit` macro:

```c
CAMLreturn(Val_unit);
```

That's all for this function. But if we want to return a, say, long int, you can the use `Val_long` to wrap an int type into `value` type.
In the Owl core C code, we normally finish the all the computation and copy the result in-place, and then returns `Val_unit`, as shown in this example.

Now that we finish `float32_sin`, we can copy basically all the code above and implement the rest three functions: `float64_sin`, `complex32_sin`, and `complex64_sin`.
However, this kind of coding practice is apparently not ideal.
Instead, in the core implementation, Owl utilises the macros and templates of C.
In the above implementation, we abstract out the only three special part: function name, math function used, and data type. We assign macro `FUN` to the first one, `MAPFN` to the next, and `NUMBER` to the third. Then the function is written as a template:

```c
CAMLprim value FUN(value vN, value vX, value vY) {
  ...
  NUMBER *X_data = (NUMBER *) X->data;
  ...
  *start_y = (MAPFN(x));
  ...
}
```

This template is defined in the file `owl_ndarray_maths_map.h` file.
In anther stub C file, these macros are defined as:

```c
#define FUN float32_sin
#define NUMBER float
#define MAPFN(X) (sinf(X))
#include "owl_ndarray_maths_map.h"
```

In this way, we can easily extend this template to other data types.
To extend it to complex number, we can use the `_Complex float` and `_Complex double` as number type, and the `csinf` and `csin` for math function function on complex data type.

```c
#define FUN4 complex32_sin
#define NUMBER _Complex float
#define MAPFN(X) (csinf(X))
#include "owl_ndarray_maths_map.h"
```

Once finished the template, we can find that, this template does not only apply to `sin`, but also the other triangular functions, and many more other similar unary math function that accept one input, such as `exp` and `log`, etc.

```c
#define FUN float32_log
#define NUMBER float
#define MAPFN(X) (logf(X))
#include "owl_ndarray_maths_map.h"
```

Of course, the template can become quite complex for other types of function.
But by utilising the template and macros, the core C code of Owl is much simplified.
A brief recap: in the core module we are talking about three files.
The first one is a ocaml file that contains functions like `_owl_sin` that interfaces to C code using `external` keyword.
Then the C implementation is divided into the template file, normally as a `.h` header file, and is named as `*_impl.h`.
The stub that finally utilises these templates to generate functions are put into `*_stub.c` files.

Note that if the input parameters are more than 5, then two primitives should be implemented.
The first `bytecode` function takes two arguments: a pointer to a list of `value` type arguments, and an integer that indicating the number of arguments provided.
The other `native` function takes its arguments directly.
The syntax of using `external` should also be changed to include both functions.

```
external name : type = bytecode-C-function-name native-code-C-function-name
```

For example, in our implementation of convolution we have a pair of functions:

```c
CAMLprim value FUN_NATIVE (spatial) (
  value vInput_ptr, value vKernel_ptr, value vOutput_ptr,
  value vBatches, value vInput_cols, value vInput_rows, value vIn_channel,
  value vKernel_cols, value vKernel_rows,
  value vOutput_cols, value vOutput_rows, value vOut_channel,
  value vRow_stride,  value vCol_stride,
  value vPadding, value vRow_in_stride, value vCol_in_stride
) {
  ....
}

CAMLprim value FUN_BYTE (spatial) (value * argv, int argn) {
  return FUN_NATIVE (spatial) (
    argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
    argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14],
    argv[15], argv[16]
  );
}
```

In the stub we define the function name macros:

```
#define FUN_NATIVE(dim) stub_float32_ndarray_conv ## _ ## dim  ## _ ## native
#define FUN_BYTE(dim) stub_float32_ndarray_conv ## _ ## dim  ## _ ## bytecode
```

And therefore in the OCaml interfacing code we interface to C code with:

```
external owl_float32_ndarray_conv_spatial
  :  ('a, 'b) owl_arr -> ('a, 'b) owl_arr -> ('a, 'b) owl_arr -> int -> int -> int -> int -> int -> int -> int -> int -> int -> int -> int -> int -> int -> int -> unit
  = "stub_float32_ndarray_conv_spatial_bytecode" "stub_float32_ndarray_conv_spatial_native"
```

More details of interfacing to C code OCaml can be found in the OCaml [documentation](https://caml.inria.fr/pub/docs/manual-ocaml/intfc.html).
Another approach is to use the Foreign Function Interface, as explained [here](https://dev.realworldocaml.org/foreign-function-interface.html).

## Optimisation Techniques

There is a big room for optimising the C code.
We are trying to push the performance forward with multiple techniques.
We mainly use the multiprocessing with OpenMP and parallel computing using SIMD intrinsics when possible.
In this section, We choose some representative operations to demonstrate our optimisation of the core ndarray operations.
Besides them, we have also applied other basic C code optimisation techniques such as avoiding redundant computation in for-loop.

To show how these optimisation works, we compare performance of an operation, in different numerical libraries: Owl, NumPy, Julia, and Eigen.
The purpose is two-fold: first, to bring insight into the low-level structure design; second, to demonstrate the possible optimisations in implementing these operations.

In the performance measurements, we use multiple input sizes, and observe the execution time and memory usage.
The experiments are conducted on both a laptop (Thinkpad T460s, Core i5 CPU) and a Raspberry Pi (rPi) 3B model. They represent different CPU architectures and computation power.


### Map Operations

The `map` operations are a family of operations that accept ndarray as input, and apply a function on all the elements in the ndarray.
Again, we use the trigonometric `sin` operation as a representative map arithmetic operation in this section. It requires heavy computation.
In the implementation, it directly calls the low-level C functions via a single template.
The performance of such operation is mainly decided by the linked low level library.
Map function can also benefit from parallel execution on the multi-core CPU, such as using OpenMP.

OpenMP is one of the most common parallel programming models in use today.
Unlike pthread, the low-level API to work with threads, OpenMP operate at a high-level and is much more portable.
It uses a "Fork–join model" where the master thread spawns other threads as necessary, as shown in [@fig:core-opt:fork-join].

![Fork-join model used by OpenMP](images/core-opt/fork_join.png "fork"){width=90%s #fig:core-opt:fork-join}

In the C code we can create threads with the `omp parallel` pragma. For example, to create a four-thread parallel region, we can use:

```c
omp_set_num_threads(4);
#pragma omp parallel
{
  /* parallel region */
  ...
}
```

The task in the region is assigned to the four threads and get executed in parallel.
The most frequently used pattern in our core code is to move a for-loop into the parallel region. Each thread is assigned part of the whole input array, and apply the math computation on each element in parallel.
Taking the implementation code from previous chapter, we only need to add a single line of OpenMP compiler directive:

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    NUMBER x = *(start_x + i);
    *(start_y + i) = (MAPFN(x));
}
```

The for-loop is included in parallel region, and the `N` elements are scheduled to each thread.
In the code we use the `static` scheduling, which means scheduling is done at compile time.
It works best when the each iterations take roughly equal time.
Otherwise we can consider using the "dynamic" scheduling that happens at runtime, or "auto" scheduling when the runtime can learn from previous executions of the same loop.

That's all. We apply it simple techniques to the templates of many map function.
Note that OpenMP comes with a certain overhead.
What if we don't want to use the OpenMP version?


Our solution is to provide two sets of C templates and switch depending on configuration flags.
For example, for the map functions, we have the normal template file "owl_ndarray_maths_map.h", and then a similar one "owl_ndarray_maths_map_omp.h" where each template uses the OpenMP derivative.
We can then switch between these two implementation by simply define or un-define the `_OPENMP` macro, which can easily be done in the configuration file.

```c
#ifdef _OPENMP
#define OWL_NDARRAY_MATHS_MAP  "owl_ndarray_maths_map_omp.h"
#else
#define OWL_NDARRAY_MATHS_MAP  "owl_ndarray_maths_map.h"
#endif
```

OpenMP is surely not only utilised in the map function. We also implement OpenMP-enhanced templates for the fold operations, comparison operations, slicing, and matrix swap, etc.

Another optimisation is to remove the memory copy phase by applying mutable operations.
A mutable operation does not create new memory space before calculation, but instead utilise existing memory space of input
ndarray.
This kind of operations does not involve the C code, but rather in the ndarray module. For example:

```text
let sin_ ?out x =
  let out =
    match out with
    | Some o -> o
    | None   -> x
  in
  _owl_sin (kind x) (numel x) x out
```

The `_owl_sin` function is still the same, but in this mutable function `sin_` we choose the destination array and source array to be the same.
Therefore, the existing memory is utilised and we don't have to copy the previous content to a new memory space before the calculation.

Both vectorisation and parallelisation techniques can be utilised to improve its performance.
Computation-intensive operations such as sine in a for-loop can
be vectorised using SIMD instructions.
The computation performance can be boosted by executing single instruction on multiple data in the input ndarray.
In that way, with only one core, 4 or 8 elements in the for-loop can be processed at the same time.
However, unlike OpenMP, we cannot say "apply sine operation on these 4 elements".
The SIMD intrinsics, such as the ones provided by [Intel](https://software.intel.com/sites/landingpage/IntrinsicsGuide/), only support basic operations such as copy, add, etc.
To implement functions such as sine and exponential is non-trivial task.
One simple implementation using the SSE instruction set is [here](http://gruntthepeon.free.fr/ssemath/).
More and more libraries such as the Intel MKL provide SIMD version of these basic math operations instead of that provided in the standard C library.

Let's look at how our implementation of the `sin` operation performs compared with the other libraries.
To measure performance, we compare the sine operation in Owl, NumPy, Julia, and C. The compiling flags in C and Owl are set to the same level 3 optimisation. The input is a vector of single-precision float numbers.
We increase the input size from 100,000 to 5,000,000 gradually.
The comparison results are shown in [@fig:core-opt:op_eval_sin].

![Sin operation performance.](images/core-opt/opeval_sin.png){width=100% #fig:core-opt:op_eval_sin}

It can be seen that the execution time in Owl grows linearly with input size, and very similar to that of C library. Julia has large deviation, but it performs fastest on rPi, even faster than C.
It is because of Julia utilises NEON, the SIMD architecture extension on ARM.
In some cases, NumPy can be compiled with MKL library. The MKL Vector Math functions provide highly optimised routines for trigonometric operations.
In this evaluation we use NumPy library that is not compiled with MKL, and it performs close to Owl and C, with slightly larger deviation.

### Convolution Operations

The convolution operations take up the majority of computation involved in deep neural network, and therefore is the main target of our core optimisation.
We have seen how the convolution works and the Neural Network chapter.
In this section, we would like to go a bit deeper and talk about its implementation.
Starting with ta short recap of how the convolution works.

A convolution operation takes two ndarrays as input: image ($I$) and kernel ($F$). In a 2-dimensional convolution, both ndarrays are of four dimensions.
The image ndarray has $B$ batches, each image has size $H\times W$, and has $IC$ channels.
The kernel ndarray has $R$ rows, $C$ columns, the same input channel $IC$, and output channel $K$. The convolution can then be expressed as:

$$CONV_{b,h,w,k} = \sum_{ic=1}^{IC}\sum_{r=1}^{R}\sum_{c=1}^{C}I_{b,h+r,w+c,ic}F_{r,c,ic,k}.$$ {#eq:core-opt:conv}

A naive convolution algorithm is to implement [@eq:core-opt:conv] with nested for-loops. It is easy to see that this approach does not benefit from any parallelisation, and thus not suitable for production code.

The next version of implementation uses the `im2col` method. A `im2col`-based convolution transforms the input ndarray into a matrix with redundancy.
This process can be explained clearly with [@fig:core-opt:im2col] ([src](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/making_faster.html)).

![Basic implementation algorithm of convolution: im2col](images/core-opt/im2col.png "im2col"){width=95% #fig:core-opt:im2col}

You can see that originally at each channel the filter moves along the input and apply element-wise multiplication.
Instead, the chosen input elements can be flatten into one single column, and so is the kernel.
One single dot multiplication is the same as previous steps.
Moving filter to a new position means creating a new column and a new column.
Therefore, the convolution can be implemented as the dot product of two matrices, which are that flattened kernel matrix and the new input matrices that we have created column by column.
The advantage of this approach is that it is easy to understand and benefit from the general matrix multiply operation provided by highly optimised linear algebra packages such as OpenBLAS.

However, this algorithm requires generating a large temporary
intermediate matrix.
It's row number is `kernel_col * kernel_rowl * input_channel`, and its column number is `output_col * output_row * batches`.
Even for a mediocre size convolution layer, the size of this intermediate input matrices is not small, not to mention for larger input/kernel sizes and with tens and hundreds of convolution layers together in a neural network.
The memory usage can easily reach Gigabytes in DNN applications.


There are several methods proposed to mitigate this problem.
If you look closely at the intermediate matrix, you will find that it contains a lot of redundant information: the columns overlap too much.
Algorithms such as [Memory-efficient Convolution](https://arxiv.org/abs/1706.06873) aims to reduce the size of this intermediate matrix based on not generating the whole intermediate matrix, but only part of it to efficiently utilise the overlapped content.
But even so, it may still fail with very large input or kernel sizes.

The implementation in Eigen provides another solution.
Eigen a C++ template library for linear algebra. Think of it as an alternative to BLAS etc.
Based on its core functionalities, it implements convolution operations as a unsupported module.
It was used in TensorFlow for its CPU convolution implementation.
The convolution operation is first implemented in Owl by interfacing to the Eigen library.
We later tun to C implementation since interfacing to this C++ library proves to be problematic and leads to a lot of installation issues.
In its implementation, Eigen solves this memory usage problem according to the method proposed in [@goto2008anatomy].

It still generally follows the previous matrix multiplication approach, but instead of generating the whole intermediate matrix, it cuts the input and kernel matrices into small blocks one at a time so that the memory usage is limited no matter how large the input and kernel are.

Specifically, the block size can be chosen in a way to fit into the L1/L2 cache of CPU to do high-performance computation.
Multiplication of two matrices can be divided into multiplication of small blocks.
The L1/L2/L3 cache sizes are retrieved using the `CPUID` instruction on x86 architecture, and predefined const value for non-x86 architectures.

To further improve the performance, we use the SIMD intrinsics during building those small temporary matrices from input ndarray.
We focus on the main operation that copy input ndarray into the new input matrix: loading data, storing data, and adding two vectors.
Currently we only support the most recent Advanced Vector Extensions (AVX) x86 extension on Intel and AMD architectures.
We detect if the AVX extension is supported by detecting if the `__AVX__` is detected in GCC.
If so, we include the header filer `immintrin.h`.

The extends commands to 256 bits, so that we can process eight float or 4 double elements at the same time.
In the code we mainly use the `_mm256_store_ps`, `_mm256_load_ps`, and `_mm256_add_ps` intrinsics, for storing 256-bits variable from source into memory, loading 256-bits to memory, or adding two 256-bits into destination variable.
Note that the load and store intrinsics require the source or destination address to be aligned on a 32-byte boundary.
If not, we need to use the unaligned version `_mm256_storeu_ps` and `_mm256_loadu_ps`, with degraded performance.

To maximise the performance of caching, we need to make the memory access as concecutive as possible.
Depending on the input channel is divisible by the supported data length of SIMD (e.g. 8 float numbers for AVX), we provide two set of implementations for filling the temporary blocks.
If input channel is divisible by data length, the input matrix can always be loaded consecutively at a step of data length with the AVX intrinsics, otherwise we have to build the temporary matrix blocks with less AVX intrinsics, on only part of the matrix, and then take care of the edge cases.

We have described the implementation method we use to optimise the convolution operations.
We recommend reading the full code in `owl_ndarray_conv_impl.h` file for more details.
One more optimisation is that, we have shown the `im2col` method and its disadvantage with memory usage. Howver, it is still straightforward and fast with small input sizes.
Therefore, we set a pre-defined threshold to decide if we use the `im2col` implementation or the one that inspired by Eigen.

As you know, convolution operations consists of three types: `Conv`, `ConvBackwardKernel`, `ConvBackwardInput`.
The `Conv` operation calculates the output given input image and kernel. Similarly, `ConvBackwardKernel` calculates the kernel given the input and output ndarrays, and `ConvBackwardInput` gets input ndarray from kernel and output.
The last two are mainly used in the backpropagation phase in training a DNN, but all three operations share a similar
calculation algorithm.
The backward convs are actually also implemented as matrix multiplication.
For `ConvBackwardKernel`, it first reshape the output ndarray as matrix, and multiply it with the intermediate input matrix.
Similarly, in ``ConvBackwardInput`, we need to first multiply the kernel and output matrix to get the intermeidate input matrix, and then re-construct the input ndarray based it.

These implementation can then be easily extended to the three dimension and one dimension cases.
Besides, the transpose convolutions and diluted convolutions are only variate of normal convolutin and the code only needs to be slightly changed.
At the OCaml level, mutable convolution operations are also provided, so as to further improve performance by utilising existing memory space.

To measure the performance of my convolution implementation, we compare the three convolution operations on both the labtop and rPi as described before.
We use two settings: fixed input size with varying kernel size; and fixed kernel size with varying input size. The Owl code is interfaced to existing implementation and Eigen library. The results are in
[@fig:core-opt:op_eval_eigen_conv_tp] and [@fig:core-opt:op_eval_eigen_conv_rpi].

![Measure the performance of Conv2D operation on Owl and Eigen on laptop](images/core-opt/eigen_tp_conv2d.png){width=90% #fig:core-opt:op_eval_eigen_conv_tp}

![Measure the performance of Conv2D Backward kernel operation on Owl and Eigen on rPi](images/core-opt/eigen_rpi_conv2d_bk.png){width=90% #fig:core-opt:op_eval_eigen_conv_rpi}

![Measure the memory usage of Conv2D Backward Input operation on Owl and Eigen](images/core-opt/eigen_rpi_conv2d_bi.png){width=90% #fig:core-opt:op_eval_eigen_conv_mem}

The results show that, our `Conv2D` implementation is as efficient as that in Eigen, and the `Conv2DBackwardKernel` operation is faster on the rPi.
In [@fig:core-opt:op_eval_eigen_conv_mem] it is shown that our proposed implementation of `Conv2DBackwardInput` operation uses less memory than Eigen.

### Reduction Operations

We have also improved the algorithm of certain operations.
For the reduction operation and the repeat operations, we update the algorithms to reduce the memory usage caused by intermediate results.

Reduction operations such as `sum` and `max` accumulate values in a ndarray along certain axes by certain functions. For example, `sum` is used for implementing BatchNormalisation neuron, which is a frequently used neuron in DNN.

A naive implementation of multi-axes `sum` operation is to repeat sum operation along one axis for each axis specified.
However, it creates extra temporary intermediate results. In applications such as DNN, the inefficiency of reduction operation becomes a memory and performance bottleneck.
The implementation is shown below.


**TODO**: Explain the code instead of just showing some code.

```c
int iy = 0;
int cnt = 0;

for (int ix = 0; ix < N;) {
  for (int k = 0; k < innersize; k++) {
    ACCFN(y[iy+k], x[ix+k]);
  }

  ix += innersize;
  cnt++;

  if (cnt == loopsize) {
    cnt = 0;
    int residual;
    int iterindex = ix;
    int pre_iteridx = ix;

    for (int i = ndim - 1; i >= 0; i--) {
      iterindex /= x_shape[i];
      residual = pre_iteridx - iterindex * x_shape[i];
      iy += residual * strides[i];
      pre_iteridx = iterindex;
    }
  }
}
```

The `ACCFN(y, x)` function in this template is the accumulation function that is unique to each operation.
For `sum`, it adds `y` and `x` and accumulate the value to `y`.
In this algorithm, the elements in original ndarray `x` and the target reduced ndarray `y` are iterated one-by-one, but at different steps, indicating by `iy` and `ix`.

One optimisation step before this algorithm is to combine adjacent axes.
For example, if an ndarray of shape $[2,3,4,5]$ is to be reduced along the second and third axis, then it can be simplified to reducing an ndarray of shape $[2,12,5]$.

![Sum reduction operation on laptop](images/core-opt/opeval_tp_sum_reduce_mem_00.png){width=60% #fig:core-opt:opeval_sumreduce}

Since it involves multiple axes, to evaluate the reduction operation, we use a four-dimensional ndarray of float numbers as input.
All four dimensions are of the same length. We measure the peak memory usage with increasing length, each for axis equals to 0, 1, and both 0 and 2 dimension.
The evaluation result compared with NumPy and Julia is shown in [@fig:core-opt:opeval_sumreduce].


### Repeat Operations

`Repeat` is another operation that is frequently used in DNN. It is used for implementing the Upsampling and BatchNormalisation
neurons.
The `repeat` operation repeats elements of an ndarray along each axis for specified times. It consists of inner repeat and outer
repeat (or `tile`).
The former repeats elements of an input ndarray, while the later constructs an ndarray by repeating the whole input ndarray by specified number of times along each axis.

Similar to the reduction functions, a multi-axes repeat function can hardly achieve ideal performance by simply using existing operation for multiple times. We implement multi-axes repeat in Owl and it outperforms NumPy and Julia.

The optimisation we use in the algorithm follows two patterns.
The first is to provide multiple implementations for different inputs.
For example, if only one axis is used, then a specific implementation for that case would be much faster than a general solution.
The second is to reduce intermediate results. Similar to the reduction operations, a multiple-axes `repeat` could be implemented by multiple single axis operation, but it would lead to extra memory usage and much slower execution speed.

The core code of my proposed repeat algorithm is shown below.
Here we define `HD` to be the highest non-one-repeat dimension, copy the HD dimension from source ndarray to target ndarray, and then copy the lower dimensions within target ndarray.


**TODO**: Explain the code instead of just showing some code.

```c
for (int i = 0; i < num_hd; ++i) {
  int ofsy_sub = ofsy;
  if (block_sz == 1) {
    COPYFUN(slice_x[HD], x, ofsx, 1, y, ofsy, 1);
  } else {
    for (int j = 0; j < slice_x[HD]; ++j) {
      COPYFUN(block_sz, x, ofsx + j, 0, y, ofsy_sub, 1);
      ofsy_sub += block_sz;
    }
  }

  /* Increase index */
  ofsx += slice_x[HD];
  ofsy += stride_y[HD - 1] * reps[HD - 1];
  for (int j = HD - 1; j > 0; --j) {
    int c = counter[j];
    if (c + 1 == block_num[j]) {
      ofsy += stride_y[j - 1] * (reps[j - 1] - 1);
    }
    counter[j] = (c + 1 == block_num[j] ? 0 : c + 1);
  }
}
```

The evaluation of `repeat` is similar to that of reduction operations. We use a four-dimensional ndarray of float numbers as input. All four dimensions are of the same length. We measure the speed for increasing length, the repetition times is set to 2 on all dimensions.

![Repeat operation speed](images/core-opt/opeval_repeat.png){width=95% #fig:core-opt:opeval_repeat}

![Repeat operation memory usage comparison](images/core-opt/opeval_tp_repeat_mem_00.png){width=80% #fig:core-opt:opeval_tp_repeat_mem_00}

The evaluation results compared with NumPy and Julia are shown in [@fig:core-opt:opeval_repeat].
We also measure the peak memory usage in[@fig:core-opt:opeval_tp_repeat_mem_00].
As can be seen, my repeat operation achieves about half of that in NumPy with regard to both execution speed and memory usage.
The outer repeat operation in NumPy is implemented using the single axis version, and thus is less efficient.
The repeat operation in Julia is much slower. One reason is that `repeat` is not a computation-intensive operation, so the optimisation techniques such as static compilation and vectorisation are of less importance than algorithm design.

## Summary

## References
