# Testing Framework

Every proper software requires testing, and so is Owl. All too often we have found that testing can help use to find potential errors we had not anticipated during coding.

In this chapter, we introduce the philosophy of testing in Owl, the tool we use for conducting the unit test, and examples to demonstrate how to do that in Owl.
Issues such as using functors in test, and other things to notice in writing test code for Owl etc. are also discussed in this chapter.

## Unit Test

There are multiple ways to perform tests on your code. One common way is to use assertion or catching/raising errors in the code.
These kinds of tests are useful, but embedded in the code itself, while we need separate test modules that check the implementation of functions against expected behaviours.

In Owl, we apply *unit test* to make sure the correctness of numerical routines as much as possible.
Unit test is a software test method that check the behaviour of individual units in the code.
In our case the "unit" often means a single numerical function.

There is an approach of software development that is called Test Driven Development, where you write test code even before you implement the function to be tested itself.
Though we don't enforce such approach, there are certain testing philosophy we follow during the development of Owl.
For example, we generally don't trust code that is not tested, so in a PR it is always a good practice to accompany your implementation with unit test in the `test/` directory in the source code.
Besides, try to keep the function short and simple, so that a test case can focus on a certain aspect.

We use the `alcotest` framework for testing in Owl. `alcotest` is a lightweight test framework with simple interfaces. It exposes a simple `TESTABLE` module type, a `check` function to assert test predicates and a `run` function to perform a list of `unit -> unit` test callbacks.

## Example

Let's look at an example of using `alcotest` in Owl. Suppose you have implemented some functions in the linear algebra module, including the functions such as rank, determinant, inversion, etc., and try to test if before make a PR.
The testing code should look something like this:

```ocaml
open Owl
open Alcotest

module M = Owl.Linalg.D

(* Section #1 *)

let approx_equal a b =
  let eps = 1e-6 in
  Stdlib.(abs_float (a -. b) < eps)

let x0 = Mat.sequential ~a:1. 1 6

(* Section #2 *)

module To_test = struct
  let rank () =
    let x = Mat.sequential 4 4 in
    M.rank x = 2

  let det () =
    let x = Mat.hadamard 4 in
    M.det x = 16.

  let vecnorm_01 () =
    let a = M.vecnorm ~p:1. x0 in
    approx_equal a 21.

  let vecnorm_02 () =
    let a = M.vecnorm ~p:2. x0 in
    approx_equal a 9.539392014169456

  let is_triu_1 () =
    let x = Mat.of_array [| 1.; 2.; 3.; 0.; 5.; 6.; 0.; 0.; 9. |] 3 3 in
    M.is_triu x = true

  let mpow () =
    let x = Mat.uniform 4 4 in
    let y = M.mpow x 3. in
    let z = Mat.(dot x (dot x x)) in
    approx_equal Mat.(y - z |> sum') 0.

end

(* Section #3 *)

let rank () = Alcotest.(check bool) "rank" true (To_test.rank ())

let det () = Alcotest.(check bool) "det" true (To_test.det ())

let vecnorm_01 () = Alcotest.(check bool) "vecnorm_01" true (To_test.vecnorm_01 ())

let vecnorm_02 () = Alcotest.(check bool) "vecnorm_02" true (To_test.vecnorm_02 ())

let is_triu_1 () = Alcotest.(check bool) "is_triu_1" true (To_test.is_triu_1 ())

let mpow () = Alcotest.(check bool) "mpow" true (To_test.mpow ())

(* Section #4 *)

let test_set =
  [ "rank", `Slow, rank
  ; "det", `Slow, det
  ; "vecnorm_01", `Slow, vecnorm_01
  ; "vecnorm_02", `Slow, vecnorm_02
  ; "is_triu_1", `Slow, is_triu_1
  ; "mpow", `Slow, mpow ]

```

There are generally four sections in a test file.
In the first section, you specify the required precision and some predefined input data.
Here we use `1e-6` as precision threshold. Two ndarrays are deemed the same if the sum of their difference is less than `1e-6`, as shown in `mpow`.
The predefined input data can also be defined in each test case, as in `is_triu_1`.

In the second section, a test module need to be built, which contains a series of test functions.
The most common test function used in Owl has the type `unit -> bool`.
The idea is that each test function compare a certain aspect of a function with expected results.
If there are multiple test cases for the same function, such the case in `vecnorm`, we tend to build different test cases instead of using one large test function to include all the cases.
The common pattern of these function can be summarised as:

```
let test_func () =
    let expected = expected_value in
    let result = func args in
    assert (expected = result)
```

It is important to understand that the equal sign does not necessarily mean the two values have to be the same; in fact, for the float-point number is involved, which is quite often the case, we only need the two values to be approximately equal enough.
If that's case, you need to pay attention to which precision you are using, double or float. The same threshold might be enough for float number, but could still be a large error for double precision computation.

In the third section wraps these functions with `alcotest` by stating the expected output. Here we expect all the test functions to return `true`, though `alcotest` does support testing returning a lot of other types such as string, int, etc. Please refer to the [source file](https://github.com/mirage/alcotest/blob/master/src/alcotest/alcotest.mli) for more detail.

In the final section, we take functions from section 3 and put them into a list of test set. The test set specify the name and mode of the test.
The test mode is either `Quick` or `Slow`.
Quick tests are ran on any invocations of the test suite.
Slow tests are for stress tests that are ran only on occasion, typically before a release or after a major change.

After this step, the whole file is named `unit_linalg.ml` and put under the `test/` directory, as with all other unit test files.
Now the only thing left is to add it in the `test_runner.ml`:

```
let () =
  Alcotest.run "Owl"
    [ "linear algebra", Unit_linalg.test_set;
    ...
    ]
```

That's all. Now you can try `make test` and check if the functions are implemented well:

![All tests passes](images/testing/example_00.png){#fig:testing:example_00}

What if one of the test functions does not pass? Let's intentionally make a wrong test, such as asserting the matrix in the `rank` test is 1 instead of the correct answer 2, and run the test again:

![Error in tests](images/testing/example_01.png){#fig:testing:example_01}


## What Could Go Wrong

> Who's Watching the Watchers?

Beware that the test code itself is still code, and thus can also be wrong. We need to be careful in implementing the testing code.
There are certain cases that you may want to check.

### Corner Cases

Corner cases involves situations that occur outside of normal operating parameters.
That is obvious in the testing of convolution operations.

As the core operation in deep neural networks, convolution is complex: it contains input, kernel, strides, padding, etc. as parameters.
Therefore, special cases such as `1x1` kernel, strides of different height and width etc. are tested in various combinations, sometimes with different input data.

```
module To_test_conv2d_back_input = struct
    (* conv2D1x1Kernel *)
    let fun00 () = ...

    (* conv2D1x2KernelStride3Width5 *)
    let fun01 () = ...

    (* conv2D1x2KernelStride3Width6 *)
    let fun02 () = ...

    (* conv2D1x2KernelStride3Width7 *)
    let fun03 () = ...

    (* conv2D2x2KernelC1Same *)
    let fun04 () = ...

    ...

    (* conv2D2x2KernelStride2Same *)
    let fun09 () = ...
```

### Test Coverage

Another issue is the test coverage. It means the percentage of code for which an associated test has exist.
Though we don't seek a strict 100% coverage for now, a wider test coverage is always a good idea.

For example, in our implementation of the `repeat` operation, depending on whether the given axes contains one or multiple integers, the implementation changes. Therefore in the test functions it is crucial to cover both cases.


## Use Functor

Note that you can still benefit from all the powerful features OCaml such as functor.
For example, in testing the convolution operation, we hope to the implementation of both that in the core library (which implemented in C), and that in the base library (in pure OCaml).
Apparently there is no need to write the same unit test code twice for these two set of implementation.

To solve that problem, we have a test file `unit_conv2d_genericl.ml` that has a large module that contains all the previous four sections:

```
module Make (N : Ndarray_Algodiff with type elt = float) = struct
    (* Section #1 - #4 *)
    ...
end
```

And in the specific testing file for core implementation `unit_conv2d.ml`, it simply contains one line of code:

```
include Unit_conv2d_generic.Make (Owl_algodiff_primal_ops.S)
```

Or in the test file for base library `unit_base_conv2d.ml`:

```
include Unit_conv2d_generic.Make (Owl_base_algodiff_primal_ops.S)
```

## Summary
