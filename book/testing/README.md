# Testing Framework

Every proper software requires testing, and so is Owl. All too often we have found that testing can help use to find potential errors we had not anticipated during coding. 

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

let inv () = Alcotest.(check bool) "inv" true (To_test.inv ())

let vecnorm_01 () = Alcotest.(check bool) "vecnorm_01" true (To_test.vecnorm_01 ())

let vecnorm_02 () = Alcotest.(check bool) "vecnorm_02" true (To_test.vecnorm_02 ())

let is_triu_1 () = Alcotest.(check bool) "is_triu_1" true (To_test.is_triu_1 ())

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

![](images/testing/example_00.png)

What if one of the test functions does not pass? Let's intentionally make a wrong test, such as asserting the matrix in the `rank` test is 1 instead of the correct answer 2, and run the test again:

![](images/testing/example_01.png)


## Use functor

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