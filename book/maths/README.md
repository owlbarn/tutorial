# Mathematical Functions

Though the central data structure in Owl is `ndarray`, we provide support for scalar math functions. 

## Basic Functions

**GUIDE**: categorise unary functions (basic, triangular, log, etc. ), each with some introduction of history and application etc.; not just how this function is used. 

### Binary Functions

Binary functions `f x y` has the signature in the form of `val f : float -> float -> float`

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`add`         `x + y`

`sub`         `x - y`

`mul`         `x * y`

`div`         `x / y`

`fmod`        `x % y`

`atan2`       returns $\arctan(y/x)$, accounting for the sign of the 
              arguments; this is the angle to the vector $(x, y)$ counting from the x-axis.
------------  -------------------------------------------------------
: Binary math functions {#tbl:maths:binary}

### Basic Unary Math Functions 

Unary functions with signature `val f : float -> float`

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`abs`         `|x|`

`neg`         `-x`

`reci`        `1/x`

`floor`       the largest integer that is smaller than `x`

`ceil`        the smallest integer that is larger than `x`

`round`       rounds `x` towards the bigger integer when on the fence

`trunc`       integer part of `x`

`sqr`         $x^2$

`sqrt`        $\sqrt{x}$

`pow`         $x^y$

`hypot`       $\sqrt{x^2 + y^2}$
------------  -------------------------------------------------------
: Basic unary math functions {#tbl:maths:basic_unary}


### Exponential and Logarithmic Functions

Introduction to exponential and logarithm functions 

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`exp`         exponential $e^x$

`exp2`        $2^x$

`exp10`       $10^x$

`expm1`       returns $\exp(x) - 1$ but more accurate for $x \sim 0$ 

`log`         $log_e~x$

`log2`        $log_2~x$

`log10`       $log_10~x$

`logn`        $log_n~x$

`log1p`       Inverse of `expm1`

`logabs`      $\log(|x|)$

`xlogy`       $x \log(y)$

`xlog1py`     $x \log(y+1)$

`logit`       $\log(p/(1-p))$

`expit`       $1/(1+\exp(-x))$

`log1mexp`    $log(1-exp(x))$

`log1pexp`    $log(1+exp(x))$
------------  -------------------------------------------------------
: Exponential and logarithmic math functions {#tbl:maths:explog}

### Triangular Functions

Introduction of triangular functions

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`sin`         $\sin(x)$

`cos`         $\cos(x)$

`tan`         $\tan(x)$

`cot`         $1/\tan(x)$

`sec`         $1/\cos(x)$

`csc`         $1/\sin(x)$

`asin`        $\arcsin(x)$

`acos`        $\arcsin(x)$

`atan`        $\arctan(x)$

`acot`        Inverse function of `cot`

`asec`        Inverse function of `sec`

`acsc`        Inverse function of `csc`

`sinh`        $\sinh(x)$

`cosh`        $\cosh(x)$

`tanh`        $\tanh(x)$

`coth`        $\coth(x)$

`sech`        $1/\cosh(x)$

`csch`        $1/\sinh(x)$

`asinh`       Inverse function of `sinh`

`acosh`       Inverse function of `cosh`

`atanh`       Inverse function of `tanh`

`acoth`       Inverse function of `coth`

`asech`       Inverse function of `sech`

`acsch`       Inverse function of `csch`

`sinc`        returns $\sin(x)/x$ and 1 for $x=0$

`logsinh`     returns $\log(\sinh(x))$ but handles large $|x|$

`logcosh`     returns $\log(\cosh(x))$ but handles large $|x|$

`sindg`       Sine of angle given in degrees

`cosdg`       Cosine of the angle given in degrees

`tandg`       Tangent of angle given in degrees

`cotdg`       Cotangent of the angle given in degrees
------------  -------------------------------------------------------
: Triangular math functions {#tbl:maths:triangular}

### Other Unary Math Functions

```
val sigmoid : float -> float
(** ``sigmoid x`` returns the logistic sigmoid function
:math:`1 / (1 + \exp(-x))`. *)

val signum : float -> float
(** ``signum x`` returns the sign of :math:`x`: -1, 0 or 1. *)

val softsign : float -> float
(** Smoothed sign function. *)

val softplus : float -> float
(** ``softplus x`` returns :math:`\log(1 + \exp(x))`. *)

val relu : float -> float
(** ``relu x`` returns :math:`\max(0, x)`. *)
```

## Special Functions

The definition of numerous special functions of mathematical physics. 
It interfaces to the [Cephes Mathematical Functions Library](http://www.netlib.org/cephes/).

### Airy Functions

In the physical sciences, the Airy function (or Airy function of the first kind) Ai(x) is a special function named after the British astronomer George Biddell Airy. (COPY)
It is the solution of differential equation:

$$y''(x) = xy(x).$$

This differential equation has two linearly independent solutions `Ai` and `Bi`. 
Owl provides the `airy` to do that:

```
val airy : float -> float * float * float * float
```

The four returned numbers are `Ai`, its derivative `Ai'`, `Bi`, and its derivative `Bi'`.
Let's look at an example. 

```ocaml
let x = Mat.linspace (-15.) 5. 200

let y0 = Mat.map (fun x -> 
	let ai, _, _, _ = Maths.airy x in ai
) x 

let y1 = Mat.map (fun x -> 
	let _, _, bi, _ = Maths.airy x in bi
) x 

let _ = 
  let h = Plot.create "special_airy.png" in
  Plot.(plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2. ] x y0);
  Plot.(plot ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2. ] x y1);
  Plot.(set_yrange h (-0.5) 1.);
  Plot.(legend_on h ~position:SouthEast [|"Ai"; "Bi"|]);
  Plot.output h
```

![Examples of the two solutions of an Airy equation](images/maths/example_airy.png "airy"){width=75% #fig:algodiff:airy}

APPLICATION description

### Bessel Functions 

Bessel functions, first defined by the mathematician Daniel Bernoulli and then generalized by Friedrich Bessel, are canonical solutions y(x) of Bessel's differential equation:

$$x^2y''+xy'+(x^2 - \alpha^2)y = 0.$$

The complex number $\alpha$ is called the *order* of the bessel function.

The Bessel functions can be divided into two "kinds". 
Bessel functions of the first kind $J$ are solutions of Bessel's differential equation that are finite at $x=0$ integer or positive order and diverge as $x$ approaches zero for negative non-integer order.
Bessel functions of the second kind are solutions of the Bessel differential equation that have a singularity at $x=0$ and are multivalued. (COPY ALERT)

A special case is when $x$ is purely imaginary. In this case, the solutions to the Bessel equation are called the *modified Bessel functions*. These modified Bessel functions can also be categorised as first kind and second kind. 

Based on these category, Owl provides these functions. 

-------- ------------------------- ---------------------------------------------
Function Interface                 Explanation  
-------- ------------------------- ---------------------------------------------
`j0`     `float -> float`          Bessel function of the first kind of order 0

`j1`     `float -> float`          Bessel function of the first kind of order 1

`jv`     `float -> float -> float` Bessel function of the first kind of real order

`y0`     `float -> float`          Bessel function of the second kind of order 0

`y1`     `float -> float`          Bessel function of the second kind of order 1

`yv`     `float -> float -> float` Bessel function of the second kind of real order

`yn`     `int -> float -> float`   Bessel function of the second kind of integer order

`i0`     `float -> float`          Modified Bessel function of order 0

`i1`     `float -> float`          Modified Bessel function of order 1

`iv`     `float -> float -> float` Modified Bessel function of real order

`i0e`    `float -> float`          Exponentially scaled modified Bessel function of order 0

`i1e`    `float -> float`          Exponentially scaled modified Bessel function of order 1

`k0`     `float -> float`          Modified Bessel function of the second kind of order 0

`k1`     `float -> float`          Modified Bessel function of the second kind of order 1

`k0e`    `float -> float`          Exponentially scaled modified Bessel function of the second kind of order 0

`k1e`    `float -> float`          Exponentially scaled modified Bessel function of the second kind of order 1
-------- ------------------------- ---------------------------------------------
: Bessel functions {#tbl:maths:bessel}

Let's look at one example.

```ocaml
let x = Mat.linspace (0.) 20. 200

let y0 = Mat.map Maths.j0 x

let y1 = Mat.map Maths.j1 x

let y2 = Mat.map (Maths.jv 2.) x

let _ =
  let h = Plot.create "example_bessel.png" in
  Plot.(plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2. ] x y0);
  Plot.(plot ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2. ] x y1);
  Plot.(plot ~h ~spec:[ RGB (244, 180,  0); LineStyle 3; LineWidth 2. ] x y2);
  Plot.(legend_on h ~position:NorthEast [|"j0"; "j1"; "j2"|]);
  Plot.output h
```

![Examples of Bessel function of the first kind, with different order](images/maths/example_bessel.png "bessel"){width=75% #fig:algodiff:bessel}

(More examples can be added if we want to expand)

Bessel's equation arises when finding separable solutions to Laplace's equation and the Helmholtz equation in cylindrical or spherical coordinates. Bessel functions are therefore especially important for many problems of wave propagation and static potentials. In solving problems in cylindrical coordinate systems, one obtains Bessel functions of integer order or half integer order. 
For example, electromagnetic waves in a cylindrical waveguide, pressure amplitudes of inviscid rotational flows, heat conduction in a cylindrical object, etc.  (COPY ALERT)

### Elliptic Functions

------------------------- ------------------------------------------------------
Function                  Explanation  
------------------------- ------------------------------------------------------
`ellipj u m`              Jacobian elliptic functions of parameter `m` between 0 and 1, and real argument `u`.

`ellipk m`                Complete elliptic integral of the first kind

`ellipkm1 p`              Complete elliptic integral of the first kind around m = 1

`ellipkinc phi m`         Incomplete elliptic integral of the first kind

`ellipe m`                Complete elliptic integral of the second kind

`ellipeinc phi m`         Incomplete elliptic integral of the second kind
------------------------- ------------------------------------------------------
: Elliptic functions {#tbl:maths:elliptic}


The Jacobian elliptic functions are found in the description of the motion of a pendulum, as well as in the design of the electronic elliptic filters. These functions are periodic, with quarter-period on the real axis equal to the complete elliptic integral.
There are twelve Jacobi elliptic functions and `ellipj` returns three of them: sn, cn, dn. And the fourth result `phi` is called the amplitude of `u`. (COPY)

Elliptic integrals arose from the attempts to find the perimeter of an ellipse.
elliptic integral. A Elliptic integral function can be expressed in the form of:
$$f(x)=\int_c^xR(t, \sqrt(P(t)))dt,$$
where $R$ is a rational function of its two arguments, $P$ is a polynomial of degree 3 or 4 with no repeated roots, and $c$ is a constant.
Incomplete elliptic integrals are functions of two arguments; complete elliptic integrals are functions of a single argument. (COPY)

In general, integrals in this form cannot be expressed in terms of elementary functions. Exceptions to this general rule are when P has repeated roots, or when $R(x,y)$ contains no odd powers of y. However, with the appropriate reduction formula, every elliptic integral can be brought into a form that involves integrals over rational functions and the three Legendre canonical forms (i.e. the elliptic integrals of the first, second and third kind). (COPY)

We can use `ellipe` to compute the circumference of an ellipse. To compute that requires calculus, and the elliptic functions provides a solution.
Suppose an ellipse has semi-major axis $a=4$ and semi-minor axis $b=3$. We an compute its circumference using $4a\textrm{ellipe}(1 - \frac{b^2}{a^2})$.

```ocaml
# let a = 4. 
val a : float = 4.
# let b = 3.
val b : float = 3.
# let c = 4. *. a *. Maths.(ellipe (1. -. pow (b /. a) 2.))
val c : float = 22.1034921607095072
```

### Gamma Functions

For a positive integer n, the Gamma function is the factorial function. 

$$\Gamma(n) = (n-1)!$$

For a complex numbers $z$ with a positive real part, 

$$\Gamma(z) = \int_0^{\infty}x^{z-1}e^{-x}dx.$$

The Gamma function is widely used in a range of areas such as fluid dynamics, geometry, astrophysics, etc. It is especially suitable for describing a common pattern of processes that decay exponentially in time or space. 
The Gamma function and related function provided in Owl are list in [@tbl:maths:gamma].

------------------------- ------------------------------------------------------
Function                  Explanation  
------------------------- ------------------------------------------------------
`gamma z`                 Returns the value of the Gamma function

`rgamma z`                Reciprocal of the Gamma function

`loggamma z`              Principal branch of the logarithm of the Gamma function

`gammainc a x`            Regularized lower incomplete gamma function

`gammaincinv a y`         Inverse function of `gammainc`

`gammaincc a x`           Complemented incomplete gamma integral

`gammainccinv a y`        Inverse function of `gammaincc`

`psi z`                   The digamma function
------------------------- ------------------------------------------------------
: Gamma functions {#tbl:maths:gamma}

The incomplete gamma functions are similarly to the gamma function but with different or "incomplete" integral limits. The gamma function is defined as an integral from zero to infinity. This contrasts with the lower incomplete gamma function, which is defined as an integral from zero to a variable upper limit. Similarly, the upper incomplete gamma function is defined as an integral from a variable lower limit to infinity. 
The digamma function is defined as the logarithmic derivative of the gamma function. (COPY)

Here is an example of using `gamma`.

```ocaml
let x = Mat.linspace (-3.5) 5. 2000

let y = Mat.map Maths.gamma x

let _ =
  let h = Plot.create "example_gamma.png" in
  Plot.(plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2. ] x y);
  Plot.(set_yrange h (-10.) 20.);
  Plot.output h
```

![Examples of Gamma function along part of the real axis](images/maths/example_gamma.png "gamma"){width=75% #fig:algodiff:gamma}

(TODO: this figure should not have the vertical lines)


### Beta Functions

```
val beta : float -> float -> float
(**
Beta function.
 *)

val betainc : float -> float -> float -> float
(** Incomplete beta integral. *)

val betaincinv : float -> float -> float -> float
(** Inverse function of ``betainc``. *)
```

### Factorials

```
val fact : int -> float
(** Factorial function ``fact n`` calculates :math:`n!`. *)

val log_fact : int -> float
(** Logarithm of factorial function ``log_fact n`` calculates :math:`\log n!`. *)

val doublefact : int -> float
(** Double factorial function ``doublefact n`` calculates
:math:`n!! = n(n-2)(n-4)\dots 2` or :math:`\dots 1` *)

val log_doublefact : int -> float
(** Logarithm of double factorial function. *)

val permutation : int -> int -> int
(** ``permutation n k`` returns the number :math:`n!/(n-k)!` of ordered subsets
 * of length :math:`k`, taken from a set of :math:`n` elements. *)

val permutation_float : int -> int -> float
(**
``permutation_float`` is like ``permutation`` but deals with larger range.
 *)

val combination : int -> int -> int
(** ``combination n k`` returns the number :math:`n!/(k!(n-k)!)` of subsets of k elements
    of a set of n elements. This is the binomial coefficient
    :math:`\binom{n}{k}` *)

val combination_float : int -> int -> float
(** ``combination_float`` is like ``combination`` but can deal with a larger range. *)

val log_combination : int -> int -> float
(** ``log_combination n k`` returns the logarithm of :math:`\binom{n}{k}`. *)
```

### Error Functions 

```
val erf : float -> float
(** Error function. :math:`\int_{-\infty}^x \frac{1}{\sqrt(2\pi)} \exp(-(1/2) y^2) dy` *)

val erfc : float -> float
(** Complementary error function, :math:`\int^{\infty}_x \frac{1}{\sqrt(2\pi)} \exp(-(1/2) y^2) dy` *)

val erfcx : float -> float
(** Scaled complementary error function, :math:`\exp(x^2) \mathrm{erfc}(x)`. *)

val erfinv : float -> float
(** Inverse function of ``erf``. *)

val erfcinv : float -> float
(** Inverse function of ``erfc``. *)
```

### Struve Functions

```
val struve : float -> float -> float
(** ``struve v x`` returns the value of the Struve function of
order :math:`v` at :math:`x`. The Struve function is defined as,

.. math::
  H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},

where :math:`\Gamma` is the gamma function. :math:`x` must be positive unless :math:`v` is an integer

 *)
```

### Zeta Functions 

```
val zeta : float -> float -> float
(** ``zeta x q`` returns the Hurwitz zeta function :math:`\zeta(x, q)`, which
    reduces to the Riemann zeta function :math:`\zeta(x)` when :math:`q=1`. *)

val zetac : float -> float
(** Riemann zeta function minus 1. *)
```

### Other Functions

```
val is_prime : int -> bool
(** returns true if x is a prime number. 
The function is deterministic for all numbers representable by an int. The function uses the Rabin-Miller primality test.
*)

val fermat_fact : int -> int * int
(**
``fermat_fact x`` performs Fermat factorisation over ``x``, i.e. into two
roughly equal factors. ``x`` must be an odd number.
 *)

val nextafter : float -> float -> float
(** ``nextafter from to`` returns the next representable double precision value
of ``from`` in the direction of ``to``. If ``from`` equals ``to``, this value
is returned.
 *)

val nextafterf : float -> float -> float
(** ``nextafter from to`` returns the next representable single precision value
of ``from`` in the direction of ``to``. If ``from`` equals ``to``, this value
is returned.
 *)

```

## Interpolation and Extrapolation

## Integration

### Dawson and Fresnel Integrals

```
val dawsn : float -> float
(** Dawson's integral. *)

val fresnel : float -> float * float
(** Fresnel trigonometric integrals. ``fresnel x`` returns a tuple consisting of
``(Fresnel sin integral, Fresnel cos integral)``. *)
```

### Other Special Integrals

```
val expn : int -> float -> float
(** Exponential integral :math:`E_n`. *)

val shichi : float -> float * float
(** Hyperbolic sine and cosine integrals, ``shichi x`` returns
 * :math:`(\mathrm{shi}, \mathrm{chi})``. *)

val shi : float -> float
(** Hyperbolic sine integral. *)

val chi : float -> float
(** Hyperbolic cosine integral. *)

val sici : float -> float * float
(** Sine and cosine integrals, ``sici x`` returns :math:`(\mathrm{si}, \mathrm{ci})`. *)

val si : float -> float
(** Sine integral. *)

val ci : float -> float
(** Cosine integral. *)
```

## Complex Number
