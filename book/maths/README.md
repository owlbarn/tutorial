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

### Airy Functions

In the physical sciences, the Airy function (or Airy function of the first kind) Ai(x) is a special function named after the British astronomer George Biddell Airy. (COPY)


```
val airy : float -> float * float * float * float
(**
Airy function ``airy x`` returns ``(Ai, Ai', Bi, Bi')`` evaluated at :math:`x`.
``Ai'`` is the derivative of ``Ai`` whilst ``Bi'`` is the derivative of ``Bi``.
*)
```

### Bessel Functions 

Bessel functions, first defined by the mathematician Daniel Bernoulli and then generalized by Friedrich Bessel, are canonical solutions y(x) of Bessel's differential equation.

```
val j0 : float -> float
(** Bessel function of the first kind of order 0. *)

val j1 : float -> float
(** Bessel function of the first kind of order 1. *)

val jv : float -> float -> float
(** Bessel function of real order. *)

val y0 : float -> float
(** Bessel function of the second kind of order 0. *)

val y1 : float -> float
(** Bessel function of the second kind of order 1. *)

val yv : float -> float -> float
(** Bessel function of the second kind of real order. *)

val yn : int -> float -> float
(** Bessel function of the second kind of integer order. *)

val i0 : float -> float
(** Modified Bessel function of order 0. *)

val i0e : float -> float
(** Exponentially scaled modified Bessel function of order 0. *)

val i1 : float -> float
(** Modified Bessel function of order 1. *)

val i1e : float -> float
(** Exponentially scaled modified Bessel function of order 1. *)

val iv : float -> float -> float
(** Modified Bessel function of the first kind of real order. *)

val k0 : float -> float
(** Modified Bessel function of the second kind of order 0, :math:`K_0`.*)

val k0e : float -> float
(** Exponentially scaled modified Bessel function K of order 0. *)

val k1 : float -> float
(** Modified Bessel function of the second kind of order 1, :math:`K_1(x)`. *)

val k1e : float -> float
(** Exponentially scaled modified Bessel function K of order 1. *)
```

### Elliptic Functions 

```
val ellipj : float -> float -> float * float * float * float
(** Jacobian Elliptic function ``ellipj u m`` returns ``(sn, cn, dn, phi)``. *)

val ellipk : float -> float
(** ``ellipk m`` returns the complete elliptic integral of the first kind. *)

val ellipkm1 : float -> float
(** FIXME. Complete elliptic integral of the first kind around :math:`m = 1`. *)

val ellipkinc : float -> float -> float
(** ``ellipkinc phi m`` incomplete elliptic integral of the first kind. *)

val ellipe : float -> float
(** ``ellipe m`` complete elliptic integral of the second kind. *)

val ellipeinc : float -> float -> float
(** ``ellipeinc phi m`` incomplete elliptic integral of the second kind. *)
```

### Gamma Functions

```
val gamma : float -> float
(**
``gamma z`` returns the value of the Gamma function.
The gamma function is often referred to as the generalized factorial since
:math:`z\ gamma(z) = \gamma(z+1)` and :math:`gamma(n+1) = n!`
for natural number :math:`n`.
 *)

val rgamma : float -> float
(** Reciprocal Gamma function. *)

val loggamma : float -> float
(** Logarithm of the gamma function. *)

val gammainc : float -> float -> float
(** Incomplete gamma function. *)

val gammaincinv : float -> float -> float
(** Inverse function of ``gammainc``. *)

val gammaincc : float -> float -> float
(** Complemented incomplete gamma integral. *)

val gammainccinv : float -> float -> float
(** Inverse function of ``gammaincc``. *)

val psi : float -> float
(** The digamma function. *)
```

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
