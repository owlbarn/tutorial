# Mathematical Functions

TBD

## Basic Functions

Note that functions in this chapter works on scalar values. 
The N-dimensional array module introduced in later chapters contains these basic functions that work on n-dimensional arrays, including vectors and matrices.

### Basic Unary Math Functions

Many basic math functions takes one float number as input and returns one float number. We call them *unary* functions.
You can use these unary functions easily from the `Maths` module. For example:

```ocaml
# Maths.sqrt 2. 
- : float = 1.41421356237309515
```

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
------------  -------------------------------------------------------
: Basic unary math functions {#tbl:maths:basic_unary}

### Basic Binary Functions

Binary functions takes two floats as inputs and returns one float as return. 
The most common arithmetic functions belong to this category.

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`add`         `x + y`

`sub`         `x - y`

`mul`         `x * y`

`div`         `x / y`

`fmod`        `x % y`

`pow`         $x^y$

`hypot`       $\sqrt{x^2 + y^2}$

`atan2`       returns $\arctan(y/x)$, accounting for the sign of the 
              arguments; this is the angle to the vector $(x, y)$ counting from the x-axis.
------------  -------------------------------------------------------
: Binary math functions {#tbl:maths:binary}

### Exponential and Logarithmic Functions

The constant $e = \sum_{n=0}^{\infty}\frac{1}{n!}$ is what we called the "natural constant". 
It is called this way because the exponential function and it inverse function logarithm are so frequently used in nature and our daily life: logarithmic spiral, population growth, carbon date ancient artifacts, computing bank investments, etc.

We also have this beautiful Euler's formula that connects the two most frequently used constants and the base of complex number and natural numbers:

$$e^{i\pi}+ 1=0.$$

As an example, in a scientific experiment about bacteria, we can assume the number of bacterial follows an exponential function $n(t) = Ce^rt$ where $C$ is the initial population and $r$ is the daily increase rate. 
With this model, we can predict how the population of bacterial grows within certain time.

The full list of exponential and logarithmic functions, together with some variants, are presented in [@tbl:maths:explog].

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

### Trigonometric Functions

In mathematics, the trigonometric functions are real functions which relate an angle of a right-angled triangle to ratios of two side lengths. They are widely used in all sciences that are related to geometry, such as navigation, solid mechanics, celestial mechanics, geodesy, and many others. They are among the simplest periodic functions, and as such are also widely used for studying periodic phenomena, through Fourier analysis. The most widely used trigonometric functions are the sine, the cosine, and the tangent. Their reciprocals are respectively the cosecant, the secant, and the cotangent, which are less used in modern mathematics. ([COPY](https://en.wikipedia.org/wiki/Trigonometric_functions))

The triangular functions are all unary functions, for example:

```ocaml
# Maths.sin (Owl_const.pi /. 2.)
- : float = 1.
```

And they are all included in the math module in Owl, as shown in [@tbl:maths:triangular].

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
------------  -------------------------------------------------------
: Trigonometric math functions {#tbl:maths:triangular}


![Relationship between different trigonometric functions](images/maths/trio.png "trio"){width=80% #fig:algodiff:trio}

([Figure src](https://zh.wikipedia.org/wiki/%E5%8F%8C%E6%9B%B2%E5%87%BD%E6%95%B0))

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
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
------------  -------------------------------------------------------
: Hyperbolic Trigonometric math functions {#tbl:maths:triangular_hyper}


![Relationship between different hyperbolic trigonometric functions](images/maths/hyper_trio.png "hyper_trio"){width=80% #fig:algodiff:hyper_trio}

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`sinc`        returns $\sin(x)/x$ and 1 for $x=0$

`logsinh`     returns $\log(\sinh(x))$ but handles large $|x|$

`logcosh`     returns $\log(\cosh(x))$ but handles large $|x|$

`sindg`       Sine of angle given in degrees

`cosdg`       Cosine of the angle given in degrees

`tandg`       Tangent of angle given in degrees

`cotdg`       Cotangent of the angle given in degrees
------------  -------------------------------------------------------
: Other Trigonometric math functions {#tbl:maths:triangular_other}

### Other Math Functions

There are some other function that may not be very common in traditional math.
Functions such as `sigmoid` and `relu` are frequently used in Deep Learning as the activation functions in a neural network.
The activation functions are crucial to the neural network regarding various aspects, including output result, accuracy, convergence speed, etc.

------------  -------------------------------------------------------
Function      Explanation  
------------  -------------------------------------------------------
`sigmoid x`   $1 / (1 + \exp(-x))$

`signum x`    Returns the sign of `x`: -1, 0, or 1.

`softsign x`  Smoothed `sign` function

`relu x`      $\max(0, x)$
------------  -------------------------------------------------------
: Other math functions {#tbl:maths:others}


## Special Functions

The definition of numerous special functions of mathematical physics. 
Special functions are particular mathematical functions that have more or less established names and notations due to their importance in mathematical analysis, functional analysis, physics, or other applications.(COPY)
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

Beta function is defined as:

$$B(x,y) = \int_0^1t^{x-1}(1-t)^{y-1}dt = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$$

The incomplete beta function extends this definition to:

$$B(x, a, b) = \int_0^xt^{a-1}(1-t)^{b-1}dt.$$

They are both included in the special functions provided by Owl.

------------------------- ------------------------------------------------------
Function                  Explanation  
------------------------- ------------------------------------------------------
`beta x y`                Beta function     

`betainc a b x`           Incomplete Beta integral

`betaincinv a b y`        Inverse function of `betainc`
------------------------- ------------------------------------------------------
: Beta functions {#tbl:maths:beta}

The Beta function has several properties:

```ocaml
# let x = Maths.beta 3. 4. 
val x : float = 0.0166666666666666664
# let y = Maths.((gamma 3.) *. (gamma 4.) /. (gamma (7.))) 
val y : float = 0.0166666666666666664
```

This validate the relationship between beta funtion and gamma function.
Another property of beta function is it is symmetric, which means $B(x,y) = B(y, x)$.

```ocaml
# let x = Maths.beta 3. 4. 
val x : float = 0.0166666666666666664
# let y = Maths.beta 4. 3. 
val y : float = 0.0166666666666666664
```

Beta function is the first known scattering amplitude in String theory in physics. It can also be used to model a preferential attachment process, which describes the distribution of resources among individuals based on the resource amount they already have. (COPY)

### Struve Functions

The Struve function is defined as:
$$H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},$$

where $\Gamma$ is the gamma funcction. $x$ must be positive unless $v$ is an integer.
The function `struve v x` returns the value of Struve function. The paramter $v$ is called the *order* of this Struve function.
Here is an example.

```ocaml
let _ =
  let h = Plot.create "example_struve.png" in
  Plot.(plot_fun ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.] (Maths.struve 0.) (-12.) 12.);
  Plot.(plot_fun ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2.] (Maths.struve 1.) (-12.) 12.);
  Plot.(plot_fun ~h ~spec:[ RGB (244, 180,  0); LineStyle 3; LineWidth 2.] (Maths.struve 2.) (-12.) 12.);
  Plot.(plot_fun ~h ~spec:[ RGB (77,  81,  57); LineStyle 1; LineWidth 2.] (Maths.struve 3.) (-12.) 12.);
  Plot.(plot_fun ~h ~spec:[ RGB (111, 51, 129); LineStyle 2; LineWidth 2.] (Maths.struve 4.) (-12.) 12.);
  Plot.(set_yrange h (-3.) 5.);
  Plot.(legend_on h ~position:SouthEast [|"H0"; "H1"; "H2"; "H3"; "H4"|]);
  Plot.output h
```

![Examples of Struve function for different orders.](images/maths/example_struve.png "struve"){width=75% #fig:algodiff:struve}

Struve functions have some specific uses across many different fields of physics in a wide variety of applications. For example, they can be found in water-wave and surface-wave problems (specifically flow of liquid near a turning ship) as well as calculations to do with the distribution of fluid pressure over a vibrating disk and other unsteady aerodynamics. They also crop up when considering aspects of optical diffraction, plasma stability (specifically resistive magnetohydrodynamics instability theory), quantum dynamical studies of spin decoherence and excitation in carbon nanotubes. ([COPY](https://www.nag.co.uk/content/struve-functions))


### Zeta Functions

The Hurwitz zeta function `zeta x q` returns the Hurwitz zeta function:

$$\zeta(x, q) = \sum_{k=0}^{\infty}\frac{1}{(k+q)^x}.$$

When $q$ is set to 1, this function is reduced to Riemann zeta function. 
The function `zetac x` returns Riemann zeta function minus 1.
We can evaluate the zeta function at certain points, for example:

```ocaml
# Maths.zeta 4. 1. 
- : float = 1.08232323371113837

# (Maths.pow Owl_const.pi 4.) /. 90.
- : float = 1.08232323371113792
```

The Riemann zeta function plays a pivotal role in analytic number theory and has applications in physics, probability theory, and applied statistics.
Zeta function regularization is used as one possible means of regularization of divergent series and divergent integrals in quantum field theory. In one notable example, the Riemann zeta-function shows up explicitly in one method of calculating the Casimir effect. The zeta function is also useful for the analysis of dynamical systems. (COPY)

### Error Functions 

The error functions are not about error processing in programming.
In mathematics, it is defined as:
$$\frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2})dt.$$

------------------------- ------------------------------------------------------
Function                  Explanation  
------------------------- ------------------------------------------------------
`erf x`                   Error function

`erfc x`                  Complementary error function: $1 - \textrm{erf}(x)$

`erfcx x`                 Scaled complementary error function: $\exp(x^2) \mathrm{erfc}(x)$

`erfinv x`                Inverse function of `erf`

`erfcinv x`               Inverse function of `erfc`
------------------------- ------------------------------------------------------
: Error functions {#tbl:maths:error}

The error function is a sigmoid function. We can observe its shape by the code below.

```ocaml
let _ =
  let h = Plot.create "example_erf.png" in
  Plot.(plot_fun ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.] Maths.erf (-3.) 3.);
  Plot.output h
```

![Plot of the Error function.](images/maths/example_erf.png "struve"){width=75% #fig:algodiff:erf}

The error function occurs often in probability, statistics, and partial differential equations describing diffusion. In statistics, for nonnegative values of x, the error function has the following interpretation: for a random variable `Y` that is normally distributed with mean 0 and variance 0.5, then `erf x` is the probability that `Y` falls in the range `[-x, x]`. (COPY)

### Integral Functions

Owl also provides several special integral functions. 
The Dawson function is defined as:
$$D(x) = e^{-x^2}\int_0^x~e^{t^2}dt$$

And the Fresnel trigonometric integral returns a tuple that contains two parts:
$$S(x) = \int_0^x~sin(t^2)dt, C(x) = \int_0^x~cos(t^2)dt.$$

We can observe the functions of these integrals with plots.

```ocaml
let _ =
  let h = Plot.create ~m:1 ~n:2 "example_integrals.png" in
  Plot.subplot h 0 0;
  Plot.(plot_fun ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.] Maths.dawsn (-5.) 5.);
  Plot.set_ylabel h "dawsn(x)";
  Plot.subplot h 0 1;
  Plot.(plot_fun ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.] (fun x -> let s, _ = Maths.fresnel x in s) 0. 5.);
  Plot.(plot_fun ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2.] (fun x -> let _, c = Maths.fresnel x in c) 0. 5.);
  Plot.(legend_on h ~position:SouthEast [|"S(x)"; "C(x)"|]);
  Plot.set_ylabel h "fresnel(x)";
  Plot.output h
```

![Plot of the Dawson and Fresnel integral function.](images/maths/example_integrals.png "integrals"){width=100% #fig:algodiff:integrals}

Besides these two, other type of special integral functions are also provided, as shown in [@tbl:maths:integral].

----------------- -----------------------------------------------------------
Function          Explanation  
----------------- -----------------------------------------------------------
`expn n x`        Generalized exponential integral $E_n(x) = x^{n-1}\int_x^{\infty}\frac{e^{-t}}{t^n}dt$

`shi x`           Hyperbolic sine integral: $\int_0^x~\frac{\sinh~t}{t}dt$

`chi x`           Hyperbolic cosine integral: $\gamma + \log(x) + \int_0^x~\frac{\cosh~t -1}{t}dt$

`shichi x`        (`shi x`, `chi x`)

`si x`            Sine integral: $\int_0^x~\frac{\sin~t}{t}dt$

`ci x`            Cosine integral: $\gamma + \log(x) + \int_0^x~\frac{\cos~t -1}{t}dt$

`sici x`          (`si x`, `ci x`)
----------------- -----------------------------------------------------------
: Integral functions {#tbl:maths:integral}

Dawson integrals is motivated by research on the electromagnetic radiation propagation across the surface of earth.
The Fresnel integrals were originally used in the calculation of the electromagnetic field intensity in an environment where light bends around opaque objects. More recently, they have been used in the design of highways and railways, specifically their curvature transition zones. Other applications are roller coasters or calculating the transitions on a velodrome track to allow rapid entry to the bends and gradual exit. ([COPY](https://en.wikipedia.org/wiki/Fresnel_integral))


## Factorials

The definition of *factorials* is simple:

$F(n) = n! = n \times (n - 1) \times (n-2) \ldots \times 1$

The factorial function, together with several variants, are contained in the math module.

----------------- -----------------------------------------------------------
Function          Explanation  
----------------- -----------------------------------------------------------
`fact n`          Factorial function $!n$

`log_fact n`      Logarithm of factorial function

`doublefact n`    Double factorial function calculates $n!! = n(n-2)(n-4)\dots 2$ (or 1)

`log_doublefact n` Logarithm of double factorial function
----------------- -----------------------------------------------------------
: Factorial functions {#tbl:maths:factorial}

The factorial functions accepts integer as input, for example:

```ocaml
# Maths.fact 5 
- : float = 120.
```

The factorials are applied in many areas of mathematics, most notably the combinatorics.
The permutation and combination are both defined in factorials. 
The permutation returns the number $n!/(n-k)!$ of ordered subsets of length $k$, taken from a set of $n$ elements. 
THe combination returns the number ${n\choose k} = n!/(k!(n-k)!)$ of subsets of $k$ elements of a set of $n$ elements.
[@tbl:maths:perm] provides the combinatorics functions you can use in the math module.


----------------------  -----------------------------------------------------------
Function                Explanation  
----------------------  -----------------------------------------------------------
`permutation n k`       Permutation number 

`permutation_float n k` Similar to `permutation` but deals with larger range and returns float

`combination n k`       Combination number

`combination_float n k` Similar to `combination` but deals with larger range and returns float

`log_combination n k`   Returns the logarithm of ${n\choose k}$
----------------------  -----------------------------------------------------------
: Permutation and combination functions {#tbl:maths:perm}

We can see a simple example.

```ocaml
# let x = Maths.combination 10 2
val x : int = 45
# let y = Maths.combination_float 10 2
val y : float = 45.
```

## Interpolation and Extrapolation

Sometimes we don't know the full description of a function $f$, but only some points on it, and therefore we cannot calculate its value at an aribitrary point.
The target is to esimate the $f(x)$ for an arbitrary $x$ by drawing a smooth curve through the given data. If $x$ is within the range of the given data, this taks is called *interpolation*, otherwise it's called *extrapolation*, which is much more difficult to do.

The `Owl_maths_interpolate` module provides an `polint` function for interpolation and extrapolation:

```
val polint : float array -> float array -> float -> float * float
```

`polint xs ys x` performs polynomial interpolation of the given arrays `xs` and `ys`. Given arrays $xs[0 \ldots (n-1)]$ and $ys[0\ldots~(n-1)]$, and a value `x`. 
The function returns a value `y`, and an error estimate `dy`.
The paramter `xs` is an array of input `x` values of `P(x)`, and `ys` is an array of corresponding `y` values of `P(x)`.
It returns `(y', dy)` wherein `y'` is the returned value `y' = P(x)`, and `dy` is the estimated error.

As its name suggests, the `polint` approximate complicated curves with polynomial of lowest possible degree that passes the given points.
We can show how this interplation method works for an example. 
In the previous chapter we have introduced that the Gamma function is actually a interpolation solution to the integer function $y(x) = (n-1)!$. 
So we can specify five nodes on a plane that are generated from this factorial functions.

```ocaml env=maths:interp
# let x = [|2; 3; 4; 5; 6|]
val x : int array = [|2; 3; 4; 5; 6|]
# let y = Array.map (fun x -> Maths.fact (x - 1)) x 
val y : float array = [|1.; 2.; 6.; 24.; 120.|]
# let x = Array.map float_of_int x
val x : float array = [|2.; 3.; 4.; 5.; 6.|]
```

Now we can define the interpolation function `f` that accept on float number and returns another float number.
Also we convert the given data $x$ and $y$ into matrix format for plotting purpose.

```ocaml env=maths:interp
let f a = 
  let v, _ = Owl_maths_interpolate.polint x y a in
  v 

let xm = Mat.of_array x 1 5
let ym = Mat.of_array y 1 5
```

Now we can plot the interpolation function. We compare it to the Gamma function. 
As can be seen in [@fig:maths:interp], both lines cross the given nodes. We can see that the interpolated line fits well with the "true interpolation", i.e. the Gamma function. 
However, the extrapolation fitting where the x-value falls out of given data, is less than ideal.

```ocaml env=maths:interp
let _ =
  let h = Plot.create "interp.png" in
  Plot.(plot_fun ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.] f 2. 6.5);
  Plot.(plot_fun ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2.] Maths.gamma 2. 6.5);
  Plot.(scatter ~h ~spec:[ Marker "#[0x229a]"; MarkerSize 5. ] xm ym);
  Plot.(legend_on h ~position:NorthWest [|"Interpolation"; "Gamma function"; "Given values"|]);
  Plot.output h
```

![Plot of interpolation and corresponding Gamma function.](images/maths/interp.png "interp"){width=75% #fig:maths:interp}

## Integration

Given a function $f$ that accepts a real variable and an interval $[a, b]$ of the real line, the integral of this function

$$\int_a^bf(x)dx$$

can be thought of as the sum of signed area of the region in the cartesian plane that is bounded by the curve of f, the x-axis within the x-axis range $[a, b]$. The area above the x-axis adds to the sum and that below the x-axis subtracts from the area sum.

Owl provides several neumerical routines to help you to do integrations in `Owl_maths_quandrature` module. For example, we can compute  $\int_1^4x^2$ with the code below:

```ocaml
# Owl_maths_quadrature.trapz (fun x -> x ** 2.) 1. 4.
- : float = 21.0001344681758439
```

We can verify this result using the fundamental theorem of calculus:

$$\int_1^4x^2 = (4^3 -1^3) / 3 = 21$$.

So you might be thinking, what is this `trapz`? Why the result is not exactly `21`?

Using numerical methods (or *quadrature*) to do integration dates back to the invention of calculus or even earlier. 
The basic idea is to use summation of small areas to approximate that of an integration, as shown in [@fig:maths:integration_basic] ([src](https://www.sciencedirect.com/topics/computer-science/numerical-integration)).

![Basic method of numerical integration](images/maths/integration_basic.png "integration"){width=80% #fig:maths:integration_basic}

There exists a lot of algorithms to do numerical integration, and using the trapezoial rule is one of them. 
This classical method divide a to b into $N$ equally spaced abscissas: $x_0, x_1, \ldots, x_N$. Each area between $x_i$ and $x_j$ is seen as an Trapezoid and the area formula is computed as:

$$\int_{x_0}^{x_1}f(x)dx = h(\frac{f(x_0)}{2} + \frac{f(x_1)}{2}) + O(h^3f'').$$

Here the error term $O(h^3f'')$ indicated that the error of approximation is related with that of abscissas size $h$ and second order derivative of the original function.

Function `trapz` implements this method. It's interface is:

```
val trapz : ?n:int -> ?eps:float -> (float -> float) -> float -> float -> float
```

`trapz ~n ~eps f a b` computes the integral of `f` on the interval `[a,b]` using the trapezoidal rule.
It works by iterating for several stages, each stage improving the accuracy by adding more interior points. 
The argument $n$ specifies the maximum step which defaults to 20, and `eps` is the desired fractional accuracy threshold, which defaults to `1e-6`.

The other methods are similar to `trapz` in interface, only different in implementation.
For example, the `simpson` uses the Simpson formula:

$$\int_{x_0}^{x_2}f(x)dx = h(\frac{f(x_0)}{3} + \frac{4f(x_1)}{3} + \frac{f(x_2)}{3}) + O(h^5f(4)).$$

Then there is the *Romberg integration* (`romberg`) that can choose methods of different orders to give good accuracy, and the algorithms is normally much faster than the `trapz` and `simpson` methods.
Moreover, if the abscissas can be varied, then there is the adaptive Gaussian quadrature of fixed tolerance `gaussian` and Gaussian quadrature of fixed order `gaussian`.

As an example, we can compute the special integral function $Si(x)=\int_0^x\frac{sin(t)}{t}dt$ from previous section using the numerical integration method. Let's set $x=4$.

```ocaml
# let f t = Maths.(div (sin t) t)
val f : float -> float = <fun>
# Owl_maths_quadrature.gaussian f 0. 4.
- : float = 1.75820313914469306
# Owl_maths.si 4.
- : float = 1.75820313894905289
```

We can see the numerical method `gaussian` works well to approximate this special integral function.

## Utility Functions

Besides what we have mentioned, there are also some utitlity functions that worth mentioning. 

A prime number is a natural number greater than `1` that cannot be formed by multiplying two smaller natural numbers. 
The `is_prime` checks if an integer is a prime number.
This function is deterministic for all numbers representable by an int. It is implemented using the [Miller-Rabin primality test](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test) method.

```ocaml
# Maths.is_prime 997
- : bool = true
```

Primes are used in several routines in information technology, such as public-key cryptography, which relies on the difficulty of factoring large numbers into their prime factors. In abstract algebra, objects that behave in a generalized way like prime numbers include prime elements and prime ideals.([COPY](https://en.wikipedia.org/wiki/Prime_number))

Another number theory related idea is the *Fermat's factorization*, which represents an odd integer as the difference of two squares: $N = a^2 - b^2$, and therefore `N` can be factorised as $(a+b)(a-b)$.
The function `fermat_fact` performs Fermat factorisation over odd number `N`, i.e. into two roughly equal factors $x$ and $y$ so that $N=x\times~y$.

```ocaml
# Maths.fermat_fact 6557
- : int * int = (83, 79)
# 83 * 79
- : int = 6557
```

Next two functions concerns the precision of float numbers in computer.

TODO: Explain the mechansim of float number in a computer. 

`nextafter from to` returns the next representable double precision value of ``from`` in the direction of `to`. If `from` equals `to`, this value is returned.
The other is `nextafterf`.
`nextafter from to` returns the next representable single precision value
of `from` in the direction of `to`. If `from` equals `to`, this value
is returned.
For example:

```ocaml
# Maths.nextafterf 1. 2.;;
- : float = 1.00000011920928955
# Maths.nextafter 1. 2.;;
- : float = 1.00000000000000022
# Maths.nextafter 1. 0.;;
- : float = 0.999999999999999889
```
