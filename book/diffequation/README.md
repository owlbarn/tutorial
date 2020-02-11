# Ordinary Differential Equations


## What Is An ODE

A *differential equation* is an equation that contains a function and one or more of its derivatives. 
It is studied ever since the invention of calculus, driven by the applications in mechanics, astronomy, and geometry.
Currently it has become a important branch of mathematics study and its application is widely extended to biology, engineering, economics, and much more fields. 

In a differential equation, if the function and its derivatives are about only one variable, we call it an *Ordinary Differential Equation*(ODE). 
It is often used to model one-dimensional dynamical systems.
Otherwise it is an *Partial Differential Equation*(PDE). 
In this chapter we focus on the former one.

Generally, a ODE can be expressed as:

$$ F(x, y^{'}, y^{''}, \ldots, y^{(n)}) = 0.$$ {#eq:diffequation:ode-def}

The differential equations model dynamic systems, and the initial status of the system is often known. That is called *initial values*. 
They can be represented as:

$$y|_{x=x_0} = y_0, y^{'}|_{x=x_1} = y_1, \ldots ,$${#eq:diffequation:init}

where the $y_0$, $y_1$, etc. are known.
The highest order of derivatives that are used in [@eq:diffequation:ode-def] is the *order* of this differential equation.
A first-order differential equation can be generally expressed as: $\frac{dy}{dx}=f(x,y)$, where $f$ is any function that contains $x$ and $y$.

Solving [@eq:diffequation:ode-def] that fits given initial values as in [@eq:diffequation:init] is called the *initial value problem*.
Solving this kind of problems is the main target of many numerical ODE solvers.

### Exact Solutions 

Solving a differential equation is often complex, but we do not how to solve part of them.
Before looking at the the solvers to a random ODEs, let's turn to the math first and look at some ODE forms that we already have analytical close-form solution.

**Separable equations**: 

$$P(y)\frac{dy}{dx} + Q(x) = 0,$$

and it's close form solution is:

$$\int^{y}P(y)dy + \int^{x}Q(x)dx = C$$.

**Linear first-order equations**:

$$\frac{dy}{dx} + P(x)y = Q(x)$$

It's solution is:

EQUATION

Solving ODE analytically is not the focus of solvers.
REFER to classical math book (reference required) or full course for more detail.

### Reduce High-Order Equations

Enough to support the examples in the rest of this chapters.

### Linear Systems

Many system involves not just one unknown functions $y$. 

System of equations 
Oscillator, two-body problem.

## Solving An ODE Numerically

This section introduces the basic idea of solving the initial value problem numerically.
Let's start with an example:

$$y' = 2xy + x,$$ {#eq:diffequation:example01}

where the initial value is $y(0) = 0$.
Without going deep into the whole math calculation process (hint: it's a separable first-order ODE), we give its analytical close-form solution:

$$y = 0.5(\exp{x^2} - 1).$$ {#eq:diffequation:example01_solution}

Now, pretending we don't know the solution in [@eq:diffequation:example01_solution], and we want to answer the question: what is $y$'s value when $x = 1$ (or any other value)?
How can we solve it numerically?

Meet the *Euler Method*, a first-order numerical procedure to solve initial value problems. This method proposes to approximate the function $y$ using a sequence of iterative steps:

$$ y_{n+1} = y_n + \Delta~f(x_n, y_n),$$

where $\Delta$ is a certain step size.
This method is really easy to be implemented in OCamla, as shown below. 

```ocaml
let x = ref 0.
let y = ref 0.
let target = 1.
let step = 0.001
let f x y = 2. *. x *. y +. x

let _ = 
  while !x <= target do 
	y := !y +. step *. (f !x !y);
	x := !x +. step
  done
```

In this case, we know that the analytical solution at $x=1$ is $0.5(\exp{1^2} - 1$:

```ocaml
# (Owl_const.e -. 1.)/. 2.
- : float = 0.859140914229522545
```

and the solution given by the previous numerical code is about `0.8591862`, which is pretty close to the true answer.

However, this method is as easy as it is unsuitable to be used in practical applications.
One reason is that this method is not very accurate, despite that it works well in our example here. We will show this point soon.
Also, it is not very stable, nor does it provide error estimate.

Therefore, we can modify the Euler's method to use a "midpoint" in stepping, hoping to curb the error in the update process:

$$ s_1 = f(x_n, y_n),$$
$$ s_2 = f(x_n + \Delta~/2, y_n + s_1~\Delta~/2),$$ {#eq:diffequation:rk2}
$$ y_{n+1} = y_n + \Delta~\frac{s_1 + s_2}{2}.$$

This method is called the *Midpoint Method*, and we can also implement it in OCaml similarly.
Let's compare the performance of Euler and Midpoint in approximating the true result in [@eq:diffequation:example01_solution]:

```ocaml
let f x y = 2. *. x *. y +. x
let f' x = 0.5 *. (Maths.exp (x *. x) -. 1.)

let euler step target = 
	let x = ref 0. in
	let y = ref 0. in
	while !x <= target do 
		y := !y +. step *. (f !x !y);
		x := !x +. step
	done;
	!y

let midpoint step target = 
	let x = ref 0. in
	let y = ref 0. in
	while !x <= target do 
		let s1 = f !x !y in 
		let s2 = f (!x +. step /. 2.) (!y +. step /. 2. *. s1) in 
		y := !y +. step *. (s1 +. s2) /. 2.;
		x := !x +. step
	done;
	!y

let _ = 
	let target = 2.6 in 
	let h = Plot.create "plot_rk01.png" in 
	Plot.(plot_fun ~h ~spec:[ RGB (66,133,244); LineStyle 1; LineWidth 2.; Marker "*" ] f' 2. target);
	Plot.(plot_fun ~h ~spec:[ RGB (219,68,55); LineStyle 2; LineWidth 2.; Marker "+" ] (euler 0.01) 2. target);
	Plot.(plot_fun ~h ~spec:[ RGB (219,68,55); LineStyle 2; LineWidth 2.; Marker "." ] (euler 0.001) 2. target);
	Plot.(plot_fun ~h ~spec:[ RGB (244,180,0); LineStyle 3; LineWidth 2.; Marker "+" ] (midpoint 0.01) 2. target);
	Plot.(plot_fun ~h ~spec:[ RGB (244,180,0); LineStyle 3; LineWidth 2.; Marker "." ] (midpoint 0.001) 2. target);
	Plot.(legend_on h ~position:NorthWest [|"Close-Form Solution"; "Euler (step = 0.01)"; 
		"Euler (step = 0.001)"; "Midpoint (step = 0.01)"; "Midpoint (step = 0.001)"|]);
	Plot.output h
```

Let's see the result.

![Comparing the accuracy of Euler method and Midpoint method in approximating solution to ODE](images/diffequation/plot_rk01.png "plot_rk01"){width=70%}

We can see that the choice of step size indeed matters to the precision. We use 0.01 and 0.001 for step size in the test, and for both cases the midpoint method outperforms the simple Euler method. 

Should we stop now? Do we find a perfect solution in midpoint method? Surely no!
We can follow the existing trend and add more intermediate stages in the update sequence. 
For example, we can do this:

$$ s_1 = f(x_n, y_n),$$
$$ s_2 = f(x_n + \Delta~/2, y_n + s_1~\Delta~/2),$$
$$ s_3 = f(x_n + \Delta~/2, y_n + s_2~\Delta~/2),$$ {#eq:diffequation:rk4}
$$ s_4 = f(x_n + \Delta, y_n + s_3~\Delta),$$
$$ y_{n+1} = y_n + \Delta~\frac{s_1 + 2s_2+2s_3+s_4}{6}.$$

Here in each iteration four intermediate steps are computed, once at the initial point, once at the end, and twice at the midpoints.
This method often more accurate than the midpoint method.

We won't keep going on but you have seen the pattern.
These seemingly mystical parameters are related to the term in Taylor series expansions.
In the previous methods, e.g. Euler method, every time you update $y_n$ to $y_{n+1}$, an error is introduced into the approximation.
The *order* of a method is the exponent of the smallest power of $\Delta$ that cannot be matched.
All these methods are called *Runge-Kutta Method*.
It's basic idea is to remove the errors order by order, using the correct set of coefficients.
A higher order of error indicates smaller error.

The Euler is the most basic form of Runge-Kutta method, and the Midpoint is also called the second-order Runge-Kutta Method (rk2). 
What [@eq:diffequation:rk4] shows is a fourth-order Runge-Kutta method (rk4).
It is the most often used RK method and works surprisingly well in many cases, and it is often a good choice especially when computing $f$ is not expensive. 

However, as powerful as it may be, the classical `rk4` is still a native implementation, and a modern ODE solvers, though largely follows the same idea, adds more "ingredients". 
For example, the step size should be adaptively updated instead of being const in our example. 
Also, you may have seen solvers with names such as `ode45` in MATLAB, and in their implementation, it means that this solver gets its error estimate at each step by comparing the 4th order solution and 5th order solution and then decide the direction.

Besides, other methods also exists. For example, the Bulirsch-Stoer method is known to be both accurate and and efficient computation-wise.
(TODO: Brief introduction of Adam and BDF.)
Discussion of these advanced numerical methods and techniques are beyond this book. Please refer to [@press2007numerical] for more information.

## Owl-ODE

Obviously, we cannot just relies on these manual solutions every time in practical use. It's time we use some tools. 
Based on the computation functionalities and ndarray data structures in Owl, we provide the package *[owl_ode](https://github.com/owlbarn/owl_ode)" to perform the tasks of solving initial value problems.

Without further due, let's see it how `owl-ode` package can be used to solve ODE problem.

### Example: Linear Oscillator System

EXPLAIN: how the equation of Oscillator becomes this linear representation.

Let's see how to solve a time independent linear dynamic system that contains two states:

$$\frac{dy}{dt} = Ay, \textrm{where } A = \left[ \begin{matrix} 1 & -1 \\ 2 & -3 \end{matrix} \right].$$ {#eq:diffequation:example_01}

This equation represents an oscillator system.
In this system, $y$ is the state of the system, and $t$ is time.
The initial state at $t=0$ is $y_0 = \left[ -1, 1\right]^T$.
Now we want to know the system state at $t=2$.
The function can be expressed in Owl using the matrix module.

```ocaml env=diffequation_example01
let f y t = 
  let a = [|[|1.; -1.|];[|2.; -3.|]|]|> Mat.of_arrays in
  Mat.(a *@ y)
```

Next, we want to specify the timespan of this problem: from 0 to 2, at a step of 0.001.

```
let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 2.; dt=1E-3})
```

One last thing to solve the problem is of course the initial values:

```ocaml env=diffequation_example01
let y0 = Mat.of_array [|-1.; 1.|] 2 1
```

And finally we can provide all these information to the `rk4` solver in `Owl_ode` and get the answer:

```
let ts, ys = Owl_ode.Ode.odeint Owl_ode.Native.D.rk4 f x0 tspec ()

val ts : Owl_dense_matrix_d.mat =

   C0    C1    C2    C3    C4     C1996 C1997 C1998 C1999 C2000
R0  0 0.001 0.002 0.003 0.004 ... 1.996 1.997 1.998 1.999     2

val ys : Owl_dense_matrix_d.mat =

   C0       C1       C2       C3       C4        C1996    C1997    C1998    C1999    C2000
R0 -1   -1.002 -1.00399 -1.00599 -1.00798 ... -3.56302 -3.56451   -3.566 -3.56749 -3.56898
R1  1 0.995005 0.990022 0.985049 0.980088 ... -2.07436 -2.07527 -2.07617 -2.07707 -2.07798
```

The `rk4` solver is short for "forth-order Runge-Kutta Method" that we have introduced before.
The results shows both the steps $ts$ and the system values at each step $ys$. 
We can visualise the oscillation according to the result:

IMAGE

### Solver Structure

Hope that you have gotten the gist of how to use `Owl-ode`.
From these example, we can see that the `owl-ode` abstracts the initial value problems as four different parts:

1. a function $f$ to shows how the system evolves in equation $y'(t) = f(y, t)$;
2. a specification of the timespan;
3. system initial values;
4. and most importantly, a solver. 

If you look at the signature of a solver:

```
val rk4 : (module Types.Solver
    with type state = M.arr
    and type f = M.arr -> float -> M.arr
    and type step_output = M.arr * float
    and type solve_output = M.arr * M.arr)
```

it clear indicates these different parts.
Based on this uniform abstraction, you can choose a suitable solver and use it to solve many complex and practical ODE problems. 
Note that due to the difference of solvers, the requirement of different solver varies. 
Some requires the state to be two matrices, while others process data in a more general ndarray format.


Here is a table that lists all the solvers that are currently supported by `owl-ode`.

### Features and Limits 


Its functionality and limit.

The methods we have introduced are all included:
The interfaces

Install

Limit 
Explicit vs Implicit 

## Examples of using Owl-ODE

As with many good things in the world, Mastering solving ODE requires practice. 
After getting to know `owl-ode` in the previous section, in this section we will demonstrate more examples of using this tool.

### Explicit ODE

Now that we have this powerful tool, we can use the solver in `owl-ode` to solve the motivative problem in [@eq:diffequation:example01] with simple code.

```
let f y t = Mat.(2. $* y *$ t +$ t)

let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 1.; dt=1E-3})

let y0 = Mat.zeros 1 1

let solver = Owl_ode.Native.D.rk45 ~tol:1E-9 ~dtmax:10.0

let _, ys = Owl_ode.Ode.odeint solver f y0 tspec ()
```

The code is mostly similar to previous example, the only difference is that we can now try another solver provided: the `rk45` solver, with certain parameters specified. 
You don't have to worry about what the `tol` or `dtmax` means for now.
Note that this solver (and the previous one) requires input to be of type `mat` in Owl, and the function $f$ be of type `mat -> float -> mat`.
The result is as expected:

```
# Mat.transpose ys
- : Mat.mat =

   C0    C1          C2          C3          C4        C996    C997    C998    C999   C1000
R0  0 1E-06 4.00001E-06 9.00004E-06 1.60001E-05 ... 1.69667 1.70205 1.70744 1.71285 1.71828
```

(TODO: the result is not as expected from [@eq:diffequation:example01_solution] and previous manual solution. Find the reason)

### Another Explicit ODE

Options: logistic equation, or compound interest, or both.

### Two Body Problem

```
CODE
```

### Lorenz Attractor

```
CODE
```

## Stiffness

Explain stiff vs. non-Stiff

For some ODE problems, the step size taken by the solver is forced down to an unreasonably small level in comparison to the interval of integration, even in a region where the solution curve is smooth. These step sizes can be so small that traversing a short time interval might require millions of evaluations. This can lead to the solver failing the integration, but even if it succeeds it will take a very long time to do so.
Equations that cause this behaviour in ODE solvers are said to be stiff. (Copy alert)

**van der Pol Equation**

Sundails and odepack 

### Solve Stiff ODEs

example code and illustration

### Solve Non-stiff ODEs

example code and illustration

## Choose ODE solvers

Question: "why cannot I just use a 'best' solver for all the questions?"

Introduce various solvers in Owl-ODE and states their pros and cons. 

## Exercise

1. Implement `rk4` manually and apply to the same problem to compare it's effect.

## References
