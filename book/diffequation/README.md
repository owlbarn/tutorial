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

Solving a differential equation is often complex, but we do know how to solve part of them.
Before looking at the the computer solvers to a random ODEs, let's turn to the math first and look at some ODE forms that we already have analytical close-form solution to.

| ODE | Solution |
| :------------: |:---------------------------------- |
| $P(y)\frac{dy}{dx} + Q(x) = 0$ | $\int^{y}P(y)dy + \int^{x}Q(x)dx = C$ |
| $\frac{dy}{dx} + P(x)y = Q(x)$ | $y=e^{-\sum_{x_0}^xP(x)dx}(y_0 + \sum_{x_0}^xQ(x)e^{\sum_{x_0}^xP(x)dx}dx)$ |
: Examples of solutions to certain types of ODE {#tbl:diffequation:ode_solution}

The [@tbl:diffequation:ode_solution] shows two examples.
The first line is a type of ODEs that are called the "separable equations".
The second line represents the ODEs that are called the "linear first-order equations".
The solution to both form of ODE are already well-known, as shown in the second column.
Here $C$ is a constant decided by initial condition $x_0$ and $y_0$.

Note that in both types the derivative $dy/dx$ can be expressed explicitly as a function of $x$ and $y$, and therefore is called *explicit* ODE.
Otherwise it is called an *implicit* ODE.

High order ODEs can be reduced to the first order ones that contains only $y'$, $y$, and $x$.
For example, an ODE in the form $y^{(n)} = f(x)$ can be reduced by multiple integrations one both sizes.
If a two-order ODE is in the form $y^{''} = f(x, y')$, let $y' = g(x)$, then $y^{''} = p'(x)$. Put them into the original ODE, it can be transformed as: $p'=f(x,p)$.
This is a first-order ODE that can be solved by normal solutions.
Suppose we get $y'=p=h(x, C_0)$, then this explicit form of ODE can be integrated to get: $y = \int~h(x, C_0)dx + C_1$.

We have only scratch the surface of the ODE as traditional mathematics topic.
This chapter does not aim to fully introduce how to solve ODEs analytically or simplify high-order ODEs.
For those who interested, please refer to classical calculus books or courses.

(TODO: Explicit vs Implicit etc.: The three types of equations. This is important.)

### Linear Systems

ODEs are often used to describe various dynamic systems. In the previous examples there is only one function `y` that changes over time.
However, a real world system often contains multiple interdependent components, each can be described by a unique function that evolves over time.
In the next of this chapter, we will talk about several ODE examples in detail, such as the two-body problem and the Lorenz attractor.
For now, it suffices for us to look at [@eq:diffequation:twobody_system] and [@eq:diffequation:lorenz] in the sections below and see how they are different from the single-variant ODE so far.
For example, the Lorenz attractor system has three components that changes with time: the rate of convection in the atmospheric flow, the horizontal and vertical temperature variation.

These two systems are examples of what is called the *first-order linear system of ODE* or just the *linear system of ODE*. Generally, if we have:

$$\boldsymbol{y}(t) = \left[\begin{matrix}y_1(t) \\ \vdots \\ y_n(t) \end{matrix} \right],
\boldsymbol{A}(t) = \left[\begin{matrix}a_{11}(t) & \ldots & a_{1n}(t) \\ \vdots & \ldots & \vdots \\ a_{n1}(t) & \ldots & a_{nn}(t) \end{matrix} \right],
\textrm{and}
\boldsymbol{g}(t) = \left[\begin{matrix}g_1(t) \\ \vdots \\ g_n(t) \end{matrix} \right],
$$

then a linear system can be expressed as:

$$\boldsymbol{y'}(t) = \boldsymbol{A}(t)\boldsymbol{y}(t) + \boldsymbol{g}(t).$$
 {#eq:diffequation:linear-system}

This linear system contains $n$ time-dependent components: $y_1(t), y_2(t), \ldots, y_n(t)$.
As we will be shown soon, the first-order linear system is especially suitable for the numerical ODE solver to solve.
Therefore, transforming a high-order single-component ODE into a linear system is sometimes necessary, as we will show in the two body problem example.
But before we stride too far away, let's get back to the ground and start with the basics of solving an ODE numerically.

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
This method is really easy to be implemented in OCaml, as shown below.

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

![Comparing the accuracy of Euler method and Midpoint method in approximating solution to ODE](images/diffequation/plot_rk01.png "plot_rk01"){width=80% #fig:diffequation:plot_rk01}

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
Based on the computation functionalities and ndarray data structures in Owl, we provide the package "[owl_ode](https://github.com/owlbarn/owl_ode)" to perform the tasks of solving the initial value problems.
Without further due, let's see it how the `owl-ode` package can be used to solve ODE problem.

### Example: Linear Oscillator System

EXPLAIN: how the equation of Oscillator becomes this linear representation.

This oscillation system appears frequently in Physics and several other fields: charge flow in electric circuit, sound wave, light wave, etc.
These phenomena all follow the similar pattern of ODEs.
One example is the mass on a spring. 

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


### Features and Limits

TODO: *EXPLAIN what is symplectic solver and how it can be fit into existing framework*.

`Owl-ode` provides a wide range to solvers. It implements native solvers and symplectic solvers which are based on the step-by-step update basic idea we have discussed.
Currently there are already many mature off-the-shelf tools for solving ODEs, we choose two of them: [sundials]((https://computing.llnl.gov/projects/sundials),) and [ODEPACK](https://computing.llnl.gov/casc/odepack/).
Both methods are well implemented and widely used in practical use. (TODO: more information.)

- `sundials`: a SUite of Nonlinear and DIfferential/ALgebraic equation Solvers. It contains six solvers, and we interface to its `CVODE` solver for solving initial value problems for ordinary differential equation systems.

- `odepack`: ODEPACK is a collection of Fortran solvers for the initial value problem for ordinary differential equation systems. We interface to its LSODA solver which is for solving the explicit form ODE.

For all these solvers, `owl-ode` provides an easy-to-use unified interface, as you have seen in the examples.
[@tbl:diffequation:solvers] is a table that lists all the solvers that are currently supported by `owl-ode`.

| Solvers | Type | State | Function | Step | Note |
| ------- | ---- | ----- | -------- | ---- | ---- |
| `rk4`   | Native | `M.arr` | `M.arr -> float -> M.arr` | `M.arr * float` | |

: Solvers provided by owl-ode and their types. {#tbl:diffequation:solvers}

**Automatic inference of state dimensionality**

(COPY ALERT)

All the provided solvers automatically infer the dimensionality of the state from the initial state. Consider the Native solvers, for which the state of the system is a matrix. The initial state can be a row vector, a column vector, or a matrix, so long as it is consistent with that of %f$. If the initial state $y_0$ is a row vector with dimensions 1xN and we integrate the system for $T$ time steps, the time and states will be stacked vertically in the output (i.e. `ts` will have dimensions `Tx1` and and `ys` will have dimensions `TxN`). On the contrary, if the initial state %y_0$ is a column vector with dimensions, the results will be stacked horizontally (i.e. $ts$ will have dimensions `1xT` and $ys$ will have dimensions `NxT`).

We also support temporal integration of matrices. That is, cases in which the state $y$ is a matrix of dimensions of dimensions `NxM`. By default, in the output, we flatten and stack the states vertically (i.e., ts has dimensions Tx1 and xs has dimensions TxNM. We have a helper function `Native.D.to_state_array` which can be used to pack $ys$ into an array of matrices.

**Custom Solvers**

We can define new solver module by creating a module of type Solver. For example, to create a custom Cvode solver that has a relative tolerance of 1E-7 as opposed to the default 1E-4, we can define and use `custom_cvode` as follows:

```
let custom_cvode = Owl_ode_sundials.cvode ~stiff:false ~relative_tol:1E-7 ~abs_tol:1E-4
(* usage *)
let ts, xs = Owl_ode.Ode.odeint custom_cvode f x0 tspec ()
```

Here, we use the `cvode` function construct a solver module `Custom_Owl_Cvode`.
Similar helper functions like cvode have been also defined for native and symplectic solvers.

**Multiple Backends**

The owl-ode-base contains implementations that are purely written in OCaml. As such, they are compatible for use in Mirage OS or in conjunction with js_of_ocaml, where C library linking is not supported.

**Limit**

Note that currently the `owl-ode` is still at development phase. Due to lack of vector-valued root finding functions, it is limited to solving initial value problems for the explicit ODE of form $y' = f(y, x)$.

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

### Damped Oscillation

(TODO: check if the section title is suitable)

TODO: explain the problem; explain why the symplectic solver is used.

```
let damped_noforcing a (xs, ps) _ : Owl.Mat.mat =
  Owl.Mat.((xs *$ -1.0) + (ps *$ (-1.0 *. a)))


let a = 1.0
let dt = 0.1

let plot_sol fname t sol1 sol2 sol3 =
  let open Owl in
  let h = Plot.create fname in
  let open Plot in
  set_foreground_color h 0 0 0;
  set_background_color h 255 255 255;
  set_title h fname;
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1 ] t (Mat.col sol1 0);
  plot ~h ~spec:[ RGB (0, 255, 0); LineStyle 1 ] t (Mat.col sol2 0);
  plot ~h ~spec:[ RGB (255, 0, 0); LineStyle 1 ] t (Mat.col sol3 0);
  legend_on h ~position:NorthEast [| "Leapfrog"; "Ruth3"; "Symplectic Euler" |];
  output h


let () =
  let x0 = Owl.Mat.of_array [| -0.25 |] 1 1 in
  let p0 = Owl.Mat.of_array [| 0.75 |] 1 1 in
  let t0, duration = 0.0, 15.0 in
  let f = damped_noforcing a in
  let tspec = T1 { t0; duration; dt } in
  let t, sol1, _ = Ode.odeint (module Symplectic.D.Leapfrog) f (x0, p0) tspec () in
  let _, sol2, _ = Ode.odeint Symplectic.D.ruth3 f (x0, p0) tspec () in
  let _, sol3, _ =
    Ode.odeint (module Symplectic.D.Symplectic_Euler) f (x0, p0) tspec ()
  in
  plot_sol "damped.png" t sol1 sol2 sol3
```

IMAGE

### Two Body Problem

In classical mechanics, the *two-body problem* is to predict the motion of two massive objects. It is assumed that the only force that are considered comes from each other, and both objects are not affected by any other object.
This problem can be seen in the astrodynamics where the objects of interests are planets, satellites, etc. under the influence of only gravitation.
Another case is the trajectory of electron around Atomic nucleus in a atom.

This classic problem is one of the earliest investigated mechanics problems, and was long solved from the age of Newton. It is also a typical integrable problem in classical mechanics.
In this example, let's consider a simplified version of this problem.
We assume that the two objects interact on a 2-dimensional plane, and one of them is so much more massive than the other one that it can be thought of as being static (think about electron and nucleus) and sits at the zero point of a Cartesian coordinate system (0, 0) in the plane.
In this system, let's consider the trajectory of the lighter object.
This "one-body" problem is basis of the two body problem. For many forces, including gravitational ones, a two-body problem can be divided into a pair of one-body problems.

Given the previous assumption and newton's equation, it can be [proved](https://people.sc.fsu.edu/~jburkardt/m_src/two_body_simulation/two_body_simulation.html) that the location of the lighter object [$y_0$, $y_1$] with regard to time $t$ can be described by:

$$y_0^{''}(t) = -\frac{y_0}{r^3},$$
$$y_1^{''}(t) = -\frac{y_1}{r^3},$$ {#eq:diffequation:twobody}

where $r=\sqrt{y_0^2 + y_1^2}$.
These are a second-order ODEs, and to make it solvable using our tool, we need to make them into a first-order explicit ordinary differential equation system:

$$y_0^{'} = y_2,$$
$$y_1^{'} = y_3,$$
$$y_2^{'} = -\frac{y_0}{r^3},$$ {#eq:diffequation:twobody_system}
$$y_3^{'} = -\frac{y_1}{r^3},$$

Based on [@eq:diffequation:twobody_system], we can build up our code as below:

```
let f y _t =
  let y = Mat.to_array y in
  let r = Maths.(sqrt ((sqr y.(0)) +. (sqr y.(1)))) in
  let y0' = y.(2) in
  let y1' = y.(3) in
  let y2' = -.y.(0) /. (Maths.pow r 3.) in
  let y3' = -.y.(1) /. (Maths.pow r 3.) in
  [| [|y0'; y1'; y2'; y3'|] |] |> Mat.of_arrays

let y0 = Mat.of_array [|-1.; 0.; 0.5; 0.5|] 1 4
let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 20.; dt=1E-2})
let custom_solver = Native.D.rk45 ~tol:1E-9 ~dtmax:10.0
```

Here the `y0` provides initial status of the system: first two numbers denote the initial location of object, and the next two numbers indicate the initial momentum to this object. (TODO: check if this is true or a better word should be used.)
After building the function, initial status, timespan, and solver, we can then solve the system and visualise it.

```
let _ =
  let ts, ys = Ode.odeint custom_solver f y0 tspec () in
  let h = Plot.create "two_body.png" in
  let open Plot in
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1 ] (Mat.col ys 0) (Mat.col ys 1);
  scatter ~h ~spec:[ Marker "#[0x229a]"; MarkerSize 5. ] (Mat.zeros 1 1) (Mat.zeros 1 1);
  text ~h ~spec:[ RGB (51,51,51)] (-.0.3) 0. "Massive Object";
  output h
```

![The trajectory of lighter object orbiting the massive object in a simplified two-body problem](images/diffequation/two-body.png "two-body"){ width=80% #fig:diffequation:two-body }

One example of this simplified two-body problem is the "planet-sun" system where a planet orbits the sun.
Kepler's law states that in this system the planet goes around the sun in an ellipse shape, with the sun at a focus of the ellipse.
The orbiting trajectory in the result visually follows this theory.


### Lorenz Attractor

Lorenz equations are one of the most thoroughly studied ODEs.
This system of ODEs is proposed by Edward Lorenz in 1963 to model flow of fluid (the air in particular) from hot area to cold area.
Lorenz simplified the numerous atmosphere factors into the simple equations below.

$$x'(t) = \sigma~(y(t)- x(t))$$
$$y'(t) = x(t)(\rho - z(t)) - y(t)$$ {#eq:diffequation:lorenz}
$$z'(t) = x(t)y(t) - \beta~z(t)$$

Here $x$ is proportional to the rate of convection in the atmospheric flow, $y$ and $z$ are proportional to the horizontal and vertical temperature variation.
Parameter $\sigma$ is the Prandtl number, and $\rho$ is the normalised Rayleigh number.
$\beta$ is related to the geometry of the domain.
The most commonly used parameter values are: $\sigma = 10, \rho=20$, and $\beta = \frac{8}{3}$.
Based on these information, we can use `owl-ode` to express the Lorenz equations with code.

```ocaml
let sigma = 10.
let beta = 8. /. 3.
let rho = 28.

let f y _t =
  let y = Mat.to_array y in
  let y0' = sigma *. (y.(1) -. y.(0)) in
  let y1' = y.(0) *. (rho -. y.(2)) -. y.(1) in
  let y2' = y.(0) *. y.(1) -. beta *. y.(2) in
  [| [|y0'; y1'; y2'|] |] |> Mat.of_arrays
```

We set the initial values of the system to `-1`, `-1`, and `1` respectively.
The simulation timespan is set to 30 seconds, and keep using the `rk45` solver.

```
let y0 = Mat.of_array [|-1.; -1.; 1.|] 1 3
let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 30.; dt=1E-2})
let custom_solver = Native.D.rk45 ~tol:1E-9 ~dtmax:10.0
```

Now, we can solve the ODEs system and visualise the results.
In the plots, we first show how the value of $x$, $y$ and $z$ changes with time; next we show the phase plane plots between each two of them.

```
let _ =
  let ts, ys = Ode.odeint custom_solver f y0 tspec () in
  let h = Plot.create ~m:2 ~n:2 "lorenz_01.png" in
  let open Plot in
  subplot h 0 0;
  set_xlabel h "time";
  set_ylabel h "value on three axes";
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1 ] ts (Mat.col ys 2);
  plot ~h ~spec:[ RGB (219, 68,  55); LineStyle 1 ] ts (Mat.col ys 1);
  plot ~h ~spec:[ RGB (244, 180,  0); LineStyle 1 ] ts (Mat.col ys 0);
  subplot h 0 1;
  set_xlabel h "x-axis";
  set_ylabel h "y-axis";
  plot ~h ~spec:[ RGB (66, 133, 244) ] (Mat.col ys 0) (Mat.col ys 1);
  subplot h 1 0;
  set_xlabel h "y-axis";
  set_ylabel h "z-axis";
  plot ~h ~spec:[ RGB (66, 133, 244) ] (Mat.col ys 1) (Mat.col ys 2);
  subplot h 1 1;
  set_xlabel h "x-axis";
  set_ylabel h "z-axis";
  plot ~h ~spec:[ RGB (66, 133, 244) ] (Mat.col ys 0) (Mat.col ys 2);
  output h
```

![Three components and phase plane plots of Lorenz attractor](images/diffequation/lorenz_01.png "lorenz_01"){ width=100% #fig:diffequation:lorenz_01 }

From [@fig:diffequation:lorenz_01], we can image that the status of system keep going towards two "voids" in a three dimensional space, jumping from one to the other.
These two voids are a certain type of *attractors* in this dynamic system, where a system tends to evolve towards.

Now, about Lorenz equation, there is an interesting question: "what would happen if I change the initial value slightly?"
For some systems, such as a pendulum, that wouldn't make much a difference, but not here. We can see that clearly in Owl.
Keep function and timespan the same, let's change only 0.1% of initial value and then solve the system.

```
let y00 = Mat.of_array [|-1.; -1.; 1.|] 1 3
let y01 = Mat.of_array [|-1.001; -1.001; 1.001|] 1 3
let ts0, ys0 = Ode.odeint custom_solver f y00 tspec ()
let ts1, ys1 = Ode.odeint custom_solver f y01 tspec ()
```

To make later calculation easier, we can make the two resulting matrices to be of the same shape using slicing.

```
let r0, c0 = Mat.shape ys0
let r1, c1 = Mat.shape ys1
let r  = if (r0 < r1) then r0 else r1
let ts = if (r0 < r1) then ts0 else ts1
let ys0 = Mat.get_slice [[0; r-1]; []] ys0
let ys1 = Mat.get_slice [[0; r-1]; []] ys1
```

Now, we can compare the euclidean distance between the status of these two systems at certain time. Also, we shows the value change of three components with time after changing initial values.

```
let _ =
  (* plot the distance between two systems *)
  let h = Plot.create ~m:1 ~n:2 "lorenz_02.png" in
  let open Plot in
  subplot h 0 0;
  set_xlabel h "time";
  set_ylabel h "value on three axes";
  plot ~h ~spec:[ RGB (244, 180,  0); LineStyle 1 ] ts (Mat.col ys1 0);
  plot ~h ~spec:[ RGB (219, 68,  55); LineStyle 1 ] ts (Mat.col ys1 1);
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1 ] ts (Mat.col ys1 2);
  subplot h 0 1;
  let diff = Mat.(
    sqr ((col ys0 0) - (col ys1 0)) +
    sqr ((col ys0 1) - (col ys1 1)) +
    sqr ((col ys0 2) - (col ys1 2))
    |> sqrt
  )
  in
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1 ] ts diff;
  set_xlabel h "time";
  set_ylabel h "distance of two systems";
  output h
```

![Change the initial states on three dimension by only 0.1%, and the value of Lorenz system changes visibly.](images/diffequation/lorenz_02.png "lorenz_02"){ width=100% #fig:diffequation:lorenz_02 }

According to [@fig:diffequation:lorenz_02], the first figure shows that, initially the systems looks quite like that in [@fig:diffequation:lorenz_01], but after about 15 seconds, the system state begins to change.
This change is then quantified using the euclidean distance between these two systems.
Clearly the difference two system changes sharply after a certain period of time, with no sign fo converge.
You can try to extend the timespan longer, and the conclusion will still be similar.

This result shows that, in the Lorenz system, even a tiny bit of change in the initial state can lead to a large and chaotic change of future state after a while.
It partly explains why weather prediction is difficult to do: you can only accurately predict the weather for a certain period of time, any day longer and the weather will be extremely sensitive to a tiny bit of perturbations at the beginning, such as ..., well, such as the flapping of the wings of a distant butterfly several weeks earlier.
You are right, the Lorenz equation is closely related to the idea we now call "butterfly effect" in the pop culture.

## Stiffness

*Stiffness* is an important concept in the numerical solution of ODE.
Think about a function that has a "cliff" where at a point its nearby value changes rapidly.
Therefore, to find the solution with normal method as we have used, it may requires such a extremely small stepping size that traversing the whole timespan may takes a very long time and lot of computation.

TODO: the very basic idea of stiff algorithms.

The **Van der Pol equation** is a good example to show both non-stiff and stiff cases.
In dynamics, the Van der Pol oscillator is a non-conservative oscillator with non-linear damping.
Its behaviour with time can be described with a high order ODE:

$$y^{''} - \mu~(1-y^2)y' + y = 0,$$ {#eq:diffequation:vanderpol_0}

where $\mu$ is a scalar parameter indicating the non-linearity and the strength of the damping.
To make it solvable using our tool, we can change it into a pair of explicit one-order ODEs in a linear system:

$$y_0^{'} = y_1,$$
$$y_1^{'} = \mu~(1-y_0^2)y_1 - y_0.$$ {#eq:diffequation:vanderpol_1}

As we will show shortly, by varying the damping parameter, this group of equations can be either non-still or stiff.

We provide both stiff (`Owl_Cvode_Stiff`) and non-still (`Owl_Cvode`) solver by interfacing to Sundials, and the `LSODA` solver of ODEPACK can automatically switch between stiff and non-stiff algorithms.
We will try both in the example.

Here we start with the basic function code that are shared by both cases.

```
open Owl
open Owl_ode
open Owl_ode.Types
open Owl_plplot

let van_der_pol mu =
  fun y _t ->
    let y = Mat.to_array y in
    [| [| y.(1); mu *. (1. -. Maths.sqr y.(0)) *. y.(1) -.y.(0) |] |]
    |> Mat.of_arrays
```


### Solve Non-Stiff ODEs

When set the parameter to 1, the equation is a normal non-stiff one, and let's try to use the `Cvode` solver from sundials to do this job.

```
let f_non_stiff = van_der_pol 1.

let y0 = Mat.of_array [| 0.02; 0.03 |] 1 2

let tspec = T1 { t0 = 0.0; dt = 0.01; duration = 30.0 }

let ts, ys = Ode.odeint (module Owl_ode_sundials.Owl_Cvode) f_stiff y0 tspec ()
```

Everything seems normal. To see the "non-stiffness" clearly, we can plot how the two system states change over time, and a phase plane plot of their trajectory on the plane, using the two states as x- and y-axis values.
The result is shown in [@fig:diffequation:nonstiff].

```
let () =
  let fname = "vdp_sundials_nonstiff.png" in
  let h = Plot.create ~n:2 ~m:1 fname in
  let open Plot in
  set_foreground_color h 0 0 0;
  set_background_color h 255 255 255;
  subplot h 0 0;
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1 ] (Mat.col ys 0) (Mat.col ys 1);
  subplot h 0 1;
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1 ] ts Mat.(col ys 1);
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 3 ] ts Mat.(col ys 0);
  output h

```

![Solving Non-Stiff Van der Pol equations with Sundial CVode solver](images/diffequation/vdp_sundials_nonstiff.png "vdp_sundials_nonstiff"){width=100% #fig:diffequation:nonstiff}

### Solve Stiff ODEs

Change the parameters to 1000, and now this function becomes *stiff*.
We follow the same procedure as before, but now we use the `Lsoda` solver from odepack, and the timespan is extended to 3000.
From [@fig:diffequation:stiff] we can see clearly what "stiff" means.
Both lines in this figure contain very sharp "cliffs".

```
let f_stiff = van_der_pol 1000.
let y0 = Mat.of_array [| 2.; 0. |] 1 2
let tspec = T1 { t0 = 0.0; dt = 0.01; duration = 3000.0 }

let () =
  let ts, ys = Ode.odeint (module Owl_ode_odepack.Lsoda) f_stiff y0 tspec () in
  let fname = "vdp_odepack_stiff.png" in
  let h = Plot.create fname in
  let open Plot in
  set_foreground_color h 0 0 0;
  set_background_color h 255 255 255;
  set_yrange h (-2.) 2.;
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1 ] ts Mat.(col ys 1);
  plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 3 ] ts Mat.(col ys 0);
  output h

```

![Solving Stiff Van der Pol equations with ODEPACK LSODA solver.](images/diffequation/vdp_odepack_stiff.png "vdp_odepack_stiff"){ width=70% #fig:diffequation:stiff }

## Choose ODE solvers

Question: "why cannot I just use a 'best' solver for all the questions?"

TODO: Introduce the basic principles of how to choose solvers


## Summary

## References
