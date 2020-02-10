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

## Solving An ODE Numerically

This chapter should be built around examples:

- make very clear what kind of problems we can solve at the beginning. Even MATLAB can only solve a certain kinds of ODEs, and that's no problem at all;
- a lot of examples 
- explain how these solvers work 
- finally, real problems: compound interest, chemical reactions etc. and solve them. From textbooks; compare with analytical solutions.


### Basic Methods

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
$$ s_2 = f(x_n + h/2, yn + s_1~h/2),$$ {#eq:diffequation:rk2}
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

![Comparing the accuracy of Euler method and Midpoint method in approximating solution to ODE](images/diffequation/plot_rk01.png){widht=80%, #fig:diffequation:rk01}

This is called a "midpoint" method.
It has error estimation.


Obviously, we don't have to stop here. We can go to use four steps:
Equations. 
We leave the implementation as exercise. 


What you have seen is the the *Runge-Kutta Method*.
Its benefit: basic but competitive with other methods; stable and always succeeds; a good choice especially when computing $f$ is not expensive. 

More advanced methods. 

`ODE45`: explain how this method works; it's always a good first choice.

Bulirsch-Stoer: introduce briefly with one paragraph.
We are not going to introduce them any further. Refer to NR book.
Now it's finally the time we use some tools. 


### Owl-ODE

A general introduction of Owl-ODE. Its functionality and limit.

The methods we have introduced are all included. 

Install

TODO: how to use ODE 

One simple example

### Choose ODE solvers

Question: "why cannot I just use a 'best' solver for all the questions?"

Introduce various solvers in Owl-ODE with examples to show their pros and cons. 

## Solvers in Action

Examples. A LOT of examples.

Explain stiff vs. non-Stiff

### Solve Stiff ODEs

For some ODE problems, the step size taken by the solver is forced down to an unreasonably small level in comparison to the interval of integration, even in a region where the solution curve is smooth. These step sizes can be so small that traversing a short time interval might require millions of evaluations. This can lead to the solver failing the integration, but even if it succeeds it will take a very long time to do so.
Equations that cause this behaviour in ODE solvers are said to be stiff. (Copy alert)

REFER: matlab doc 

### Solve Non-stiff ODEs

REFER: matlab doc 

## Exercise

1. Implement `rk4` manually and apply to the same problem to compare it's effect.

## References
