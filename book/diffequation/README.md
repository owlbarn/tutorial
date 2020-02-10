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

equation 

analytical solution

Now how can we solve it numerically?

The Euler method. Equations.

Why it works. 

Let's see an simple example:

```
CODE
```

Hmm, the result is ... kind of OK, but still large error, but the euler method does not provide an error estimate

Let's improve it a bit (Equations).

```
CODE
```

Let's see the result. Much better.

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
