# Differential Equations


## What Is An ODE

A differential equation is an equation for a function that relates the values of tfunction to the values of its derivatives. 
An *Ordinary Differential Equation* (ODE) is a differential equation that contains functions and their derivatives of a single variable. 

The *order* of an ODE is the highest order of derivatives included. 
A first-order differential equation can be generally written as:

$$\frac{dy}{dx}=f(x,y),$$

where $f$ is any function that contains $x$ and $y$.


### Exact Solutions 

Some differential equations have solutions that can be written in an exact and closed form.
We list 1-2 as example, but more can be seen in math textbooks.

**Separable equations**: 

$$P(y)\frac{dy}{dx} + Q(x) = 0,$$

and it's close form solution is:

$$\int^{y}P(y)dy + \int^{x}Q(x)dx = C$$.


**Linear first-order equations**:

$$\frac{dy}{dx} + P(x)y = Q(x)$$


## How To Solve An ODE Numerically

This chapter should be built around examples:

- make very clear what kind of problems we can solve at the beginning. Even MATLAB can only solve a certain kinds of ODEs, and that's no problem at all;
- a lot of examples 
- explain how these solvers work 
- finally, real problems: compound interest, chemical reactions etc. and solve them.

### Owl-ODE

TODO: how to use ODE; simple example of using Owl-ODE

### Choose ODE solvers

TODO: introduce various solvers in Owl-ODE

## Numerical Integration of ODE

REFER: NR book chapter 

(Could be removed if not yet implemented)

## A Concrete Example

### Solve Stiff ODEs

For some ODE problems, the step size taken by the solver is forced down to an unreasonably small level in comparison to the interval of integration, even in a region where the solution curve is smooth. These step sizes can be so small that traversing a short time interval might require millions of evaluations. This can lead to the solver failing the integration, but even if it succeeds it will take a very long time to do so.
Equations that cause this behaviour in ODE solvers are said to be stiff. (Copy alert)

REFER: matlab doc 

### Solve Non-stiff ODEs

REFER: matlab doc 
