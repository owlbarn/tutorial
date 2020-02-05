# Optimisation

(The basic idea of this chapter: give a general introduction of this topic regardless of Owl; when Owl has corresponding function, we provide a simple example.)

## Introduction 

Mathematical optimisation deals with the problem of finding numerically minimums, maximums (or zeros) of a function. An optimisation problem has the form:

$$\textrm{minimise} f_0(\mathbf{x}),$$
$$\textrm{subject to} f_i(\mathbf{x}) \leq b_i, i = 1, 2, \ldots, m. $$ {#eq:optimisation:def}

Here $\mathbf{x}$ is a vector that contains all the *optimisation variable*: $\mathbf{x} = [x_0, x_1, ... x_n]$. Function $f_0 : \mathbf{R}^n \rightarrow \mathbf{R}$ is the optimisation target, and is called an *objective function*, or *cost function*.
A optimisation problem could be bounded by zero or more *constraints*. $f_i : \mathbf{R}^n \rightarrow \mathbf{R}$ in a constraint is called a *constraint function*, which are bounded by the $b_i$'s.
The target is to find the optimal variable values $\mathbf{x}^{*}$ so that $f_0$ can take on a maximum or minimum value.

A optimisation problem formalises the idea "maximum benefit/minimise cost with given constraint", which is a widely applicable topic in many real world problems: scheduling computation/network resources, optimisation of investment portfolio, fitting math model based on observed data, logistics, aero engineering, competitive games...
Optimisation has already been applied in many areas.

In [@eq:optimisation:def], if for all the objective function and constraint function, we have:

$$ f_i(\alpha~x+\beta~y) = \alpha~f_i(x) + \beta~f_i(y),$$ {#eq:optimisation:linear}

the optimisation problem is then called *linear optimisation*. It is an important class of optimisation problems. 
If we change the "$=$" to "$\leq$" in [@eq:optimisation:linear] make all the functions to be *convex*, and the problem then becomes *convex optimisation*, which can be seen as a generalised linear optimisation.

Linear optimisation is important because non-negativity is a usual constraint on real world quantities, and that people are often interested in additive bounds. Besides, many problems can be approximated by a linear model. 
Though still limited by actual problem size, the solution of most linear optimisation problems are already known and provided by off-the-shelf software tools. 
The text book [@boyd2004convex] focus exclusively on the topic of convex optimisation.

Compare to linear optimisation, finding solutions to *non-linear optimisation* problems are still very challenging.
Finding a *global* solution that maximise or minimise the non-linear objective function can be quite time-consuming, even for only a small set of variables. Therefore, global optimisation of a non-linear problem is normally only used when absolutely necessary.
For example, if a system pressure test is modelled as an optimisation problem, given a small number of variants in the system, and a global extreme value has to find to test if the system is robust enough. 
Otherwise, a *local* maximum or minimum is normally used instead as an approximation. In most engineering applications, a local extreme value is good enough.
Though optimisation cannot promise a true extremism, and is easily affected by algorithm parameters and initial guess in iterative algorithms, as a trade-off, local optimisation is much faster and thus still widely used.

Looking back at [@eq:optimisation:def], if we remove the constraints, then it becomes an *unconstrained optimisation* problem. 
If $f$ is convex and differentiable, this problem can be seen as finding the root of the derivative of $f$ so that $f'(x^*) = 0$.
As for the constrained version, we have introduced the linear programming where all the functions are linear. There are also other types of optimisations such as quadratic programming, semi-definite programming, etc. 
One subset of constrained optimisation, the *equality constrained optimisation* where all the constraints are expressed in the form of equality $Ax=b$. This set of problem can be simplified into the corresponding unconstrained problems.

You can see that the topic of optimisation covers a wide range of topics and we can only give a very brief introduction here. 
In this chapter, we mostly cover the unconstrained and local optimisation. 
We will cover the other more advanced content briefly in the end of this chapter, and refer readers to classic books such as [@boyd2004convex] and [@fletcher2013practical] for more information.

(NOTE: if we decide to add linear programming later, we can extend the constrained)


(NOTE: We need a lot of illustrations to show the gradient process)

## Numerical Differentiation VS. Algorithm Differentiation

The idea of derivative/gradient will be used extensively in optimisation. 

We have talked about Algodiff in the previous chapter. 

What is numerical differentiation? It's simple ...

Owl has provided numerical differentiation. It's close to the interface of that of Algodiff:

```
val diff : (elt -> elt) -> elt -> elt

val grad : (arr -> elt) -> arr -> arr

```

So what's their difference? What's the pro's and con's?

EXAMPLE to show the accumulating round-off/truncating error.

For the rest of this chapter, whenever we need to compute gradient/derivatives, we prefer Algodiff, but of course you can also use the numerical differentiation.

## Root Finding

(Reference: Matlab NC book.)

Root Finding is not exactly optimisation, however, the two topics are closely related. Both need to find target on a function iteratively.

Understanding the method used in root finding is very helpful for understanding the method in optimisation.

### Newton, Secant, and IQI

First, bisect method. 

In Newton, we use algodiff to compute the derivative; actually we can use the owl example here.

If derivative is not available, Secant replaces the derivative evaluation in Newton. 

In Secant, we use two points to get to the next one; IQI, we use three.

### Brent's Method

It's a combination of those three. 
Generally considered the best of the root-finding routines.

We implement it in Owl (put some Code here..?)

It is also what we use in the examples in "Math" chapter.

## Scalar Function Optimisation

Now after we understand the root-finding, let's look at optimisation problem. 
Let's start with the simple case that only one variable in the objective function.

### Use Derivative

Extreme value can be found where the derivatives equals 0:

$$f'(x) = 0$$

Let's use algodiff to do that. 

A simple example.

### Golden Section Search

But what if derivative is not available?

It's an optimisation method that does NOT require computing derivative.

Basic idea: in root-finding, you move two number to locate the zero point. Here you need three. And Golden search is an efficient way to do that. 

If your function has a discontinuous first or second derivative, then use this.

## Multivariate Function Optimisation

When things become more complex...

### Nelder-Mead Simplex Method

Gradient is the popular way, but first, let's briefly look at the method that does not require gradient.

Also mention Powell's Method if we have space left

### Gradient Descent Methods

Find the zero point: Gradient Descent.

### Conjugate Gradient Method

### Quasi-Newton Methods

BFGS

## Global Optimisation and Constrained Optimisation

So far we have talked about unconstrained optimisation, mostly to find local optimal.
In the rest of this chapter we will give a very very brief introduction to global optimisation and constrained optimisation 

The basic idea of global optimisation.

The type of problems covered constrained optimisation; applications. Currently can they be solved and how to solve them with existing tools.
