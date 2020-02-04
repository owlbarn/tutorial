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

(NOTE: if we decide to add optimisation later, we can extend the constrained)


REFERENCE: Numerical Recipes

## Scalar Function Optimisation

### Golden Section Search 

### Brent's Method

without and with derivative

## Multivariate Function Optimisation

### Nelder-Mead Simplex Method

And also Powell's Method, since they both do not require gradient

### Gradient Descent Methods

Also include Conjugate Gradient Method. We can decide how to divide sections later.

### Quasi-Newton Methods

BFGS

## Root Finding

### Bisection

### Brent

### Newton Method


## Optimisation in Practice: Training Neural Network

Adagrad etc in action: they are all gradient descent algorithms.


NOTE: perhaps also add in global optimisation section. Linear Programming?...

Also, I need to at least explain what is hessian/jacobian etc. in this chapter.
