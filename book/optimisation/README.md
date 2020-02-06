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

The derivative/gradient will be used extensively in solving optimisation problems. Therefore, it would do no harm to start this chapter with understanding the difference of the two ways to compute derivatives: *algorithm differentiation* and *numerical differentiation*.

We have talked about algorithm differentiation* in detail in the previous chapter.
What is this numerical differentiation then? 
It's actually simple according to the definition of derivative itself:

$$f'(x) = \lim_{\delta~\to~0}\frac{f(x+\delta) - f(x)}{\delta}.$$ {#eq:optimisation:numdiff}

This method is pretty easy to follow: evaluate the given $f$ at point $x$, and then choose a suitable small amount $\delta$, add it to the original $x$ and then re-evaluate the function. Then the derivative can be calculated using [@eq:optimisation:numdiff]. 
We can implement this method easily using OCaml:

```ocaml
let _eps = 0.00001

let diff f x = (f (x +. _eps) -. f (x -. _eps)) *. _eps
```

We can apply it to a simple case:

```
CODE
```

Looks good. 

Owl has provided numerical differentiation. It's close to the interface of that of Algodiff:

```
val diff : (elt -> elt) -> elt -> elt
(** derivative of ``f : scalar -> scalar``. *)

val diff2 : (elt -> elt) -> elt -> elt
(** second order derivative of ``f : float -> float``. *)

val grad : (arr -> elt) -> arr -> arr
(** gradient of ``f : vector -> scalar``. *)

val jacobian : (arr -> arr) -> arr -> arr
(** jacobian of ``f : vector -> vector``. *)

val jacobianT : (arr -> arr) -> arr -> arr
(** transposed jacobian of ``f : vector -> vector``. *)
```

Looks nice, much easier than Algodiff's approach, right?

No. 

There are two source of errors: truncating error (explain) and roundoff error (explain). You must be very careful and apply some numerical techniques. 
While Algodiff guarantees a true derivative value without loss of accuracy.
you can see the difference in this example:

```
CODE (how to show the difference)?
```

For the rest of this chapter, we prefer to use the algorithm differentiation to compute gradient/derivatives when required, but of course you can also use the numerical differentiation.

## Root Finding

We have seen some examples of root finding in the Math chapter.
*Root finding* is the process by which to find zeroes or *roots* of continuous functions. 
It is not an optimisation problem, but these two topics are closely related.
I would be beneficial for users to learn about the methods used in optimisation if they understand how the root finding algorithm work, e.g. how to the root by bracketing and how to find target in an iterative manner.

### Bisect, Newton, Secant, and IQI

First, the Bisection method. Use $\sqrt{2}$ as an example, just show the string of number here:$1\frac{1}{2}, 1\frac{1}{4}, 1\frac{3}{8}, 1\frac{5}{16} \ldots$. (DETAIL)
Owl provides `Owl_maths_root.bisec` method. (NOTE: we can have a example or even paste the Owl impl. here if we want to beef this section up, but let's keep thing concise for now.)
This method converges slowly, but it is a solid and reliable method.

Newton method utilises the derivative of objective function $f$. It starts with a initial value $x_0$, and follows this process:

$$x_{n+1} = x_{n} - \frac{f(x_n)}{f'(x_n)}.$$ {#eq:optimisation:newton}

We can use the Algorithm Differential module in Owl to do that.
In the next example we find the root of $x^2 - 2 = 0$, i.e., find an approximate value of $\sqrt{2}$.
The Owl code is just a plain translation of [@eq:optimisation:newton].

```ocaml
open Algodiff.D

let f x = Maths.(x ** (F 2.) - (F 2.))

let _ = 
	let x = ref 1. in
	for _ = 0 to 6 do
		let g = diff f (F !x) |> unpack_elt in 
		let v = f (F !x) |> unpack_elt in 
		x := !x -. v /. g;
		Printf.printf "%.15f\n" !x
	done
```

The resulting sequence is very short compared to the bisection method:
```
1.500000000000000
1.416666666666667
1.414215686274510
1.414213562374690
1.414213562373095
1.414213562373095
1.414213562373095
```

The Newton method can is efficient: it is quadratic convergence which means the square of the error at one iteration is proportional to the error at the next iteration. 
It is the basis of many powerful numerical methods (such as?)

If $f$ is not smooth or computing derivative is not always available, we need to approximate the tangent at one point with a secant through two points. This is called a *Secant Method*:

$$ f'(x) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}.$${#eq:optimisation:secant}

In the Secant method, two points are used at each iteration, the *Inverse Quadratic Interpolation* (IQI) method then uses three points to do that. 
DETAIL.
Benefit/Limit.

### Brent's Method

The *Brent's Method* is generally considered the best of the root-finding routines.
It combines the robustness of Bisection methods, and the iteration speed of Secant and IQI methods. 
The idea is to use the fast algorithm if possible, and turn to the slow but reliable method when in doubt.

A description of this method. 

This method is what we use in the examples in "Math" chapter.
We have also implemented it in Owl. (Paste the code if necessary, but for now just keep the root-finding section short)

## Univariate Function Optimisation

Now that we have briefly introduced how root-finding works and some classic methods, let's move on to the main topic of this chapter: unconstrained optimisation problems. 
Let's start with the simple case that only one variable in the objective function.
We will introduce the optimisation methods for multivariate functions in the next section, and they all apply for the univariate case, but the specific algorithms can work faster. Besides, understanding the optimisation of univariate functions can be a good step before getting to know the multivariate ones.

### Use Derivatives

If a function is continuous and differentiable, then one obvious solution to find extreme values is to locate where the derivatives equals 0:

$$f'(x) = 0$$

This leads us back to our root finding solutions. 
Let's look at an example:

```
CODE (with Algodiff) with explanation/images.
```

### Golden Section Search

Here we face the similar question again: what if computing derivative of the function is difficult or not available? (NOTE: give specific examples.)

There is a close analogue of bisection method in solving optimisation problems: *Gold Section Search*.
It's an optimisation method that does not require computing derivative.
It is one choice to do optimisation if your function has a discontinuous first or second derivative.

Explain the basic idea: in root-finding, you move two number to locate the zero point. Here you need three. And Golden search is an efficient way to do that. 

The Brent method in root finding can also be applied here. 
EXPLAIN

## Multivariate Function Optimisation

The methods for univariate scenarios can be extended to solving multivariate optimisation problems. 
The analogue of derivative here is a ndarray called *gradient*.
Similarly, you have two options: to use gradient, or not.

### Nelder-Mead Simplex Method

First, similar to the Golden Section Search or Brent's, you can always opt for a non-gradient method, which is as slow as it is robust.
One such method we can use is the *Nelder-Mead Simplex Method*. 
As its name shows, it is probably the simplest way to minimize a fairly well-behaved function.
It simply goes downhill in a straightforward way, without special assumptions about the objective function. 

EXPLAIN the algorithm in detail, perhaps with illustration.

There are some other method that does not rely on computing gradients such as Powell's method.
If the function is kind of smooth, this method can find the direction in going downhill, but instead of computing gradient, it relies on a one-dimensional optimisation method to do that, and therefore faster than the simplex method. 
We will not talk about this method in detail.

### Gradient Descent Methods

Compared to the previous solutions, gradient descent is one of the most widely used algorithms to perform optimisation and the most common way to optimize neural networks (we will talk about it in the Neural Network chapter). 

A *descent method* is an iterative optimisation process.
The idea is to start from a initial value, and then find a certain *search direction* along a function to decrease the value by certain *step size* until it converges to a local minimum. 
This process can be illustrated in [@fig:optimisation:gradient].

![Reach the local minimum by iteratively moving downhill](images/optimisation/gradient.png){#fig:optimisation:gradient}

Therefore, we can describe the $n$-th iteration of descent method as:

1. calculate a descent direction $d$;
2. choose a step size $\alpha$;
3. update the location: $x_{n+1} = x_n + \alpha~d$.

Repeat this process until a stop condition is met, such as the update is smaller than a threshold. 

Based on this process, *Gradient Descent* method uses the function gradient to decide its direction $d$.
The precess can be described as:

1. calculate a descent direction $-\nabla~f(x_n)$;
2. choose a step size $\alpha$;
3. update the location: $x_{n+1} = x_n + \alpha~\nabla~f(x_n)$.

Here $\nabla$ denotes the gradient, and the distance $\alpha$ it moves along certain direction is also called *learning rate*.
In a gradient descent process, when looking for the minimum, the point always follow the direction that is against the direction (represented by the negative gradient), as shown in [@fig:optimisation:gd].

![Example of gradient descent process](images/optimisation/gd.png){width=50%, #fig:optimisation:gd}

We can easily implement this process with the algorithm differentiation module in Owl. 
Let's look at an example.

```ocaml
open Algodiff.S
module N = Dense.Ndarray.S

let a = Mat.uniform 1 2

(* the first simple test function *)
let f0 x = Maths.(sin x |> sum')

(* the Rosenbrock function *)
let f1 a =
	let x = Mat.get a 0 0 in 
	let y = Mat.get a 0 1 in
	Maths.( (F 100.) * (y - (x ** (F 2.))) ** (F 2.) + (F 1. - x) ** (F 2.) |> sum')

let rec desc ?(eta=F 0.01) ?(eps=2e-5) f x =
  let g = grad f x in
  let arr = unpack_arr g in 
  (* N.print arr; *)
  if (N.sum' arr) < eps then x
  else desc ~eta ~eps f Maths.(x - eta * g)

let _ = desc f0 a |> unpack_arr
```

IMAGE: contour image, how the circles, and how the dots moves.

In Owl we provide a `minimise_fun` function to do that. 

```
val minimise_fun :  ?state:Checkpoint.state -> Params.typ -> (t -> t) -> t  -> Checkpoint.state * t
```

This function minimises `f : x -> y` w.r.t `x`; `x` is a ndarray, and ``y`` is a scalar value.
This function is implemented using gradient descent. 

EXPLAIN in detail, such as checkpoint etc. 

TODO: explain, perhaps with a bit theory or visual aid, to show why gradient descent is much more efficient that the previous non-gradient methods.

### Conjugate Gradient Method

One problem with the Gradient Descent is that it does not perform well on all functions.
For example, if the function forms a steep and narrow value, then using gradient descent will take many small steps to reach the minimum, even if the function is in a perfect quadratic form.

The *Conjugate Gradient* method can solve this problem. 
(HISTORY.)
It is similar to Gradient Descent, but the new direction does not follow the new gradient, but somehow *conjugated* to the old gradients and to all previous directions traversed.


EXPLAIN in detail.

Instead of $-\nabla~f(x_n)$, CG choose another way to calculate the descent direction:
EQUATION of CG

Both GD and CG are abstracted in a module in Owl:

Let's look at an example.

```
CODE: it's good if we can minimise a two dimensional example an visualise it, comparing GD and CG.
```

IMAGE: compare the optimisation route of two methods.

Besides the classic gradient descent and conjugate gradient, there are more methods that can be use to specify the descent direction: CD by Fletcher, NonlinearCG.... 

```
let run = function
    | GD -> fun _ _ _ _ g' -> Maths.neg g'
    | CG ->
        fun _ _ g p g' ->
          let y = Maths.(g' - g) in
          let b = Maths.(sum' (g' * y) / (sum' (p * y) + _f 1e-32)) in
          Maths.(neg g' + (b * p))
    | CD ->
        fun _ _ g p g' ->
          let b = Maths.(l2norm_sqr' g' / sum' (neg p * g)) in
          Maths.(neg g' + (b * p))
    ...
```
Explain

We also Give them a brief introduction here, but refer to paper and book for more details. 

### Newton and Quasi-Newton Methods

There is also a Newton Method in optimisation (it is not to be confused with the newton method used in root-finding). Still following the basic process of descent method, newton method starts from a initial point and then repeat the process:

1. compute the descent direction: $d = -\frac{\nabla~f(x_n)}{\nabla^{2}~f(x_n)}$
2. choose a step size $\alpha$;
3. update the location: $x_{n+1} = x_n + \alpha~d$.

Here $\nabla^{2}~f(x_n)$ denotes the second-order derivatives of function $f$. 
For a scalar-valued function, it can be represented by a square matrix called *Hessian matrix*, denoted by $\mathbf{H}$. Therefore the update process of newton method can be expressed as:

$$x_{n+1} = x_n - \alpha~\mathbf{H_n}^{-1}\nabla~f(x_n).$$

This can also be implemented in Owl with algorithm differentiation.

```ocaml
(* Update this example to be clearer *)
open Owl
open Algodiff.D

let rec newton ?(eta=F 0.01) ?(eps=1e-6) f x =
  let g, h = (gradhessian f) x in
  if (Maths.l2norm' g |> unpack_flt) < eps then x
  else newton ~eta ~eps f Maths.(x - eta * g *@ (inv h))

let _ =
  (* [f] must be [f : vector -> scalar]. *)
  let f x = Maths.(cos x |> sum') in
  let y = newton f (Mat.uniform 1 2) in
  Mat.print y
```

EXPLAIN: benefit of newton method.

However, one big problem with the newton method is the problem size. 
In the real world applications, it is not rare to see optimisation problems with thousands, millions or more variants. In these cases, it is impractical to compute the Hessian matrix, not to mention its inverse. 

Towards this end, the *Quasi-newton* methods are proposed. 
The basic idea is to iteratively build up an approximation of the inverse of Hessian matrix.
The most important method in this category is BFGS, named after its four authors. 

EXPLAIN briefly.

The Limited-BFGS (L-BFGS) address the memory usage issue in BFGS.

## Global Optimisation and Constrained Optimisation

So far we have talked about unconstrained optimisation, mostly to find local optimal.
In the rest of this chapter we will give a very very brief introduction to global optimisation and constrained optimisation 

The basic idea of global optimisation.

The type of problems covered constrained optimisation; applications. Currently can they be solved and how to solve them with existing tools.


## Exercise 

1. Newton method can be unstable and trapped in a loop: try to solve $f(x) = \textrm{sign}(x-2)\sqrt{|x-2|}$ in the range of [0, 4]. And try to apply the secant method on the same problem.

## References
