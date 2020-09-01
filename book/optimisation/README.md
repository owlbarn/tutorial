# Optimisation

Optimisation is one of the  fundamental functionality in numerical computation.
In this chapter, we will briefly introduce the optimisation methods.
We will use Owl to implement some of these methods.


## Introduction

Mathematical optimisation deals with the problem of finding minimums or maximums of a function. The solution can be numerical if closed-form expression does not exist. An optimisation problem has the form:

$$\textrm{minimise } f_0(\mathbf{x}),$$
$$\textrm{subject to } f_i(\mathbf{x}) \leq b_i, i = 1, 2, \ldots, m. $$ {#eq:optimisation:def}

Here $\mathbf{x}$ is a vector that contains all the *optimisation variable*: $\mathbf{x} = [x_0, x_1, ... x_n]$. Function $f_0 : \mathbf{R}^n \rightarrow \mathbf{R}$ is the optimisation target, and is called an *objective function*, or *cost function*.
An optimisation problem could be bounded by zero or more *constraints*. $f_i : \mathbf{R}^n \rightarrow \mathbf{R}$ in a constraint is called a *constraint function*, which are bounded by the $b_i$'s.
The target is to find the optimal variable values $\mathbf{x}^{*}$ so that $f_0$ can take on a maximum or minimum value.

An optimisation problem formalises the idea "maximum benefit/minimise cost with given constraint", which is a widely applicable topic in many real world problems: scheduling computation/network resources, optimisation of investment portfolio, fitting math model based on observed data, logistics, aero engineering, competitive games...
Optimisation has already been applied in many areas.

An optimisation problem can be categorised into different types.
In [@eq:optimisation:def], if for all the objective functions and constraint functions, we have:

$$ f_i(\alpha~x+\beta~y) = \alpha~f_i(x) + \beta~f_i(y),$$ {#eq:optimisation:linear}

the optimisation problem is then called *linear optimisation*. It is an important class of optimisation problems.
If we change the "$=$" to "$\leq$" in [@eq:optimisation:linear], it would make all the functions to be *convex*, and the problem then becomes *convex optimisation*, which can be seen as a generalised linear optimisation. 
In the optimisation world, convexity is considered as the watershed between easy and difficult problems; because for most convex problems, there exist efficient algorithmic solutions.

Linear optimisation is important because non-negativity is a usual constraint on real world quantities, and that people are often interested in additive bounds. Besides, many problems can be approximated by a linear model.
Though still limited by actual problem size, the solution of most linear optimisation problems are already known and provided by off-the-shelf software tools.
The text book [@boyd2004convex] focuses exclusively on the topic of convex optimisation.

Compare to linear optimisation, solving *non-linear optimisation* problems can still be very challenging.
Finding a *global* solution that maximises or minimises the non-linear objective function is often quite time-consuming, even for only a small set of variables. Therefore, global optimisation of a non-linear problem is normally only used when it is absolutely necessary.
For example, if a system pressure test is modelled as an optimisation problem, given a small number of variants in the system, and a global extreme value has to find to test if the system is robust enough.
Otherwise, a *local* maximum or minimum is normally used instead as an approximation. In most engineering applications, a local extreme value is good enough.
Even though optimisation cannot promise a true extremism, and is easily affected by algorithm parameters and initial guess in iterative algorithms, as a trade-off, local optimisation is much faster and thus still widely used.

Looking back at [@eq:optimisation:def], if we remove the constraints, it becomes an *unconstrained optimisation* problem.
If $f$ is convex and differentiable, this problem can be seen as finding the root of the derivative of $f$ so that $f'(x^*) = 0$.
As for the constrained version, one commonly used type is the *linear programming problem* where all the functions are linear. There are also other types of optimisations such as quadratic programming, semi-definite programming, etc.
One subset of constrained optimisation is the *equality constrained optimisation* where all the constraints are expressed in the form of equality $Ax=b$. This set of problem can be simplified into the corresponding unconstrained problems.

Optimisation covers a wide range of topics and we can only give a very brief introduction here.
In the rest of this chapter, we mostly cover the unconstrained and local optimisation.
We will cover the other more advanced content briefly in the end of this chapter, and refer readers to classic books such as [@boyd2004convex] and [@fletcher2013practical] for more information.

This chapter uses differentiation techniques we have introduced in the previous chapter. 
Compared to numerical differentiation, the algorithmic differentiation guarantees a true derivative value without loss of accuracy.
For the rest of this chapter, we prefer to use the algorithmic differentiation to compute derivatives when required, but of course you can also use the numerical differentiation.

(NOTE: if we decide to add linear programming later, we can extend the constrained)

## Root Finding

We have seen some examples of root finding in the Math chapter.
*Root finding* is the process which tries to find zeroes or *roots* of continuous functions.
It is not an optimisation problem, but these two topics are closely related.
I would be beneficial for users to learn about the methods used in optimisation if they understand how the root finding algorithm work, e.g. how to the root by bracketing and how to find target in an iterative manner.

### Bisect, Newton, Secant, and IQI

First, the Bisection method. Use $\sqrt{2}$ as an example, just show the string of number here:$1\frac{1}{2}, 1\frac{1}{4}, 1\frac{3}{8}, 1\frac{5}{16} \ldots$. (DETAIL)
Owl provides `Owl_maths_root.bisec` method. (NOTE: we can have a example or even paste the Owl impl. here if we want to beef this section up, but let's keep thing concise for now.)
This method converges slowly, but it is a solid and reliable method.

Newton method utilises the derivative of objective function $f$. It starts with a initial value $x_0$, and follows this process:

$$x_{n+1} = x_{n} - \frac{f(x_n)}{f'(x_n)}.$$ {#eq:optimisation:newton}

We can use the Algorithm Differentiation module in Owl to do that.
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

The Newton method is very efficient: it has quadratic convergence which means the square of the error at one iteration is proportional to the error at the next iteration.
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
We will introduce the optimisation methods for multivariate functions in the next section, and they all apply to the univariate case, but the specific algorithms can work faster. Besides, understanding the optimisation of univariate functions can be a good step before getting to know the multivariate ones.

### Use Derivatives

If a function is continuous and differentiable, then one obvious solution to find extreme values is to locate where the derivatives equals 0:

$$f'(x) = 0$$

This leads us back to our root finding solutions.
If you already know the analytical form of $f'(x)$, it's good. For example, if $f(x) = x^2 - x$, then you can directly find root for $g(x) = 2x-1$.
Otherwise, you can use the differentiation functions in owl.
Let's look at an example. The objective function is in a hump shape:

$$f(x) = \frac{1}{(x-0.3)^2 + 0.01} + \frac{1}{(x-0.9)^2 + 0.04} -6$$

```ocaml env=optimisation_00
open Algodiff.D

let f x = Maths.(
	(F 1.) / ((x - F 0.3) ** (F 2.) + F 0.01) +
	(F 1.) / ((x - F 0.9) ** (F 2.) + F 0.04) - F 6.)

let g = diff f

let f' x = f (F x) |> unpack_flt

let g' x = g (F x) |> unpack_flt
```

Visualise the image:

```ocaml env=optimisation_00

let _ =
  let h = Plot.create ~m:1 ~n:2 "plot_hump.png" in
  Plot.set_pen_size h 1.5;
  Plot.subplot h 0 0;
  Plot.plot_fun ~h f' 0. 2.;
  Plot.set_ylabel h "f(x)";
  Plot.subplot h 0 1;
  Plot.plot_fun ~h g' 0. 2.;
  Plot.set_ylabel h "f'(x)";
  Plot.output h
```

![The hump function and its derivative function](images/optimisation/plot_hump.png){width=90% #fig:optimisation:hump}

And then you can find the extreme values using the root finding algorithm, such as Brent's:

```ocaml env=optimisation_00
# Owl_maths_root.brent g' 0. 0.4
- : float = 0.30037562625819042
# Owl_maths_root.brent g' 0.4 0.7
- : float = 0.63700940626897
# Owl_maths_root.brent g' 0.7 1.0
- : float = 0.892716303287079405
```

The issue is that you cannot be certain which is maximum and which is minimum.

### Golden Section Search

Here we face the similar question again: what if computing derivative of the function is difficult or not available?
That leads us to some search-based approach.
We have seen how we can keep reducing a pair of range to find the root of a function.
A close analogue in optimisation is also a search method called *Gold Section Search*.
It's an optimisation method that does not require calculating derivatives.
It is one choice to do optimisation if your function has a discontinuous first or second derivative.

The basic idea is simple. It also relies on keep reducing a "range" until it is small enough.
The difference is that, instead of using only two numbers, this search method uses three numbers: `[a, b, c]`.
It contains two ranges: `[a,b]` and `[b, c]`.
For every iteration, we need to find a new number `d` within one of the the two ranges.
For example, if we choose the `d` within `[b, c]`, and if $f(b) > f(d)$, then the new triplet becomes `[b, d, c]`, otherwise the new triplet is chosen as `[a, b, d]`.
With the approach, the range of this triplet keep reducing until it is small enough and the minimum value can thus be found.

Then the only question is: how to choose the suitable `d` point at each step.
This approach first chooses the larger the two ranges, either `[a, b]` or `[b, c]`. And then instead of choosing the middle point in that range, it uses the fractional distance 0.38197 from the central point of the triplet.
The name comes from the ratio and length of range is closely related with the golden ratio.
This method is slow but robust. It guarantees that each new iteration will bracket the minimum to an range just 0.61803 times the size of the previous one.


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
According to the Numerical Recipe, "the downhill simplex method now takes a series of steps, most steps just moving the point of the simplex where the function is largest (“highest point”) through the opposite face of the simplex to a lower point. These steps are called reflections, and they are constructed to conserve the volume of the simplex (and hence maintain its nondegeneracy). When it can do so, the method expands the simplex in one or another direction to take larger steps. When it reaches a “valley floor”, the method contracts itself in the transverse direction and tries to ooze down the valley."

There are some other method that does not rely on computing gradients such as Powell's method.
If the function is kind of smooth, this method can find the direction in going downhill, but instead of computing gradient, it relies on a one-dimensional optimisation method to do that, and therefore faster than the simplex method.
We will not talk about this method in detail.

### Gradient Descent Methods


A *descent method* is an iterative optimisation process.
The idea is to start from a initial value, and then find a certain *search direction* along a function to decrease the value by certain *step size* until it converges to a local minimum.
This process can be illustrated in [@fig:optimisation:gradient]([source](https://cedar.buffalo.edu/~srihari/talks/NCC1.ML-Overview.pdf)).

![Reach the local minimum by iteratively moving downhill ](images/optimisation/gradient.png){width=80% #fig:optimisation:gradient}

Therefore, we can describe the $n$-th iteration of descent method as:

1. calculate a descent direction $d$;
2. choose a step size $\alpha$;
3. update the location: $x_{n+1} = x_n + \alpha~d$.

Repeat this process until a stopping condition is met, such as the update is smaller than a threshold.

Among the descent methods, the *Gradient Descent* method is one of the most widely used algorithms to perform optimisation and the most common way to optimize neural networks.
we will talk about it in the Neural Network chapter.

Based on this process, Gradient Descent method uses the function gradient to decide its direction $d$.
The precess can be described as:

1. calculate a descent direction $-\nabla~f(x_n)$;
2. choose a step size $\alpha$;
3. update the location: $x_{n+1} = x_n + \alpha~\nabla~f(x_n)$.

Here $\nabla$ denotes the gradient, and the distance $\alpha$ it moves along certain direction is also called *learning rate*.
In a gradient descent process, when looking for the minimum, the point always follow the direction that is against the direction (represented by the negative gradient)

We can easily implement this process with the algorithmic differentiation module in Owl.
Let's look at an example.
Here we use define the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) which is usually used as performance test for optimisation problems.
The function is defined as:

$$f(x, y) = (a - x)^2 + b(y-x^2)^2.$$ {#eq:optimisation:rosenbrock}

The parameters are usually set as $a=1$ and $b=100$.

```ocaml env=optimisation:gd
open Algodiff.D
module N = Dense.Ndarray.D

let rosenbrock a =
	let x = Mat.get a 0 0 in
	let y = Mat.get a 0 1 in
	Maths.( (F 100.) * (y - (x ** (F 2.))) ** (F 2.) + (F 1. - x) ** (F 2.) |> sum')
```

Now we hope to apply the gradient descent method and observe the optimisation trajectory.

```ocaml env=optimisation:gd
let a = N.of_array [|2.; -0.5|] [|1; 2|]
let traj = ref (N.copy a)
let a = ref a
let eta = 0.0001
let n = 200
```

As preparation, we use the initial starting point `[2, -0.5]`. The step size `eta` is set to `0.0001`, and the iteration number is 100.
Then we can perform the iterative descent process. You can also run this process in a recursive manner.

```ocaml env=optimisation:gd
let _ =
  for i = 1 to n - 1 do
	let u = grad rosenbrock (Arr !a) |> unpack_arr in
	a := N.(sub !a (scalar_mul eta u));
	traj := N.concatenate [|!traj; (N.copy !a)|]
  done
```

We apply the `grad` method on the Rosenbrock function iteratively, and the updated data `a` is stored in the `traj` array.
Finally, let's visualise the trajectory of the optimisation process.

```text
let _ =
	let a, b = Dense.Matrix.D.meshgrid (-2.) 2. (-1.) 3. 50 50 in
	let c = N.(scalar_mul 100. (pow_scalar (sub b (pow_scalar a 2.)) 2.) + (pow_scalar (scalar_sub 1. a) 2.)) in

	let h = Plot.create ~m:1 ~n:2 "plot_gradients.png" in
	Plot.subplot h 0 0;
	Plot.(mesh ~h ~spec:[ NoMagColor ] a b c);

	Plot.subplot h 0 1;
	Plot.contour ~h a b c;

	let vx = N.get_slice [[]; [0]] !traj in
	let vy = N.get_slice [[]; [1]] !traj in
	Plot.plot ~h vx vy;
	Plot.output h
```

We first create a meshgrid based on the Rosenbrock function  to visualise the 3D image, and then on the 2D contour image of the same function we plot how the result of the optimisation is updated, from the initial starting porint towards a local minimum point.
The visualisation results are shown in [@fig:optimisation:gd_rosenbrock].
On the right figure the black line shows the moving trajectory. You can image it moving downwards along the slope in the right side figure.

![Optimisation process of gradient descent on multivariate function](images/optimisation/gd_rosenbrock.png "gd_rosenbrock"){width=100% #fig:optimisation:gd_rosenbrock}

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

[@fig:optimisation:gradients] compares the different descent efficiency of the conjugate gradient with gradient descent.
([src](https://www.researchgate.net/publication/221533635_A_gradient-based_algorithm_competitive_with_variational_Bayesian_EM_for_mixture_of_Gaussians))

![Compare conjugate gradient and gradient descent](images/optimisation/gradients.png "gradients"){width=60% #fig:optimisation:gradients}

Both GD and CG are abstracted in a module in Owl
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

This can also be implemented in Owl with algorithmic differentiation.

```ocaml
open Owl
open Algodiff.D

let rec newton ?(eta=F 0.01) ?(eps=1e-6) f x =
  let g, h = (gradhessian f) x in
  if (Maths.l2norm' g |> unpack_flt) < eps then x
  else newton ~eta ~eps f Maths.(x - eta * g *@ (inv h))

let _ =
  let f x = Maths.(cos x |> sum') in
  let y = newton f (Mat.uniform 1 2) in
  Mat.print y
```

Once nice property about the newton's method is its rate of convergence: it converges quadratically.
However, one big problem with the newton method is the problem size.
In the real world applications, it is not rare to see optimisation problems with thousands, millions or more variants. In these cases, it is impractical to compute the Hessian matrix, not to mention its inverse.

Towards this end, the *Quasi-newton* methods are proposed.
The basic idea is to iteratively build up an approximation of the inverse of Hessian matrix.
Their convergence is fast, but not as efficient as newton method. It takes about $n$ quasi-newton iterations to progress similarly as the newton method.
The most important method in this category is BFGS (Broyden-Fletcher-Goldfarb-Shanno), named after its four authors.

EXPLAIN briefly.

The Limited-BFGS (L-BFGS) address the memory usage issue in BFGS.
Instead of propagating updates over all iterations, this method only keeps updates from the last $m$ iterations.

## Global Optimisation and Constrained Optimisation

This chapter mainly focuses on unconstrained optimisation, mostly to find local optimal.
In the rest of this chapter we will give a very very brief introduction to global optimisation and constrained optimisation

The basic idea of global optimisation is to provide effective search methods and heuristics to traverse the search space effectively.
One method is to start from sufficient number of initial points and find the local optimal, the choose the smallest/largest value from them.
Another heuristic is to try stepping away from a local optimal value by taking a finite amplitude step away from it, perform the optimisation method, and see if it leads to a better solution or still the same.

One example of algorithm: Simulated Annealing Methods.

The constrained optimisation is another large topic we haven't covered in this chapter.

Application of linear programming and non-linear programming.

Their current status (how difficult to solve etc.), classic methods, and some existing tools.

Their connection with the unconstrained method we have introduced.

Refer to these book for more detail.

## Summary

## References

REFERENCE BOOK in writing: Numerical methods in engineering with Python 3, by Jaan Kiusalaas.
