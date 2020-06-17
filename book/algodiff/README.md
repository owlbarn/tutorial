# Algorithmic Differentiation

TBD

## Introduction

In science and engineering it is often necessary to study the relationship between two or more quantities, where change of one quantity leads to change of others.
For example, in describing the motion an object, we describe velocity $v$ of an object with the change of the distance regarding time:

$$v = \lim_{\Delta~t}\frac{\Delta~s}{\Delta~t} = \frac{ds}{dt}.$$ {#eq:algodiff:def}

This relationship $\frac{ds}{dt}$ can be called "*derivative* of $s$ with respect to $t$".
This process can be extended to higher dimensional space.
For example, think about a solid block of material, placed in a cartesian axis system. You heat it at some part of it and cool it down at some other place, and you can imagine that the temperature $T$ at different position of this block: $T(x, y, z)$.
In this field, we can describe this change with partial derivatives along each axis:

$$\nabla~T = (\frac{\partial~T}{\partial~x}, \frac{\partial~T}{\partial~y}, \frac{\partial~T}{\partial~z}).$$ {#eq:algodiff:grad}

Here, we call the vector $\nabla~T$ *gradient* of $T$.
The procedure to calculating derivatives and gradients is refferred to as *differentiating*.

Differentiation is crucial in many scientific related fields:
find maximum or minimum values using gradient descent (see later chapter);
ODE (see later chapter);
Non-linear optimisation such as KKT optimality conditions is still a prime application.
One new crucial application is in machine learning.

TODO: detail description of application.

### Chain Rule

Before diving into how to do differentiation on computers, let's recall how to do it with a pencil and paper from our Calculus 101.

One of the most important rules in performing differentiation is the *chain rule*.
In calculus, the chain rule is a formula to compute the derivative of a composite function.
Suppose we have two functions $f$ and $g$, then the Chain rule states that:

$$F'(x)=f'(g(x))g'(x).$$ {#eq:algodiff:chainrule01}

This seemingly simple rule is one of the most fundamental rules in calculating derivatives.
For example, let $y = x^a$, where $a$ is a real number, and then we can get $y'$ using the chain rule.
Specifically, let $y=e^{\ln~x^a} = e^{a~\ln~x}$, and then we can set $u= a\ln{x}$ so that now $y=e^u$. By applying the chain rule, we have:

$$y' = \frac{dy}{du}~\frac{du}{dx} = e^u~a~\frac{1}{x} = ax^{a-1}.$$

Besides the chain rule, it's helpful to remember some basic differentiation equations, as shown in [@tbl:algodiff:chainrule02].
Here $x$ is variable and both $u$ and $v$ are functions with regard to $x$. $C$ is constant.
These equations are the building blocks of differentiating more complicated ones.
Of course, this very short list is incomplete. Please refer to calculus textbooks for more information.
Armed with chain rule and these basic equations, wen can begin to solve more differentiation problems than you can imagine.

----------------------  --------------------------------------
Function                Derivatives
----------------------  --------------------------------------
$(u(x) + v(x))'$        $u'(x) + v'(x)$

$(C\times~u(x))'$       $C\times~u'(x)$

$(u(x)v(x))'$           $u'(x)v(x) + u(x)v'(x)$

$(\frac{u(x)}{v(x)})'$  $\frac{u'(x)v(x) - u(x)v'(x)}{v^2(x)}$

$\sin(x)$               $\cos(x)$

$e^x$                   $e^x$

$log_a(x)$              $\frac{1}{x~\textrm{ln}~a}$
----------------------  --------------------------------------
: A Short Table of Basic Derivatives {#tbl:algodiff:chainrule02}


### Differentiation Methods

As the models and algorithms become increasingly complex, sometimes the function being implicit, it is impractical to perform manual differentiation.
Therefore, we turn to computer-based automated computation methods.
There are three different ways widely used to automate differentiation: numerical differentiation, symbolic differentiation, and algorithmic differentiation.

**Numerical Differentiation**

The numerical differentiation comes from the definition of derivative in [@eq:algodiff:def].
It uses a small step $\delta$ to approximate the limit in the definition:

$$f'(x) = \lim_{\delta~\to~0}\frac{f(x+\delta) - f(x)}{\delta}.$$

As long as you knows how to evaluate function $f$, this method can be applied. The function $f$ per se can be treated a black box.
The differentiation coding used in this method is also straightforward.
However, the problem with this method is prone to truncation errors and round-off errors.
The truncation errors is introduced by truncating an infinite sum and approximating it by a finite sum;
the round-off error is then caused by representing numbers approximately in numerical computation during this process.
Besides, this method is also slow due to requiring multiple evaluation of function $f$.
We'll discuss it later in the optimisation chapter, since optimisation using gradient is a very important application of differentiation.
Some discussion about numerically solving derivative related problems is also covered in the Ordinary Differentiation Equation chapter, where we focus on introducing solving these equations numerically, and how the impact of these errors can be reduced.


**Symbolic Differentiation**

Symbolic Differentiation is the opposite of numerical solution. It does not involve numerical computation, only math symbol manipulation.
The rules we have introduced in [@tbl:algodiff:chainrule02] are actually expressed in symbols.
Think about this function: $f(x_0, x_1, x_2) = x_0 * x_1 * x_2$. If we compute $\nabla~f$ symbolically, we end up with:

$$\nabla~f = (\frac{\partial~f}{\partial~x_0}, \frac{\partial~f}{\partial~x_1}, \frac{\partial~f}{\partial~x_2}) = (x_1 * x_2, x_0 * x_2, x_1 * x_2).$$

It is nice and accurate, leaving limited space for numerical errors.
However, you can try to extend the number of variables from 3 to a large number $n$, which means $f(x) = \prod_{i=0}^{n-1}x_i$, and then try to perform the symbolic differentiation again.

The point is that, symbolic computations tends to give a very large result for even not very complex functions.
It's easy to have duplicated common sub computations, and produce exponentially large symbolic expressions.
Therefore, as intuitive as it is, the symbolic differentiation method can easily takes a lot of memory in computer, and is slow.

The explosion of computation complexity is not the only limitation of symbolic differentiaton. In contrast to the numerical differentiation, we have to treat the fuction in symbolic differentiation as a whitebox, knowing exactly what is inside of the function. This further indicates that it cannot be used for arbitrary functions.


**Algorithmic Differentiation**

Algorithmic differentiation (AD) is a chain-rule based technique for calculating the derivatives with regards to input variables of functions defined in a computer programme.
It is also known as automatic differentiation, though strictly speaking AD does not fully automate differentiation and can lead to inefficient code.

AD can generate exact results with superior speed and memory usage, therefore highly applicable in various real world applications.
Even though AD also follows the chain rule, it directly applies numerical computation for intermediate results.  
It is now important to point out that AD is neither numerical nor symbolic differentiation, it actually takes the best parts of both worlds, as we will see in the next section.
Actually, according to [@griewank1989automatic], the reverse mode of AD yields any gradient vector at no more than five times the cost of evaluating the function $f$ itself.
AD has already been implemented in various popular languages, including the [`ad`](https://pythonhosted.org/ad/) in Python, [`JuliaDiff`](https://www.juliadiff.org/) in Julia, and [`ADMAT`](http://www.cayugaresearch.com/admat.html) in MATLAB, etc.
In the rest of this chapter, we focus on introducing the AD module in Owl.


## How Algorithmic Differentiation Works

We have seen the chain rules being applied on simple functions such as $y=x^a$. Now let's check how this rule can be applied to more complex computations.
Let's look at the function below:

$$y(x_0, x_1) = (1 + e^{x_0~x_1 + sin(x_0)})^{-1}.$$ {#eq:algodiff:example}

This functions is based on a sigmoid function. Our goal is to compute the partial derivative $\frac{\partial~y}{\partial~x_0}$ and $\frac{\partial~y}{\partial~x_1}$.
To better illustrate this process, we express [@eq:algodiff:example] as a graph, as shown in [@fig:algodiff:example_01].
At the right side of the figure, we have the final output $y$, and at the roots of this graph are input variables.
The nodes between them indicate constants or intermediate variables that are gotten via basic functions such as `sine`.
All nodes are labelled by $v_i$.
An edge between two nodes represents an explicit dependency in the computation.

![Graph expression of function](images/algodiff/example_01.png "example_01"){ width=100% #fig:algodiff:example_01}

Based on this graphic representation, there are two major ways to apply the chain rules: the forward differentiation mode, and the reverse differentiation mode (not "backward differentiation", which is a method used for solving ordinary differential equations).
Next, we introduce these two methods.

### Forward Mode

Our target is to calculate $\frac{\partial~y}{\partial~x_0}$ (partial derivative regarding $x_1$ should be similar).
But hold your horse, let's start with some earlier intermediate results that might be helpful.
For example, what is $\frac{\partial~x_0}{\partial~x_1}$? 1, obviously. Equally obvious is $\frac{\partial~x_1}{\partial~x_1} = 0$. It's just elementary.
Now, things gets a bit trickier: what is $\frac{\partial~v_3}{\partial~x_0}$? Now it is a good time to use the chain rule:

$$\frac{\partial~v_3}{\partial~x_0} = \frac{\partial~(x_0~x_1)}{\partial~x_0} = x_1~\frac{\partial~(x_0)}{\partial~x_0} + x_0~\frac{\partial~(x_1)}{\partial~x_0} = x_1.$$

After calculating $\frac{\partial~v_3}{\partial~x_0}$, we can then processed with derivatives of $v_5$, $v_6$, all the way to that of $v_9$ which is also the output $y$ we are looking for.
This process starts with the input variables, and ends with output variables. Therefore, it is called *forward differentiation*.
We can do simplify the math notations in this process by letting $\dot{v_i}=\frac{\partial~(v_i)}{\partial~x_0}$.
The $\dot{v_i}$ here is called *tangent* of function $v_i(x_0, x_1, \ldots, x_n)$ with regard to input variable $x_0$, and the original computation results at each intermediate point is called *primal* values.
The forward differentiation mode is sometimes also called "tangent linear" mode.

Now we can present the full forward differentiation calculation process, as shown in [@tbl:algodiff:forward].
Two simultaneous computing processes take place, represented as two separated columns: on the left hand side is the computation procedure specified by [@eq:algodiff:example];
on the right side shows computation of derivative for each intermediate variable with regard to $x_0$.
Let's find out $\dot{y}$ when setting $x_0 = 1$, and $x_1 = 1$.

---- --------------------------  ---------------------------------
Step Primal computation          Tangent computation            
---- --------------------------  ---------------------------------
0    $v_0 = x_0 = 1$             $\dot{v_0}=1$

1    $v_1 = x_1 = 1$             $\dot{v_1}=0$

2    $v_2 = sin(v_0) = 0.84$     $\dot{v_2} = cos(v_0)*\dot{v_0} = 0.54 * 1 = 0.54$   

3    $v_3 = v_0~v_1 = 1$         $\dot{v_3} = v_0~\dot{v_1} + v_1~\dot{v_0} = 1 * 0 + 1 * 1 = 1$

4    $v_4 = v_2 + v3 = 1.84$     $\dot{v_4} = \dot{v_2} + \dot{v_3} = 1.54$

5    $v_5 = 1$                   $\dot{v_5} = 0$

6    $v_6 = \exp{(v_4)} = 6.30$  $\dot{v_6} = \exp{(v_4)} * \dot{v_4} = 6.30 * 1.54 = 9.70$

7    $v_7 = 1$                   $\dot{v_7} = 0$

8    $v_8 = v_5 + v_6 = 7.30$    $\dot{v_8} = \dot{v_5} + \dot{v_6} = 9.70$

9    $y = v_9 = \frac{1}{v_8}$   $\dot{y} = \frac{-1}{v_8^2} * \dot{v_8} = -0.18$
---- --------------------------  ---------------------------------
: Computation process of forward differentiation {#tbl:algodiff:forward}

This procedure shown in this table can be illustrated in [@fig:algodiff:example_01_forward].

![Example of forward accumulation with computational graph](images/algodiff/example_01_forward.png "example_01_forward"){ width=100% #fig:algodiff:example_01_forward}

Of course, all the numerical computation here are approximated with only two significant figures.  
We can validate this result with algorithmic differentiation module in Owl. If you don't understand the code, don't worry. We will cover the details of this module in later sections.

```ocaml
# open Algodiff.D

# let f x =
    let x1 = Mat.get x 0 0 in
    let x2 = Mat.get x 0 1 in
    Maths.(div (F 1.) (F 1. + exp (x1 * x2 + (sin x1))))
val f : t -> t = <fun>

# let x = Mat.ones 1 2
val x : t = [Arr(1,2)]

# let _ = grad f x |> unpack_arr
- : A.arr =
          C0        C1
R0 -0.181974 -0.118142

```

**TODO:** introduce dual number

### Reverse Mode

Now let's rethink about this problem from the other direction, literally.
Question remains the same, i.e. calculating $\frac{\partial~y}{\partial~x_0}$.
We still follow the same "step by step" idea from the forward mode, but the difference is that, we calculate it backward.
For example, here we reduce the problem in this way: since in this graph $y = v_7 / v_8$, if only we can have $\frac{\partial~y}{\partial~v_7}$ and $\frac{\partial~y}{\partial~v_8}$, then this problem should be one step closer towards my target problem.

First of course, we have $\frac{\partial~y}{\partial~v_9} = 1$, since $y$ and $v_9$ are the same.
Then how do we get $\frac{\partial~y}{\partial~v_7}$? Again, time for chain rule:

$$\frac{\partial~y}{\partial~v_7} = \frac{\partial~y}{\partial~v_9} * \frac{\partial~v_9}{\partial~v_7} = 1 * \frac{\partial~v_9}{\partial~v_7} = \frac{\partial~(v_7 / v_8)}{\partial~v_7} = \frac{1}{v_8}.$$ {#eq:algodiff:reverse_01}

Hmm, let's try to apply a substitution to the terms to simplify this process. Let

$$\bar{v_i} = \frac{\partial~y}{\partial~v_i}$$

be the derivative of output variable $y$ with regard to intermediate node $v_i$.
It is called the *adjoint* of variable $v_i$ with respect to the output variable $y$.
Using this notation, [@eq:algodiff:reverse_01] can be expressed as:

$$\bar{v_7} = \bar{v_9} * \frac{\partial~v_9}{\partial~v_7} = 1 * \frac{1}{v_8}.$$

Note the difference between tangent and adjoint.
In the forward mode, we know $\dot{v_0}$ and $\dot{v_1}$, then we calculate $\dot{v_2}$, $\dot{v3}$, .... and then finally we have $\dot{v_9}$, which is the target.
Here, we start with knowing $\bar{v_9} = 1$, and then we calculate $\bar{v_8}$, $\bar{v_7}$, .... and then finally we have $\bar{v_0} = \frac{\partial~y}{\partial~v_0} = \frac{\partial~y}{\partial~x_0}$, which is also exactly our target.
Again, $\dot{v_9} = \bar{v_0}$ in this example, given that we are talking about derivative regarding $x_0$ when we use $\dot{v_9}$.
Following this line of calculation, the reverse differentiation mode is also called *adjoint mode*.

With that in mind, let's see the full steps of performing reverse differentiation.
First, we need to perform a forward pass to compute the required intermediate values, as shown in [@tbl:algodiff:reverse_01].

---- --------------------------
Step Primal computation        
---- --------------------------
0    $v_0 = x_0 = 1$           

1    $v_1 = x_1 = 1$           

2    $v_2 = sin(v_0) = 0.84$   

3    $v_3 = v_0~v_1 = 1$       

4    $v_4 = v_2 + v3 = 1.84$   

5    $v_5 = 1$                 

6    $v_6 = \exp{(v_4)} = 6.30$

7    $v_7 = 1$                  

8    $v_8 = v_5 + v_6 = 7.30$  

9    $y = v_9 = \frac{1}{v_8}$
---- --------------------------
: Forward pass in the reverse differentiation mode {#tbl:algodiff:reverse_01}

You might be wondering, this looks the same as the left side of [@tbl:algodiff:forward].
You are right. These two are exactly the same, and we repeat it again to make the point that, this time you cannot perform the calculation with one pass.
You must compute the required intermediate results first, and then perform the other "backward pass", which is the key point in reverse mode.

---- ---------------------------------------------------------------------------------
Step Adjoint computation
---- ---------------------------------------------------------------------------------
10   $\bar{v_9} = 1$

11   $\bar{v_8} = \bar{v_9}\frac{\partial~(v_7/v_8)}{\partial~v_8} = 1 * \frac{-v_7}{v_8^2} = \frac{-1}{7.30^2} = -0.019$

12   $\bar{v_7} = \bar{v_9}\frac{\partial~(v_7/v_8)}{\partial~v_7} = \frac{1}{v_8} = 0.137$   

13   $\bar{v_6} = \bar{v_8}\frac{\partial~v_8}{\partial~v_6} = \bar{v_8} * \frac{\partial~(v_6 + v5)}{\partial~v_6} =  \bar{v_8}$

14   $\bar{v_5} = \bar{v_8}\frac{\partial~v_8}{\partial~v_5} = \bar{v_8} * \frac{\partial~(v_6 + v5)}{\partial~v_5} = \bar{v_8}$

15   $\bar{v_4} = \bar{v_6}\frac{\partial~v_6}{\partial~v_4} = \bar{v_8} * \frac{\partial~\exp{(v_4)}}{\partial~v_4} = \bar{v_8} * e^{v_4}$

16   $\bar{v_3} = \bar{v_4}\frac{\partial~v_4}{\partial~v_3} = \bar{v_4} * \frac{\partial~(v_2 + v_3)}{\partial~v_3} = \bar{v_4}$

17   $\bar{v_2} = \bar{v_4}\frac{\partial~v_4}{\partial~v_2} = \bar{v_4} * \frac{\partial~(v_2 + v_3)}{\partial~v_2} = \bar{v_4}$

18   $\bar{v_1} = \bar{v_3}\frac{\partial~v_3}{\partial~v_1} = \bar{v_3} * \frac{\partial~(v_0*v_1)}{\partial~v_1} = \bar{v_4} * v_0 = \bar{v_4}$

19   $\bar{v_{02}} = \bar{v_2}\frac{\partial~v_2}{\partial~v_0} = \bar{v_2} * \frac{\partial~(sin(v_0))}{\partial~v_0} = \bar{v_4} * cos(v_0)$

20   $\bar{v_{03}} = \bar{v_3}\frac{\partial~v_3}{\partial~v_0} = \bar{v_3} * \frac{\partial~(v_0 * v_1)}{\partial~v_0} = \bar{v_4} * v_1$

21   $\bar{v_0} = \bar{v_{02}} + \bar{v_{03}} = \bar{v_4}(cos(v_0) + v_1) = \bar{v_8} * e^{v_4}(0.54 + 1) = -0.019 * e^{1.84} * 1.54 = -0.18$
---- ---------------------------------------------------------------------------------
: Computation process of the backward pass in reverse differentiation {#tbl:algodiff:reverse_02}

Note that things a bit different for $x_0$. It is used in both intermediate variables $v_2$ and $v_3$.
Therefore, we compute the adjoint of $v_0$ with regard to $v_2$ (step 19) and $v_3$ (step 20), and accumulate them together (step 20).
(TODO: Explain why adding these two adjoints.)


Similar to the forward mode, reverse differentiation process in [] can be clearly shown in figure [@fig:algodiff:example_01_reverse].

![Example of reverse accumulation with computational graph](images/algodiff/example_01_reverse.png "example_01_reverse"){ width=100% #fig:algodiff:example_01_reverse}

This result $\bar{v_0} = -0.18$ agrees what we have have gotten using the forward mode.
However, if you still need another fold of insurance, we can use Owl to perform a numerical differentiation.
The code would be similar to that of using algorithmic differentiation as shown before.

```ocaml env=algodiff_reverse_example_00
module D = Owl_numdiff_generic.Make (Dense.Ndarray.D);;

let x = Arr.ones [|2|]

let f x =
	let x1 = Arr.get x [|0|] in
	let x2 = Arr.get x [|1|] in
	Maths.(div 1. (1. +. exp (x1 *. x2 +. (sin x1))))
```

And then we can get the differentiation result at the point $(x_0, x_1) = (0, 0)$, and it agrees with the previous results.

```ocaml env=algodiff_reverse_example_00
# D.grad f x
- : D.arr =
         C0        C1
R -0.181973 -0.118142

```

Before we move on, did you notice that we get $\frac{\partial~y}{\partial~x_1}$ for "free" while calculating $\frac{\partial~y}{\partial~x_0}$. Noticing this will help you to understad the next section, namely how to decide which mode (forward or backward) to use in practice.


### Forward or Reverse?

Since both can be used to differentiate a function then the natural question is which mode we should choose in practice. The short answer is: it depends on your function.

In general, given a function that you want to differentiate, the rule of thumb is:

* if input variables >> output variables, then use reverse mode;
* if input variables << output variables, then use forward mode.

Later we will show example of this point.

For each input variable, we need to seed individual variable and perform one forward pass. The number of forward passes increase linearly as the number of inputs increases. However, for backward mode, no matter how many inputs there are, one backward pass can give us all the derivatives of the inputs. I guess now you understand why we need to use backward mode for `f`. One real-world example of `f` is machine learning and neural network algorithms, wherein there are many inputs but the output is often one scalar value from loss function.

Backward mode needs to maintain a directed computation graph in the memory so that the errors can propagate back; whereas the forward mode does not have to do that due to the algebra of dual numbers.


## A Strawman AD Engine

TODO: revise code, especially naming such as `sin_ad`.

Surely you don't want to make these tables every time you are faced with a new computation.
Now that you understand how to use forward and reverse propagation to do algorithmic differentiation, let's look at how to do it with computer programmes.
In this section, we will introduce how to implement the differentiation modes using OCaml code.
Of course, these will be elementary straw man implementation compared to the industry standard module provided by Owl, but nevertheless important to the understanding of the latter.

We will again use the function in [@eq:algodiff:example] as example, and we limit the computation in our small AD engine to only these operations: `add`, `div`, `mul`,

### Simple Forward Implementation

How can we represent [@tbl:algodiff:forward]? A intuitive answer is to build a table when traversing the computation graph.
However, that's not a scalable: what if there are hundreds and thousands of computation steps?
A closer look at the [@tbl:algodiff:forward] shows that a intermediate node actually only need to know the computation results (primal value and tangent value) of its parents nodes to compute its own results.
Based on this observation, we can define a data type that preserve these two values:

```ocaml env=algodiff_simple_impl_forward
type df = {
 	mutable p: float;
 	mutable t: float
}

let primal df = df.p
let tangent df = df.t
```

And now we can define operators that accept type `df` as input and outputs the same type:

**liang: change your variable name from df to just x**

```ocaml env=algodiff_simple_impl_forward
let sin_ad df =
    let p = primal df in
    let t = tangent df in
    let p' = Owl_maths.sin p in
    let t' = (Owl_maths.cos p) *. t in
    {p=p'; t=t'}
```

EXPLAIN

Now you can easily extend towards the `exp` operation:


```ocaml env=algodiff_simple_impl_forward
let exp_ad df =
    let p = primal df in
    let t = tangent df in
    let p' = Owl_maths.exp p in
    let t' = p' *. t in
    {p=p'; t=t'}
```

But what about operators that accept multiple inputs? Let's see multiplication.

```ocaml env=algodiff_simple_impl_forward
let mul_ad dfa dfb =
    let pa = primal dfa in
    let ta = tangent dfa in
    let pb = primal dfb in
    let tb = tangent dfb in
    let p' = pa *. pb in
    let t' = pa *. tb +. ta *. pb in
    {p=p'; t=t'}
```

Similarly, you can extend that towards similar operations: the `add` and `div`.

```ocaml env=algodiff_simple_impl_forward
let add_ad dfa dfb =
    let pa = primal dfa in
    let ta = tangent dfa in
    let pb = primal dfb in
    let tb = tangent dfb in
    let p' = pa +. pb in
    let t' = ta +. tb in
    {p=p'; t=t'}

let div_ad dfa dfb =
    let pa = primal dfa in
    let ta = tangent dfa in
    let pb = primal dfb in
    let tb = tangent dfb in
    let p' = pa /. pb in
    let t' = (ta *. pb -. tb *. pa) /. (pb *. pb) in
    {p=p'; t=t'}
```

Based on these functions, we can provide a tiny wrapper named `diff`:

```ocaml env=algodiff_simple_impl_forward
let diff f =
  let f' x y =
    let r = f x y in
    primal r, tangent r
  in
  f'
```

And that's all! Now we can do differentiation on our previous example.

```ocaml env=algodiff_simple_impl_forward
let x0 = {p=1.; t=1.}
let x1 = {p=1.; t=0.}
```

These are inputs.
We know the tangent of x1 with regard to x0 is zero, and so are the other constants used in the computation.

```ocaml env=algodiff_simple_impl_forward
# let f x0 x1 =
    let v2 = sin_ad x0 in
    let v3 = mul_ad x0 x1 in
    let v4 = add_ad v2 v3 in
    let v5 = {p=1.; t=0.} in
    let v6 = exp_ad v4 in
    let v7 = {p=1.; t=0.} in
    let v8 = add_ad v5 v6 in
    let v9 = div_ad v7 v8 in
    v9
val f : df -> df -> df = <fun>

# let pri, tan = diff f x0 x1
val pri : float = 0.13687741466075895
val tan : float = -0.181974376561731321
```

Just as expected.

### Simple Reverse Implementation

(TODO: revise the code e.g. naming; explain the code as clear as possible since the adjoint function is very tricky. Use graph if necessary.)

The reverse mode is a bit more complex.
As shown in the previous section, forward mode only needs one pass, but the reverse mode requires two passes, a forward pass followed by a backward pass.
This further indicates that, besides computing primal values, we also need to "remember" the operations along the forward pass, and then utilise these information in the backward pass.
There can be multiple ways to do that, e.g. a stack or graph structure.
What we choose here is bit different though.
Start with the data types we use.


```ocaml env=algodiff_simple_impl_reverse
type dr = {
 	mutable p: float;
 	mutable a: float ref;
  mutable adj_fun : float ref -> (float * dr) list -> (float * dr) list
}

let primal dr = dr.p
let adjoint dr = dr.a
let adj_fun dr = dr.adj_fun
```

The `p` is for primal while `a` stands for adjoint. It's easy to understand.
The `adj_fun` is a bit tricky. Let's see an example:


```ocaml env=algodiff_simple_impl_reverse
let sin_ad dr =
    let p = primal dr in
    let p' = Owl_maths.sin p in
    let adjfun' ca t =
        let r = !ca *. (Owl_maths.cos p) in
        (r, dr) :: t
    in
    {p=p'; a=ref 0.; adj_fun=adjfun'}
```

It's an implementation of `sin` operation.
The `adj_fun` here can be understood as a placeholder for the adjoint value we don't know yet in the forward pass.
The `t` is a stack of intermediate nodes to be processed in the backward process.
It says that, if I have the adjoint value `ca`, I can then get the new adjoint value of my parents `r`.
This result, together with the original data `dr`, is pushed to the stack `t`.
This stack is implemented in OCaml list.  

Let's look at the `mul` operation:


```ocaml env=algodiff_simple_impl_reverse
let mul_ad dr1 dr2 =
    let p1 = primal dr1 in
    let p2 = primal dr2 in

    let p' = Owl_maths.mul p1 p2 in
    let adjfun' ca t =
        let r1 = !ca *. p2 in
        let r2 = !ca *. p1 in
        (r1, dr1) :: (r2, dr2) :: t
    in
    {p = p'; a = ref 0.; adj_fun = adjfun'}
```

Both of its parents are added to the task stack.

For the input data, we need a helper function:

```ocaml env=algodiff_simple_impl_reverse
let make_reverse v =
    let a = ref 0. in
    let adj_fun _a t = t in
    {p=v; a; adj_fun}
```

With this function, we can do the forward pass like this:

```ocaml env=algodiff_simple_impl_reverse
# let x = make_reverse 1.
val x : dr = {p = 1.; a = {contents = 0.}; adj_fun = <fun>}
# let y = make_reverse 2.
val y : dr = {p = 2.; a = {contents = 0.}; adj_fun = <fun>}
# let v = mul_ad (sin_ad x) y
val v : dr = {p = 1.68294196961579301; a = {contents = 0.}; adj_fun = <fun>}
```

After the forward pass, we have the primal values at each intermediate node, but their adjoint values are all set to zero, since we don't know them yet.
And we have this adjoin function. Noting that executing this function would create a list of past computations, which in turn contains its own `adj_fun`.
This resulting `adj_fun` remembers all the required information, and know we need to recursively calculate the adjoint values we want.

```ocaml env=algodiff_simple_impl_reverse
let rec reverse_push xs =
    match xs with
    | [] -> ()
    | (v, dr) :: t ->
        let aa = adjoint dr in
        let adjfun = adj_fun dr in
        aa := !aa +. v;
        let stack = adjfun aa t in
        reverse_push stack
```

The `reverse_push` does exactly that. Stating from a list, it get the top element `dr`, get the adjoint value we already calculated `aa`, update it with `v` (explain why), get the `adj_fun`. Now that we know the adjoint value, we can use that as input parameter to the `adj_fun` to execute the data of current task and recursively execute more nodes until the task stack is empty.

Now, let's add some other required operations:


```ocaml env=algodiff_simple_impl_reverse
let exp_ad dr =
    let p = primal dr in
    let p' = Owl_maths.exp p in
    let adjfun' ca t =
        let r = !ca *. (Owl_maths.exp p) in
        (r, dr) :: t
    in
    {p=p'; a=ref 0.; adj_fun=adjfun'}


let add_ad dr1 dr2 =
    let p1 = primal dr1 in
    let p2 = primal dr2 in
    let p' = Owl_maths.add p1 p2 in
    let adjfun' ca t =
        let r1 = !ca in
        let r2 = !ca in
        (r1, dr1) :: (r2, dr2) :: t
    in
    {p = p'; a = ref 0.; adj_fun = adjfun'}


let div_ad dr1 dr2 =
    let p1 = primal dr1 in
    let p2 = primal dr2 in

    let p' = Owl_maths.div p1 p2 in
    let adjfun' ca t =
        let r1 = !ca /. p2 in
        let r2 = !ca *. (-.p1) /. (p2 *. p2) in
        (r1, dr1) :: (r2, dr2) :: t
    in
    {p = p'; a = ref 0.; adj_fun = adjfun'}
```

We can express the differentiation function `diff` with the reverse mode, with first a forward pass and then a backward pass.

```ocaml env=algodiff_simple_impl_reverse
let diff f =
  let f' x =
    (* forward pass *)
    let r = f x in
    (* backward pass *)
    reverse_push [(1., r)];
    (* get result values *)
    let x0, x1 = x in
    primal x0, !(adjoint x0), primal x1, !(adjoint x1)
  in
  f'
```

Now we can do the calculation, which are the same as before, only difference is the way to build constant values.

```ocaml env=algodiff_simple_impl_reverse
# let x1 = make_reverse 1.
val x1 : dr = {p = 1.; a = {contents = 0.}; adj_fun = <fun>}
# let x0 = make_reverse 1.
val x0 : dr = {p = 1.; a = {contents = 0.}; adj_fun = <fun>}

# let f x =
    let x0, x1 = x in
    let v2 = sin_ad x0 in
    let v3 = mul_ad x0 x1 in
    let v4 = add_ad v2 v3 in
    let v5 = make_reverse 1. in
    let v6 = exp_ad v4 in
    let v7 = make_reverse 1. in
    let v8 = add_ad v5 v6 in
    let v9 = div_ad v7 v8 in
    v9
val f : dr * dr -> dr = <fun>
```

Now let's do the differentiation:

```ocaml env=algodiff_simple_impl_reverse
# let pri_x0, adj_x0, pri_x1, adj_x1 = diff f (x0, x1)
val pri_x0 : float = 1.
val adj_x0 : float = -0.181974376561731321
val pri_x1 : float = 1.
val adj_x1 : float = -0.118141988016545588
```

Their adjoint values are just as expected.

### Unified Implementations

We have shown how to implement forward and reverse AD from scratch separately. But in the real world applications, we often need a system that supports both differentiation modes. How can we build it then?
We start with combining the previous two record data types `df` and `dr` into a new data type `t`:

```ocaml env=algodiff_simple_impl_unified_00
type t =
    | DF of float * float  
    | DR of float * float ref * adjoint

and adjoint = float ref -> (float * t) list -> (float * t) list

let primal = function
  | DF (p, _) -> p
  | DR (p, _, _) -> p

let tangent = function
  | DF (_, t) -> t
  | DR (_, _, _) -> failwith "error: no tangent for DR"

let adjoint = function
  | DF (_, _) ->  failwith "error: no adjoint for DF"
  | DR (_, a, _) -> a


let make_forward p a = DF (p, a)

let make_reverse p =
    let a = ref 0. in
    let adj_fun _a t = t in
    DR (p, a, adj_fun)

let rec reverse_push xs =
  match xs with
  | [] -> ()
  | (v, x) :: t ->
    (match x with
    | DR (_, a, adjfun) ->
      a := !a +. v;
      let stack = adjfun a t in
      reverse_push stack
    | _ -> failwith "error: unsupported type")
```

Now we operate on one unified data type. Based on this new data type, we can then combine the forward and reverse mode into one single function, using a `match` clause.

```ocaml env=algodiff_simple_impl_unified_00
let sin_ad x =
  let ff = Owl_maths.sin in
  let df p t = (Owl_maths.cos p) *. t in
  let dr p a = !a *. (Owl_maths.cos p) in
  match x with
  | DF (p, t) ->
    let p' = ff p in
    let t' = df p t in
    DF (p', t')
  | DR (p, _, _) ->
    let p' = ff p in
    let adjfun' a t =
      let r = dr p a in
      (r, x) :: t
    in
    DR (p', ref 0., adjfun')
```

The code is mostly taken from the previous two implementations, so should be not very alien to you now.
Similarly we can also build the multiplication operator:

```ocaml env=algodiff_simple_impl_unified_00
let mul_ad xa xb =
  let ff = Owl_maths.mul in
  let df pa pb ta tb = pa *. tb +. ta *. pb in
  let dr pa pb a = !a *. pb, !a *. pa in
  match xa, xb with
  | DF (pa, ta), DF (pb, tb) ->
    let p' = ff pa pb in
    let t' = df pa pb ta tb in
    DF (p', t')
  | DR (pa, _, _), DR (pb, _, _) ->
    let p' = ff pa pb in
    let adjfun' a t =
      let ra, rb = dr pa pb a in
      (ra, xa) :: (rb, xb) :: t
    in
    DR (p', ref 0., adjfun')
  | _, _ -> failwith "unsupported op"
```

Now before moving forward, let's pause and think about what's so different among different math functions.
First, they may take different number of input arguments; they could be unary functions or binary functions.
Second, they have different computation rules.
Specifically, three type of computations are involved:
`ff`, which computes the primal value; `df`, which computes the tangent value; and `dr`, which computes the adjoint value.
The rest are mostly fixed.

Based on this observation, we can utilise the first-class citizen in OCaml: module, to reduce a lot of copy and paste in code.
We can start with two types of modules: unary and binary:

```ocaml env=algodiff_simple_impl_unified_00
module type Unary = sig
  val ff : float -> float

  val df : float -> float -> float

  val dr : float -> float ref -> float
end

module type Binary = sig
  val ff : float -> float -> float

  val df : float -> float -> float -> float -> float

  val dr : float -> float -> float ref -> float * float
end
```

They express both points of difference: first, the two module differentiate between unary and binary ops; second, each module represents the three core operations: `ff`, `df`, and `dr`.
We can focus on the computation logic of each computation in each module:


```ocaml env=algodiff_simple_impl_unified_00
module Sin = struct
  let ff = Owl_maths.sin
  let df p t = (Owl_maths.cos p) *. t
  let dr p a = !a *. (Owl_maths.cos p)
end

module Exp = struct
  let ff = Owl_maths.exp
  let df p t = (Owl_maths.exp p) *. t
  let dr p a = !a *. (Owl_maths.exp p)
end

module Mul = struct
  let ff = Owl_maths.mul
  let df pa pb ta tb = pa *. tb +. ta *. pb
  let dr pa pb a = !a *. pb, !a *. pa
end

module Add = struct
  let ff = Owl_maths.add
  let df _pa _pb ta tb = ta +. tb
  let dr _pa _pb a = !a, !a
end

module Div = struct
  let ff = Owl_maths.div
  let df pa pb ta tb = (ta *. pb -. tb *. pa) /. (pb *. pb)
  let dr pa pb a =
     !a /. pb, !a *. (-.pa) /. (pb *. pb)
end
```

Now we can provide a template to build math functions:

```ocaml env=algodiff_simple_impl_unified_00
let unary_op (module U: Unary) = fun x ->
  match x with
  | DF (p, t) ->
    let p' = U.ff p in
    let t' = U.df p t in
    DF (p', t')
  | DR (p, _, _) ->
    let p' = U.ff p in
    let adjfun' a t =
      let r = U.dr p a in
      (r, x) :: t
    in
    DR (p', ref 0., adjfun')


let binary_op (module B: Binary) = fun xa xb ->
  match xa, xb with
  | DF (pa, ta), DF (pb, tb) ->
    let p' = B.ff pa pb in
    let t' = B.df pa pb ta tb in
    DF (p', t')
  | DR (pa, _, _), DR (pb, _, _) ->
    let p' = B.ff pa pb in
    let adjfun' a t =
      let ra, rb = B.dr pa pb a in
      (ra, xa) :: (rb, xb) :: t
    in
    DR (p', ref 0., adjfun')
  | _, _ -> failwith "unsupported op"
```

Each template accepts a module, and then returns the function we need. Let's see how it works with concise code.

```ocaml env=algodiff_simple_impl_unified_00
let sin_ad = unary_op (module Sin : Unary)

let exp_ad = unary_op (module Exp : Unary)

let mul_ad = binary_op (module Mul : Binary)

let add_ad = binary_op (module Add: Binary)

let div_ad = binary_op (module Div : Binary)
```

As you can expect, the `diff` function can also be implemented in a combined way. In this implementation we focus on the tangent and adjoint value of `x0` only.

```ocaml env=algodiff_simple_impl_unified_00
let diff f =
  let f' x =
    let x0, x1 = x in
    match x0, x1 with
    | DF (_, _), DF (_, _)    ->
      f x |> tangent  
    | DR (_, _, _), DR (_, _, _) ->
      let r = f x in
      reverse_push [(1., r)];
      !(adjoint x0)
    | _, _ -> failwith "error: unsupported operator"
  in
  f'
```

That's all. We can move on once again to our familiar examples.

```ocaml env=algodiff_simple_impl_unified_00
# let x0 = make_forward 1. 1.
val x0 : t = DF (1., 1.)
# let x1 = make_forward 1. 0.
val x1 : t = DF (1., 0.)
# let f_forward x =
    let x0, x1 = x in
    let v2 = sin_ad x0 in
    let v3 = mul_ad x0 x1 in
    let v4 = add_ad v2 v3 in
    let v5 = make_forward 1. 0. in
    let v6 = exp_ad v4 in
    let v7 = make_forward 1. 0. in
    let v8 = add_ad v5 v6 in
    let v9 = div_ad v7 v8 in
    v9
val f_forward : t * t -> t = <fun>
# diff f_forward (x0, x1)
- : float = -0.181974376561731321
```

That's just forward mode. With only tiny change of how the variables are constructed, we can also do the reverse mode.


```ocaml env=algodiff_simple_impl_unified_00
# let x0 = make_reverse 1.
val x0 : t = DR (1., {contents = 0.}, <fun>)
# let x1 = make_reverse 1.
val x1 : t = DR (1., {contents = 0.}, <fun>)
# let f_reverse x =
    let x0, x1 = x in
    let v2 = sin_ad x0 in
    let v3 = mul_ad x0 x1 in
    let v4 = add_ad v2 v3 in
    let v5 = make_reverse 1. in
    let v6 = exp_ad v4 in
    let v7 = make_reverse 1. in
    let v8 = add_ad v5 v6 in
    let v9 = div_ad v7 v8 in
    v9
val f_reverse : t * t -> t = <fun>

# diff f_reverse (x0, x1)
- : float = -0.181974376561731321
```

Once again, the results agree and are just as expected.

## Forward and Reverse Propagation API

So far we have talked a lot about what is Algorithmic Differentiation and how it works, with theory, example illustration, and code.
Now finally let's turn to how to use it in Owl.

Owl provides both numerical differentiation (in [Numdiff.Generic](https://github.com/owlbarn/owl/blob/master/src/base/optimise/owl_numdiff_generic_sig.ml) module) and algorithmic differentiation (in [Algodiff.Generic](https://github.com/owlbarn/owl/blob/master/src/base/algodiff/owl_algodiff_generic_sig.ml) module).
We have briefly used them in previous sections to validate the calculation results of our manual forward and reverse differentiation.

Algorithmic Differentiation is a core module built in Owl, which is one of Owl's special feature among other similar numerical libraries.
`Algodiff.Generic` is a functor that accept a Ndarray modules.
By plugging in `Dense.Ndarray.S` and `Dense.Ndarray.D` modules we can have AD modules that support `float32` and `float64` precision respectively.

```ocaml
module S = Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.S)
module D = Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.D)
```

EXPLAIN what is this `Owl_algodiff_primal_ops` thing.
We will mostly use the double precision `Algodiff.D` module, but of course using other choices is also perfectly fine.

### Expressing Computation

Let's look at the the previous example of [@eq:algodiff:example], and express it in the AD module.
Normally, the code below should do.

```ocaml
open Owl

let f x =
	let x0 = Mat.get x 0 0 in
	let x1 = Mat.get x 0 1 in
	Owl_maths.(1. /. (1. +. exp (sin x0 +. x0 *. x1)))
```

This function accept a vector and returns a float value; exactly what we are looking for.
However, the problem is that we cannot directly differentiate this programme. Instead, we need to do some minor but important change:

```ocaml env=algodiff_example_02
module AD = Algodiff.D

let f x =
	let x0 = AD.Mat.get x 0 0 in
	let x1 = AD.Mat.get x 0 1 in
	AD.Maths.((F 1.) / (F 1. + exp (sin x0 + x0 * x1)))
```

This function looks very similar, but now we are using the operators provided in the AD module, including the `get` operation and math operations.
In AD, all the input/output are of type `AD.t`. There is no difference between scalar, matrix, or ndarray for the type checking.

The `F` is special packing mechanism in AD. It makes a type `t` float. Think about wrap a float number inside a container that can be recognised in this factory called "AD". And then this factory can produce what differentiation result you want.
The `Arr` operator is similar. It wraps an ndarray (or matrix) inside this same container.
And as you can guess, there are also unpacking mechanisms. When this AD factory produces some result, to see the result you need to first unwrap this container with the functions `unpack_flt` and `unpack_arr`.
It can be illustrated in the figure below.

IMAGE: a box with packing and unpacking channels

For example, we can directly execute the AD functions, and the results need to be unpacked before being used.

```
open AD

# let input = Arr (Dense.Matrix.D.ones 1 2)
# let result = f input |> unpack_flt
val result : float = 0.13687741466075895
```

Despite this slightly cumbersome number packing mechanism, we can now perform. **liang: what?**
Next, we show how to perform the forward and reverse propagation on this computation in Owl.

### Example: Forward Mode

The forward mode is implemented with the `make_forward` and `tangent` function:

```
val make_forward : t -> t -> int -> t

val tangent : t -> t
```

The forward process is straightforward.

```
open AD

# let x = make_forward input (F 1.) (tag ());;  (* seed the input *)
# let y = f x;;                                  (* forward pass *)
# let y' = tangent y;;                           (* get all derivatives *)
```

All the derivatives are ready whenever the forward pass is finished, and they are stored as tangent values in `y`.
We can retrieve the derivatives using `tangent` function.

### Example: Reverse Mode

The reverse mode consists of two parts:

```
val make_reverse : t -> int -> t

val reverse_prop : t -> t -> unit
```

Let's look at the code snippet below.

```
open AD

# let x = Mat.ones 1 2;;              (* generate random input *)
# let x' = make_reverse x (tag ());;  (* init the reverse mode *)
# let y = f x';;                      (* forward pass to build computation graph *)
# let _ = reverse_prop (F 1.) y;;     (* backward pass to propagate error *)
# let y' = adjval x';;                (* get the gradient value of f *)

- : A.arr =
          C0        C1
R0 -0.181974 -0.118142
```

`make_reverse` function does two things for us: 1) wrap `x` into type `t` that Algodiff can process 2) generate a unique tag for the input so that input numbers can have nested structure.
By calling `f x'`, we construct the computation graph of `f` and the graph structure is maintained in the returned result `y`. Finally, `reverse_prop` function propagates the error back to the inputs.
In the end, the gradient of `f` is stored in the adjacent value of `x'`, and we can retrieve that with `adjval` function.
The result agrees with what we have calculated manually.

## High-Level APIs

What we have seen is the basic of AD modules.
There might be cases you do need to operate these low-level functions to write up your own applications (e.g., implementing a neural network), then knowing the mechanisms behind the scene is definitely a big plus.
However, using these complex low level function hinders daily use of algorithmic differentiation in numerical computation task.
In reality, you don't really need to worry about forward or reverse mode if you simply use high-level APIs such as `diff`, `grad`, `hessian`, and etc.
They are all built on the forward or reverse mode that we have seen, but provide clean interfaces, making a lot of details transparent to users.
In this chapter we will introduce how to use these core high level APIs with examples.

### Derivative

The most basic and commonly used differentiation functions is used for calculating the *derivative* of a function.
The AD module provides `diff` function for this task.
Given a function `f` that takes a scalar as input and also returns a scalar value, we can calculate its derivative at a point `x` by `diff f x`, as shown in this function signature.

```
val diff : (t -> t) -> t -> t
```

The physical meaning of derivative is intuitive. The function `f` can be expressed as a curve in a cartesian coordinate system, and the derivative at a point is the tangent on a function at this point.
It also indicate the rate of change at this point.

Suppose we define a function `f0` to be the triangular function `tanh`, we can calculate its derivative at position $x=0.1$ by simply calling:

```ocaml env=algodiff_00
open Algodiff.D

let f0 x = Maths.(tanh x)
let d = diff f0 (F 0.1)
```

Moreover, the AD module is much more than that; we can easily chains multiple `diff` together to get a function's high order derivatives.
For example, we can get the first to fourth order derivatives of `f0` by using the code below.

```ocaml env=algodiff_00
let f0 x = Maths.(tanh x);;
let f1 = diff f0;;
let f2 = diff f1;;
let f3 = diff f2;;
let f4 = diff f3;;
```

We can further plot these five functions using Owl, and the result is show in [@fig:algodiff:plot00].

```ocaml env=algodiff_00
let map f x = Owl.Mat.map (fun a -> a |> pack_flt |> f |> unpack_flt) x;;

let x = Owl.Mat.linspace (-4.) 4. 200;;
let y0 = map f0 x;;
let y1 = map f1 x;;
let y2 = map f2 x;;
let y3 = map f3 x;;
let y4 = map f4 x;;

let h = Plot.create "plot_00.png" in
Plot.plot ~h x y0;
Plot.plot ~h x y1;
Plot.plot ~h x y2;
Plot.plot ~h x y3;
Plot.plot ~h x y4;
Plot.output h;;
```

![Higher order derivatives](images/algodiff/plot_00.png "plot 00"){ width=70% #fig:algodiff:plot00 }

If you want, you can play with other functions, such as $\frac{1-e^{-x}}{1+e^{-x}}$ to see what its derivatives look like.

### Gradient

As we have introduced in [@eq:algodiff:grad], gradient generalise derivatives to multivariate functions.
Therefore, for a function that accept a vector (where each element is a variable), and returns a scalar, we can use the `grad` function to find it gradient at a point.
Imagine a 3D surface. At each points on this surface, a gradient consists of three element that each represents the derivative along the x, y or z axis.
This vector shows the direction and magnitude of maximum change of a multivariate function.

One important application of gradient is the *gradient descent*, a widely used technique to find minimum values on a function.
The basic idea is that, at any point on the surface, we calculate the gradient to find the current direction of maximal change at this point, and move the point along this direction by a small step, and then repeat this process until the point cannot be further moved.
We will talk about it in detail in the Regression an Optimisation chapters in our book.

As an example, we calculate the gradient of a physical function.
The fourth chapter of [@feynman1964feynman] describes an electronic fields. It consists two point charges, `+q` and `-q`, separated by the distance $d$. The z axis goes through the charges, and the origin is set to halfway between these two charges.
The potential from the two charges can be described by

$$\phi(x,y,z)=\frac{1}{4\pi~\epsilon_0}\left(\frac{q}{\sqrt{(z-d/2)^2 + x^2 + y^2}} + \frac{-q}{\sqrt{(z+d/2)^2 + x^2 + y^2}}\right)$$

TODO: finish this example?

### Jacobian

Just like gradient extends derivative, the gradient can also be extended to the *Jacobian matrix*.
The `grad` can be applied on functions with vector as input and scalar as output.
The `jacobian` function on the hand, deals with functions that has both input and output of vectors.
Suppose the input vector is of length $n$, and contains $m$ output variables, the jacobian matrix is defined as:

$$ \mathbf{J}(y) = \left[ \begin{matrix} \frac{\partial~y_1}{\partial~x_1} & \frac{\partial~y_1}{\partial~x_1} & \ldots & \frac{\partial~y_1}{\partial~x_n} \\ \frac{\partial~y_2}{\partial~x_0} & \frac{\partial~y_2}{\partial~x_1} & \ldots & \frac{\partial~y_2}{\partial~x_n} \\ \vdots & \vdots & \ldots & \vdots \\ \frac{\partial~y_m}{\partial~x_0} & \frac{\partial~y_m}{\partial~x_1} & \ldots & \frac{\partial~y_m}{\partial~x_n} \end{matrix} \right]$$

The intuition of Jacobian is similar to that of the gradient.
At a particular point in the domain of the target function, If you give it a small change in the input vector, the Jacobian matrix shows how the output vector changes.
One application field of Jacobian is in the analysis of dynamical systems.
In a dynamic system $\vec{y}=f(\vec{x})$, suppose $f: \mathbf{R}^n \rightarrow \mathbf{R}^m$ is differentiable and its jacobian is $\mathbf{J}$.

According to the [Hartman-Grobman](https://en.wikipedia.org/wiki/Hartman%E2%80%93Grobman_theorem) theorem, the behaviour of the system near a stationary point is related to the eigenvalues of $\mathbf{J}$.
Specifically, if the eigenvalues all have real parts that are negative, then the system is stable near the stationary point, if any eigenvalue has a real part that is positive, then the point is unstable. If the largest real part of the eigenvalues is zero, the Jacobian matrix does not allow for an evaluation of the stability. (COPY ALERT)

Let's revise the two-body problem from Ordinary Differential Equation Chapter. This dynamic system is described by a group of differential equations:

$$y_0^{'} = y_2,$$
$$y_1^{'} = y_3,$$
$$y_2^{'} = -\frac{y_0}{r^3},$$ {#eq:algodiff:twobody_system}
$$y_3^{'} = -\frac{y_1}{r^3},$$

We can express this system with code:

```ocaml env=algodiff_jacobian
open Algodiff.D

let f y =
  let y0 = Mat.get y 0 0 in
  let y1 = Mat.get y 0 1 in
  let y2 = Mat.get y 0 2 in
  let y3 = Mat.get y 0 3 in

  let r = Maths.(sqrt ((sqr y0) + (sqr y1))) in
  let y0' = y2 in
  let y1' = y3 in
  let y2' = Maths.( neg y0 / pow r (F 3.)) in
  let y3' = Maths.( neg y1 / pow r (F 3.)) in

  let y' = Mat.ones 1 4 in
  let y' = Mat.set y' 0 0 y0' in
  let y' = Mat.set y' 0 1 y1' in
  let y' = Mat.set y' 0 2 y2' in
  let y' = Mat.set y' 0 3 y3' in
  y'
```

For this functions $f: \mathbf{R}^4 \rightarrow \mathbf{R}^4$, we can then find its Jacobian matrix. Suppose the given point of interest of where all four input variables equals one. Then we can use the `Algodiff.D.jacobian` function in this way.

```text
let y = Mat.ones 1 4
let result = jacobian f y

let j = unpack_arr result;;
- : A.arr =

         C0       C1 C2 C3
R0        0        0  1  0
R1        0        0  0  1
R2 0.176777  0.53033  0  0
R3  0.53033 0.176777  0  0
```

Next, we find the eigenvalues of this jacobian matrix with the Linear Algebra module in Owl that we have introduced in previous chapter.

```text
let eig = Owl_linalg.D.eigvals j

val eig : Owl_dense_matrix_z.mat =

               C0              C1             C2              C3
R0 (0.840896, 0i) (-0.840896, 0i) (0, 0.594604i) (0, -0.594604i)
```

It turns out that one eigenvalue is real and positive, so the corresponding component of the solutions is growing.
One eigenvalue is real and negative, indicating a decaying component.
The other two eigenvalues are pure imaginary numbers. representing oscillatory components.
(COPY ALERT)
The analysis result shows that at current point the system is unstable.

### Hessian and Laplacian

Another way to extend the gradient is to find the second order derivative of a multivariate function which takes $n$ input variables and outputs a scalar.
Its second order derivatives can be organised as a matrix:

$$ \mathbf{H}(y) = \left[ \begin{matrix} \frac{\partial^2~y_1}{\partial~x_1^2} & \frac{\partial^2~y_1}{\partial~x_1~x_2} & \ldots & \frac{\partial^2~y_1}{\partial~x_1~x_n} \\ \frac{\partial^2~y_2}{\partial~x_2~x_1} & \frac{\partial^2~y_2}{\partial~x_2^2} & \ldots & \frac{\partial^2~y_2}{\partial~x_2~x_n} \\ \vdots & \vdots & \ldots & \vdots \\ \frac{\partial^2~y_m}{\partial^2~x_n~x_1} & \frac{\partial^2~y_m}{\partial~x_n~x_2} & \ldots & \frac{\partial^2~y_m}{\partial~x_n^2} \end{matrix} \right]$$

TODO: intuition of hessian.

As an example of using Hessian matrix, consider the *newton's method*.
It is also used for solving the optimisation problem, i.e. to find the minimum value on a function.
Instead of following the direction of the gradient, the newton method combines gradient and second order gradients: $\frac{\nabla~f(x_n)}{\nabla^{2}~f(x_n)}$.
Specifically, starting from a random position $x_0$, and it can be iteratively updated by repeating this procedure until converge, as shown in [@eq:algodiff:newtons].

$$x_(n+1) = x_n - \alpha~\mathbf{H}^{-1}\nabla~f(x_n)$$ {#eq:algodiff:newtons}

This process can be easily represented using the `Algodiff.D.hessian` function.

```ocaml env=algodiff_hessian_example
open Algodiff.D

let rec newton ?(eta=F 0.01) ?(eps=1e-6) f x =
  let g = grad f x in
  let h = hessian f x in
  if (Maths.l2norm' g |> unpack_flt) < eps then x
  else newton ~eta ~eps f Maths.(x - eta * g *@ (inv h))
```

We can then apply this method on a two dimensional triangular function to find one of the local minimum values, staring from a random initial point.
Note that here the functions has to take a vector as input and output a scalar.

```ocaml env=algodiff_hessian_example
let _ =
  let f x = Maths.(cos x |> sum') in
  newton f (Mat.uniform 1 2)
```

We will come back to this example in the in Optimisation chapter with more details.

Another useful and related function is `laplacian`, it calculate the *Laplacian operator* $\nabla^2~f$, which is the the trace of the Hessian matrix:

$$\nabla^2~f=trace(H_f)= \sum_{i=1}^{n}\frac{\partial^2f}{\partial~x_i^2}.$$

The Laplacian occurs in differential equations that describe many physical phenomena, such as electric and gravitational potentials, the diffusion equation for heat and fluid flow, wave propagation, and quantum mechanics. The Laplacian represents the flux density of the gradient flow of a function. For instance, the net rate at which a chemical dissolved in a fluid moves toward or away from some point is proportional to the Laplacian of the chemical concentration at that point; expressed symbolically, the resulting equation is the diffusion equation. For these reasons, it is extensively used in the sciences for modelling all kinds of physical phenomena. (COPY ALERT)


### Other APIs

Besides, there are also many helper functions, such as `jacobianv` for calculating jacobian vector product; `diff'` for calculating both `f x` and `diff f x`, and etc.
They will come handy in certain cases for the programmers.
Besides the functions we have already introduced, the complete list of APIs can be found below.

```
  val diff' : (t -> t) -> t -> t * t
  (** similar to ``diff``, but return ``(f x, diff f x)``. *)

  val grad' : (t -> t) -> t -> t * t
  (** similar to ``grad``, but return ``(f x, grad f x)``. *)

  val jacobian' : (t -> t) -> t -> t * t
  (** similar to ``jacobian``, but return ``(f x, jacobian f x)`` *)

  val jacobianv : (t -> t) -> t -> t -> t
  (** jacobian vector product of ``f`` : (vector -> vector) at ``x`` along ``v``, forward
      ad. Namely, it calcultes ``(jacobian x) v`` *)

  val jacobianv' : (t -> t) -> t -> t -> t * t
  (** similar to ``jacobianv'``, but return ``(f x, jacobianv f x v)`` *)

  val jacobianTv : (t -> t) -> t -> t -> t
  (** transposed jacobian vector product of ``f : (vector -> vector)`` at ``x`` along
      ``v``, backward ad. Namely, it calculates ``transpose ((jacobianv f x v))``. *)

  val jacobianTv' : (t -> t) -> t -> t -> t * t
  (** similar to ``jacobianTv``, but return ``(f x, transpose (jacobianv f x v))`` *)

  val hessian' : (t -> t) -> t -> t * t
  (** simiarl to ``hessian``, but return ``(f x, hessian f x)`` *)

  val hessianv : (t -> t) -> t -> t -> t
  (** hessian vector product of ``f`` : (scalar -> scalar) at ``x`` along ``v``. Namely,
      it calculates ``(hessian x) v``. *)

  val hessianv' : (t -> t) -> t -> t -> t * t
  (** similar to ``hessianv``, but return ``(f x, hessianv f x v)``. *)

  val laplacian' : (t -> t) -> t -> t * t
  (** simiar to ``laplacian``, but return ``(f x, laplacian f x)``. *)

  val gradhessian : (t -> t) -> t -> t * t
  (** return ``(grad f x, hessian f x)``, ``f : (scalar -> scalar)`` *)

  val gradhessian' : (t -> t) -> t -> t * t * t
  (** return ``(f x, grad f x, hessian f x)`` *)

  val gradhessianv : (t -> t) -> t -> t -> t * t
  (** return ``(grad f x v, hessian f x v)`` *)

  val gradhessianv' : (t -> t) -> t -> t -> t * t * t
  (** return ``(f x, grad f x v, hessian f x v)`` *)
```


**More Examples in Book**
Differentiation is an important topic in scientific computing, and therefore is not limited to only this chapter in our book.
As we have already shown in previous examples, we use AD in the newton method to find extreme values in optimisation problem in the Optimisation chapter.
It is also used in the Regression chapter to solve the linear regression problem with gradient descent.
More importantly, the algorithmic differentiation is core module in many modern deep neural libraries such as PyTorch.
The neural network module in Owl benefit a lot from our solid AD module.
We will elaborate these aspects in the following chapters. Stay tuned!

## Internal of Algorithmic Differentiation

### Go Beyond Simple Implementation

Now that you know the basic implementation of forward and reverse differentiation, and you have also seen the high level APIs that Owl provided. You might be wondering, how are these APIs implemented in Owl?

It turns out that, the simple implementation we have is not very far away from the industry-level implementation in Owl.
There are of course many details that need to be taken care of in Owl, but by now you should be able to understand the gist of it.
Without digging too deep into the code details, in this section we will give an overview of some of the differences between the Owl implementation and the simple version.  


A FIGURE that covers five parts: basic type and helper functions (level 1), op, builder, reverse (level 2), and high-level API (level 3).

![Architecture of the AD module](images/algodiff/architecture.png "architecture"){width=90% #fig:algodiff:architecture}

This figure shows the structure of AD module in Owl.
It consists of five parts. EXPLAIN.

Let's start with the type definition.

```ocaml
  type t =
    | F   of A.elt
    | Arr of A.arr
    (* primal, tangent, tag *)
    | DF  of t * t * int
    (* primal, adjoint, op, fanout, tag, tracker *)
    | DR  of t * t ref * op * int ref * int * int ref

  and adjoint = t -> t ref -> (t * t) list -> (t * t) list

  and register = t list -> t list

  and label = string * t list

  and op = adjoint * register * label
```

You will notice some difference. First, besides DF and DR, it also contains two constructors: `F` and `Arr`.
This points out one shortcoming of our simple implementation: we cannot process ndarray as input, only float.
That's why, if you look back how our computation is constructed, we have to explicitly say that the computation takes two variable as input.
In a real-world application, we only need to pass in a `1x2` vector as input.


You can also note that some extra information fields are included in the DF and DR data types.
The most important one is `tag`, it is mainly used to solve the problem of high order derivative and nested forward and backward mode. This problem is called *perturbation confusion* and is important in any AD implementation.
Here we only scratch the surface of this problem.

Here is the example: what if I want to compute the derivative of:

$$f(x) = x\frac{d(x+y)}{dy},$$

i.e. a function that contains another derivative function? It's simple, since $\frac{d(x+y)}{dy} = 1$, so $f'(x) = x' = 1$. Elementary. There is no way we can do it wrong, right?

Well, not exactly.
Let's follow our previous simple implementation:

```ocaml env=algodiff_simple_impl_unified_00
# let diff f x =
    match x with
    | DF (_, _)    ->
      f x |> tangent
    | DR (_, _, _) ->
      let r = f x in
      reverse_push [(1., r)];
      !(adjoint x)
val diff : (t -> t) -> t -> float = <fun>

# let f x =
    let g = diff (fun y -> add_ad x y) in
    mul_ad x (make_forward (g (make_forward 2. 1.)) 1.)
val f : t -> t = <fun>

# diff f (make_forward 2. 1.)
- : float = 4.
```

Hmm, the result is 3 at point $(x=2, y=2)$ but the result should be 1 at any point as we have calculated, so what has gone wrong?

Notice that `x=DF(2,1)`. The tangent value equals to 1, which means that $frac{dx}{dx}=1$. Now if we continue to use this same `x` value in function `g`, whose variable is y, the same `x=DF(2,1)` can be translated by the AD engine as $\frac{dx}{dy}=1$, which is apparently wrong.
Therefore, when used within function `g`, `x` should actually be treated as `DF(2,0)`.

The tagging technique is proposed to solve this nested derivative problem. The basic idea is to distinguish derivative calculations and their associated attached values by using a unique tag for each application of the derivative operator.
More details of method is explained in [@siskind2005perturbation].

TODO: should the other fields also be discussed in length?

Now we move on to the higher level. It's structure should be familiar to you now.
The `builder` module abstract out the general process of forward and reverse modes, while `ops` module contains all the specific calculation methods for each operations.
Keep in mind that not all operation can follow exact math rules to perform differentiation. For example, what is the tangent and adjoint of the convolution or concatenation operations? These are all includes in the `ops.ml`.

We have shown how the unary and binary operations can be built by providing two builder modules.
But of course there are many operators that have other type of signatures.
Owl abstracts more operation types according to their number of input variables and output variables.
For example, the `qr` operations calculates QR decomposition of an input matrix. This operation uses the SIPO (single-input-pair-output) builder template.

In `ops`, each operation specifies three kinds of functions for calculating the primal value (`ff`), the tangent value (`df`), and the adjoint value (`dr`). However, actually some variants are required.
In our simple examples, all the constants are either `DF` or `DR`, and therefore we have to define two different functions `f_forward` and `f_reverse`, even though only the definition of constants are different.
Now that the float number is included in the data type `t`, we can define only one computation function for both modes:

```
  let f_forward x =
    let x0, x1 = x in
    let v2 = sin_ad x0 in
    let v3 = mul_ad x0 x1 in
    let v4 = add_ad v2 v3 in
    let v5 = F 1. in (* change *)
    let v6 = exp_ad v4 in
    let v7 = F 1. in (* change *)
    let v8 = add_ad v5 v6 in
    let v9 = div_ad v7 v8 in
    v9
```

Now, we need to consider the question: how to compute `DR` and `F` data types together? To do that, we need to consider more cases in an operation.
For example, in the previous implementation an multiplication use three functions:

```
module Mul = struct
  let ff a b = Owl_maths.mul a b
  let df pa pb ta tb = pa *. tb +. ta *. pb
  let dr pa pb a = !a *. pb, !a *. pa
end
```

But now things get more complicated. For `ff a b`, we need to consider, e.g. what if `a` is float and `b` is an ndarray, or vice versa.
For `df` and `dr`, we need to consider what happens if one of the input is constant (`F` or `Arr`).
The builder module also has to take these factors into consideration accordingly.

In `reverse.ml`, we have the `reverse_push` functions.
Compared to the simple implementation, it performs an extra `shrink` step. This step checks adjoint `a` and its update `v`, ensuring rank of a must be larger than or equal with rank of `v`.
Also, an extra `reverse_reset` functions is actually required before the reverse propagation begins to reset all adjoint values to the initial zero values.

EXAMPLE: what happens if we don't do that.

For the high level APIs, one thing we need to notice is that although `diff` functions looks straightforward to be implemented using the forward and backward mode, the same cannot be said of other functions, especially `jacobian`.

Another thing is that, in our simple implementation, we don't support multiple precisions.
It is solved by functors in Owl.
In `Algodiff.Generic`, all the APIs are encapsulated in a module named `Make`.
This module takes in an ndarray-like module and generate operations with corresponding precision.
If it accepts a `Dense.Ndarray.S` module, it generate AD modules of single precision; if it is `Dense.Ndarray.D` passed in, the functor generates AD module that uses double precision.
(This description is not correct actually; the required functions actually has to follow the signatures specified in `Owl_types_ndarray_algodiff.Sig`, which also contains operation about scalar, matrix, and linear algebra, besides ndarray operations.)

### Extend AD module

The module design shown above brings one large benefit: it is very flexible in supporting adding new operations on the fly.
Let's look at an example: suppose the Owl does not provide the operation `sin` in AD module, and to finish our example in [@eq:algodiff:example], what can we do?

We can use the `Builder` module in AD.

```ocaml env=algodiff:extend_ad
open Algodiff.D

module Sin = struct
  let label = "sin"
  let ff_f a = F A.Scalar.(sin a)
  let ff_arr a = Arr A.(sin a)
  let df _cp ap at = Maths.(at * cos ap)
  let dr a _cp ca = Maths.(!ca * cos (primal a))
end
```

As a user, you need to know how the three types of functions work for the new operation you want to add.
These are defined in a module called `Sin` here.
This module can be passed as parameters to the builder to build a required operation.
We call it `sin_ad` to make it different from what the AD module actually provides.

```ocaml env=algodiff:extend_ad
let sin_ad = Builder.build_siso (module Sin : Builder.Siso)
```

That's all! Now we can use this function as if it is an native operation.

```ocaml env=algodiff:extend_ad
# let f x =
    let x1 = Mat.get x 0 0 in
    let x2 = Mat.get x 0 1 in
    Maths.(div (F 1.) (F 1. + exp (x1 * x2 + (sin_ad x1))))
val f : t -> t = <fun>

# let x = Mat.ones 1 2
val x : t = [Arr(1,2)]

# let _ = grad f x |> unpack_arr
- : A.arr =
          C0        C1
R0 -0.181974 -0.118142

```

This new operator works seamlessly with existing ones.

### Lazy Evaluation

TODO: Check if current understanding of lazy evaluation is correct.

Using the `Builder` enables users to build new operations conviniently, and it greatly improve the clarity of code.
However, with this mechanism comes a new problem: efficiency.

Imagine that a large computation that consists of hundreds and thousands of operations, with a function occurs many times in these operations. (Though not discussed yet, but in a neural network which utilises AD, it is quite common to create a large computation where basic functions such as `add` and `mul` are repeated tens and hundreds of times.
With the current `Builder` approach, every time this operation is used, it has to be created by the builder again. This is apparently not effiient.
We need some mechanism of caching.

This is where the *lazy evaluation* in OCaml comes to help.

```
val lazy: 'a -> 'a lazy_t

module Lazy :
  sig
    type 'a t = 'a lazy_t
    val force : 'a t -> 'a
  end
```

OCaml provides a built-in function `lazy` that accept an input of type `'a` and returns a `'a lazy_t` object.
It is a value of type `'a` whose computation has been delayed. This lazy expression won't be evaluated until it is called by `Lazy.force`.
The first time it is called by `Lazy.force`, the expresion is evaluted and the result is saved; and thereafter every time it is called by `Lazy.force`, the saved results will be returned without evaluation.

Here is an example:

```
(* TODO: check the code *)
# let x = Printf.printf "hello world!"; 42
# let lazy_x = lazy (Printf.printf "hello world!"; 42)
# let _ = Lazy.force lazy_x
# let _ = Lazy.force lazy_x
# let _ = Lazy.force lazy_x
```

In this example you can see that building `lazy_x` does not evalute the content, which is delayed to ther first `Lazy.force`. After that, ever time `force` is called, only the value is returned, but the `x` itself, including the `printf` function, will not be evaluted.

We can use this mechanism to improve the implementation of our AD code. Back to our previous section where we need to add a `sin` operation that the AD module supposedly "does not provide". We can still do:

```ocaml env=algodiff_lazy
open Algodiff.D

module Sin = struct
  let label = "sin"
  let ff_f a = F A.Scalar.(sin a)
  let ff_arr a = Arr A.(sin a)
  let df _cp ap at = Maths.(at * cos ap)
  let dr a _cp ca = Maths.(!ca * cos (primal a))
end
```

This part is the same, but now we need to utilise the lazy evalutation:

```
let _sin_ad = lazy Builder.build_siso (module Sin : Builder.Siso)

let sin_ad = Lazy.force _sin_ad
```

Int this way, reglardless how many times this `sin` function is called in a massive computation, the `Builder.build_siso` process is only invoked once.

(TODO: maybe an example to show the performance before and after applying the lazy evaluation? The problem is that, the non-forced part is not visible to users.)

What we have talked about is the lazy evaluation at the compiler level, and do not mistake it with another kind of lazy evaluation that are also related with the AD.
Think about that, instead of computing the specific numbers, each step accumulates on a graph, so that computation like primal, tangent, adjoint etc. all generate a graph as result, and evaluation of this graph can only be executed when we think it is suitable.
This leads to delayed lazy evaluation.
Remember that the AD functor takes an ndarray-like module to produce the `Algodiff.S` or `Algodiff.D` modules, and to do what we have described, we only need to plugin another ndarray-like module that returns graph instead of numerical value as compuation result.
This module is called the *computation graph*. It is an quite important idea, and we will talk about it in length in the second part of this book.


## Summary

## References
