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

$$\nabla~T = (\frac{\partial~T}{\partial~x}, \frac{\partial~T}{\partial~y}, \frac{\partial~T}{\partial~z}).$$

Here, the call the vector $\nabla~T$ *gradient* of $T$.
The procedure to calculating derivatives and gradients is called *differentiating*.

Differentiation is crucial in many scientific related fields:
find maximum or minimum values using gradient descent (see later chapter);
ODE (see later chapter);
Non-linear optimisation such as KKT optimality conditions is still a prime application.
One new crucial application is in machine learning.

TODO: detail description of application.

### Chain Rule

Before diving into how to do differentiation on computers, let's recall how to do it manually from our Calculus 101.

One of the most rule in performing differentiation is the *chain rule*.
In calculus, the chain rule is a formula to compute the derivative of a composite function.
Suppose we have two functions $f$ and $g$, then the Chain rule states that:

$$F'(x)=f'(g(x))g'(x).$$ {#eq:algodiff:chainrule01}

This seemingly simple rule is one of the basic rule in calculating derivatives.
For example, let $y = x^a$, where $a$ is a real number, and then we can get $y'$ using the chain rule.
Specifically, let $y=e^{ln~x^a} = e^{a~ln~x}$, and then we can set $u= alnx$ so that now $y=e^u$. By applying the chain rule, we have:

$$y' = \frac{dy}{du}~\frac{du}{dx} = e^u~a~\frac{1}{x} = ax^{a-1}.$$ 

Besides the chain rule, it's helpful to remember some basic differentiation equations, as shown in [@tbl:algodiff:chainrule02]. 
Here $x$ is variable and both $u$ and $v$ are functions with regard to $x$. $C$ is constant.
Of course, this very short list is incomplete. Please refer to calculus textbooks for more information.
Armed with chain rule and these basic equations, wen can begin to solve more differentiation problem than you can imagine. 

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
There are three: numerical differentiation, symbolic differentiation, and algorithmic differentiation.

**Numerical Differentiation**

The numerical differentiation comes from the definition of derivative in [@eq:algodiff:def].
It uses a small step $\delta$ to approximate the limit in the definition:

$$f'(x) = \lim_{\delta~\to~0}\frac{f(x+\delta) - f(x)}{\delta}.$$ 

As long as you knows how to evaluate function $f$, this method can be applied, and coding this method is also straightforward.
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

**Algorithmic Differentiation**

Algorithmic differentiation (AD) is a chain-rule based technique for calculating the derivatives with regards to input variables of functions defined in a computer programme.
It is also known as automatic differentiation, though strictly speaking AD does not fully automate differentiation and can lead to inefficient code.

It is important to realise that AD is not symbolic differentiation, as we will see in the next section.
Even though AD also follows the chain rule, it directly applies numerical computation for intermediate results. 
Therefore, AD can generate exact results with acceptable speed and memory usage, and therefore highly applicable in various real world applications. 
Actually, according to [@griewank1989automatic], the reverse mode of AD yields any gradient vector at no more than five times the cost of evaluating the function $f$ itself.
AD has already been implemented in various popular languages, including the [`ad`](https://pythonhosted.org/ad/) in Python, [`JuliaDiff`](https://www.juliadiff.org/) in Julia, and [`ADMAT`](http://www.cayugaresearch.com/admat.html) in MATLAB, etc.
In the rest of this chapter, we focus on introducing the AD module in Owl. 


## How Algorithmic Differentiation Works

We have seen the chain rules being applied on simple functions such as $y=x^a$. Now let's check how this rule can be applied on more complex computations. 
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
But don't be so hurry, let's start with some earlier intermediate results that might be helpful.
For example, what is $\frac{\partial~x_0}{\partial~x_1}$? 1, obviously. Equally obvious is $\frac{\partial~x_1}{\partial~x_1} = 0$. It's just elementary.
Now, things gets a bit trickier: what is $\frac{\partial~v_3}{\partial~x_0}$? Not is a good time to use the chain rule:

$$\frac{\partial~v_3}{\partial~x_0} = \frac{\partial~(x_0~x_1)}{\partial~x_0} = x_1~\frac{\partial~(x_0)}{\partial~x_0} + x_0~\frac{\partial~(x_1)}{\partial~x_0} = x_1.$$

After calculating $\frac{\partial~v_3}{\partial~x_0}$, we can then processed with derivatives of $v_5$, $v_6$, all the way to that of $v_9$ which is also the output $y$ we are looking for. 
This process starts with the input variables, and ends with output variables. Therefore, it is called *forward differentiation*.
We can do simplify the math notations in this process by letting $\dot{v_i}=\frac{\partial~(v_i)}{\partial~x_0}$. 
The $\dot{v_i}$ here is called *tangent* of function $v_i(x_0, x_1, \ldots, x_n)$ with regard to input variable $x_0$.
The forward differentiation mode is sometimes also called "tangent linear" mode.

Now we can present the full forward differentiation calculation process, as shown in [@tbl:algodiff:forward].
Two simultaneous lines of computing happen: on the left hand side is the computation procedure specified by [@eq:algodiff:example]; 
on the right side shows computation of derivative for each intermediate variable with regard to $x_0$.
Let's find out $\dot{y}$ when setting $x_0 = 1$, and $x_1 = 1$.

---- --------------------------  --------------------------------- 
Step Intermediate computation    Tangent computation            
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

This procedure show in this table can be illustrated in [@fig:algodiff:example_01_forward].

![Example of forward accumulation with computational graph](images/algodiff/example_01_forward.png "example_01_forward"){ width=100% #fig:algodiff:example_01_forward}

Of course, all the numerical computation here are approximated with only two significant figures.  
We can validate this result with algorithmic differentiation module in Owl. If you don't understand the code, don't worry. We will cover the detail of this module in detail later.

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

Now let's think this problem from the other direction, literally.
The same questions: to calculate $\frac{\partial~y}{\partial~x_0}$. 
We still follow the same "step by step" idea from the forward mode, but the difference is that, we think it backward. 
For example, here we reduce the problem in this way: since in this graph $y = v_7 / v_8$, if only we can have $\frac{\partial~y}{\partial~v_7}$ and $\frac{\partial~y}{\partial~v_8}$, then this problem should be one step closer towards my target problem.

First of course, we have $\frac{\partial~y}{\partial~v_9} = 1$, since $y$ and $v_9$ are the same. 
Then how do we get $\frac{\partial~y}{\partial~v_7}$? Again, time for chain rule:

$$\frac{\partial~y}{\partial~v_7} = \frac{\partial~y}{\partial~v_9} * \frac{\partial~v_9}{\partial~v_7} = 1 * \frac{\partial~v_9}{\partial~v_7} = \frac{\partial~(v_7 / v_8)}{\partial~v_7} = \frac{1}{v_8}.$$ {#eq:algodiff:reverse_01}

Hmm, let's try to apply a notation to simplify this process. Let 

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
Step Intermediate computation    
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

### Forward or Reverse?

Since both can be used to differentiate a function then the natural question is which mode we should choose in practice. The short answer is: it depends on your function.

In general, given a function that you want to differentiate, the rule of thumb is:

* if input variables >> output variables, then use reverse mode;
* if input variables << output variables, then use forward mode.

Later we will show example of this point.

**Theoretical Basis:**
first derivative, higher derivative, etc.

## Implementing Algorithmic Differentiation

### Native Implementation 

A most simple one, `toy_forward`, `toy_reverse`, support only small number of operators. 

### Updated Implementations 

2-3 times of updates

### Design of Algorithmic Differentiation in Owl 

The structure of main engine: recursive, node, module, etc.

### Advanced feature: Lazy Evaluation 

### Advanced feature: Extend AD module

"There is no spoon"

## APIs of Algorithmic Differentiation Module

Owl provides both numerical differentiation (in [Numdiff.Generic](https://github.com/owlbarn/owl/blob/master/src/base/optimise/owl_numdiff_generic_sig.ml) module) and algorithmic differentiation (in [Algodiff.Generic](https://github.com/owlbarn/owl/blob/master/src/base/algodiff/owl_algodiff_generic_sig.ml) module).
We have briefly used them in previous sections to validate the calculation results of our manual forward and reverse differentiation.

`Algodiff.Generic` is a functor that accept a Ndarray module.
By plugging in `Dense.Ndarray.S` and `Dense.Ndarray.D` modules we can have AD modules that support `float32` and `float64` precision respectively. 

```
module S = Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.S)
module D = Owl_algodiff_generic.Make (Owl_algodiff_primal_ops.D)
```

In this section, we will use examples to demonstrate some of the most important APIs that the Algorithmic Differentiation module provides. 
We first introduce the *low level APIs*, i.e. those for performing forward and reverse propagations.
We then introduce some of the most important *high level APIs*, including `diff`, `grad`, `jacobian`, `hessian`, and `laplacian`.
We will mostly use the double precision `Algodiff.D` module, but of course using other choices is also perfectly fine.

### APIs for Forward and Reverse Modes

`Algodiff` has implemented both forward and backward mode of AD. 

```
val make_forward : t -> t -> int -> t

val make_reverse : t -> int -> t

val reverse_prop : t -> t -> unit
```

Let's look at the two simple functions `f` and `g` defined below. `f` falls into the first category we mentioned before, i.e., inputs is more than outputs; whilst `g` falls into the second category.

```ocaml

  open Algodiff.D;;

  (* f : vector -> scalar *)
  let f x =
    let a = Mat.get x 0 0 in
    let b = Mat.get x 0 1 in
    let c = Mat.get x 0 2 in
    Maths.((sin a) + (cos b) + (sqr c))
  ;;

  (* g : scalar -> vector *)
  let g x =
    let a = Maths.sin x in
    let b = Maths.cos x in
    let c = Maths.sqr x in

    let y = Mat.zeros 1 3 in
    let y = Mat.set y 0 0 a in
    let y = Mat.set y 0 1 b in
    let y = Mat.set y 0 2 c in
    y
  ;;

```

According to the rule of thumb, we need to use backward mode to differentiate `f`, i.e., calculate the gradient of `f`. How to do that then? Let's look at the code snippet below.

```ocaml

  let x = Mat.uniform 1 3;;           (* generate random input *)
  let x' = make_reverse x (tag ());;  (* init the backward mode *)
  let y = f x';;                      (* forward pass to build computation graph *)
  reverse_prop (F 1.) y;;             (* backward pass to propagate error *)
  let y' = adjval x';;                (* get the gradient value of f *)

```

`make_reverse` function does two things for us: 1) wrap `x` into type `t` that Algodiff can process using type constructor `DF`; 2) generate a unique tag for the input so that input numbers can have nested structure. By calling `f x'`, we construct the computation graph of `f` and the graph structure is maintained in the returned result `y`. Finally, `reverse_prop` function propagates the error back to the inputs.

In the end, the gradient of `f` is stored in the adjacent value of `x'`, and we can retrieve that with `adjval` function.

How about function `g` then, the function represents those having a small amount of inputs but a large amount of outputs. According to the rule of thumb, we are suppose to use the forward pass to calculate the derivatives of the outputs w.r.t its inputs.

```ocaml

  let x = make_forward (F 1.) (F 1.) (tag ());;  (* seed the input *)
  let y = g x;;                                  (* forward pass *)
  let y' = tangent y;;                           (* get all derivatives *)

```

Forward mode appears much simpler than the backward mode. `make_forward` function does almost the same thing as `make_reverse` does for us, the only exception is that `make_forward` uses `DF` type constructor to wrap up the input. All the derivatives are ready whenever the forward pass is finished, and they are stored as tangent values in `y`. We can retrieve the derivatives using `tangent` function, as we used `adjval` in the backward mode.

OK, how about we abandon the rule of thumb? In other words, let's use forward mode to differentiate `f` rather than using backward mode. Please check the solution below.

```text
  Need to be fixed!

  let x0 = make_forward x (Arr Vec.(unit_basis 3 0)) (tag ());;  (* seed the first input variable *)
  let t0 = tangent (f x0);;                                      (* forward pass for the first variable *)

  let x1 = make_forward x (Arr Vec.(unit_basis 3 1)) (tag ());;  (* seed the second input variable *)
  let t1 = tangent (f x1);;                                      (* forward pass for the second variable *)

  let x2 = make_forward x (Arr Vec.(unit_basis 3 2)) (tag ());;  (* seed the third input variable *)
  let t2 = tangent (f x2);;                                      (* forward pass for the third variable *)

```

As we can see, for each input variable, we need to seed individual variable and perform one forward pass. The number of forward passes increase linearly as the number of inputs increases. However, for backward mode, no matter how many inputs there are, one backward pass can give us all the derivatives of the inputs. I guess now you understand why we need to use backward mode for `f`. One real-world example of `f` is machine learning and neural network algorithms, wherein there are many inputs but the output is often one scalar value from loss function.

Similarly, you can try to use backward mode to differentiate `g`. I will just this as an exercise for you. One last thing I want to mention is: backward mode needs to maintain a directed computation graph in the memory so that the errors can propagate back; whereas the forward mode does not have to do that due to the algebra of dual numbers.

In reality, you don't really need to worry about forward or backward mode if you simply use high-level APIs such as `diff`, `grad`, `hessian`, and etc. However, there might be cases you do need to operate these low-level functions to write up your own applications (e.g., implementing a neural network), then knowing the mechanisms behind the scene is definitely a big plus.


### Derivative

```
val diff : (t -> t) -> t -> t
  (* calculate derivative for f : scalar -> scalar *)
```

The following code first defines a function `f0`, then calculates from the first to the fourth derivative by calling `Algodiff.AD.diff` function.

```ocaml env=algodiff_00
open Algodiff.D;;

let map f x = Owl.Mat.map (fun a -> a |> pack_flt |> f |> unpack_flt) x;;

(* calculate derivatives of f0 *)
let f0 x = Maths.(tanh x);;
let f1 = diff f0;;
let f2 = diff f1;;
let f3 = diff f2;;
let f4 = diff f3;;

let x = Owl.Mat.linspace (-4.) 4. 200;;
let y0 = map f0 x;;
let y1 = map f1 x;;
let y2 = map f2 x;;
let y3 = map f3 x;;
let y4 = map f4 x;;

(* plot the values of all functions *)
let h = Plot.create "plot_00.png" in
Plot.plot ~h x y0;
Plot.plot ~h x y1;
Plot.plot ~h x y2;
Plot.plot ~h x y3;
Plot.plot ~h x y4;
Plot.output h;;
```

Start your `utop`, then load and open `Owl` library. Copy and past the code above, the generated figure will look like this.

![Higher order derivatives](images/algodiff/plot_00.png "plot 00"){ width=90% #fig:algodiff:plot00 }

If you replace `f0` in the previous example with the following definition, then you will have another good-looking figure :)

```ocaml env=algodiff_00
let f0 x = Maths.(
  let y = exp (neg x) in
  (F 1. - y) / (F 1. + y)
);;
```

As you see, you can just keep calling `diff` to get higher and higher-order derivatives. E.g., 

```ocaml env=algodiff_00
let f'''' f = f |> diff |> diff |> diff |> diff
```

The code above will give you the fourth derivative of `f`, i.e. `f''''`.


### Gradient

```
val grad : (t -> t) -> t -> t
(* calculate gradient for f : vector -> scalar *)
```

Example: Gradient Descent, only briefly.


### Jacobian 

```
 val jacobian : (t -> t) -> t -> t
  (* calculate jacobian for f : vector -> vector *)
```

Example: ??? 


### Hessian and Laplacian 

```
  val hessian : (t -> t) -> t -> t
  (* calculate hessian for f : scalar -> scalar *)

  val laplacian : (t -> t) -> t -> t
  (* calculate laplacian for f : scalar -> scalar *)
```

Example: 
1) Newton method 
2) Laplacian: ???


### Other APIs

Besides, there are also more helper functions such as `jacobianv` for calculating jacobian vector product; `diff'` for calculating both `f x` and `diff f x`, and etc.

The complete list of APIs can be found in [owl_algodiff_generic.mli](https://github.com/ryanrhymes/owl/blob/ppl/src/base/optimise/owl_algodiff_generic.mli). The core APIs are listed below.


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


### More Examples in Book

Differentiation is an important topic in scientific computing, and therefore is not limited to only this chapter in our book.
We use AD in the newton method to find extreme values in optimisation problem in the Optimisation chapter.
It is also used in the Regression chapter to solve the linear regression problem with gradient descent. 
More importantly, the algorithmic differentiation is core module in many modern deep neural libraries such as PyTorch.
The neural network module in Owl benefit a lot from our solid AD module. 
We will elaborate these aspects in the following chapters. Stay tuned! 

## References
