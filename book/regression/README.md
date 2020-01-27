# Regression

An important part of Machine Learning (supervised learning).
General idea: given data, make predictions.

## Model and Error

### Standard Errors of Regression Coefficients

### Model Selection

Feature selection: REFER to ISL book Chap 6.

## Linear Regression

### Example

We have a set of simple data (link) for linear regression.
A motivational question: house price, etc. as long as it fits the data. Now you need to find out the relationship for better choice-making. 

Your hypothesis is that this relationship can be formalised as:
$$ y = \theta_0~ + \theta_1~x_1$$

Look at the data. Some notation: $n$ is number of training examples, $x$ is input variable, and $y$ is target variable.
Image: Visualise of the data, including a line, and a line-to-dot represent distance. 

The target is to choose $\Theta$ so that the line is *close* to data we observed. 
Define "close":

$$E(\Theta) = \frac{1}{2n}\sum_{i=1}^{n}(h_{\Theta}(x_i - y_i)^2)$$

We call it the cost function.
It's physical meaning; maximise likelihood

### Solving Problem with Gradient Descent

Why gradient descent? We have this surf image:

IMAGE

It shows how cost function changes with both $\theta_0$ and $\theta_1$. We can thus use GD to locate the local minimal value.

How GD works:

$$ \theta_j \leftarrow \theta_j - \alpha~\frac{\partial}{\partial \theta_j}~J(\theta_0, \theta_1) $$

The second part means the direction, and the second part gives the step size.

This seemingly terrible partial derivative is actually simple to do:

EQUATION

Following these equations, you can perform the process until converges:

CODE

And now we get the result.

PLOT: the resulting line against data samples. 


Of course, we can also directly use the GD optimisation method in Owl:

CODE

The result would be similar.

## Multiple Regression

Now the same problem, but variables goes from one to multiple. What would you do?
Basically similar, but instead of $\theta_0$, $\theta_1$, $\theta_2$, ... now we need the vectorised representation: $\Theta$:

$$ J(\Theta) = \frac{1}{2n}(X\Theta - y)^2$$.

Next we focus on some issues.

### Feature Normalisation

One factor is hundred times larger than the other variables. That's bad.

Regularisation

### Regularisation

### Ols, Ridge, Lasso, and Elastic_net 

[REFER](https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net)

### Analytical Solution

Besides GD, there is actually one close form solution to Linear Regression:

$$\Theta = (X^T~X)^{-1}X^Ty$$

Try this solution, compare the result with that from GD.

Where does this solution come from.

It's pros and cons vs GD.

## Non-linear regressions 

Polynomial
CODE
IMAGE: result visualisation

Exponential
CODE
IMAGE: result visualisation

## Logistic Regression

So far we have been predicting a value for our problems, but what if we don't care about is not the value, but a classification? For example, we want to know if this tumour is cancer or not given previous data. 

We can of course continue to use linear regression to represent the possibility of one of these result, but one issue is that, it could well be out of the bounds of [0, 1]. 

### Sigmoid Function 

EQUATION + IMAGE

With this function, instead of $h = \Theta~X$,  the problem can be modelled as:

$$h(\Theta) = g(\Theta~X)$$

Now we can interpret this model easily.
If it is larger than 0.5, then ... else ...

### Decision Boundary 

Linear 

IMAGE

Non-Linear

IMAGE

Let's use the non-linear one as practice example.

### Cost Function 

With the new model comes new cost function. 

Measure the *distance*:
Previously for linear regression we have the Euclidean distance of prediction and real value.

Now we defined it this way:
Cost equation that involves $log$ function.
Explain how it comes: maximise the log likelihood

Therefore, we have this cost function:

$$ J(\Theta) = \frac{1}{n}\sum_{i=1}^{n}\textrm{Cost}(h_{\Theta}(x_i), y_i)$$

### Gradient Descent

How to solve this terrible equation?

The sigmoid has a nice property: its derivative is simple:

EQUATION

Therefore, similar to LR, we only need to repeat this step until converges:

EQUATION

Let's write that in Owl:

CODE #1: plain code 

CODE #2: use existing function in Owl 

Plotting the boundaries.

### Multi-class classification 

Similar to the LR problem, you hardly stop at 2 parameters. What if we need to classified an object into one fo multiple classes?

One popular classification problem is the hand-written recognition task. It is...
It is a widely used ABC task for Neural Networks, and we will also cover it later in Chapter DNN.
For now, we solve that from the logistic regression line of thought. 

Dataset description
Visualise

Similarly, we extend the cost function towards multi-class:

EQUATION

We can also use the generalised version of GD as before, or directly apply GD method in Owl:

CODE

Let's apply the model on test data:

result.

## Support Vector Machine

It's a similar idea to logistic regression.

Explain the history and basic idea about SVM. It's difference with Log Reg.

Apply the SVM to the previous problem, with multiple choices of kernel, and then plot the result.


## Exercise 

Regularisation of logistic regression could be used as an excise
