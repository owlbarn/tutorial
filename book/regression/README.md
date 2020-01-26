# Regression

An important part of Machine Learning (supervised learning).
General idea: given data, make predictions.

## Model and Error

### Standard Errors of Regression Coefficients

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
It's physical meaning.

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

### Feature Regularisation

One factor is hundred times larger than the other variables. That's bad.

Regularisation

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


## Support Vector Machine
