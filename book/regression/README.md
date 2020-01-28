# Regression

Regression is an important topic in statistical modelling and machine learning. 
It's about modelling problems that includes one or more variables (also called "features" or "predictors") and making predictions of another variable ("output variable") based on previous data of predictors. 

Regression analysis includes a wide range of models, from linear regression to isotonic regression, each with different theory background and application fields.
Introducing all these models are beyond this book.
In this chapter, we focus on several common form of regressions, mainly linear regression and logistic regression. We introduce their basic ideas, how they are supported in Owl, and how to use them to solve problems. 

## Model and Error

### Standard Errors of Regression Coefficients

### Model Selection

Feature selection: REFER to ISL book Chap 6.

## Linear Regression

Linear regression models the relationship of the features and output variable with a linear model. 
Let's start with a simple problem where only one feature needs to be considered. 

### Problem: Where to locate a new McDonald's restaurant?

McDonald's is no doubt one of the most successful fast food chains in the world. Up to 2018, it has already had more than 37, 000 stores world wide, and surely more is being built as you are reading.
One question then is: where to locate a new McDonald's restaurant? 

According to its [website](https://www.mcdonalds.com/gb/en-gb/help/faq/18665-how-do-you-decide-where-to-open-a-new-restaurant.html#), a lot of factors are in play: area population, existing stores in the area, proximity to retail parks, shopping centres, etc. 
Now let's simplified this problem by asserting that the potential profit is only related to area population. 
Suppose you are the decision maker in McDonald's, and also have access to data of each branch store (profit, population around this branch). 
Now linear regression would be a good friend when you are deciding where to locate your next branch.

Here list a part of the data (TODO: link to csv file):

| Profit | Population |
| :----: | :--------: |
| 20.27 | 21.76 |
| 5.49  | 4.26 |
| 6.32  | 5.18 |
| 5.56  | 3.08 |
| 18.94 | 22.63 |
| 12.82 | 13.50 |
| ... | ... |


Visualising these data can present a clear view.

TODO: CODE and IMAGE in Owl 

According to this figure, there is a clear trend that larger population and larger profit are co-related together. But precisely how?

### Cost Function

Let's start with a linear model that assumes the the relationship between these two variables be formalised as: 
$$ y = \theta_0~ + \theta_1~x_1 + \epsilon$$,
where $y$ denotes the profit we want to predict, and input variable $x_1$ is the population number in this example. 
Since modelling can hardly make a perfect match with the real data, we use $\epsilon$ to denote the error between our prediction and the data. 
Specifically, we represent the prediction part as $h(\theta_0, \theta_1)$:
$$h(\theta_0, \theta_1) = \theta_0~ + \theta_1~x_1$$

The $\theta_0$ and $\theta_1$ are the parameters of this model. Mathematically they decide a line on a plain. 
We can now choose randomly these parameters and see how the result works, and some of these guesses are just bad intuitively.
Our target is to choose suitable parameters so that the line is *close* to data we observed. 

TODO: Figures with data, and also some random lines. Maybe three figures, and two of them are bad fit.

How do we define the line being "close" to the observed data then?
One frequently used method is to use the *ordinary least square* to minimizes the sum of squared distances between the data and line.
We have shown the "$x$-$y$" pairs in the data above, and we represent the total number of data pairs with $n$, and thus the $i$'th pair of data can be represented with $x_i$ and $y_i$.
With these notations, we can represent a metric to represent the *closeness* as:

$$J(\theta_0, \theta_1) = \frac{1}{2n}\sum_{i=1}^{n}(h_{\theta_1, \theta_0}(x_i^2) - y_i)$$

In regression, we call this function the *cost function*. It measures how close the models are to ideal cases, and our target is thus clear: find suitable $\theta$ parameters to minimise the cost function. 

**TIPS**: Why do we use least square in the cost function? Physically, the cost function $J$ represents the average distance of each data point to the line -- by "distance" we mean the the euclidean distace. between a data point and the point on the line with the same x-axis. 
A reasonable solution can thus be achieved by minimising this average distance.
On the other hand, from the statistical point of view, minimizing the sum of squared errors leads to maximizing the likelihood of the data.
TODO: explain the relationship between maximum likelihood estimation and least square.

### Solving Problem with Gradient Descent

To give a clearer view, we can visualise the cost function with a contour graph:

IMAGE (CODE if we can do that in Owl)

We can see that cost function varies with parameters $\theta_0$ and $\theta_1$ with a bowl-like shape curve surface. 
It is thus natural to recall the gradient descent we have introduced in the previous chapter, and use it to find the minimal point in this bowl-shape surface.

Recall from previous chapter that gradient descent works by starting at one point on the surface, and move gradually towards certain *direction* at some *step size*, and hopefully can converge at a local minimum. 
Let's use a fixed step size $\alpha$, and the direction at certain point on the surface can be gotten by using partial derivative on the surface. 
Therefore, what we need to do is to apply this update process iteratively for both $\theta$ parameters:
$$ \theta_j \leftarrow \theta_j - \alpha~\frac{\partial}{\partial \theta_j}~J(\theta_0, \theta_1), $$
where $i$ is 1 or 2.

This process may seem terrible at first sight, but we can calculate it as:

$$ \theta_0 \leftarrow \theta_0 - \frac{\alpha}{n}\sum_{i=1}^{m} (h_{\theta_0, \theta_1}(x_i) - y_i)x_{i0}, $$
and 
$$ \theta_1 \leftarrow \theta_1 - \frac{\alpha}{n}\sum_{i=1}^{m} (h_{\theta_0, \theta_1}(x_i) - y_i)x_{i1}.$$

Here the $x_i0$ and $x_i1$ are just different input features of the $i$-th row in data. Since currently we only focus on one feature in our problem, $x_i0 = 1$ and $x_i1 = x_i$.
Following these equations, you can manually perform the gradient descent process until it converges.
Here is the code. 

TODO: CODE and explanation.

By executing the code, we can get a pair of parameters: $\theta_0 = xxx$ and $\theta-1 = xxx$.
To check if they indeed are suitable parameters, we can visualise them against the input data.
The resulting figure shows a line that aligns with input data.

TODO: PLOT: the resulting line against data samples. 

Of course, there is no need to use to manually solve a linear regression problem with Owl. 
It has already provides high-level regression functions for use. 
For example, `ols` uses the odinary least square method we have introduced to perform linear regression.

```
val ols : ?i:bool -> arr -> arr -> arr array
```

And we can use that to directly solve the problem, and the resulting parameters are similar to what we have get manually.

CODE and result (no need to figure).

Another approach is from the perspective of function optimisation instead of regression. We can use the gradient descent optimisation method in Owl and apply it directly on the cost function.

CODE and result (no need to figure).

## Multiple Regression

Now the same problem, but variables goes from one to multiple. What would you do?
Basically similar, but instead of $\theta_0$, $\theta_1$, $\theta_2$, ... now we need the vectorised representation: $\Theta$:

$$ J(\Theta) = \frac{1}{2n}(X\Theta - y)^2$$.

Next we focus on some issues.

### Feature Normalisation

One factor is hundred times larger than the other variables. That's bad.

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

Discussion on accuracy and possible improvement. Leave for exercise. 

## Support Vector Machine

It's a similar idea to logistic regression.

Explain the history and basic idea about SVM. It's difference with Log Reg.

Apply the SVM to the previous problem, with multiple choices of kernel, and then plot the result.


## Exercise 

Regularisation of logistic regression could be used as an excise
