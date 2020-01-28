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

```
TODO: CODE and explanation.
```

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
```
CODE and result (no need to figure).
```

Another approach is from the perspective of function optimisation instead of regression. We can use the gradient descent optimisation method in Owl and apply it directly on the cost function.

```
CODE and result (no need to figure).
```
## Multiple Regression

Back to our McDonald's problem. We have seen how a new store's profit can be related to the population of it's surrounding, and we can even predict it given previous data. 
Now, remember that in the real world, population is not the only input features that affects the store's profit. Other factors such as existing stores in the area, proximity to retail parks, shopping centres, etc. also play a role. 
In that case, how can we extend our one-variable linear regression to the case of multiple variables?

The answer is very straight forward. We just use more parameters, so the model becomes:

$$h(\theta_0, \theta_1, \theta_2, \theta_3, ...) = \theta_0~ + \theta_1~x_1 + \theta_2~x_2 + \theta_3~x_3 ... $$

However, to list all the parameters explicitly is not a good idea, especially when the question requires considering thousands or even more features. 
Therefore, we use the vectorised format in the model:
$$h(\Theta) = \Theta~X^{(i)}$$,
where $\Theta = [\theta_0, \theta_1, \theta_2, \theta_3, ...]$, and $X^{(i)} = [1, x_1, x_2, x_3, ...]^T$ contains all the features from the $i$th row in data.

Accordingly, the cost function can be represented as:
$$ J(\Theta) = \frac{1}{2n}\sum_{i=1}^{n}(\Theta~X^{(i)} - y^{(i)})^2$$, 
where $y^{(i)}$ is the output variable value on the $i$th row of input data. 

The derivative and manual gradient descent are left as exercise. 
Here we only show an example of using the regression function Owl has provided.
Similar to the previous problem, we provide some data to this multiple variable problem. 
Part of the data are listed below:

| $x_1$ | $x_2$  | y |
| :--: | :--: | :--: |
| 1888 | 2 | 255000  |
| 1604 | 3 | 242900  |
| 1962 | 4 | 259900  |
| 3890 | 3 | 573900  |
| 1100 | 3 | 249900  |
| 1458 | 3 | 464500  |
| 2526 | 3 | 469000  |
| 2200 | 3 | 475000  |
| ...  | ... | ...   |

The problem has two different features. Again, by using the `ols` regression function in Owl, we can easily get the multi-variable linear model.
```
CODE + result
```

### Feature Normalisation

However, getting a result doesn't mean the end. Using the multi-variable regression problem as example, we would like to discuss an important issue about regression: feature normalisation.

Let's look at the multi-variable data again. Apparently, the first feature is magnitude larger than the second feature.
That means the model and cost function are dominated by the first feature, and a minor change of this column will have a disproportionally large impact on the model. 

To overcome this problem, we hope to pre-process the data before the regression, and normalise every features within about [-1, 1]. 
This step is also called feature scaling. 
There are many ways to do this, and one of them is the *mean normalisation*: for a column of features, calculate its mean, and divided by the difference between the largest value and smallest value, as shown in the code:

```
CODE: normalisation
```

Another benefit of performing data normalisation is that gradient descent can be accelerated. The illustration below shows the point.

IMAGES: two image, one ellipses, one circles. with arrows showing the steps towards center. 

Normalisation is not only used in regression, but also may other data analysis and machine learning tasks. 
For example, in computer vision tasks, an image is represented as an ndarray with three dimension. Each element represents an pixel in the image, with a value between 0 and 255. 
More often than not, this ndarray needs to be normalised in data pre-processed for the next step processing such as image classification.

### Analytical Solution

Before taking a look at some other forms of regression, let's discuss solution to the linear regression besides gradient descent.
It turns out that there is actually one close form solution to linear regression problems:

$$\Theta = (X^T~X)^{-1}X^Ty$$

Suppose the linear model contains $m$ features, and the input data contains $n$ rows, then here $X$ is a $n\times~(m+1)$ matrix representing the features data, and the output data $y$ is a $n\times~1$ matrix.
The reason there is m+1 columns in $X$ is that we need an extra constant feature for each data, and it equals to one for each data point. 

TODO: where does this equation come from?

With this method, we don't need to iterate the solutions again and again until converge. We can just compute the result with one pass with the given input data. 
This calculation can be efficiently performed in Owl using its Linear Algebra module.
Let's use the dataset from multi-variable regression again and perform the computation. 

```
CODE: close form solution.
```

TODO: compare the result with previous GD solution. 

Compared to the gradient descent solution, the methods does not require multiple iterations, and you also don't need to worry about hyper-parameters settings such as the choice of learning rate. 
On the other hand, however, this approach has its own problems. 

When the size of $X$, or the input data, becomes very large, the computation of large linear algebra operations such as matrix multiplication and inversion could become really slow.
Or even worse: your computer might don't even have enough memory to perform the computation.
Compare to it, gradient descent proves to work well even when the dataset is large. 

Besides, there could be no solution at all using this method. That's when the $X^T~X$ matrix is non-invertible, e.g. a singular matrix. 
That could be caused by multiple reasons. Perhaps some of the features are linear dependent, or that there are many redundant features. 
Then techniques such as choosing feature or regularisation are required. 

Most importantly, there is not always a close-form solution for you to use in other regression or machine learning problems. Gradient descent is a much more general solution. 

## Non-linear regressions 

If only the world is as simple as linear regression. But that's not to be. 
A lot of data can follow other patterns than a linear one. 
For example, checkout the dataset below:

IMAGE, the dataset that follows a convex curve. 

You can try to fit a line into these data, but it's quite likely that the result would be very fitting. 
And that requires non-linear models. 

In this section, we present two common non-linear regressions: the polynomial regression, and exponential regression. 
We shows how to use them with examples, and won't go into details of the math. Refer to [reference] for more details. 

In polynomial regression, the relationship between the feature $x$ and the output variable is modelled as an nth degree polynomial in the feature $x$:

$$ h(\Theta) = \theta_0 + \theta_1~x + \theta_2~x^2 + \theta_3~x^3 \ldots $$.

The model for exponential regression takes two parameters:

$$ h(\theta_0, \theta_1) = \theta_0~\theta_1^x$$.

Owl provides functions to do both form of regressions:

```
val exponential : ?i:bool -> arr -> arr -> elt * elt * elt

val poly : arr -> arr -> int -> arr
```

Let's look at how to use them in the code. The dataset is the same as in previous figure, contained in the file [data_03.csv](Link).

```
CODE: Polynomial. We limit that to 3th order. 
```

The result we get is: ... . That gives us the polynomial model $y = x + x^2 + x^3 + \epsilon$.

The code for exponential regression is similar:

```
CODE: exponential reg.
```

The result we get is ... That leads to a model: $y = ab^x + \epsilon$.

Let's see show the models works in fitting data:

IMAGE: data scatter point with two curves. 

## Regularisation

Regularisation is an important issue in regression, and is widely used in various regression models. 
The motivation of using regularisation comes from the problem of *over-fitting* in regression.
In statistics, over-fitting means a model is tuned too closely to a particular set of data and it may fail to predict future observations reliably.
Let' use the polynomial regression as an example.

IMAGE: two graphs with the same data, one is fit to the 2nd order, the other fit to the 4th order. 

Apparently, the second model fit too closely with the given data, and you can see that it won't make a good prediction of future output values.

To reduce the effect of higher order parameters, we can penalize these parameters in the cost function. We design the cost function so that the large parameter values leads to higher cost, and therefore by minimising the cost function we keep the parameters relatively small. 
Actually we don't need to change the cost functions dramatically. All we need is to add some extra bit at the end, for example, we can do this:

$$J(\Theta)=\frac{1}{2n}\left[ \sum_{i=1}{n}(h_{\Theta}(x^{(i)} - y^{(i)}))^2 + \lambda\sum_{j=1}^{m}\theta_j^2 \right].$$

Here the sum of squared parameter values is the penalty we add to the original cost function, and $lambda$ is a regularisation control parameter. 

That leads to a bit of change in the derivative of $J(\Theta)$ in using gradient descent:

$$\theta_j \leftarrow \theta_j - \frac{\alpha}{n} \left[ \sum_{i=1}^{m} (h_{\Theta}(x_i) - y_i)x_{i}^{(j)} - \lambda~\theta_j \right].$$

We can now apply the new update procedure in gradient descent code, with a polynomial model up to 4th order.

```
CODE and IMAGE (data, old overfitted line; new regularised line)
However, it is quite likely that we need to use a multiple variable regression as example, 
since I'm not sure if the functions in the next sections supports polynomial regression;
that requires the overfitting based on these multi-variable data be obvious.
```

We can see that by using regularisation the over-fitting problem is solved.
Note that we use linear regression in the equation, but regularisation is widely use in all kinds of regressions.

### Ols, Ridge, Lasso, and Elastic_net 

You might notice that Owl provides a series of functions other than `ols`:

```
val ridge : ?i:bool -> ?alpha:float -> arr -> arr -> arr array

val lasso : ?i:bool -> ?alpha:float -> arr -> arr -> arr array

val elastic_net : ?i:bool -> ?alpha:float -> ?l1_ratio:float -> arr -> arr -> arr array
```

What are these functions? The short answer is that: they are for regularisation in  regression using different methods.
The `ridge` cost function adds the L2 norm of $\theta$ as the penalty term: $\lambda\sum\theta^2$, which is what we have introduced. 
The `lasso` cost function is similar. It add the L1 norm, or absolute value of the parameter as penalty: $\lambda\sum|\theta|$.
This difference makes `lasso` to be able to allow for some coefficients to be zero, which is very useful for feature selection.
The `elastic_net` is proposed (by whom?) to combine the penalties of the previous two. What it adds is this:
$$\lambda(\frac{1-a}{2}\sum\theta^2 + a\sum|\theta|),$$ 
where $a$ is a control parameter between `ridge` and `lasso`.
The elastic net method aims to make the feature selection less dependent on input data. 

We can thus choose one of these functions to perform regression with regularisation on the dataset in the previous chapter.

```
CODE using ridge. 
```

## Logistic Regression

So far we have been predicting a value for our problems, whether using linear, polynomial or exponential regression. 
What if we don't care about is not the value, but a classification? For example, we have some historical medical data, and want to decide if a tumour is cancer or not based on several features.  

We can try to continue using linear regression, and the model can be interpreted as the possibility of one of these result.
But one problem is that, the prediction value could well be out of the bounds of [0, 1]. Then maybe we need some way to normalise the result to this range?

### Sigmoid Function 

The solution is to use the sigmoid function (or logistic function): $f(x) = \frac{1}{1 + e^{-x}}$.

As shown in the figure, this function project value within the range of [0, 1].
Applying this function on the returned value of a regression, we can get a model returns value within [0, 1].

$$h(\Theta) = f(\Theta~X) = \frac{1}{1 + e^{-\Theta~x}}$$.

Now we can interpret this model easily. The function value can be seen as possibility. If it is larger than 0.5, then the classification result is 0, otherwise it returns 1.
Remember that in logistic regression we only care about the classification. So for a 2-class classification, returning 0 and 1 is enough.

### Decision Boundary 

The physical meaning of classification is to draw a decision boundary in a hyper-plain. 
For example, if we are using a linear model $h$ within the logistic function, the linear model itself divide the points into two halves in the plain, as shown in the figure.  

IMAGE

If we use a non-linear polynomial model, then the plane is divided by curve lines. 
Suppose $h(x) = \theta_0 + \theta_1~x + \theta_2~x^2$.
According to the property of sigmoid function, "y=1 if g(h(x)) > 0.5" equals to "y=1 if h(x)>0", and thus the classification is divided by a circle:

IMAGE

Logistic regression uses the linear model as kernel.
If you believe your data won't be linearly separable, or you need to be more robust to outliers, you should look at SVM (see sections below) and look at one of the non-linear kernels. 

### Cost Function 

With the new model comes new cost function. 
Previously in linear regression we measure the cost with least square, or euclidean distance. 
Now in the logistic regression, we define its cost function as:

$$J_{\Theta}(h(x), y) = -log(h(x)), if y = 1, $$ or 
$$J_{\Theta}(h(x), y) = -log(1 - h(x)), if y = 0.$$ 

TODO: explain how to come up with this equation. About maximise the log likelihood. Refer to book scratch.

### Gradient Descent

Again the question is how to solve this terrible equation? 
Luckily, The sigmoid function has a nice property: its derivative is simple. 

$$\frac{\partial J(\Theta)}{\partial \theta_j} = \frac{1}{2n}\sum_{i=1}^{n}(\Theta~X^{(i)} - y^{(i)})^2$$, 

This gradient looks the same to that in linear regression, but it's actually different, since the definition of $h$ is actually different. 
Therefore, similar to linear regression, we only need to repeat this gradient descent step until converges.
The process is similar to that in linear regression so we will not dig into details again. 
Instead, we will use the function that Owl provides:

```
val logistic : ?i:bool -> arr -> arr -> arr array
```

We have prepared some data in [data_04.csv](Link). We can perform the regression with these data. 

```
CODE: logistic regression; using polynomial kernel.
```

IMAGE: plot the data and resulting boundary. 


### Multi-class classification 

Similar to the LR problem, we can hardly stop at only two parameters. What if we need to classified an object into one fo multiple classes?

One popular classification problem is the hand-written recognition task. It requires the model to recognise a 28x28 grey scale image, representing a hand-written number, to be one of ten numbers, from 0 to 9.
It is a widely used ABC task for Neural Networks, and we will also cover it later in Chapter DNN.
For now, we solve that from the logistic regression line of thought. 

TODO: Dataset description and Visualise

Similarly, we extend the cost function towards multi-class:

EQUATION

We can also use the generalised version of GD as before, or directly apply GD method in Owl:

```
CODE
```

Let's apply the model on test data:

Result.

Discussion on accuracy and possible improvement. Leave for exercise. 

## Support Vector Machine

Support Vector Machine (SVM) is a similar model to logistic regression, but uses non-linear kernel functions. 
(TODO: explain kernel).
SVMs are supervised learning models with associated learning algorithms that analyse data used for classification and regression analysis. 
Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. 
An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall. (COPY alert)

Explain the history and basic idea about SVM.

TODO: Apply the SVM to the previous problem, with multiple choices of kernel, and then plot the result.

## Exercise 

1. Manual gradient descent and optimizer on multiple variable problem
1. Regularisation of logistic regression could be used as an excise
1. In regularisation, what would happen if the $\lambda$ is extremely large?
