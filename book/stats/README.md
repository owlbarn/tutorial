# Statistical Functions

Statistics is an indispensable tool for data analysis, it helps us to gain the insights from data. The statistical functions in Owl can be categorised into three groups: descriptive statistics, distributions, and hypothesis tests.


## Descriptive Statistics

Descriptive statistics are used to summarise the characteristics of data. The commonly used ones are mean, variance, standard deviation, skewness, kurtosis, and etc.

We first draw one hundred random numbers which are uniformly distributed between 0 and 10. Here we use `Stats.uniform_rvs` function to generate numbers following uniform distribution.

```ocaml env=stats_00
let data = Array.init 100 (fun _ -> Stats.uniform_rvs 0. 10.);;
```

Then We use `mean` function calculate sample average. As we can see, it is around 5. We can also calculate higher moments such as variance and skewness easily with corresponding functions.

```ocaml env=stats_00
# Stats.mean data
- : float = 5.39282409364656168
# Stats.std data
- : float = 2.82512008501752376
# Stats.var data
- : float = 7.98130349476942147
# Stats.skew data
- : float = -0.100186978462459622
# Stats.kurtosis data
- : float = 1.90668234297861261
```

The following code calculates different central moments of `data`. A central moment is a moment of a probability distribution of a random variable about the random variable's mean. The zeroth central moment is always 1, and the first is close to zero, and the second is close to the variance. 

```ocaml env=stats_00
# Stats.central_moment 0 data
- : float = 1.
# Stats.central_moment 1 data
- : float = 5.3290705182007512e-17
# Stats.central_moment 2 data
- : float = 7.90149045982172549
# Stats.central_moment 3 data
- : float = -2.25903009746890815
```


## Correlations

Correlation studies how strongly two variables are related. There are different ways of calculating correlation. For the first example, let's look at Pearson correlation. 

`x` is our explanatory variable and we draw 50 random values uniformly from an interval between 0 and 10. Both `y` and `z` are response variables with a linear relation to `x`. The only difference is that we add different level of noise to the response variables. The noise values are generated from Gaussian distribution.

```ocaml env=stats_01
let noise sigma = Stats.gaussian_rvs ~mu:0. ~sigma;;
let x = Array.init 50 (fun _ -> Stats.uniform_rvs 0. 10.);;
let y = Array.map (fun a -> 2.5 *. a +. noise 1.) x;;
let z = Array.map (fun a -> 2.5 *. a +. noise 8.) x;;
```

It is easier to see the relation between two variables from a figure. Herein we use Owl's Plplot module to make two scatter plots.

```ocaml env=stats_01
(* convert arrays to matrices *)

let x' = Mat.of_array x 1 50;;
let y' = Mat.of_array y 1 50;;
let z' = Mat.of_array z 1 50;;

(* plot the figures *)

let h = Plot.create ~m:1 ~n:2 "plot_01.png" in

  Plot.subplot h 0 0;
  Plot.set_xlabel h "x";
  Plot.set_ylabel h "y (sigma = 1)";
  Plot.scatter ~h x' y';

  Plot.subplot h 0 1;
  Plot.set_xlabel h "x";
  Plot.set_ylabel h "z (sigma = 8)";
  Plot.scatter ~h x' z';

  Plot.output h;;
```

The subfigure 1 shows the functional relation between `x` and `y` whilst the subfiture 2 shows the relation between `x` and `z`. Because we have added higher-level noise to `z`, the points in the second figure are more diffused.

<img src="images/stats/plot_01.png" alt="plot 01" title="Plot 01" width="700px" />


Intuitively, we can easily see there is stronger relation between `x` and `y` from the figures. But how about numerically? In many cases, numbers are preferred because they are easier to compare with by a computer. The following snippet calculates the Pearson correlation between `x` and `y`, as well as the correlation between `x` and `z`. As we see, the smaller correlation value indicates weaker linear relation between `x` and `z` comparing to that between `x` and `y`.

```ocaml env=stats_01
# Stats.corrcoef x y
- : float = 0.987944913889222565
# Stats.corrcoef x z
- : float = 0.757942970751708911
```

...


## Distributions

Stats module supports many distributions. For each distribution, there is a set of related functions using the distribution name as their common prefix. Here we use Gaussian distribution as an exmaple to illustrate the naming convention.

* `gaussian_rvs` : random number generator.
* `gaussian_pdf` : probability density function.
* `gaussian_cdf` : cumulative distribution function.
* `gaussian_ppf` : percent point function (inverse of CDF).
* `gaussian_sf` : survival function (1 - CDF).
* `gaussian_isf` : inverse survival function (inverse of SF).
* `gaussian_logpdf` : logarithmic probability density function.
* `gaussian_logcdf` : logarithmic cumulative distribution function.
* `gaussian_logsf` : logarithmic survival function.

We generate two data sets in this example, and both contain 999 points drawn from different Gaussian distribution $\mathcal{N} (\mu, \sigma^{2})$. For the first one, the configuration is $(\mu = 1, \sigma = 1)$; whilst for the second one, the configuration is $(\mu = 12, \sigma = 3)$.

```ocaml env=stats_02
let noise sigma = Stats.gaussian_rvs ~mu:0. ~sigma;;
let x = Array.init 999 (fun _ -> Stats.gaussian_rvs ~mu:1. ~sigma:1.);;
let y = Array.init 999 (fun _ -> Stats.gaussian_rvs ~mu:12. ~sigma:3.);;
```

We can visualise the data sets using histogram plot as below. When calling `histogram`, we also specify 30 bins explicitly. You can also fine tune the figure using `spec` named parameter to specify the colour, x range, y range, and etc. We will discuss in details on how to use Owl to plot in a separate chapter.

```ocaml env=stats_02
(* convert arrays to matrices *)

let x' = Mat.of_array x 1 999;;
let y' = Mat.of_array y 1 999;;

(* plot the figures *)

let h = Plot.create ~m:1 ~n:2 "plot_02.png" in

  Plot.subplot h 0 0;
  Plot.set_ylabel h "frequency";
  Plot.histogram ~bin:30 ~h x';
  Plot.histogram ~bin:30 ~h y';

  Plot.subplot h 0 1;
  Plot.set_ylabel h "PDF p(x)";
  Plot.plot_fun ~h (fun x -> Stats.gaussian_pdf ~mu:1. ~sigma:1. x) (-2.) 6.;
  Plot.plot_fun ~h (fun x -> Stats.gaussian_pdf ~mu:12. ~sigma:3. x) 0. 25.;

  Plot.output h;;
```

In subplot 1, we can see the second data set has much wider spread. In subplot 2, we also plot corresponding the probability density functions of the two data sets.

<img src="images/stats/plot_02.png" alt="plot 02" title="Plot 02" width="700px" />

As an exercise, you can also try out other distributions like gamma, beta, chi2, and student-t distribution.


## Hypothesis Tests


## Order Statistics



TBD
