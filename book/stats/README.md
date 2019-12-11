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

```ocaml env=stats_01
let noise sigma = Stats.gaussian_rvs ~mu:0. ~sigma;;
let x = Array.init 50 (fun _ -> Stats.uniform_rvs 0. 10.);;
let y = Array.map (fun a -> 2.5 *. a +. noise 1.) x;;
let z = Array.map (fun a -> 2.5 *. a +. noise 8.) x;;
```

Plot the figure ...

```ocaml env=stats_01
let x' = Mat.of_array x 1 50;;
let y' = Mat.of_array y 1 50;;
let z' = Mat.of_array z 1 50;;

let h = Plot.create ~m:1 ~n:2 "plot_01.png" in

  Plot.subplot h 0 0;
  Plot.set_xlabel h "x";
  Plot.set_ylabel h "y (sigma = 1)";
  Plot.scatter ~h x' y';

  Plot.subplot h 0 1;
  Plot.set_xlabel h "x";
  Plot.set_ylabel h "z (sigma = 2)";
  Plot.scatter ~h x' z';

  Plot.output h;;
```

The figure is as below ...

<img src="images/stats/plot_01.png" alt="plot 01" title="Plot 01" width="700px" />


Now let us have a look at the correlation between `x` and `y`, as well as the correlation between `x` and `z`.

```ocaml env=stats_01
# Stats.corrcoef x y
- : float = 0.987944913889222565
# Stats.corrcoef x z
- : float = 0.757942970751708911
```


...


## Distributions


## Hypothesis Tests


TBD
