(**
  * URL: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
  * The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
 prices and the demand for clean air', J. Environ. Economics & Management,
 vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
 ...', Wiley, 1980.   N.B. Various transformations are used in the table on
 pages 244-261 of the latter.

 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
  *)

let data = Owl_io.read_csv ~sep:' ' "boston.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let lstat = Mat.get_slice [[];[12]] data
let medv = Mat.get_slice [[];[13]] data


let plot_01 () =
  let h = Plot.create "boston.png" in
  Plot.scatter ~h lstat medv;
  Plot.output h


let poly () = 
  let a = Regression.D.poly lstat medv 2 in 
  let a0 = Mat.get a 0 0 in 
  let a1 = Mat.get a 1 0 in 
  let a2 = Mat.get a 2 0 in 
  fun x -> a0 +. a1 *. x +. a2 *. x *. x 


let poly4 () = 
  let a = Regression.D.poly lstat medv 4 in 
  let a0 = Mat.get a 0 0 in 
  let a1 = Mat.get a 1 0 in 
  let a2 = Mat.get a 2 0 in 
  let a3 = Mat.get a 3 0 in 
  let a4 = Mat.get a 4 0 in 
  fun x -> a0 +. a1 *. x +. a2 *. x *. x
    +. a3 *. x *. x *. x +. a4 *. x *. x *. x *. x 


let poly6 () = 
  let a = Regression.D.poly lstat medv 6 in 
  let a0 = Mat.get a 0 0 in 
  let a1 = Mat.get a 1 0 in 
  let a2 = Mat.get a 2 0 in 
  let a3 = Mat.get a 3 0 in 
  let a4 = Mat.get a 4 0 in 
  let a5 = Mat.get a 5 0 in 
  let a6 = Mat.get a 6 0 in 
  fun x -> a0 +. a1 *. x +. a2 *. x *. x
    +. a3 *. x *. x *. x +. a4 *. x *. x *. x *. x 
    +. a5 *. x *. x *. x *. x *. x 
    +. a6 *. x *. x *. x *. x *. x *. x 


let plot_poly () = 
  let h = Plot.create "reg_poly.png" in
  Plot.scatter ~h lstat medv;
  Plot.plot_fun ~h (poly ()) 2. 36.;
  Plot.output h


let plot_poly4 () = 
  let h = Plot.create "reg_poly4.png" in
  Plot.scatter ~h lstat medv;
  Plot.plot_fun ~h (poly4 ()) 2. 36.;
  Plot.output h


let plot_poly6 () = 
  let h = Plot.create "reg_poly6.png" in
  Plot.scatter ~h lstat medv;
  Plot.plot_fun ~h (poly6 ()) 2. 36.;
  Plot.output h