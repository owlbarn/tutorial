let data = Owl_io.read_csv ~sep:' ' "boston.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let rm = Mat.get_slice [[];[5]] data
let medv = Mat.get_slice [[];[13]] data

let plot () =
  let h = Plot.create "foo.png" in
  Plot.scatter ~h rm medv;
  Plot.output h

let exp () = 
  let a = Regression.D.exponential rm medv in 
  let a0 = Mat.get a 0 0 in 
  let a1 = Mat.get a 1 0 in 
  let a2 = Mat.get a 2 0 in 
  fun x -> a0 +. a1 *. x +. a2 *. x *. x 

let plot_exp () = 
  let h = Plot.create "reg_exp.png" in
  Plot.scatter ~h lstat medv;
  Plot.plot_fun ~h (exp ()) 4. 8.;
  Plot.output h