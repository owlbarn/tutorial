
let data = Owl_io.read_csv ~sep:',' "data_01.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let x = Mat.get_slice [[];[1]] data
let y = Mat.get_slice [[];[0]] data

let plot_01 () =
  let h = Plot.create "regdata.png" in
  Plot.scatter ~h x y;
  Plot.output h


let plot_02 () =
  let h = Plot.create ~m:1 ~n:3 "reg_options.png" in
  Plot.subplot h 0 0;
  Plot.scatter ~h x y;
  Plot.plot_fun ~h ~spec:[ RGB (0,255,0) ] (fun a -> a +. 2.) 3. 20.;
  Plot.subplot h 0 1;
  Plot.scatter ~h x y;
  Plot.plot_fun ~h ~spec:[ RGB (0,255,0) ] (fun a -> a *. (-1.) +. 15.) 0. 12.;
  Plot.subplot h 0 2;
  Plot.scatter ~h x y;
  Plot.plot_fun ~h ~spec:[ RGB (0,255,0) ] (fun a -> a *. (0.5) +. 10.) 0. 20.;
  Plot.output h 