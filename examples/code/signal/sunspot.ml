let data = Owl_io.read_csv ~sep:';' "sunspot_full.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let x = Mat.get_slice [[];[0]] data
let y = Mat.get_slice [[];[1]] data


let plot_sunspot x y = 
  let h = Plot.create "plot_sunspot.png" in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "Date";
  Plot.set_ylabel h "Sunspot number";
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] x y;
  Plot.output h

let y' = Owl_fft.D.rfft ~axis:0 y
let y' = Dense.Ndarray.Z.get_slice [[1; (Dense.Ndarray.Z.shape y').(0) - 1];[]] y'
let p = Dense.Ndarray.Z.abs y' |> Dense.Ndarray.Z.re
let n = (Arr.shape p).(0)
let f = Arr.(mul_scalar (linspace 0. 1. n') 0.5)


let plot_sunspot_freq f p = 
  let h = Plot.create ~m:1 ~n:2 "plot_sunspot_freq.png" in
  Plot.set_pen_size h 3.;
  Plot.subplot h 0 0;
  Plot.set_xlabel h "cycle/year";
  Plot.set_ylabel h "squared power";
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] f p;

  Plot.subplot h 0 1;
  Plot.set_xlabel h "year/cycle";
  Plot.set_ylabel h "squared power";
  let f' = Arr.(scalar_div 1. (get_slice [[1; Stdlib.(n'-1)]] f)) in
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] f' p;
  Plot.set_xrange h 0. 40.;
  Plot.output h


(**
let foo = Dense.Ndarray.Z.re y'
let bar = Dense.Ndarray.Z.im y'

let plot_test x y = 
  let h = Plot.create "test.png" in
  Plot.scatter ~h x y;
  Plot.output h
 *)