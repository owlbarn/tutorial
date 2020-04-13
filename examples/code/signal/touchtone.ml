let plot_tone x y filename = 
  let h = Plot.create filename in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "time(s)";
  Plot.set_ylabel h "signal magnitude";
  Plot.plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1] x y;
  Plot.output h


let plot_tone_freq x y filename = 
  let h = Plot.create filename in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "frequency";
  Plot.set_ylabel h "";
  Plot.plot ~h ~spec:[ RGB (0, 0, 255); LineStyle 1] x y;
  Plot.output h

(** Part I *)

let data = Owl_io.read_csv ~sep:',' "touchtone.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays
let data = Mat.div_scalar data 128.
let fs = 8192.
let x = Mat.div_scalar (Mat.sequential 1 (Arr.shape data).(1)) fs

let _ = plot_tone x y "plot_tone.png"

let yf = Owl_fft.D.rfft data
let y' = Dense.Ndarray.Z.(abs yf |> re)
let n = (Arr.shape y').(1)
let x' = Mat.linspace 0. (fs /. 2.) n

let _ = plot_tone_freq x' y' "plot_tone_freq.png"

(** Part II *)

let data2 = Arr.get_slice [[];[0; 4999]] data
let x2 = Mat.div_scalar (Mat.sequential 1 (Arr.shape data2).(1)) fs

let _ = plot_tone x y "plot_tone2.png"

let yf2 = Owl_fft.D.rfft data2
let y2' = Dense.Ndarray.Z.(abs yf2 |> re)
let n2 = (Arr.shape y2').(1)
let x2' = Mat.linspace 0. (fs /. 2.) n2

let _ = plot_tone_freq x' y' "plot_tone_freq2.png"