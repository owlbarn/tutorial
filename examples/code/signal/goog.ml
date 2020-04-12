let data = Owl_io.read_csv ~sep:',' "goog.csv"
let data = Array.map (fun x -> 
    Array.map float_of_string (Array.sub x 1 6))
    (Array.sub data 1 (Array.length data - 1)) 
    |> Mat.of_arrays

let y = Mat.get_slice [[];[3]] data

(** 1. *)

let filter = Mat.of_array (Array.make 10 0.1 ) 1 10

let y' = Mat.mapi (fun i _ ->
  let r = Mat.get_fancy [R [i; i+9]; R []] y in 
  Mat.dot filter r |> Mat.sum'
) (Mat.get_slice [[0; (Arr.shape y).(0) - 10]; []] y)


let plot_goog y y' = 
  let n = (Arr.shape x).(0) in 
  let x = Mat.sequential n 1 in 
  let h = Plot.create "plot_goog.png" in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "date";
  Plot.set_ylabel h "Google stock price ($)";
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] x y;
  Plot.plot ~h ~spec:[ RGB (0,0,255); LineStyle 2] x y';
  Plot.(legend_on h ~position:NorthWest [|"original"; "smooth"|]);
  Plot.output h


(** 2. *)

let yf = Owl_fft.D.rfft ~axis:0 y

let n = (Dense.Ndarray.Z.shape yf).(0)

let z = Dense.Ndarray.Z.zeros [|n-5; 1|]

let _ = Dense.Ndarray.Z.set_slice [[5;n-1];[]] yf z

let y2 = Owl_fft.D.irfft ~axis:0 yf

(* plot_goog y y2 *)


(** 3. *)

let y3  = Arr.reshape y [|1;251;1|]
let f3  = Arr.reshape filter [|10;1;1|]
let y3' = Arr.conv1d y3 f3 [|1|]