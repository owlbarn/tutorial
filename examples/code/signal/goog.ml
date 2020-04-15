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
  let n = (Arr.shape y).(0) in 
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


let plot_goog_fft y y' = 
  let n = (Arr.shape y).(0) in 
  let x = Mat.sequential n 1 in 
  let h = Plot.create "plot_goog_fft.png" in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "date";
  Plot.set_ylabel h "Google stock price ($)";
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] x y;
  Plot.plot ~h ~spec:[ RGB (0,0,255); LineStyle 2] x y';
  Plot.(legend_on h ~position:NorthWest [|"original"; "FFT smooth"|]);
  Plot.output h

(* plot_goog_fft y y2 *)


(** 3. *)

let y3  = Arr.reshape y [|1;251;1|]
let f3  = Arr.reshape filter [|10;1;1|]
let y3' = Arr.conv1d y3 f3 [|1|]


(** Gaussian *)

let gaussian_kernel sigma = 
  let truncate = 4. in 
  let radius = truncate *. sigma +. 0.5 |> int_of_float in
  let r = float_of_int radius in 
  let x = Mat.linspace (-.r) r (2 * radius + 1) in 
  let f a = Maths.exp (-0.5 *. a ** 2. /. (sigma *. sigma)) in 
  let x = Mat.map f x in
  Mat.(div_scalar x (sum' x))

let filter = gaussian_kernel 3.

let n = (Arr.shape filter).(1)

let y' = Mat.mapi (fun i _ ->
  let r = Mat.get_fancy [R [i; i+n-1]; R []] y in 
  Mat.dot filter r |> Mat.sum'
) (Mat.get_slice [[0; (Arr.shape y).(0) - n]; []] y)


let plot_goog2 y y' = 
  let n = (Arr.shape y).(0) in 
  let x = Mat.sequential n 1 in 
  let ny = (Arr.shape y').(0) in
  let x' = Mat.sequential ~a:10. ny 1 in (* ~a:10 is a hack *)
  let h = Plot.create "plot_goog_gauss.png" in
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.set_xlabel h "date";
  Plot.set_ylabel h "Google stock price ($)";
  Plot.plot ~h ~spec:[ RGB (255,0,0); LineStyle 1] x y;
  Plot.plot ~h ~spec:[ RGB (0,0,255); LineStyle 2] x' y';
  Plot.(legend_on h ~position:NorthWest [|"original"; "Gaussian smooth"|]);
  Plot.output h


let plot_goog2 x y y' = 
  let n = (Arr.shape x).(0) in 
  let x = Mat.sequential n 1 in 
  let h = Plot.create "plot_goog_gauss.png" in
  Plot.plot ~h  x y;
  Plot.output h