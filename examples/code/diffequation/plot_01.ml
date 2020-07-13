open Owl 

let f y t =
  let a = [|[|1.; -1.|];[|2.; -3.|]|]|> Mat.of_arrays in
  Mat.(a *@ y)

let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 2.; dt=1E-3})

let x0 = Mat.of_array [|-1.; 1.|] 2 1;;

let ts, ys = Owl_ode.Ode.odeint Owl_ode.Native.D.rk4 f x0 tspec ()

let _ =
  let h = Plot.create "plot_rk00.png" in
  let open Plot in
  set_xlabel h "ts";
  set_ylabel h "ys";
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.5; ] ts (Mat.row ys 0);
  plot ~h ~spec:[ RGB (219, 68,  55); LineStyle 2; LineWidth 2.5; ] ts (Mat.row ys 1);
  Plot.legend_on h [|"y[0]"; "y[1]"|];
  output h

let _ =
  let h = Plot.create "plot_02.png" in
  let open Plot in
  plot ~h ~spec:[ RGB (66, 133, 244); LineStyle 1; LineWidth 2.; ] (Mat.row ys 0) (Mat.row ys 1);
  output h