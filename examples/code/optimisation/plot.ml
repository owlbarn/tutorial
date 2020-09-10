
let six_hump x y =
  (4. -. 2.1 *. x *. x +. (Maths.pow x 4.) /. 3.) *. x *. x +.
  x *. y +. 4. *. (y *. y -. 1.) *. y *. y

let plot_01 () =
  let x, y = Mat.meshgrid (-2.) 2. (-1.) 1. 60 60 in
  let z = Mat.(map2 six_hump x y) in
  let h = Plot.create "gradient.png" in
  Plot.(mesh ~h ~spec:[ NoMagColor; Contour ] x y z);
  Plot.set_xlabel h "";
  Plot.set_ylabel h "";
  Plot.set_zlabel h "";
  Plot.output h

let plot_02 () =
  let x, y = Mat.meshgrid (-2.) 2. (-1.) 1. 60 60 in
  let z = Mat.(map2 six_hump x y) in
  let h = Plot.create "contour.png" in
  Plot.(contour ~h x y z);
  Plot.set_xlabel h "x0";
  Plot.set_ylabel h "x1";
  Plot.set_zlabel h "";
  Plot.output h
