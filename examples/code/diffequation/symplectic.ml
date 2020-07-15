
(**
 Src: 
 [1] https://blogs.mathworks.com/cleve/2012/06/19/symplectic-spacewar/
 [2] https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/exm/chapters/orbits.pdf
 *)


(* Normal *)
let f y t =
  let a = [|[|0.; 1.|];[|-1.; 0.|]|]|> Mat.of_arrays in
  Mat.(a *@ y)

let x0 = Mat.of_array [|1.; 0.|] 2 1;;
let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 10.; dt=1E-2});;
let ts, ys = Owl_ode.Ode.odeint Owl_ode.Native.D.euler f x0 tspec ()


(* Symplectic *)
let f (x, p) _ : Owl.Mat.mat =
  Mat.(div_scalar (add (pow_scalar x 2.) (pow_scalar p 2.)) 2.)
let x0 = Owl.Mat.of_array [| 1. |] 1 1
let p0 = Owl.Mat.of_array [| 0. |] 1 1
let tspec = Owl_ode.Types.(T1 {t0 = 0.; duration = 2.; dt=1E-2});;
let t, y1, y2 = Ode.odeint Symplectic.D.ruth3 f (x0, p0) tspec ()