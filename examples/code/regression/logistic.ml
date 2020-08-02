
let plot_sigmoid () = 
  let h = Plot.create "sigmoid.png" in 
  Plot.plot_fun ~h ~spec:[ LineWidth 2.; RGB (0,0,255) ] Maths.sigmoid (-6.) 6.;
  Plot.output h


let data = Owl_io.read_csv ~sep:',' "ex2data1.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let x = Mat.get_slice [[];[0;1]] data
let y = Mat.get_slice [[];[2]] data

let theta = Regression.D.logistic ~i:true x y

(* the result is 
- : Owl_algodiff_primal_ops.D.arr array =
[|
        C0 
R0    5.63 
R1 4.45631 
; 
        C0 
R0 4.75192 
|]

which feels wrong. 
*)

(** SVM *)

let plot_logistic data f range = 
  let neg_idx = Mat.filter_rows (fun m -> Mat.get m 0 2 = 0.) data in 
  let neg_data = Mat.get_fancy [ L (Array.to_list neg_idx); R [] ] data in 
  let pos_idx = Mat.filter_rows (fun m -> Mat.get m 0 2 = 1.) data in 
  let pos_data = Mat.get_fancy [ L (Array.to_list pos_idx); R [] ] data in 
  let h = Plot.create "reg_svm.png" in
  Plot.(scatter ~h ~spec:[ Marker "#[0x2217]"; MarkerSize 5. ] 
      (Mat.get_slice [[];[0]] neg_data) 
      (Mat.get_slice [[];[1]] neg_data));
  Plot.(scatter ~h ~spec:[ Marker "#[0x2295]"; MarkerSize 5. ] 
      (Mat.get_slice [[];[0]] pos_data) 
      (Mat.get_slice [[];[1]] pos_data));
  (* Plot.plot_fun ~h f range.(0) range.(1); *)
  Plot.output h


(**
val theta : Owl_algodiff_primal_ops.D.arr array =
  [|
         C0 
R0 0.864249 
R1 0.963497 
; 
         C0 
R0 0.907506 
|]
 *)

(* let f x = -.(0.86 *. x +. 0.91) /. 0.96



let data = Owl_io.read_csv ~sep:',' "ex2data2.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let x = Mat.get_slice [[];[0;1]] data
let y = Mat.get_slice [[];[2]] data *)


let generate_data () =
  let open Mat in
  let c = 500 in
  let x1 = (gaussian c 2 *$ 2.) in
  let a, b = float_of_int (Random.int 15), float_of_int (Random.int 15) in
  let x1 = map_at_col (fun x -> x +. a) x1 0 in
  let x1 = map_at_col (fun x -> x +. b) x1 1 in
  let x2 = (gaussian c 2 *$ 2.) in
  let a, b = float_of_int (Random.int 15), float_of_int (Random.int 15) in
  let x2 = map_at_col (fun x -> x +. a) x2 0 in
  let x2 = map_at_col (fun x -> x +. b) x2 1 in
  let y1 = create c 1 ( 1.) in
  let y2 = create c 1 ( -1.)in
  let x = concat_vertical x1 x2 in
  let y = concat_vertical y1 y2 in
  x, y

let x, y = generate_data () 

let theta = Regression.D.svm ~i:true x y 

let p = Mat.(theta.(0) @= theta.(1))

let f x = (Mat.get p 0 0 *. x +. Mat.get p 2 0) /. (Mat.get p 1 0 *. (-1.)) 

let plot_svm data f range = 
  let neg_idx = Mat.filter_rows (fun m -> Mat.get m 0 2 < 0.) data in 
  let neg_data = Mat.get_fancy [ L (Array.to_list neg_idx); R [] ] data in 
  let pos_idx = Mat.filter_rows (fun m -> Mat.get m 0 2 > 0.) data in 
  let pos_data = Mat.get_fancy [ L (Array.to_list pos_idx); R [] ] data in 
  let h = Plot.create "reg_svm.png" in
  Plot.(scatter ~h ~spec:[ Marker "#[0x2217]"; MarkerSize 5. ] 
      (Mat.get_slice [[];[0]] neg_data) 
      (Mat.get_slice [[];[1]] neg_data));
  Plot.(scatter ~h ~spec:[ Marker "#[0x2295]"; MarkerSize 5. ] 
      (Mat.get_slice [[];[0]] pos_data) 
      (Mat.get_slice [[];[1]] pos_data));
  Plot.plot_fun ~h f range.(0) range.(1);
  Plot.output h

let data = Mat.concat_horizontal x y;;