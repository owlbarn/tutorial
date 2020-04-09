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


let plot_logistic data = 
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
  Plot.output h

let data = Owl_io.read_csv ~sep:',' "ex2data2.csv"
let data = Array.map (fun x -> Array.map float_of_string x) data |> Mat.of_arrays

let x = Mat.get_slice [[];[0;1]] data
let y = Mat.get_slice [[];[2]] data

