
let alpha01 = [|0.1; 0.1|]
let alpha02 = [|3.; 3.|]

let n = 100

let generate_data alpha =
  let x = Mat.zeros 1 n in
  let y = Mat.zeros 1 n in
  for i = 0 to n - 1 do
    let a = Stats.dirichlet_rvs ~alpha in
    Mat.set x 0 i (a.(0));
    Mat.set y 0 i (a.(1));
  done;
  x, y

let _ =
  let h = Plot.create ~m:1 ~n:2 "dirichlet.png" in
  Plot.subplot h 0 0;
  let x, y = generate_data alpha01 in
  Plot.set_title h "alpha=0.1";
  Plot.(scatter ~h ~spec:[ MarkerSize 3. ] x y);
  Plot.subplot h 0 1;
  Plot.set_title h "alpha=3";
  let x, y = generate_data alpha02 in
  Plot.(scatter ~h ~spec:[ MarkerSize 3. ] x y);
  Plot.output h
