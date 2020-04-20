

let _ = 

let h = Plot.create ~m:1 ~n:3 "activations.png" in
Plot.set_pen_size h 1.5;
Plot.subplot h 0 0;
Plot.set_title h "tanh";
Plot.plot_fun ~h ~spec:[ RGB (66,133,244) ] Maths.tanh (-10.) 10.;

Plot.subplot h 0 1;
Plot.set_title h "relu";
Plot.plot_fun ~h ~spec:[ RGB (66,133,244) ] Maths.relu (-10.) 10.;

Plot.subplot h 0 2;
Plot.set_title h "softsign";
Plot.plot_fun ~h ~spec:[ RGB (66,133,244) ]
    (fun x -> x /. (1. +. Maths.abs x)) (-10.) 10.;

Plot.output h
