Visualisation
===========================================================

This chapter teaches you how to use visualisation functionality in Owl.

```ocaml
# let f x = Maths.sin x /. x in
  let h = Plot.create "plot_003.png" in

  Plot.set_foreground_color h 0 0 0;
  Plot.set_background_color h 255 255 255;
  Plot.set_title h "Function: f(x) = sine x / x";
  Plot.set_xlabel h "x-axis";
  Plot.set_ylabel h "y-axis";
  Plot.set_font_size h 8.;
  Plot.set_pen_size h 3.;
  Plot.plot_fun ~h f 1. 15.;

  Plot.output h;;
- : unit = ()
```

The generated figure is as below.

<img src="images/visualisation/plot_003.png" alt="plot 003" title="Plot example one" width="300px" />

Another example follows,

```ocaml
# let x, y = Mat.meshgrid (-2.5) 2.5 (-2.5) 2.5 50 50 in
  let z = Mat.(sin ((x * x) + (y * y))) in
  let h = Plot.create ~m:2 ~n:3 "plot_023.png" in

  Plot.subplot h 0 0;
  Plot.(mesh ~h ~spec:[ ZLine XY ] x y z);

  Plot.subplot h 0 1;
  Plot.(mesh ~h ~spec:[ ZLine X ] x y z);

  Plot.subplot h 0 2;
  Plot.(mesh ~h ~spec:[ ZLine Y ] x y z);

  Plot.subplot h 1 0;
  Plot.(mesh ~h ~spec:[ ZLine Y; NoMagColor ] x y z);

  Plot.subplot h 1 1;
  Plot.(mesh ~h ~spec:[ ZLine Y; Contour ] x y z);

  Plot.subplot h 1 2;
  Plot.(mesh ~h ~spec:[ ZLine XY; Curtain ] x y z);

  Plot.output h;;
- : unit = ()
```

<img src="images/visualisation/plot_023.png" alt="plot 023" title="Plot example two" width="600px" />

Finished.
