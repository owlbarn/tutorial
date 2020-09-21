#require "owl-top,owl-plplot";;
#require "owl-ode";;

open Owl
open Owl_plplot
open Bigarray
open Owl_ode
(* open Owl_ode_odepack
open Owl_ode_sundials *)

let () = Printexc.record_backtrace false
let () =
  Owl_base_stats_prng.init 89;
  Owl_stats_prng.init 89
