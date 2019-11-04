open Dataframe

let fname = "estate.csv" in
let d = (of_csv ~sep:',' fname)
  .?(fun row -> unpack_string row.(7) = "Condo")
  .?(fun row -> unpack_string row.(4) = "2")
in
Owl_pretty.pp_dataframe Format.std_formatter d