open Dataframe

let fname = "estate.csv" in
let df = Dataframe.of_csv fname in
Owl_pretty.pp_dataframe Format.std_formatter df