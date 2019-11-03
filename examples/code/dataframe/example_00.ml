let fname = "funding.csv" in
let types =  [|"s";"s";"f";"s";"s";"s";"s";"f";"s";"s"|] in
let df = Dataframe.of_csv ~sep:',' ~types fname in
Owl_pretty.pp_dataframe Format.std_formatter df