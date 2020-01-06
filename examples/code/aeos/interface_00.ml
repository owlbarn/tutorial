module Sin = struct
  type t =
    { mutable name : string
    ; mutable param : string
    ; mutable value : int
    ; mutable input : int array array
    ; mutable y : float array
    }

  let make () =
    { name = "sin"
    ; param = "OWL_OMP_THRESHOLD_SIN"
    ; value = 0
    ; input = [| [| 0 |] |]
    ; y = [| 0. |]
    }


  let tune _t = ()

  let save_data _t = ()

  let to_string _t =  ""
end