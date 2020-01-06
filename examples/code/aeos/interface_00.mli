

module Sin : sig
    type t = {
        mutable name  : string;
        mutable param : string;
        mutable value : int;
        mutable input : int array array;
        mutable y     : float array
    }
    (* Tuner type definition. *)
    val make : unit -> t
    (* Create the tuner. *)
    val tune : t -> unit 
    (* Tuning process. *)
    val save_data : t -> unit
    (* Save tuned data to csv file for later analysis. *)
    val to_string : t -> string
    (* Convert the tuned parameter(s) to string to be written on file *)
end