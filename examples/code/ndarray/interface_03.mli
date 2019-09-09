open Owl.Dense.Ndarray.Generic

val init : ('a, 'b) kind -> int array -> (int -> 'a) -> ('a, 'b) t

val init_nd : ('a, 'b) kind -> int array -> (int array -> 'a) -> ('a, 'b) t