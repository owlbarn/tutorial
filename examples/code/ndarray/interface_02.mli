open Owl.Dense.Ndarray.Generic

val sequential : ('a, 'b) kind -> ?a:'a -> ?step:'a -> int array -> ('a, 'b) t

val linspace : ('a, 'b) kind -> 'a -> 'a -> int -> ('a, 'b) t

val logspace : ('a, 'b) kind -> ?base:float -> 'a -> 'a -> int -> ('a, 'b) t