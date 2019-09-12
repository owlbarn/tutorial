# Dataframe for Tabular Data


Dataframe is a popular way to manipulate data. It originates from R's dataframe and is widely implemented in many mainstream libraries such as Pandas. Essentially, a dataframe is simple container of the data that can be represented as a table.

Different from the matrices in numerical computing, data stored in a dataframe are not necessarily numbers but a mixture of different types. The flexibility of dataframe largely comes from the dynamic typing inherently offered in a language. Due to OCaml's static type checking, this poses greatest challenges to Owl when I was trying to introduce the similar functionality.

It becomes an art when balancing between flexibility and efficiency in designing the programming interface. This article covers the design of Dataframe module and its basic usage.


## Basic Concepts

The dataframe functionality is implemented in Owl's `Dataframe <https://github.com/owlbarn/owl/blob/master/src/base/misc/owl_dataframe.mli>`_ module. Owl views a dataframe as a collection of time series data, and each series corresponds to one column in the table. All series must have the same length and each has a unique column head. In the following, we use series and column interchangeably.

Owl packs each series into a unified type called `series` and stores them in an array. As you can already see, dataframe is column-based so accessing columns is way more efficient than accessing rows. The Dataframe module only provides basic functionality to create, access, query, iterate the data in a frame. We need to combine dataframe with the numerical functions in `Stats` module to reach its full capability. Essentially, Pandas is a bundle of table manipulation and basic statistical functions.


## Create Frames

Dataframes can be created in various ways. ``Dataframe.make`` is the core function if we can to create a frame dynamically. For example, the following code creates a frame consisting of three columns include "name", "age", and "salary" of four people.


```ocaml env=env_dataframe_1
  let name = Dataframe.pack_string_series [|"Alice"; "Bob"; "Carol"; "David"|]
  let age = Dataframe.pack_int_series [|20; 25; 30; 35|]
  let salary = Dataframe.pack_float_series [|2200.; 2100.; 2500.; 2800.|]
  let frame = Dataframe.make [|"name"; "age"; "salary"|] ~data:[|name; age; salary|]
```

If you run the code in `utop`, Owl can pretty print out the dataframe in the following format. If the frame grows too long or too wide, Owl is smart enough to truncate them automatically and present the table nicely in the toplevel.

```ocaml env=env_dataframe_1
# Owl_pretty.pp_dataframe Format.std_formatter frame;;
  +-----+---+------
    name age salary 
  +-----+---+------
R0 Alice  20  2200. 
R1   Bob  25  2100. 
R2 Carol  30  2500. 
R3 David  35  2800. 
- : unit = ()
```

In fact, you do not necessarily need to pass in the data when calling ``make`` function. You can make an empty frame by just passing in head names.

.. code-block:: ocaml

  let empty_frame = Dataframe.make [|"name"; "age"; "salary"|];;


Try the code, you will see Owl prints out an empty table.


## Manipulate Frames

There are a comprehensive set of table manipulation functions implemented in Dataframe module. I will go through them briefly in this section.

Now that Owl allows us to create empty frames, it certainly provides functions to dynamically add new columns.

```ocaml env=env_dataframe_1
  let job = Dataframe.pack_string_series [|"Engineer"; "Driver"; "Lecturer"; "Manager"|] in
  Dataframe.append_col frame job "job";;

  let gender = Dataframe.pack_string_series [|"female"; "male"; "female"; "male"|] in
  Dataframe.append_col frame gender "gender";;

  let location = Dataframe.pack_string_series [|"Cambridge, UK"; "Helsinki, FIN"; "London, UK"; "Prague, CZ"|] in
  Dataframe.append_col frame location "location";;
```

From the output, we can see that the "job" column has been appended to the end of the previously defined dataframe.

```ocaml env=env_dataframe_1
# Owl_pretty.pp_dataframe Format.std_formatter frame;;

  +-----+---+------+--------+------+-------------
    name age salary      job gender      location 
  +-----+---+------+--------+------+-------------
R0 Alice  20  2200. Engineer female Cambridge, UK 
R1   Bob  25  2100.   Driver   male Helsinki, FIN 
R2 Carol  30  2500. Lecturer female    London, UK 
R3 David  35  2800.  Manager   male    Prague, CZ 
- : unit = ()
```

We can even concatenate two dataframes. Depending on concatenating direction, there are a couple of things worth our attention.

- When two dataframes are concatenated vertically, they must have the same number of columns and consistent column types. The head names of the first argument will be used in the new dataframe.
- When two dataframes are concatenated horizontally, they must have the same number of rows. All the columns of two dataframes must have unique names.

For example, the following code add two new entries to the table by concatenating two dataframes vertically.


```ocaml env=env_dataframe_1
  let name = Dataframe.pack_string_series [|"Erin"; "Frank"|];;
  let age = Dataframe.pack_int_series [|22; 24|];;
  let salary = Dataframe.pack_float_series [|3600.; 5500.;|];;
  let job = Dataframe.pack_string_series [|"Researcher"; "Consultant"|];;
  let gender = Dataframe.pack_string_series [|"male"; "male"|];;
  let location = Dataframe.pack_string_series [|"New York, US"; "Beijing, CN"|];;
  let frame_1 = Dataframe.make [|"name"; "age"; "salary"; "job"; "gender"; "location"|]
                ~data:[|name; age; salary; job; gender; location|];;
  let frame_2 = Dataframe.concat_vertical frame frame_1;;
```

The new dataframe looks like the following.

```ocaml env=env_dataframe_1
# Owl_pretty.pp_dataframe Format.std_formatter frame_2;;

  +-----+---+------+----------+------+-------------
    name age salary        job gender      location 
  +-----+---+------+----------+------+-------------
R0 Alice  20  2200.   Engineer female Cambridge, UK 
R1   Bob  25  2100.     Driver   male Helsinki, FIN 
R2 Carol  30  2500.   Lecturer female    London, UK 
R3 David  35  2800.    Manager   male    Prague, CZ 
R4  Erin  22  3600. Researcher   male  New York, US 
R5 Frank  24  5500. Consultant   male   Beijing, CN 
- : unit = ()
```


However, if you just want to append one or two rows, the previous method seems a bit overkill. Instead, you can call `Dataframe.append_row` function.


.. code-block:: ocaml

  let new_row = Dataframe.([|pack_string "Erin"; pack_int 22; pack_float 2300.|]);;
  Dataframe.append_row frame new_row;;


There are also functions allow you to retrieve the properties.


.. code-block:: ocaml

  val copy : t -> t          (* return the copy of a dataframe. *)

  val row_num : t -> int     (* return the number of rows. *)

  val col_num : t -> int     (* return the number of columns. *)

  val shape : t -> int * int (* return the shape of a dataframe. *)

  val numel : t -> int       (* return the number of elements. *)

  ...


The module applies several optimisation techniques to accelerate the operations on dataframes. Please refer to the API reference for the complete function list.


## Query Frames

## Iterate, Map, and Filter

## Read/Write CSV Files

## Infer Type and Separator

## What Is Next

TBD
