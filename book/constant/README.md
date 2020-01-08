# Constants and Metric System

In many scientific computing problems, numbers are not abstract but reflect the realistic meanings. In other words, these numbers only make sense on top of a well-defined metric system.



## What Is A Metric System

For example, when we talk about the distance between two objects, I write down a number `30`. But what does `30` mean in reality? Is it meters, kilometers, miles, or lightyears? Another example, what is the speed of light? Well, this is really depends on what metrics you are using, e.g., `km/s`, `m/s`, `mile/h` ...

Things can get really messy in computation if we do not unify the metric system in a numerical library. The translation between different metrics are often important in real-world application. I do not intend to dig deep into the metric system here, so please read online articles to find out more, e.g., [Wiki: Outline of the metric system](https://en.wikipedia.org/wiki/Outline_of_the_metric_system).


## Four Systems in Owl

There are four metrics adopted in Owl, and all of them are wrapped in `Owl.Const` module.

* `Const.SI`: [International System of Units](https://en.wikipedia.org/wiki/International_System_of_Units)

* `Const.MKS`: [MKS system of units](https://en.wikipedia.org/wiki/MKS_system_of_units)

* `Const.CGS`: [Centimetre–gram–second system of units](https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units)

* `Const.CGSM`: [Electromagnetic System of Units](https://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units#CGS_approach_to_electromagnetic_units)

All the metrics defined in these four systems can be found in the interface file [owl_const.mli](https://github.com/owlbarn/owl/blob/master/src/base/core/owl_const.mli).

In general, SI is much newer and recommended to use. International System of Units (French: Système international d'unités, SI), historically also called the MKSA system of units for metre–kilogram–second–ampere. The SI system of units extends the MKS system and has 7 base units, by expressing any measurement of physical quantities using fundamental units of Length, Mass, Time, Electric Current, Thermodynamic Temperature, Amount of substance and Luminous Intensity, which are Metre, Kilogram, Second, Ampere, Kelvin, Mole and Candela respectively.

Here is a nice [one-page poster](http://www.npl.co.uk/upload/pdf/units-of-measurement-poster.pdf) from NPL to summarise what have talked about SI.


<img src="images/constant/metric_001.png" alt="owl_constants_001" title="Owl constants" width="700px" />


## SI Prefix

As a computer scientist, you must be familiar with `kilo`, `mega`, `giga` these prefixes. SI system includes the definition of these prefixes as well. But be careful (especially for computer science guys), the base is `10` instead of 2.

These prefixes are defined in `Const.Prefix` module.

```ocaml

  Const.Prefix.peta;;
  Const.Prefix.tera;;
  Const.Prefix.giga;;
  Const.Prefix.mega;;
  Const.Prefix.kilo;;
  Const.Prefix.hecto;;

```

## Some Examples

Now we can safely talk about the distance between two objects, light of speed, and many other real-world stuff with atop of a well-defined metric system in Owl. See the following examples.

```ocaml

  Const.SI.light_year;;     (* light year in SI system *)
  Const.MKS.light_year;;    (* light year in MKS system *)
  Const.CGS.light_year;;    (* light year in CGS system *)
  Const.CGSM.light_year;;   (* light year in CGSM system *)

```

How about Planck's constant?

```ocaml

  Const.SI.plancks_constant_h;;     (* in SI system *)
  Const.MKS.plancks_constant_h;;    (* in MKS system *)
  Const.CGS.plancks_constant_h;;    (* in CGS system *)
  Const.CGSM.plancks_constant_h;;   (* in CGSM system *)

```

## International System of Units

Now that you know how to use constants, we will use the International System of Units (SI) as an example to show the constants we include in Owl.

### Mathematical constants 

| Constant name  | Explanation  |
| -------------: |:-------------| 
| `pi`| Pi | 
| `e` | Natural constant |
| `euler` | Euler constant |

Besides these constants, we also provide some frequently used computations based on them, including: 

- `log2e`  ($\log_2 e$)
- `log10e`  ($\log_10 e$)
- `loge2`  ($\log_e 2$)
- `loge10`  ($\log_e 10$)
- `logepi`  ($\log_e \pi$)
- `pi2` ($2\pi$)
- `pi4` ($4\pi$)
- `pi_2` ($\pi / 2$)
- `pi_4` ($\pi / 4$)
- `sqrt1_2` ($\sqrt{\frac{1}{2}}$)
- `sqrt2` ($\sqrt{2}$)
- `sqrt3` ($\sqrt{3}$)
- `sqrtpi` ($\sqrt{\pi}$)

### Physical constants


| Constant name  | Explanation  |
| :------------- |:-------------| 
| `speed_of_light`| speed of light in vacuum |
| `gravitational_constant` | Newtonian constant of gravitation |
| `plancks_constant_h` | Planck constant | 
| `plancks_constant_hbar` | reduced Planck constant | 
| `astronomical_unit` | one astronomical unit in meters |
| `light_year` | one light year in meters |
| `parsec` | one light year in meters |
| `grav_accel` | standard acceleration of gravity |
| `electron_volt` | electron volt |
| `mass_electron` | electron mass |
| `mass_muon` | muon mass |
| `mass_proton` | proton mass |
| `mass_neutron` | neutron mass |
| `rydberg` | Rydberg constant |
| `boltzmann` | Boltzmann constant |
| `molar_gas` | molar gas constant |
| `standard_gas_volume` | molar volume of ideal gas (273.15 K, 100 kPa) |
| `bohr_radius` | Bohr radius |
| `stefan_boltzmann_constant` | Stefan-Boltzmann constant |
| `thomson_cross_section` | Thomson cross section in square metre |
| `bohr_magneton` | Bohr magneton in Joules per Tesla |
| `nuclear_magneton` | Nuclear magneton in Joules per Tesla |
| `electron_magnetic_moment` | electron magnetic moment in Joules per Tesla | 
| `proton_magnetic_moment` | proton magnetic moment in Joules per Tesla | 
| `faraday` | Faraday constant |
| `electron_charge` | electron volt in Joules |
| `vacuum_permittivity` | vacuum electric permittivity |
| `vacuum_permeability` | vacuum magnetic permeability |
| `debye` | one debye in coulomb metre |
| `gauss` | one gauss in maxwell per square metre |


### Time

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `minute` | one minute in seconds |
| `hour` | one hour in seconds |
| `day` | one day in seconds |
| `week` | one week in seconds | 

### Length

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `inch` | one inch in metres | 
| `foot` | one foot in metres | 
| `yard` | one yard in metres | 
| `mile` | one mile in metres | 
| `mil` | one mil in metres | 
| `fathom` | one fathom in metres | 
| `point` | one point in metres | 
| `micron` | one micron in metres | 
| `angstrom` | one angstrom in metres | 
| `nautical_mile` | one [nautical mile](https://en.wikipedia.org/wiki/Nautical_mile) in metres |


### Area 

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `hectare` | one hectare in square meters |
| `acre` | one acre in square meters |
| `barn` | one barn in square meters |

### Volume 

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `liter` | one liter in cubic meters |
| `us_gallon` | one gallon (US) in cubic meters |
| `uk_gallon` | one gallon (UK) in cubic meters |
| `canadian_gallon` | one Canadian gallon in cubic meters |
| `quart` | one quart (US) in cubic meters |
| `cup` | one cup (US) in cubic meters |
| `pint` |  one pint in cubic meters |
| `fluid_ounce` | one fluid ounce (US) in cubic meters |
| `tablespoon` | one tablespoon in cubic meters |

### Speed 

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `miles_per_hour` | miles per hour in metres per second |
| `kilometers_per_hour` | kilometres per hour in metres per second |
| `knot` | one knot in metres per second |


### Mass

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `pound_mass` | one pound (avoirdupous) in kg |
| `ounce_mass` | one ounce in kg |
| `metric_ton` | 1000 kg |
| `ton` | one short ton in kg |
| `uk_ton` | one long ton in kg |
| `troy_ounce` | one Troy ounce in kg |
| `carat` | one carat in kg |
| `unified_atomic_mass` | atomic mass constant |
| `solar_mass` | one solar mass in kg | 

### Force

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `newton` | base unit |
| `gram_force` | one gram force in newtons | 
| `kilogram_force` | one kilogram force in newtons | 
| `pound_force` | one pound force in newtons | 
| `poundal` | one poundal in newtons | 
| `dyne` | one dyne in newtons |


### Energy

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `joule` | base unit |
| `calorie` | one calorie (thermochemical) in Joules |
| `btu` | one British thermal unit (International Steam Table) in Joules |
| `therm` | one therm (US) in Joules |
| `erg` | one erg in Joules |

### Power

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `horsepower` | one horsepower in watts |


### Pressure

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `bar` | one bar in pascals | 
| `std_atmosphere` | standard atmosphere in pascals | 
| `torr` | one torr (mmHg) in pascals | 
| `meter_of_mercury` | one metre of mercury in pascals | 
| `inch_of_mercury` | one inch of mercury in pascals | 
| `inch_of_water` | one inch of water in pascals | 
| `psi` | one psi in pascals |


### Viscosity

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `poise` | base unit |
| `stokes` | base unit |


### Luminance 

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `stilb` | Candela per square metre, base unit |
| `lumen` | Candela square radian, base unit |
| `phot` | base unit |
| `lux` | one lux in phots |
| `footcandle` | one footcandle in phots |
| `lambert` | base unit |
| `footlambert` | one footlambert in lambert |


### Radioactivity

| Constant name  | Explanation  |
| :------------- |:-------------| 
| `curie` | one curie in nuclear transformations per second |
| `roentgen` | one roentgen in ampere second per kilogram |
| `rad` | one rad in erg per gram |
