# Signal Processing

TODO: refer to https://scipy.github.io/devdocs/tutorial/signal.html. Try to implement its examples first. 

To cover the full scope of this topic, even briefly, requires a whole book. 
This chapter we focuses on Fourier Transform. We introduce its basic idea, and then demonstrate how Owl support FFT with examples and applications.
We also cover the relationship between FFT and Convolution, and filters. 

## Discrete Fourier Transform

One theme in numerical applications is the transformation of equations into a coordinate system so that the original question can be easily decoupled and simplified. 
One of the most important such transformation is the *Fourier Transform*, which decomposes a function of time into its constituent frequencies. 
In computer based numerical computation, signals are often represented in a discrete way, i.e. a finite sequence of sampled data, instead of continuous. 
In that case, the method is called *Discrete Fourier Transform* (DFT). 

All these sound too vague. Let's look at an example.
Think about an audio that lasts for 10 seconds. 
This audio can surely be described in the *time domain*, which means plotting its sound intensity against time as x axis.
On the other hand, maybe less obviously, the sound can also be described in the *frequency domain*. For example, if all the 10 seconds are filled with only playing the A# note, then you can describe this whole audio with one frequency number: 466.16 Hz. If it's a C note, then the number is 523.25 Hz, etc. 
The thing is that, the real-world sound is not always so pure, they are quite likely compounded from different frequencies. Perhaps this 10 seconds are about water flowing, or wind whispering, what frequencies it is built from then?

That's where Fourier Transform comes into play. It captures the idea of converting the two forms of representing a signal: in time domain and in frequency domain. 
We can represent a signal with the values of some quantity $h$ as a function of time: $h(t)$, or this signal can be represented by giving its amplitude $H$ as function of frequency: $H(f)$. We can think they are two representation of the same thing, and Fourier Transform change between them:

$$ h(f) = \int H(f)\exp^{-2\pi~ift}df$$
$$ H(f) = \int h(t)\exp^{2\pi~ift}dt$$

To put it simply: suppose Alice mix a unknown number of colour together, and let Bob to guess what those colours are, then perhaps Bob need a Fourier Transform machine of sorts.

You might be wondering, it's cool that I can recognise how a sound is composed, but so what? 
Think of a classic example where you need to remove some high pitch noisy from some music. By using DFT, you can easily find out the frequency of this noisy, remove this frequency, and turn the signal back to the time domain by using something a reverse process.
Not just sound. 
You can also get a noisy image, recognise its noise by applying Fourier Transform, remove the noises, and reconstruct the image without noise. 
We will show such examples later.

Actually, the application of FT is more than on sound or image signal processing.
We all use Fourier Transform every day without knowing it: mobile phones, image and audio compression, communication networks, large scale numerical physics and engineering, etc.
It is the cornerstone fo computational mathematics. 
One important reason of its popularity is that it has an efficient algorithm in implementation: Fast Fourier Transform.


## Fast Fourier Transform

One problem with DFT is that if you follow its definition in implementation, the algorithm computation complexity would be $\mathcal{O}(n^2)$, since it involves a dense $n$ by $n$ matrix multiplication.
It means that DFT doesn't scale well with input size.
The Fast Fourier Transform algorithm, first formulated by Gauss in 1805 and then developed by James Cooley and John Tukey in 1965, drops the complexity down to $\mathcal{O}(n\log{}n)$.
To put it in a simple way, the FFT algorithm finds out that, any DFT can be represented by the sum of two sub-DFTs: one consists of the elements on even index in the signal, and the other consists of elements on odd positions, and most importantly, you only need to compute one such DFT, so on and so forth. Therefore the computation can be reduced to a $log$ level. 

To introduce Fourier Transform in detailed math and analysis of its properties is beyond the scope of this book, we encourage the readers to refer to other classic textbook on this topic [@phillips2003signals].
In this chapter, we focus on introducing how to use FFT in Owl and its applications with Owl code. Hopefully these materials are enough to interest you to investigate more. 

Owl provides these basic FFT functions:

| Functions | Description
| --------- |:----------| 
| `fft ~axis x` | Compute the one-dimensional discrete Fourier Transform |
| `ifft ~axis x` | Compute the one-dimensional inverse discrete Fourier Transform |
| `rfft ~axis otyp x` | Compute the one-dimensional discrete Fourier Transform for real input |
| `irfft ~axis ~n otyp x` | Compute the one-dimensional inverse discrete Fourier Transform for real input |

The implementation of the FFT module in Owl interfaces to the [Fastest Fourier Transform in the West(FFTW)](http://www.fftw.org/) library, which is known as, as its name indicates, the fastest free software implementation of the fast Fourier transform.
One interesting fact is that, though this is a C library, its highly optimised code is generated using OCaml.

TODO: introduce the FFTW we interface to a bit. It's a challenge to make FFT fast, and why FFTW works fast, etc.

### Examples

TODO: refer to https://scipy.github.io/devdocs/tutorial/fft.html. 

#### 1-D Discrete Fourier transforms

`fft` and `ifft`.


```text
# let a = [|1.;2.;1.;-1.;1.5;1.0|]
val a : float array = [|1.; 2.; 1.; -1.; 1.5; 1.|]
# let b = Arr.of_array a [|6|] |> Dense.Ndarray.Generic.cast_d2z
val b : (Complex.t, complex64_elt) Dense.Ndarray.Generic.t =

       C0      C1      C2       C3        C4      C5
R (1, 0i) (2, 0i) (1, 0i) (-1, 0i) (1.5, 0i) (1, 0i)

# let c = Owl_fft.D.fft b 
val c : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

         C0                 C1                 C2                  C3                C4                C5
R (5.5, 0i) (2.25, -0.433013i) (-2.75, -1.29904i) (1.5, 1.94289E-16i) (-2.75, 1.29904i) (2.25, 0.433013i)

# let d = Owl_fft.D.ifft c
val d : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

                 C0                C1                 C2                  C3                  C4                C5
R (1, 1.38778E-17i) (2, 1.15186E-15i) (1, -8.65641E-17i) (-1, -1.52188E-15i) (1.5, 1.69831E-16i) (1, 2.72882E-16i)

```

The example plots the FFT of the sum of two sines.

```text
# module G = Dense.Ndarray.Generic
module G = Owl.Dense.Ndarray.Generic

# let n = 600. (* sample points *)
val n : float = 600.
# let t = 1. /. 800. (* sample spacing *)
val t : float = 0.00125
# let x = Arr.linspace 0. (n *. t) (int_of_float n) 
val x : Arr.arr =

  C0         C1         C2         C3         C4         C595     C596     C597     C598 C599
R  0 0.00125209 0.00250417 0.00375626 0.00500835 ... 0.744992 0.746244 0.747496 0.748748 0.75

# let y1 = Arr.((50. *. 2. *. Owl_const.pi) $* x |> sin)
val y1 : Arr.arr =

  C0       C1       C2      C3       C4         C595    C596     C597     C598        C599
R  0 0.383289 0.708033 0.92463 0.999997 ... 0.999997 0.92463 0.708033 0.383289 1.27376E-14

# let y2 = Arr.(0.5 $* ((80. *. 2. *. Owl_const.pi) $* x |> sin))
val y2 : (float, float64_elt) Owl_dense_ndarray_generic.t =

  C0       C1       C2      C3       C4          C595     C596      C597      C598         C599
R  0 0.294317 0.475851 0.47504 0.292193 ... -0.292193 -0.47504 -0.475851 -0.294317 -2.15587E-14

# let y = Arr.(y1 + y2) |> G.cast_d2z 
val y : (Complex.t, complex64_elt) G.t =

       C0             C1            C2            C3            C4               C595           C596           C597            C598               C599
R (0, 0i) (0.677606, 0i) (1.18388, 0i) (1.39967, 0i) (1.29219, 0i) ... (0.707804, 0i) (0.449591, 0i) (0.232182, 0i) (0.0889723, 0i) (-8.82117E-15, 0i)

# let yf = Owl_fft.D.fft y
val yf : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

             C0                    C1                    C2                    C3                  C4                     C595                 C596                   C597                   C598                   C599
R (5.01874, 0i) (5.02225, 0.0182513i) (5.03281, 0.0366004i) (5.05051, 0.0551465i) (5.0755, 0.073992i) ... (5.108, -0.0932438i) (5.0755, -0.073992i) (5.05051, -0.0551465i) (5.03281, -0.0366004i) (5.02225, -0.0182513i)

# let z = Dense.Ndarray.Z.(abs yf |> re)
val z : Dense.Ndarray.Z.cast_arr =

       C0      C1      C2      C3      C4        C595    C596    C597    C598    C599
R 5.01874 5.02228 5.03294 5.05081 5.07604 ... 5.10886 5.07604 5.05081 5.03294 5.02228

```

Plot the result. 
```text
# let h = Plot.create "plot_001.png" in 
  let xa = Arr.linspace 1. 600. 600 in
  Plot.plot ~h xa z;
  Plot.output h 
- : unit = ()
```

![Plot example 01](images/signal/plot_001.png "plot_001"){.align-center width=70%}

`rfft` and `irfft` is for performing fft on real input.

```text
# let a = [|1.; 2.; 1.; -1.; 1.5; 1.0|]
val a : float array = [|1.; 2.; 1.; -1.; 1.5; 1.|]
# let b = Arr.of_array a [|6|]
val b : Arr.arr =
  C0 C1 C2 C3  C4 C5
R  1  2  1 -1 1.5  1


# let c = Owl_fft.D.rfft b
val c : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =
         C0                 C1                 C2        C3
R (5.5, 0i) (2.25, -0.433013i) (-2.75, -1.29904i) (1.5, 0i)

# let d = Owl_fft.D.irfft c
val d : (float, float64_elt) Owl_dense_ndarray_generic.t =

  C0 C1 C2 C3  C4 C5
R  1  2  1 -1 1.5  1

```

```text
# let a = [|1.; 2.; 1.; -1.; 1.5;|]
val a : float array = [|1.; 2.; 1.; -1.; 1.5|]
# let b = Arr.of_array a [|5|]
val b : Arr.arr =
  C0 C1 C2 C3  C4
R  1  2  1 -1 1.5

# let c = Owl_fft.D.rfft b
val c : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =
         C0                  C1                   C2
R (4.5, 0i) (2.08156, -1.6511i) (-1.83156, 1.60822i)

```

#### N-D Discrete Fourier transforms

The owl FFT functions also applies to multi-dimensional arrays, such as matrix.
Example: the fft matrix.

```text
# let a = Dense.Matrix.Z.eye 5
val a : Dense.Matrix.Z.mat =

        C0      C1      C2      C3      C4
R0 (1, 0i) (0, 0i) (0, 0i) (0, 0i) (0, 0i)
R1 (0, 0i) (1, 0i) (0, 0i) (0, 0i) (0, 0i)
R2 (0, 0i) (0, 0i) (1, 0i) (0, 0i) (0, 0i)
R3 (0, 0i) (0, 0i) (0, 0i) (1, 0i) (0, 0i)
R4 (0, 0i) (0, 0i) (0, 0i) (0, 0i) (1, 0i)

# let b = Owl_fft.D.fft a
val b : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

        C0                      C1                      C2                      C3                      C4
R0 (1, 0i)                 (1, 0i)                 (1, 0i)                 (1, 0i)                 (1, 0i)
R1 (1, 0i)  (0.309017, -0.951057i) (-0.809017, -0.587785i)  (-0.809017, 0.587785i)   (0.309017, 0.951057i)
R2 (1, 0i) (-0.809017, -0.587785i)   (0.309017, 0.951057i)  (0.309017, -0.951057i)  (-0.809017, 0.587785i)
R3 (1, 0i)  (-0.809017, 0.587785i)  (0.309017, -0.951057i)   (0.309017, 0.951057i) (-0.809017, -0.587785i)
R4 (1, 0i)   (0.309017, 0.951057i)  (-0.809017, 0.587785i) (-0.809017, -0.587785i)  (0.309017, -0.951057i)

```

IMAGE: plot x and y in to circle-like shape


## Applications of using FFT

REFER: *Numerical in Matlab* book.

Unlike the regression chapter, **make sure these examples work first**, then perhaps fill in some content. 
They don't have to be all finished for this round. But you have to be sure about the workload.
Do not dig deep into the topic FFT. That takes a whole book and more. 

### Find period of sunspots

Build data from a [dataset](http://sidc.oma.be/silso/newdataset) from the Solar Influences Data Center.
Explain the background of dataset, meaning of Wolfer index, and the dataset itself. 

IMAGE: visualise the dataset.

We can see there is a cycle. We want to know how long it is.

```
CODE
```

IMAGE: absolute fft'ed result vs. cycles per year. (periodogram)

Change x-axis into years per cycle

IMAGE

and we can see 11 years is a prominent cycle. 

### Decipher the Tone 

The tune of phone is combination of two different frequencies: 

IMAGE (Reference required)

Data set from Matlab book. Also 1-D signal. 

IMAGE: visualise the data.

The problem is to find out the dial number. 
Let's take the first segment as example. 

```
CODE
```

IMAGE: only two prominent frequencies. According to the table, we can be sure that the first digit in the phone number is xx.

The whole number can be used as exercise. 

### Image Processing

(from scipy book; NR chap 12.6; or the data-driven book; or from the elegant scipy book)

## FFT and Convolution

Explain the connection clearly. Compare FFT and existing convolution methods if possible. 


## Filtering

IMPLEMENTATION required: Currently none implemented; need to pick 1-2 to implement and demonstrate the general idea of filtering. 
Also, it's better we show the idea that "FFT is the basis of filtering" here.

## References
