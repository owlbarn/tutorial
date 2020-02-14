# Signal Processing

Signal processing is an electrical engineering sub-field that focuses on analysing, modifying and synthesizing signals such as sound, images and biological measurements. (WIKI)
It covers a wide range of techniques.

In this chapter we mainly focus on Fourier Transform, the core idea in signal processing and modern numerical computing. 
We introduce its basic idea, and then demonstrate how Owl support FFT with examples and applications.
We also cover the relationship between FFT and Convolution, and filters. 

## Discrete Fourier Transform

One theme in numerical applications is the transformation of equations into a coordinate system so that the original question can be easily decoupled and simplified. 
One of the most important such transformation is the *Fourier Transform*, which decomposes a function of time into its constituent frequencies. 

All these sound too vague. Let's look at an example.
Think about an audio that lasts for 10 seconds. 
This audio can surely be described in the *time domain*, which means plotting its sound intensity against time as x axis.
On the other hand, maybe less obviously, the sound can also be described in the *frequency domain*. For example, if all the 10 seconds are filled with only playing the A# note, then you can describe this whole audio with one frequency number: 466.16 Hz. If it's a C note, then the number is 523.25 Hz, etc. 
The thing is that, the real-world sound is not always so pure, they are quite likely compounded from different frequencies. Perhaps this 10 seconds are about water flowing, or wind whispering, what frequencies it is built from then?

That's where Fourier Transform comes into play. It captures the idea of converting the two forms of representing a signal: in time domain and in frequency domain. 
We can represent a signal with the values of some quantity $h$ as a function of time: $h(t)$, or this signal can be represented by giving its amplitude $H$ as function of frequency: $H(f)$. We can think they are two representation of the same thing, and Fourier Transform change between them:

$$ h(f) = \int H(f)\exp^{-2\pi~ift}df$$ {#eq:signal:ft01}
$$ H(f) = \int h(t)\exp^{2\pi~ift}dt$$ 

To put it simply: suppose Alice mix a unknown number of colour together, and let Bob to guess what those colours are, then perhaps Bob need a Fourier Transform machine of sorts.

In computer-based numerical computation, signals are often represented in a discrete way, i.e. a finite sequence of sampled data, instead of continuous. 
In that case, the method is called *Discrete Fourier Transform* (DFT). 
Suppose we have a complex vector $y$ as signal, which contains $n$ elements, then to get the Fourier Transform vector $Y$, the discrete form of [@eq:signal:ft01] can be expressed with:

$$ Y_k = \sum_{j=0}^{n-1}y_j~\omega^{jk},$$ {#eq:signal:dft01}
$$ y_k = \frac{1}{n}\sum_{j=0}^{n-1}Y_k~\omega^{-jk},$$

where $\omega = \exp{-2\pi~i/n}$ and $i = \sqrt{-1}$. $j$ and $k$ are indices that go from 0 to $n-1$ and $k = 0, 1, ..., n -1$.

We highly suggest you to checkout the [video](https://www.youtube.com/watch?v=spUNpyF58BY) that's named "But what is the Fourier Transform? A visual introduction" produced by 3Blue1Brown. 
It shows how this [@eq:signal:dft01] of Fourier Transform comes into being with beautiful and clear illustration.
(TODO: follow the video, explain the idea of FT clearly, not just smashing an equation into readers' faces.)

(maybe TODO: we can perhaps implement a naive DFT process, both to illustrate the theory with OCaml code, and lay the foundation for understanding FFT. Refer to: Matlab NC book, Chap 8.2)

You might be wondering, it's cool that I can recognise how a sound is composed, but so what? 
Think of a classic example where you need to remove some high pitch noisy from some music. By using DFT, you can easily find out the frequency of this noisy, remove this frequency, and turn the signal back to the time domain by using something a reverse process.

Not just sound; you can also get a noisy image, recognise its noise by applying Fourier Transform, remove the noises, and reconstruct the image without noise. 
We will show such examples later.

Actually, the application of FT is more than on sound or image signal processing.
We all use Fourier Transform every day without knowing it: mobile phones, image and audio compression, communication networks, large scale numerical physics and engineering, etc.
It is the cornerstone fo computational mathematics. 
One important reason of its popularity is that it has an efficient algorithm in implementation: Fast Fourier Transform.

## Fast Fourier Transform

One problem with DFT is that if you follow its definition in implementation, the algorithm computation complexity would be $\mathcal{O}(n^2)$, since it involves a dense $n$ by $n$ matrix multiplication.
It means that DFT doesn't scale well with input size.
The Fast Fourier Transform algorithm, first formulated by Gauss in 1805 and then developed by James Cooley and John Tukey in 1965, drops the complexity down to $\mathcal{O}(n\log{}n)$.
To put it in a simple way, the FFT algorithm finds out that, any DFT can be represented by the sum of two sub-DFTs: one consists of the elements on even index in the signal, and the other consists of elements on odd positions:

$$ Y_k = \sum_{even j}\omega^{jk}y_j + \sum_{odd j}\omega^{jk}y_j$$ {#eq:signal:fft01}
$$ = \sum_{j=0}^{n/2-1}\omega^{2jk}y_{2j} + \omega^k\sum_{j=0}^{n/2-1}\omega^{2jk}y_{2j+1}.$$

The key to this step is the fact that $\omega_{2n}^2 = \omega_n$.
According to [@eq:signal:fft01], you only need to compute FFT for half of the signal, the other half can be gotten by multiplied with $\omega^{k}$, and then concatenate these two sums. 
The half signal can further be halved, so on and so forth. Therefore the computation can be reduced to a $log$ level in a recursive process.

(TODO: but what about the $y_{2j+1}$ part in the second sum? how come the second could be the same as the first sum? Need figure it out.)

To introduce Fourier Transform in detailed math and analysis of its properties is beyond the scope of this book, we encourage the readers to refer to other classic textbook on this topic [@phillips2003signals].
In this chapter, we focus on introducing how to use FFT in Owl and its applications with Owl code. Hopefully these materials are enough to interest you to investigate more. 

The implementation of the FFT module in Owl interfaces to the [FFTPack](https://www.netlib.org/fftpack/) C implementation.
Owl provides these basic FFT functions, listed in Tabel [@tbl:signal:fftfun]

| Functions | Description
| --------- |:----------| 
| `fft ~axis x` | Compute the one-dimensional discrete Fourier Transform |
| `ifft ~axis x` | Compute the one-dimensional inverse discrete Fourier Transform |
| `rfft ~axis otyp x` | Compute the one-dimensional discrete Fourier Transform for real input |
| `irfft ~axis ~n otyp x` | Compute the one-dimensional inverse discrete Fourier Transform for real input |
: FFT functions in Owl {#tbl:signal:fftfun}


### Examples

We then show how to use these functions with some simple example. 
More complex and interesting will follow in the next section.

**1-D Discrete Fourier transforms**

Let start with the most basic `fft` and it reverse transform function `ifft`.


```ocaml env=fft_env01
# let a = [|1.;2.;1.;-1.;1.5;1.0|]
val a : float array = [|1.; 2.; 1.; -1.; 1.5; 1.|]
# let b = Arr.of_array a [|6|] |> Dense.Ndarray.Generic.cast_d2z
val b : (Complex.t, complex64_elt) Dense.Ndarray.Generic.t =

       C0      C1      C2       C3        C4      C5
R (1, 0i) (2, 0i) (1, 0i) (-1, 0i) (1.5, 0i) (1, 0i)

```

```ocaml env=fft_env01
# let c = Owl_fft.D.fft b 
val c : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

         C0                 C1                 C2                  C3                C4                C5
R (5.5, 0i) (2.25, -0.433013i) (-2.75, -1.29904i) (1.5, 1.94289E-16i) (-2.75, 1.29904i) (2.25, 0.433013i)

```

```ocaml env=fft_env01
# let d = Owl_fft.D.ifft c
val d : (Complex.t, complex64_elt) Owl_dense_ndarray_generic.t =

                 C0                C1                 C2                  C3                  C4                C5
R (1, 1.38778E-17i) (2, 1.15186E-15i) (1, -8.65641E-17i) (-1, -1.52188E-15i) (1.5, 1.69831E-16i) (1, 2.72882E-16i)

```
In the result returned by `fft`, the first half contain the positive-frequency terms, and the second half contain the negative-frequency terms, in order of decreasingly negative frequency.
Typically, only the FFT corresponding to positive frequencies is plotted.

The next example plots the FFT of the sum of two sines, showing the power of FFT to separate signals of different frequency.

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
```

![Using FFT to separate two sine signals from their mixed signal](images/signal/plot_001.png "plot_001"){.align-center width=70%}


Next let's see `rfft` and `irfft`.
Function `rfft` calculates the FFT of a real signal input and generates the complex number FFT coefficients for half of the frequency domain range.
The negative part is implied by the Hermitian symmetry of the FFT.
Similarly, `irfft` performs the reverse step of `rfft`. 
First, let's make the input even number.

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

And then we change the length of signal to odd.

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

Notice that the rfft of odd and even length signals are of the same shape. (?)

**N-D Discrete Fourier transforms**

( TODO: This is not the real N-D FFT. Verify it with SciPy examples.
IMPLEMENTATION required.
TODO: explain briefly how 2D FFT can be built with 1D. Reference: Data-Driven Book, Chap2.6. )

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


## Applications of FFT

As we said, the applications of FFT are numerous. Here we pick three to demonstrate the power of FFT and how to use it in Owl.
The first is to find the period rules in the historical data of sunspots, and the second is about analysing the content of dial number according to audio information.
Both applications are inspired by [@moler2008numerical].
The third application is about image processing.
These three applications together present a full picture about how the wide usage of FFT in various scenarios.

(A backup application can always be to separate noises; Refer to Data-Driven Book, Code 2.3.)

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

Blurring an image with a two-dimensional FFT.
IMPLEMENTATION required: FFT2D, iFFT2D, fftShift. Image utils also need to be made clean.
[Reference](https://scipython.com/book/chapter-6-numpy/examples/blurring-an-image-with-a-two-dimensional-fft/).

FFT on multi-dimensional signal is effective for image compression, because many Fourier frequencies in the image are small and can be neglected via using filters, leaving the major frequencies, and thus the image quality can be largely preserved.

We use the famous Lena image as example:

![Lena](images/signal/lena.png){width=50% #fig:signal:lena}

As the first step, we read in the image into Owl as a matrix. All the elements in this matrix are scaled to within 0 to 1. 

```
code
```

Then we take the 2-D FFT, and centre the frequencies. 

```
code: fft2 + fftshift; image output 
```

IMAGE

The result is shown in a figure. It is clear that there are several small frequency bands that can be ignored. 
Let's remove them using a Gaussian Filter.

```
code: using Mat.meshgrid to build a gaussian mask; matrix multiplication; image output 
```

IMAGE

Now that we remove the insignificant frequencies, we can rebuild the image based on this filtered frequency with inverse 2-D FFT.

```
code:ifft2; show image
```

IMAGE + some comment about it: its blur but keep basic information; maybe we can do better with ... (some advanced tricks + s-o-a if there is any).

Of course, following similar method as previous applications on 1-D signals, FFT is widely used for removing noise in images by isolate and manipulate particular frequency bands.

## Filtering

In the N-dimensional Array chapter, we have introduced the idea of `filter`, which allows to get target data from the input according to some function. 
*Filtering* in signal processing is similar. It is a generic name for any system that modifies input signal in certain ways, most likely removing some unwanted features from it. 

(Refer to "ThinkDSP" in writing.)

### Example: Smoothing

Let's start with a simple and common filter task: smoothing. 
Suppose you have have a segment of noisy signal: the stock price of Google.
In many cases we hope to remove the extreme trends and see a long-term trend from the historical data.

DATA

```
CODE: visualise dataset 
```

To compute the moving average of this signal, I'll create a *window* with 10 elements: 

```
CODE: short, create an array (not ndarray, since we want to start easy), and normalise to the same 0.1.
```

Now, we can sliding this window along input signal:

IMAGES: three images that shows the sliding. Two arrays "collide" with each other in opposite direction. 

```
CODE: moving average
```

IMAGE: original data + resulting average.

### Gaussian Filter

The filter we have used is a flat one: drops first, but then bouncing around (Check with experiment result). 
We can change to another one: the gaussian filter.

```
CODE
```

IMAGE: the result, should be a better smoothing curve. 

(IMPLEMENTATION maybe required: Currently none filter implemented; need to pick 1-2 to implement and demonstrate the general idea of filtering.)

Filters can be generally by their usage into time domain filters and frequency domain filters.
Time domain filters are used when the information is encoded in the shape of the signal's waveform, and can be used for tasks such as smoothing, waveform shaping, etc. 
It includes filter methods such as moving average and single pole.
Frequency filters are used to divide a band of frequencies from signals, and its input information is in the form of sinusoids. It includes filter methods such as Windowed-sinc and Chebyshev. 
There are many filters, each with different shape (or *impulse response*) and application scenarios, and we cannot cover them fully here. 
Please refer to some classical textbooks on signal processing such as [@smith1997scientist] for more information.

### Signal Convolution

What we have done is called *convolution*. 
Formally it is mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other:

$$f(t) * g(t) = \sum_{\tau=-\infty}{\infty}f(\tau)g(t-\tau)$$ {#eq:signal:01}

In equation [@eq:signal:01], $*$ denotes the convolution operation, and you can think of $f$ as (discrete) input signal, and $g$ be filter.
Note that for computing $f(\tau)g(t-\tau)$ for each $\tau$ requires adding all product pairs. 
You can see that this process is computation-heavy.
It is even more tricky for computing the convolution of two continuous signal following definition.


We have talked a lot about FFT, and here is the place we use it.
Fourier Transformation can greatly reduce the complexity of computing the convolution and filtering.
Specifically, the **Convolution Theorem** states that:

$$\textrm{DFT}(f * g) = \textrm{DFT}(f).\textrm{DFT}(g).$$ {#eq:signal:02}

To put equation [@eq:signal:02] into plain words, to get DFT of two signals' convolution, we can simply get the DFT of each signal separately and then multiply them element-wise.
Someone may also prefer to express it in this way: convolution in the time domain can be expressed in multiplication in the frequency domain. And multiplication we are very familiar with.
Once you have the $\textrm{DFT}(f * g)$, you can naturally apply the inverse transform and get $f * g$.

Example: the same problem, but solve with FFT (use either average filter or gaussian filter, depending on which one is easy to do):

```
CODE: fft -> ifft -> smoothed stock price data.
```

IMAGE


### FFT and Image Convolution

You might heard of the word "convolution" before, and yes you are right: convolution is also the core idea in the popular deep neural network (DNN) applications. 
The convolution in DNN is often applied on ndarrays. 
It is not complex: you start with an input image in the form of ndarray, and use another smaller ndarray called "kernel" to slide over the input image step by step, and at each position, an element-wise multiplication is applied, and the result is filled into corresponding position in an output ndarray.
This process can be best illustrated with the [@fig:signal:conv] created by [Andrej Karpathy](https://cs231n.github.io/convolutional-networks/):

![Image convolution illustration](images/signal/conv.png "conv"){width=90% #fig:signal:conv}

Owl has provided thorough support of convolution operation:



```
val conv1d : ?padding:padding -> (float, 'a) t -> (float, 'a) t -> int array -> (float, 'a) t

val conv2d : ?padding:padding -> (float, 'a) t -> (float, 'a) t -> int array -> (float, 'a) t

val conv3d : ?padding:padding -> (float, 'a) t -> (float, 'a) t -> int array -> (float, 'a) t
```

They corresponds to different dimension of inputs. 
Besides, Owl also support other derived convolution types, including dilated convolutions, transpose convolutions, and backward convolutions etc. 

It's OK if none of this makes sense to you now. We'll explain the convolution and its usage in later chapter in detail.
The point is that, if you look closely, you can find that the image convolution is but only a special high dimensional case of the convolution equation: a given input signal (the image), another similar but smaller filter signal (the kernel), and the filter slides across the input signal and perform element-wise multiplication.

Therefore, we can implement the convolution with FFT and vice versa. 
For example, we can use `conv1d` function in Owl to solve the previous smoothing problem:

```
CODE: conv1d of two vectors. 
```
Also, FFT is a popular implementation method of convolution. There has been a lot of research on optimising and comparing its performance with other implementation methods such as Winograd, with practical considerations such as kernel size and implementation details of code, but we will omit these technical discussion for now.


## References
