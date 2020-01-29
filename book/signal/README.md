# Signal Processing

TODO: refer to https://scipy.github.io/devdocs/tutorial/signal.html. Try to implement its examples first. 

So it's basically a bit of explanation + a lot examples for this chapter. 

## Fourier Transform


Think about an audio that lasts for 10 seconds. 
This audio can surely be described in the *time domain*, which means plotting its sound intensity against time as x axis.
On the other hand, maybe less obviously, the sound can also be described in the *frequency domain*. For example, if all the 10 seconds are filled with only playing the A# note, then you can describe this whole audio with one frequency number: 466.16 Hz. If it's a C note, then the number is 523.25 Hz, etc. 
The thing is that, the real-world sound is not always so pure, they are quite likely compounded from different frequencies. Perhaps this 10 seconds are about water flowing, or wind whispering, what frequencies it is built from then?

That's where Discrete Fourier Transform (DFT) comes into play. It captures the idea of converting the two form of representing a signal: in time domain and in frequency domain. 
We can represent a signal with the values of some quantity $h$ as a function of time: $h(t)$, or this signal can be represented by giving its amplitude $H$ as function of frequency: $H(f)$. We can think they are two representation of the same thing, and Fourier Transform change between them:

$$ h(f) = \int H(f)\exp^{-2\pi~ift}df$$
$$ H(f) = \int h(t)\exp^{2\pi~ift}dt$$

To put it simply: suppose Alice mix a unknown number of colour together, and let Bob to guess what those colours are, then perhaps Bob need a Fourier Transform machine of sorts.

(ADD: description of DFT)

You might be wondering, it's cool that I can recognise how a sound is composed, but so what? 
Think of a classic example where you need to remove some high pitch noisy from some music. By using DFT, you can easily find out the frequency of this noisy, remove this frequency, and turn the signal back to the time domain by using something a reverse process.

Actually, the application of DFT is more than on sound signal processing. (EXAMPLES).
It covers a very large of important computation problems that spans many fields and applications, such as music processing, data compressing, image processing, engineering, mathematics, etc. 

## Fast Fourier Transform

The Fast Fourier Transform is an algorithm that reduces the DFT computation complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n\log{}n)$.

TODO: A brief theory about how FFT drops down to log level. Make sure you explain DFT well enough in the previous section.

To introduce the algorithm itself in detailed math of DFT/FFT is beyond the scope of this book, we encourage the readers to refer to other classic textbook on this topic [@phillips2003signals].

Owl provides these basic FFT functions:

| Functions | Description
| --------- |:----------| 
| `fft ~axis x` | Compute the one-dimensional discrete Fourier Transform |
| `ifft ~axis x` | Compute the one-dimensional inverse discrete Fourier Transform |
| `rfft ~axis otyp x` | Compute the one-dimensional discrete Fourier Transform for real input |
| `irfft ~axis ~n otyp x` | Compute the one-dimensional inverse discrete Fourier Transform for real input |


TODO: introduce the FFTW we interface to a bit. It's a challenge to make FFT fast, and why FFTW works fast, etc.

## Applications of using FFT

REFER: *Numerical in Matlab* book.

Unlike the regression chapter, **make sure these examples work first**, then perhaps fill in some content. 
They don't have to be all finished for this round. But you have to be sure about the workload.
Do not dig deep into the topic FFT. That takes a whole book and more. 

### Small examples

TODO: refer to https://scipy.github.io/devdocs/tutorial/fft.html. Try all these small examples. We can remove some of them 

### Decipher the Tone 

### Find period of sunspots

[Dataset](http://sidc.oma.be/silso/newdataset) from the Solar Influences Data Center

### Image Processing

(from scipy book; NR chap 12.6; or the data-driven book; or from the elegant scipy book)

## B-Splines


## Filtering


## Kalman Filtering


## References
