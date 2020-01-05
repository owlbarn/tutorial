# Automatic Empirical Tuning

Recent research work on parameter tuning mostly focus on hyper-parameter tuning, such as optimising the parameters of stochastic gradient in machine learning applications. 
However, tuning code and parameters in low-level numerical libraries is of the same importance. 
For example, Automatically Tuned Linear Algebra Software ([ATLAS](http://math-atlas.sourceforge.net/)) and the recent Intel Math Kernel Library ([MKL](https://software.intel.com/en-us/mkl)) are both software libraries of optimised math routines for science and engineering computation. 
They are widely used in many popular high-level platforms such as Matlab and TensorFlow. 
One of the reasons these libraries can provide optimal performance is that they have adopted the paradigm of Automated Empirical Optimisation of Software, or AEOS. That is, a library chooses the best method and parameter to use on a given platform to do a required operation. 
One highly optimised routine may run much faster than a naively coded one. Optimised code is usually platform- and hardware-specific, but an optimised routine on one machine could perform badly on the other.
Though Owl currently does not plan to improve the low-level libraries it depends on, as an initial endeavour to apply the AEOS paradigm in Owl, one ideal tuning point is the parameters of OpenMP used in Owl.

Currently many computers contain shared memory multiprocessors. OpenMP is used in key operations in libraries such as Eigen and MKL. Owl has also utilised OpenMP on many mathematical operations to boost their performance by threading calculation.
For example, the figure below shows that when I apply the sine function on an ndarray in Owl, on a 4-core CPU MacBook, the OpenMP version only takes about a third of the execution time compared with the non-OpenMP version.

![](images/aeos/sin_perf.png)

However, performance improvement does not come for free. Overhead of using OpenMP comes from time spent on scheduling chunks of work to each thread, managing locks on critical sections, and startup time of creating threads, etc.
Therefore, when the input ndarray is small enough, these overheads might overtake the benefit of threading.
What is a suitable input size to use OpenMP then? This question would be easy to solve if there is one single suitable input size threshold for every operation, but that’s not the case. 

In a small experiment, I compare the performance of two operations, `abs` (calculate absolute value) and `sin`, in three cases: running them without using OpenMP, with 2 threads OpenMP, and with 4 threads OpenMP.
The result shows that, with growing input size, for sine operation, the OpenMP version outperforms the non-OpenMP version at a size of less than 1000, but for `abs` operation, that cross point is at about 1,000,000. The complexity of math operations varies greatly, and the difference is even starker when compare their performance on different machines.
This issue becomes more complex when considered in real applications such as deep neural networks, where one needs to deal with operations of vastly different complexity and input sizes. 
Thus one fixed threshold for several operations is not an ideal solution. Considering these factors, I need a fine-grained method to decide a suitable OpenMP threshold for each operation.

## Implementation
