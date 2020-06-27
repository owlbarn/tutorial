# Prologue

Owl is a software system for scientific and engineering computing. The library is (mostly) developed in the OCaml language. As a very unique functional programming language, OCaml offers us superb runtime efficiency, flexible module system, static type checking, intelligent garbage collector, and powerful type inference. Owl undoubtedly inherits these great features directly from OCaml. With Owl, you can write succinct type-safe numerical applications in a beautiful and battle-tested functional language without sacrificing performance, significantly speed up the development life-cycle, and reduce the cost from prototype to production use.



## A Brief History

Owl originated from a research project which studied the design of synchronous parallel machines for large-scale distributed computing in July 2016. I chose OCaml as the language for developing the system due to its expressiveness and superior runtime efficiency. Another obvious reason is I was working as a [PostDoc](http://www.cl.cam.ac.uk/~lw525/) in OCamlLabs.

Even though OCaml is a very well designed language, the libraries for numerical computing in OCaml ecosystem were very limited and the tooling was fragmented at that time. In order to test various analytical applications, I had to write so many numerical functions myself, from very low level algebra and random number generators to the high level stuff like algorithmic differentiation and deep neural networks. These code snippets started accumulating and eventually grew much bigger than the distributed engine itself. Therefore I decided to take these functions out and wrapped them up as a standalone library -- Owl.

Owl's architecture undertook at least a dozen of iterations in the beginning, and some of the architectural changes are quite drastic. I intentionally avoid looking into the architecture of SciPy, Julia, Matlab to minimise their influence on Owl's architecture, I really do not want *yet another xyz ...*. When the architecture became stabilised, I started implementing different numerical functions. That was a stressful but fulfilling year in 2017, I worked day and night and added over 6000 functions (over 150,000 LOC). After one-year intensive development, Owl was already capable of doing many complicated numerical tasks. e.g. see our [Google Inception V3 demo](http://demo.ocaml.xyz/) for image classification. I even held a tutorial in Oxford to demonstrate *Data Science in OCaml*.

Despite the fact that OCaml is a niche language, Owl has been attracting more and more users. I really appreciate their patience with this young but ambitious software. The community has been always supportive and provided useful feedback in these years. I hope Owl can help people to study functional programming and solve real-world problems.



## Reductionism vs. Holism

If you are from Python world and familiar with its ecosystem for numerical computing, you may see Owl as a mixture of NumPy, SciPy, Pandas, and many other libraries. You may be curious about why I have packed so much stuff together. As you learn more and more about OCaml, I am almost certain you will start wondering why Owl's design seems against *minimalist*, a popular design principle adopted by many OCaml libraries.

First of all, I must point out that having many functionalities included in one system does not necessarily indicate a monolithic design. Owl's functionalities are well defined in various modules, and each module is very self-contained and follows the minimalist design principle.

Second, I would like to argue that these functionalities should be co-designed and implemented together. This is exactly what we have learnt in the past two decades struggling to build a modern numerical system. The current co-design choice avoids a lot of redundant code and duplicated efforts, and makes optimisation a lot easier. NumPy, SciPy, Pandas, and other software spent so many years in order to well define the boundary of their functionality. In the end, NumPy becomes data representation (i.e. N-dimensional array), SciPy builds high-level analytical functions atop of such representation, Pandas evolves into a combination of table manipulation and analytical functions, and PyTorch bridges these functions between heterogeneous devices. However, there is still a significant amount of overlap if you look deep into the implementation code.

Back to the OCaml world, the co-design becomes even more important because of the language's strict static typing. Especially if every small numerical library wraps its data representation into abstract types, then they will not play together nicely when you try to build a large and complex application. This further indicates that by having a huge number of small libraries in the ecosystem will not effectively improve a programmers' productivity. Owl is supposed to address this issue with a consistent *holistic* design, with a strong focus on scientific computing.

Different choice of design principle also reveals the difference between system programming and numerical programming. System programming is built atop of a wide range of complicated and heterogeneous hardware, it abstracts out the real-world complexity by providing a (relatively) small set of APIs (recall how many system calls in the Unix operating system). On the other hand, numerical computing is built atop of a small amount of abstract number types (e.g. real and complex), then derives a rich set of advanced numerical operations for various fields (recall how many APIs in a numerical library). As a result, [reductionism](https://en.wikipedia.org/wiki/Reductionism) is preferred in system programming whereas [holism](https://en.wikipedia.org/wiki/Holism) is preferred in numerical one.



## Key Features

Owl has implemented many advanced numerical functions atop of its solid implementation of n-dimensional arrays. Compared to other numerical libraries, Owl is very unique in many perspectives, e.g. algorithmic differentiation and distributed computing have been included as integral components in the core system to maximise developers' productivity. Owl is young but grows very fast, the current features include:

* N-dimensional array (both dense and sparse)
* Various number types: ``float32``, ``float64``, ``complex32``, ``complex64``, ``int16``, ``int32``, and etc.
* Linear algebra and full interface to CBLAS and LAPACKE
* Algorithmic differentiation (or automatic differentiation)
* Neural network module for deep learning applications
* Dynamic computational graph
* Parallel and Distributed computation engine
* Advanced math and statistics functions (e.g., hypothesis tests, MCMC, etc.)
* Zoo system for efficient scripting and code sharing
* JavaScript and unikernel backends.
* Integration with other frameworks such as TensorFlow and PyTorch.
* GPU and other accelerator frameworks.

The Owl system evolves very fast, and OCaml's numerical ecosystem is booming as well, therefore your feedback is important for me to adjust future direction and the focus. In case you find some important features are missing, you are welcome to submit an issue on the [Issue Tracker](https://github.com/ryanrhymes/owl/issues).



## Contact Me

If you want to discuss about the book, the code, or other related topics, you can reach me in the following ways.

* [Email Me](mailto:liang.wang@cl.cam.ac.uk)
* [Slack Channel](https://join.slack.com/t/owl-dev-team/shared_invite/enQtMjQ3OTM1MDY4MDIwLTcxYTlkODhiNGI4YjVkN2FmMjhlZGZhYzhkMTFhZjY0OGI1NDY5M2Y2NmYzNjBhZmRhZGE0NTY1ZjA5MTk4MjI)
* [Issue Tracker](https://github.com/ryanrhymes/owl/issues)

**Student Project:** If you happen to be a student in the Computer Lab and want to do some challenging development and design, here are some [Part II Projects](http://www.cl.cam.ac.uk/research/srg/netos/stud-projs/studproj-17/#owl0).

If you are interested in more research-y topics, I offer Part III Projects and please have a look at :doc:[Owl's Sub-Projects](../project/proposal) page and contact me directly via email.

I am looking forward to hearing from you!
