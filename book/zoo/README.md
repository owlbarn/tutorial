# Scripting and Zoo System

In this chapter, we introduce the Zoo system, and focus on two aspects of it:

1. how to use it to make "small functions", then distribute and share them with other users
2. investigate the idea of computation composing and deployment based on existing script sharing function

## Introduction

Machine Learning (ML) techniques have begun to dominate data analytics applications and services.
Recommendation systems are the driving force of online service providers such as Amazon and Netflix.
Finance analytics has quickly adopted ML to harness large volume of data in such areas as fraud detection, risk-management, and compliance.
Deep Neural Network (DNN) is the technology behind voice-based personal assistance, self-driving cars, etc.

Many popular data analytics are deployed on cloud computing infrastructures.
However, they require aggregating users' data at central server for processing. This architecture is prone to issues such as increased service response latency, communication cost, single point failure, and data privacy concerns.

Recently computation on edge and mobile devices has gained rapid growth, such as personal data analytics in home, DNN application on a tiny stick, and semantic search and recommendation on web browser.
Edge computing is also boosting content distribution by supporting peering and caching.
HUAWEI has identified speed and responsiveness of native AI processing on mobile devices as the key to a new era in smartphone innovation.

Many challenges arise when moving ML analytics from cloud to edge devices.
One widely discussed challenge is the limited computation power and working memory of edge devices.
Personalising analytics models on different edge devices is also a very interesting topic.
However, one problem is not yet well defined and investigated: the deployment of data analytics services.
Most existing machine learning frameworks such as TensorFlow and Caffe focus mainly on the training of analytics models.
On the other, the end users, many of whom are not ML professionals, mainly use trained models to perform inference.
This gap between the current ML systems and users' requirements is growing.

Another challenge in conducting ML based data analytics on edge devices is model composition.
Training a model often requires large datasets and rich computing resources, which are often not available to normal users. That's one of the reasons that they are bounded with the models and services provided by large companies.
To this end we propose the idea *Composable Service*.
Its basic idea is that many services can be constructed from basic ML ones such as image recognition, speech-to-text, and recommendation to meet new application requirements.
We believe that modularity and composition will be the key to increasing usage of ML-based data analytics.

## Script Sharing with Zoo

Before start digging into more academic content, we want to briefly discuss the motivation of the Zoo system.
It is known that we can use OCaml as a scripting language as Python (at certain performance cost because the code is compiled into bytecode). Even though compiling into native code for production use is recommended, scripting is still useful and convenient, especially for light deployment and fast prototyping. In fact, the performance penalty in most Owl scripts is almost unnoticeable because the heaviest numerical computation part is still offloaded to Owl which runs native code.

While designing Owl, my goal is always to make the whole ecosystem open, flexible, and extensible. Programmers can make their own "small" scripts and share them with others conveniently, so they do not have to wait for such functions to be implemented in Owl's master branch or submit something "heavy" to OPAM.

### Typical Scenario

To illustrate how to use Zoo, let's start with a synthetic scenario. The scenario is very simple: Alice is a data analyst and uses Owl in her daily job. One day, she realised that the functions she needed had not been implemented yet in Owl. Therefore, she spent an hour in her computer and implemented these functions by herself. She thought these functions might be useful to others, e.g., her colleague Bob, she decided to share these functions using Zoo System.

Now let me see how Alice manages to do so in the following, step by step.


### Create a Script

First, Alice needs to create a folder (e.g., `myscript` folder) for her shared script. OK, what to put in the folder then?

She needs at least two files in this folder. The first one is of course the file (i.e., `coolmodule.ml`) implementing the function as below. The function `sqr_magic` returns the square of a magic matrix, it is quite useless in reality but serves as an example here.

```shell

  #!/usr/bin/env owl

  open Owl

  let sqr_magic n = Mat.(magic n |> sqr)

```

The second file she needs is a `#readme.md` which provides a brief description of the shared script. Note that the first line of the `#readme.md` will be used as a short description for the shared scripts. This short description will be displayed when you use `owl -list` command to list all the available Zoo code snippets on your computer.

```shell

  Square of Magic Matrix

  `Coolmodule` implements a function to generate the square of magic matrices.
```


### Share via Gist

Second, Alice needs to distribute the files in `myscript` folder. But how?

The distribution is done via [gist.github.com](https://gist.github.com/), so you must have `gist` installed on your computer. E.g., if you use Mac, you can install `gist` with `brew install gist`. Owl provides a simple command line tool to upload the Zoo code snippets. Note that you need to log into your Github account for `gist` and `git`.

```shell

  owl -upload myscript

```

The `owl -upload` command simply uploads all the files in `myscript` as a bundle to your [gist.github.com](https://gist.github.com/) page. The command also prints out the url after a successful upload. In our case, you can check the updated bundle on [this page](https://gist.github.com/9f0892ab2b96f81baacd7322d73a4b08).



### Import in Another Script

The bundle Alice uploaded before is assigned a unique `id`, i.e. `9f0892ab2b96f81baacd7322d73a4b08`. In order to use the `sqr_magic` function, Bob only needs to use `#zoo` directive in his script e.g. `bob.ml` in order to import the function.

```shell

  #!/usr/bin/env owl
  #zoo "9f0892ab2b96f81baacd7322d73a4b08"

  let _ = Coolmodule.sqr_magic 4 |> Owl.Mat.print

```

Bob's script is very simple, but there are a couple of things worth pointing out:

* Zoo system will automatically download the bundle of a given id if it is not cached locally;

* All the `ml` files in the bundle will be imported as modules, so you need to use `Coolmodule.sqr_magic` to access the function.

* You may also want to use `chmod +x bob.ml` to make the script executable. This is obvious if you are a heavy terminal user.


Note that to use `#zoo` directive in `utop` you need to manually load the `owl-zoo` library with `#require "owl-zoo";;`. Alternatively, you can also load `owl-top` using `#require "owl-top";;` which is an OCaml toplevel wrapper of Owl.

If you want to make `utop` load the library automatically by adding this line to `~/.ocamlinit`.


### Choose a Version of Script

Alice has modified and uploaded her scripts several times. Each version of her code is assigned a unique `version id`. Different versions of code may work differently, so how could Bob specify which version to use? Good news is that, he barely needs to change his code.

```shell

  #!/usr/bin/env owl
  #zoo "9f0892ab2b96f81baacd7322d73a4b08?vid=71261b317cd730a4dbfb0ffeded02b10fcaa5948"

  let _ = Coolmodule.sqr_magic 4 |> Owl.Mat.print

```

The only thing he needs to add is a version id using the parameter `vid`. The naming scheme of Zoo is designed to be similar with the field-value pair in a RESTful query. Version id can be obtained from a gist's [revisions page](https://gist.github.com/9f0892ab2b96f81baacd7322d73a4b08/revisions).

Besides specifying a version, it is also quite possible that Bob prefers to use the newest version Alice provides, whatever its id may be. The problem here is that, how often does Bob need to contact the Gist server to retreat the version information? Every time he runs his code? Well, that may not be a good idea in many cases considering the communication overhead and response time. Zoo caches gists locally and tends to use the cached code and data rather than downloading them all the time.

To solve this problem, Zoo provides another parameter in the naming scheme: `tol`. It is the threshold of a gist's *tolerance* of the time it exists on the local cache. Any gist that exists on a user's local cache for longer than `tol` seconds is deemed outdated and thus requires updating the latest `vid` information from the Gist server before being used. For example:

```shell

  #!/usr/bin/env owl
  #zoo "9f0892ab2b96f81baacd7322d73a4b08?tol=300"

  let _ = Coolmodule.sqr_magic 4 |> Owl.Mat.print

```

By setting the `tol` parameter to 300, Bob indicates that, if Zoo has already fetched the version information of this gist from remote server within the past 300 seconds, then keep using its local cache; otherwise contact the Gist server to check if a newer version is pushed. If so, the newest version is downloaded to local cache before being used. In the case where Bob don't want to miss every single update of Alice's gist code, he can simply set `tol` to 0, which means fetching the version information every time he executes his code.

`vid` and `tol` parameters enable users to have fine-grained version control of Zoo gists. Of course, these two parameters should not be used together. When `vid` is set in a name, the `tol` parameter will be ignored. If both are not set, as shown in the previous code snippet, Zoo will use the latest locally cached version if it exists.


### Command Line Tool

That's all. Zoo system is not complicated at all. There will be more features to be added in future. For the time being, you can check all the available options by executing `owl`.

```shell

  $ owl
  Owl's Zoo System

  Usage:
    owl [utop options] [script-file]  execute an Owl script
    owl -upload [gist-directory]      upload code snippet to gist
    owl -download [gist-id] [ver-id]  download code snippet from gist; download the latest version if ver-id not specified
    owl -remove [gist-id]             remove a cached gist
    owl -update [gist-ids]            update (all if not specified) gists
    owl -run [gist-id]                run a self-contained gist
    owl -info [gist-ids]              show the basic information of a gist
    owl -list [gist-id]               list all cached versions of a gist; list all the cached gists if gist-id not specified
    owl -help                         print out help information

```

Note that both `run` and `info` commands accept a full gist name that can contain extra parameters, instead of only a gist id.


### Examples

Despite of its simplicity, Zoo is a very flexible and powerful tool and we have been using it heavily in our daily work. We often use Zoo to share the prototype code and small shared modules which we do not want to bother OPAM, such those used in performance tests.

Moreover, many interesting examples are also built atop of Zoo system.

* [Google Inception V3 for Image Classification](https://gist.github.com/jzstark/9428a62a31dbea75511882ab8218076f)

* [Neural Style Transfer](https://gist.github.com/jzstark/6f28d54e69d1a19c1819f52c5b16c1a1)

* [Fast Neural Style Transfer](https://gist.github.com/jzstark/f937ce439c8adcaea23d42753f487299)

For example, you can use Zoo to perform DNN-based image classification in only 6 lines of code:

```shell

  #!/usr/bin/env owl
  #zoo "9428a62a31dbea75511882ab8218076f"

  let _ =
    let image  = "/path/to/your/image.png" in
    let labels = InceptionV3.infer image in
    InceptionV3.to_json ~top:5 labels

```

## System Design 

Based on these basic functionalities, we extend the Zoo system to address the composition and deployment challenges.
First, we would like to briefly introduce the workflow of Zoo as shown in [@fig:zoo:workflow]. 

![Zoo System Architecture](images/zoo/workflow.png){width=70% #fig:zoo:workflow}


### Services
Gist is a core abstraction in Zoo. It is the centre of code sharing.
However, to compose multiple analytics snippets, Gist alone is
insufficient. For example, it cannot express the structure of how
different pieces of code are composed together. Therefore, we introduce
another abstraction: `service`.

A service consists of three parts: *Gists*, *types*, and *dependency
graph*. *Gists* is the list of Gist ids this service requires. *Types*
is the parameter types of this service. Any service has zero or more
input parameters and one output. This design follows that of an OCaml
function. *Dependency graph* is a graph structure that contains
information about how the service is composed. Each node in it
represents a function from a Gist, and contains the Gist's name, id, and
number of parameters of this function.

Zoo provides three core operations about a service: create, compose, and
publish.  The *create\_service* creates a dictionary of services given a
Gist id. This operation reads the service configuration file from that
Gist, and creates a service for each function specified in the
configuration file. The *compose\_service* provides a series of
operations to combine multiple services into a new service. A compose
operation does type checking by comparing the "types" field of two
services. An error will be raised if incompatible services are composed.
A composed service can be saved to a new Gist or be used for further
composition. The *publish\_service* makes a service's code into such
forms that can be readily used by end users. Zoo is designed to support
multiple backends for these publication forms. Currently it targets
Docker container, JavaScript, and MirageOS as backends.

### Type Checking

One of the most important tasks of service
composition is to make sure the type matches. For example, suppose there
is an image analytics service that takes a PNG format image, and if we
connect to it another one that produces a JPEG image, the resulting
service will only generate meaningless output for data type mismatch.
OCaml provides primary types such as integer, float, string, and bool.
The core data structure of Owl is ndarray (or tensor as it is called in
some other data analytics frameworks). However, all these types are
insufficient for high level service type checking as mentioned. That
motives us to derive richer high-level types.

To support it, we use generalised algebraic data types (GADTs) in OCaml.
There already exist several model collections on different platforms,
e.g. Caffe and MxNet. We observe that
most current popular deep learning (DL) models can generally be
categorised into three fundamental types: `image`, `text`, and `voice`.
Based on them, we define sub-types for each: PNG and JPEG image, French
and English text and voice, i.e. `png img`, `jpeg img`, `fr text`,
`en text`, `fr voice`, and `en voice` types. More can be further added
easily in Zoo. Therefore type checking in OCaml ensures type-safe and
meaningful composition of high level services.

### Backend

Recognising the heterogeneity of edge device deployment, one key
principle of Zoo is to support multiple deployment methods.
Containerisation as a lightweight virtualisation technology has gained
enormous traction. It is used in deployment systems such as Kubernetes.
Zoo supports deploying services as Docker containers. Each container
provides RESTful API for end users to query.

Another backend is JavaScript. Using JavaScript to do analytics aside
from front end development begins to attract interests from
academia and industry, such as Tensorflow.js and
Facebook's Reason language. By exporting OCaml and Owl functions to
JavaScript code, users can do complex data analytics on web browser
directly without relying on any other dependencies.

Aside from these two backends, we also initially explore using MirageOS
as an option. Mirage is an example of Unikernel, which builds tiny
virtual machines with a specialised minimal OS that host only one target
application. Deploying to Unikernel is proved to be of low memory
footprint, and thus quite suitable for resource-limited edge devices.

### DSL

Zoo provides a minimal DSL for service composition and deployment.

**Composition**:

To acquire services from a Gist of id *gid*, we use $\$gid$ to create a
dictionary, which maps from service name strings to services. We
implement the dictionary data structure using `Hashtbl` in OCaml. The
$\#$ operator is overloaded to represent the "get item" operation.
Therefore, $$\$\textrm{gid} \# \textrm{sname}$$ can be used to get a
service that is named "sname". Now suppose we have $n$ services: $f_1$,
$f_2$, ..., $f_n$. Their outputs are of type $t_{f1}$, $t_{f2}$, ...,
$t_{fn}$. Each service $s$ accepts $m_s$ input parameters, which have
type $t_s^1$, $t_s^2$, ..., $t_s^{m_s}$. Also, there is a service $g$
that takes $n$ inputs, each of them has type $t_g^1$, $t_g^2$, ...,
$t_g^n$. Its output type is $t_o$. Here Zoo provides the `$>` operator
to compose a list of services with another:
$$[f_1, f_2, \ldots, f_n] \textrm{\$>} g$$ This operation returns a new
service that has $\sum_{s=1}^{n} m_s$ inputs, and is of output type
$t_o$. This operation does type checking to make sure that
$t_{fi} = t_g^i, \forall i \in {1, 2, \ldots, n}$.

**Deployment**:

Taking a service $s$, be it a basic or composed one, it can be deployed
using the following syntax:

$$s \textrm{\$@ backend}$$

The `$@` operator publish services to certain backend. It returns a
string of URI of the resources to be deployed.

Note that the `$>` operator leads to a tree-structure, which is in most
cases sufficient for our real-world service deployment. However, a more
general operation is to support graph structure. This will be our
next-step work.

### Service Discovery 

The services require a service discovery mechanism. For simplicity's
sake, each newly published service is added to a public record hosted on
a server. The record is a list of items, and each item contains the Gist
id that service based on, a one-line description of this service, string
representation of the input types and output type of this service, e.g.
"image -\> int -\> string -\> tex", and service URI. For the container
deployment, the URI is a DockerHub link, and for JavaScript backend, the
URI is a URL link to the JavaScript file itself. The service discovery
mechanism is implemented using off-the-shelf database.

## Use Case

To illustrate the workflow above, let's consider a synthetic scenario.
Alice is a French data analyst. She knows how to use ML and DL models in
existing platforms, but is not an expert. Her recent work is about
testing the performance of different image classification neural
networks. To do that, she needs to first modify the image using the
DNN-based Neural Style Transfer (NST) algorithm. The NST algorithm takes
two images and outputs to a new image, which is similar to the first
image in content and the second in style. This new image should be
passed to an image classification DNN for inference. Finally, the
classification result should be translated to French. She does not want
to put academic-related information on Google's server, but she cannot
find any single pre-trained model that performs this series of tasks.

Here comes the Zoo system to help. Alice find Gists that can do image
recognition, NST, and translation separately. Even better, she can
perform image segmentation to greatly improve the performance of
NST using another Gist. All she has to provide is some
simple code to generate the style images she need to use. She can then
assemble these parts together easily using Zoo.

```
open Zoo
(* Image classification *)
let s_img = $ "aa36e" # "infer";;
(* Image segmentation *)
let s_seg = $ "d79e9" # "seg";;
(* Neural style transfer *)
let s_nst = $ "6f28d" # "run";;
(* Translation from English to French *)
let s_trans = $ "7f32a" # "trans";;
(* Alice's own style image generation service *)
let s_style = $ alice_Gist_id # "image_gen";;
	
(* Compose services *)
let s = [s_seg; s_style] $> s_nst
    $> n_img $> n_trans;;
(* Publish to a new Docker Image *)
let pub = (List.hd s) $@
    (CONTAINER "alice/image_service:latest");;
```

Note that the Gist id used in the code is shorted from 32 digits to 5
due to column length limit. Once Alice creates the new service and
published it as a container, she can then run it locally and send
request with image data to the deployed machine, and get image
classification results back in French.

## Evaluation

In the evaluation section we mainly compare the performance of different
backends we use. Specifically, we observe three representative groups of
operations: (1) `map` and `fold` operations on ndarray; (2) using
gradient descent, a common numerical computing subroutine, to get
$argmin$ of a certain function; (3) conducting inference on complex
DNNs, including SqueezeNet and a VGG-like
convolution network. The evaluations are conducted on a ThinkPad T460S
laptop with Ubuntu 16.04 operating system. It has an Intel Core i5-6200U
CPU and 12GB RAM.

The OCaml compiler can produce two kinds of executables: bytecode and
native. Native executables are compiled specifically for an architecture
and are generally faster, while bytecode executables have the advantage
of being portable. A Docker container can adopt both options.

For JavaScript though, since the Owl library contains functions that are
implemented in C, it cannot be directly supported by `js-of-ocaml`, the
tool we use to convert OCaml code into JavaScript. Therefore in the Owl
library, we have implemented a "base" library in pure OCaml that shares
the core functions of the Owl library. Note that for convenience we
refer to the pure implementation of OCaml and the mix implementation of
OCaml and C as `base-lib` and `owl-lib` separately, but they are in fact
all included in the Owl library. For Mirage compilation, we use both
libraries.

![Performance of map and fold operations on ndarray on laptop and RaspberryPi](images/zoo/map_fold.png){#fig:zoo:map_fold}

[@fig:zoo:map_fold](a-b) show the performance of map and fold
operations on ndarray. We use simple functions such as plus and
multiplication on 1-d (size $< 1,000$) and 2-d arrays. The log-log
relationship between total size of ndarray and the time each operation
takes keeps linear. For both operations, `owl-lib` is faster than
`base-lib`, and native executables outperform bytecode ones. The
performance of Mirage executives is close to that of native code.
Generally JavaScript runs the slowest, but note how the performance gap
between JavaScript and the others converges when the ndarray size grows.
For fold operation, JavaScript even runs faster than bytecode when size
is sufficiently large.

![Performance of gradient descent on function $f$](images/zoo/gd_x86.png){#fig:zoo:gd}

In [@fig:zoo:gd], we want to investigate if the above
observations still hold in more complex numerical computation. We choose
to use a Gradient Descent algorithm to find the value that locally
minimise a function. We choose the initial value randomly between
$[0, 10]$. For both $sin(x)$ and $x^3 -2x^2 + 2$, we can see that
JavaScript runs the slowest, but this time the `base-lib` slightly
outperforms `owl-lib`.

We further compare the performance of DNN, which requires large amount of computation. 
We compare SqueezeNet and a VGG-like convolution network. 
They have different sizes of weight and networks structure complexities.

    Time (ms) VGG                     SqueezeNet
------------- ----------------------- --------------------------
   owl-native 7.96 ($\pm$ 0.93)       196.26($\pm$ 1.12)
     owl-byte 9.87 ($\pm$ 0.74)       218.99($\pm$ 9.05)
  base-native 792.56($\pm$ 19.95)     14470.97 ($\pm$ 368.03)
    base-byte 2783.33($\pm$ 76.08)    50294.93 ($\pm$ 1315.28)
   mirage-owl 8.09($\pm$ 0.08)        190.26($\pm$ 0.89)
  mirage-base 743.18 ($\pm$ 13.29)    13478.53 ($\pm$ 13.29)
   JavaScript 4325.50($\pm$ 447.22)   65545.75 ($\pm$ 629.10)

: Inference Speed of Deep Neural Networks {#tbl:zoo:dnn}


[@tbl:zoo:dnn] shows that, though the performance difference
between `owl-lib` and `base-lib` is not obvious, the former is much
better. So is the difference between native and bytecode for `base-lib`.
JavaScript is still the slowest. The core computation required for DNN
inference is the convolution operation. Its implementation efficiency is
the key to these differences. Current we are working on improving its
implementation in `base-lib`.

We have also conducted the same evaluation experiments on RaspberryPi 3
Model B.
[@fig:zoo:map_fold](c) shows the performance of fold operation
on ndarray. Besides the fact that all backends runs about one order of
magnitude slower than that on the laptop, previous observations still
hold. This figure also implies that, on resource-limited devices such as
RaspberryPi, the key difference is between native code and bytecode,
instead of `owl-lib` and `base-lib` for this operation. The other
figures are not presented here due to space limited, but the conclusions
are similar.

  Size (KB) native   bytecode   Mirage   JavaScript
----------- -------- ---------- -------- ------------
       base 2,437    4,298      4,602    739
     native 14,875   13,102     16,987   \-

: Size of executables generated by backends {#tbl:zoo:size}

Finally, we also briefly compare the size of executables generated by
different backends. We take the SqueezeNet for example, and the results
are shown in [@tbl:zoo:size].
It can be seen that `owl-lib` executives
have larger size compared to `base-lib` ones, and JavaScript code has
the smallest file size.
It can be seen that there does not exist a dominant method of deployment
for all these backends. It is thus imperative to choose suitable backend
according to deployment environment.

## Conclusions

In this work we identify two challenges of conducting data analytics on edge: service composition and deployment. We propose the Zoo system to address these two challenges. 
For the first one, it provides a simple DSL to enable easy and type-safe composition of different advanced services. We present a use case to show the expressiveness of the code.
For the second, to accommodate the heterogeneous edge deployment environment, we utilise multiple backends, including Docker container, JavaScript, and MirageOS. 
We thoroughly evaluate the performance of different backends using three representative groups of numerical operations as workload. The results show that no single deployment backend is preferable to the others, so deploying data analytics services requires choosing suitable backend according to the deployment environment.


## References
