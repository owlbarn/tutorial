# Internal Utility Modules

During development of Owl, we find some utility modules are immensely handy. 
In this chapter, we share some of them. 
These are not the main feature of Owl, and perhaps you can implement your own version very quickly. 
But we hope to present how these features are used in Owl. 

## Dataset Module

The dataset modules provides easy access to various datasets to be used in Owl, mainly the MNSIT and CIFAR10 datasets.
You can get all these data in Owl by executing: `Dataset.download_all ()`.
The data are downloaded in the home directory, for example,  `~/.owl/dataset` on Linux.

### MNIST

The [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits has a training set of 60,000 examples,
and a test set of 10,000 examples. Each example is of size 28 x 28.
It is a good starting point for deep neural network related tasks. 

You can get MNIST data via these Owl functions:

- `Dataset.load_mnist_train_data ()`: returns a triplet `x, y, y'`.
  + `x` is a [60000, 784] ndarray (`Owl_dense_ndarray.S.mat`) where each row represents a [28, 28] image.
  + `y` is a [60000, 1] label ndarray. Each row is an integer ranging from 0 to 9,
  indicating the digit on each image.
  + `y'` is a [60000, 10] label ndarray. Each one-hot row vector corresponds to
  one label.

- `Dataset.load_mnist_test_data ()`: returns a triplet.
  Similar to `load_mnist_train_data`, only that it returns the test set, so
  the example size is 10,000 instead of 60,000.

- `Dataset.load_mnist_train_data_arr ()`: similar to `load_mnist_train_data`,   but returns `x` as [60000,28,28,1] ndarray

- `Dataset.load_mnist_test_data_arr ()`: similar to
  `load_mnist_train_data_arr`, but it returns the test set, so the example size
  is 10, 000 instead of 60, 000.

- `Dataset.draw_samples x y n` draws `n` random examples from images ndarray `x` and label ndarray `y`.  

You can find the MNIST dataset used in training and testing a DNN in Owl:

```
let train network =
  let x, _, y = Dataset.load_mnist_train_data_arr () in
  Graph.train network x y |> ignore;
  network

let test network =
  let imgs, _, labels = Dataset.load_mnist_test_data () in
  let m = Dense.Matrix.S.row_num imgs in
  let imgs = Dense.Ndarray.S.reshape imgs [|m;28;28;1|] in

  let mat2num x = Dense.Matrix.S.of_array (
      x |> Dense.Matrix.Generic.max_rows
        |> Array.map (fun (_,_,num) -> float_of_int num)
    ) 1 m
  in

  let pred = mat2num (Graph.model network imgs) in
  let fact = mat2num labels in
  let accu = Dense.Matrix.S.(elt_equal pred fact |> sum') in
  Owl_log.info "Accuracy on test set: %f" (accu /. (float_of_int m))
```

### CIFAR-10

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) include small scale color images for more realistic complex image classification tasks.
It includes 10 classes of images: aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
It consists of 60,000 32 x 32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Due to the limit of file size on Github, the training set is cut into 5 smaller batch. You can get CIFAR-10 data using `Owl`:

- `Dataset.load_cifar_train_data batch`: returns a triplet `x, y, y'`.
  + The input paramter `batch` can range from 1 to 5, indicating which training set batch to choose.
  + `x` is an [10000, 32, 32, 3] ndarray (`Owl_dense_ndarray.S.arr`). The last
  dimension indicates color channels (first Red, then Green, finally Blue).
  + `y` is an [10000, 1] label ndarray, each number representing
  an image class.
  + `y'` is the corresponding [10000, 10] one-hot label ndarray.  

- `Dataset.load_cifar_test_data ()`: similar to `load_cifar_train_data`, only
  that it loads test data.

- `Dataset.draw_samples_cifar x y n` draws `n` random examples from images ndarray `x` and label ndarray `y`.

Note that all elements in the loaded matrices and ndarrays are of `float32` format.

The CIFAR10 dataset is similar to MNIST in training DNN:

```
let train network =
  let x, _, y = Dataset.load_cifar_train_data 1 in
  Graph.train network x y
```

### Text Datasets


## Graph Module


## Stack Module


## Heap Module


## Count-Min Sketch

