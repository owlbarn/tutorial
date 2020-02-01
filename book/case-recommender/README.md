# Case - Recommender System

Refer to [@wang2016kvasir]

## Architecture


## Collect and Aggregate


## Preprocess News Data


## Index Text Corpus

![Projection on different random lines](images/case-recommender/plot_01.png "plot_01"){ width=90%, #fig:case-recommender:projection }

![Construct a binary search tree from the reandom projection](images/case-recommender/plot_02.png "plot_01"){ width=90%, #fig:case-recommender:search }

Figure [@fig:case-recommender:projection] is a simple illustration of how to project points on a 2D plate to random lines. Figure [@fig:case-recommender:search] illustrates how binary search can be built.



### Optimise Data Structure

![Increase either leaf size or number of trees, but which is better?](images/case-recommender/plot_03.png "plot_03"){ width=100% }

![We do not need to store the actual vector at each node. Instead, we can use a random seed to generate on the fly. In a leaf cluster, only the indices of vectors in the original data set are stored.](images/case-recommender/plot_04.png "plot_04"){ width=100% }


### Optimise Index Algorithm

![Illustration of parallelising the computation.](images/case-recommender/plot_05.png "plot_05"){ width=90% }

Blue dotted lines are critical boundaries. The computations in the child-branches cannot proceed without finishing the computation in the parent node. There is no critical boundary. All the projections can be done in just one matrix multiplication. Therefore, the parallelism can be maximised.


## Search Articles


## Make It Live


## References
