# Case - Recommender System


## Introduction

Our daily life heavily relies on recommendations, intelligent content provision aims to match a user’s profile of interests to the best candidates in a large repository of options.
There are several parallel efforts in integrating intelligent content provision and recommendation in web browsing. They differentiate between each other by the main technique used to achieve the goal.

The initial effort relies on the [semantic web stack](https://en.wikipedia.org/wiki/Semantic_Web_Stack), which requires adding explicit ontology information to all web pages so that ontology-based applications (e.g., Piggy bank) can utilise ontology reasoning to interconnect content semantically.
Though semantic web has a well-defined architecture, it suffers from the fact that most web pages are unstructured or semi-structured HTML files, and content providers lack of motivation to adopt this technology to their websites.
Therefore, even though the relevant research still remains active in academia, the actual progress of adopting ontology-based methods in real-life applications has stalled in these years.

Collaborative Filtering (CF), which was first coined in 1992, is a thriving research area and also the second alternative solution. Recommenders built on top of CF exploit the similarities in users' rankings to predict one user's preference on a specific content.
CF attracts more research interest these years due to the popularity of online shopping (e.g., Amazon, eBay, Taobao, etc.) and video services (e.g., YouTube, Vimeo, Dailymotion, etc.).
However, recommender systems need user behaviour rather than content itself as explicit input to bootstrap the service, and are usually constrained within a single domain. Cross-domain recommenders have made progress lately, but the complexity and scalability need further investigation.

Search engines can be considered as the third alternative though a user needs explicitly extract the keywords from the page then launch another search. The ranking of the search results is based on multiple ranking signals such as
link analysis on the underlying graph structure of interconnected pages such as PageRank. Such graph-based link analysis is based on the assumption that those web pages of related topics tend to link to each other, and the importance of a page often positively correlates to its degree.
The indexing process is modelled as a random walk atop of the graph derived from the linked pages and needs to be pre-compiled offline.

The fourth alternative is to utilise information retrieval (IR) technique.
In general, a text corpus is transformed to the suitable representation depending on the specific mathematical models (e.g., set-theoretic, algebraic, or probabilistic models), based on which a numeric score is calculated for ranking.
Different from the previous CF and link analysis, the underlying assumption of IR is that the text (or information in a broader sense) contained in a document can very well indicate its (latent) topics. The relatedness of any two given documents can be calculated with a well-defined metric function atop
of these topics.
Since topics can have a direct connection to context, context awareness therefore becomes the most significant advantage in IR, which has been integrated into [Hummingbird](https://moz.com/learn/seo/google-hummingbird), Google’s new search algorithm.

In the rest of this chapter, we will introduce **Kvasir**, a system built on
top of latent semantic analysis.
Kvasir automatically looks for the similar articles when a user is browsing a web page and injects the search results in an easily accessible panel within the browser view for seamless integration.
Kvasir belongs to the content-based filtering and emphasises the semantics contained in the unstructured web text.
This chapter is based on the papers in [@7462177] and [@7840682], and you will find that many basic theory are already covered previously in the NLP chapter in Part I.
Henceforth we will assume you are familiar with this part.

## Architecture {#arch}

At the core, Kvasir implements an LSA-based index and search service, and its architecture can be divided into two subsystems as `frontend` and `backend`.
Figure [@fig:case-recommender:architecture] illustrates the general workflow and internal design of the system. The frontend is currently implemented as a lightweight extension in Chrome browser.
The browser extension only sends the page URL back to the KServer whenever a new tab/window is created.
The KServer running at the backend retrieves the content of the given URL then responds with the most relevant documents in a database.
The results are formatted into JSON strings. The extension presents the results in a friendly way on the page being browsed. From user perspective, a user only interacts with the frontend by checking the list of recommendations that may interest him.

![Kvasir architecture with components numbered based on their order in the workflow](images/case-recommender/architecture.png "architecture"){width=100% #fig:case-recommender:architecture}

To connect to the frontend, the backend exposes one simple *RESTful API* as below, which gives great flexibility to all possible frontend implementations. By loosely coupling with the backend, it becomes easy to mash-up new services on top of Kvasir.
In the code below, Line 1 and 2 give an example request to Kvasir service. `type=0` indicates that `info` contains a URL, otherwise `info` contains a piece of text if `type=1`.
Line 4-9 present an example response from the server, which contains the meta-info of a list of similar articles. Note that the frontend can refine or rearrange the results based on the meta-info (e.g., similarity or timestamp).

```json
POST https://api.kvasir/query?type=0&info=url

{"results": [
  {"title": document title,
   "similarity": similarity metric,
   "page_url": link to the document,
   "timestamp": document create date}
]}
```

The backend system implements indexing and searching functionality which consist of five components: *Crawler*, *Cleaner*, *DLSA*, *PANNS* and *KServer*. Three components (i.e., Cleaner, DLSA and PANNS) are wrapped into one library since all are implemented on top of Apache Spark. The library covers three phases as text cleaning, database building, and indexing. We briefly present the main tasks in each component as below.

**Crawler** collects raw documents from the web and then compiles them into two data sets.
One is the English Wikipedia dump, and another is compiled from over 300 news feeds of the high-quality content providers such as BBC, Guardian, Times, Yahoo News, MSNBC, and etc.
[@tbl:case-recommender:dataset] summarises the basic statistics of the data sets. Multiple instances of the Crawler run in parallel on different machines.
Simple fault-tolerant mechanisms like periodical backup have been implemented to improve the robustness of crawling process.
In addition to the text body, the Crawler also records the timestamp, URL and title of the retrieved news as meta information, which can be further utilised to refine the search results.

| Data set | # of entries | Raw text size | Article length |
| :-----:  | :----------- | :------------ | :------------- |
| Wikipedia | $3.9\times~10^6$ | 47.0 GB | Avg. 782 words  |
| News      | $4.6\times~10^5$ | 1.62 GB | Avg. 648 words  |
:  Two data sets are used in Kvasir evaluation {#tbl:case-recommender:dataset}

**Cleaner** cleans the unstructured text corpus and converts the corpus into term frequency-inverse document frequency (TF-IDF) model. In the preprocessing phase, we clean the text by removing HTML tags and stop words, de-accenting, tokenisation, etc.
The dictionary refers to the vocabulary of a language model. Its quality directly impacts the model performance.
To build the dictionary, we exclude both extremely rare and extremely common terms, and keep $10^5$ most popular ones as `features`. More precisely, a term is considered as rare if it appears in less than 20 documents, while a term is considered as common if it appears in more than 40\% of documents.


**DLSA** builds up an LSA-based model from the previously constructed TF-IDF model. Technically, the TF-IDF itself is already a vector space language model. The reason we seldom use TF-IDF directly is because the model contains too much noise and the dimensionality is too high to process efficiently even on a modern computer. To convert a TF-IDF to an LSA model, DLSA's algebraic operations involve large matrix multiplications and time-consuming SVD. We initially tried to use MLib to implement DLSA.
However, MLlib is unable to perform SVD on a data set of $10^5$ features with limited RAM, we have to implement our own stochastic SVD on Apache Spark using rank-revealing technique. The DLSA will be discussed in detail in later chapter.

**PANNS** builds the search index to enable fast $k$-NN search in high dimensional LSA vector spaces. Though dimensionality has been significantly reduced from TF-IDF ($10^5$ features) to LSA ($10^3$ features), $k$-NN search in a $10^3$-dimension space is still a great challenge especially when we try to provide responsive services.
A naive linear search using one CPU takes over 6 seconds to finish in a database of 4 million entries, which is unacceptably long for any realistic services.
[PANNS](https://github.com/ryanrhymes/panns) implements a parallel RP-tree algorithm which makes a reasonable tradeoff between accuracy and efficiency. PANNS is the core component in the backend system and we will present its algorithm in detail in later chapter.
PANNS is becoming a popular choice of Python-based approximate k-NN library for application developers. According to the PyPI's statistics, PANNS has achieved over 27,000 downloads since it was first published in October 2014.

**KServer** runs within a web server, processes the users requests and replies with a list of similar documents.
KServer uses the index built by PANNS to perform fast search in the database. The ranking of the search results is based on the cosine similarity metric. A key performance metric for KServer is the service time. We wrapped KServer into a Docker image and deployed multiple KServer instances on different machines to achieve better performance. We also implemented a simple round-robin mechanism to balance the request loads among the multiple KServers.

Kvasir architecture provides a great potential and flexibility for developers to build various interesting applications on different devices, e.g., semantic search engine, intelligent Twitter bots, context-aware content provision, and etc. We provide the [live demo](https://kvasira.com/demo) videos of the seamless integration of Kvasir into web browsing at the official website.
Kvasir is also available as [browser extension](https://kvasira.com/2019/10/03/Announcing-Kvasira-in-your-browser.html) on Chrome and Firefox.


## Build Topic Models

As has been explained in the previous section, the crawler and cleaner performs data collection and processing to build vocabulary and TF-IDF model.
We have already talked about this part in detail in the NLP chapter.
DLSA and PANNS are the two core components responsible for building language models and indexing the high dimensional data sets in Kvasir.
In this section, we first sketch out the key ideas in DLSA.

First, a recap of LSA from the NLP chapter.
The vector space model belongs to algebraic language models, where each document is represented with a row vector.
Each element in the vector represents the weight of a term in the dictionary calculated in a specific way. E.g., it can be simply calculated as the frequency of a term in a document, or slightly more complicated TF-IDF.
The length of the vector is determined by the size of the dictionary (i.e., number of features).
A text corpus containing $m$ documents and a dictionary of $n$ terms will be converted to an $A = m \times n$ row-based matrix.
Informally, we say that $A$ grows taller if the number of documents (i.e., $m$) increases, and grows fatter if we add more terms (i.e., $n$) in the dictionary.

The core operation in LSA is to perform SVD. For that we need to calculate the covariance matrix $C = A^T \times A$, which is a $n \times n$ matrix and is usually much smaller than $A$.
This operation poses as ad bottleneck in computing: the $m$ can be very large (a lot of documents) or the $n$ can be very large (a lot of features for each document).
For the first, we can easily parallelise the calculation of $C$ by dividing $A$ into $k$ smaller chunks of size $[\frac{m}{k}] \times n$, so that the final result can be obtained by aggregating the partial results as $C = \sum_{i=1}^{k} A^T_i \times A_i \label{eq:1}$.

However, a more serious problem is posed by the second issue. The SVD function in MLlib is only able to handle tall and thin matrices up to some hundreds of features. For most of the language models, there are often hundreds of thousands features (e.g., $10^5$ in our case). The covariance matrix $C$ becomes too big to fit into the physical memory, hence the native SVD operation in MLlib of Spark fails as the first subfigure of Figure [@fig:case-recommender:revealing] shows.

![Rank-revealing reduces dimensionality to perform in-memory SVD](images/case-recommender/plot_06.png "plot_06"){ width=90%, #fig:case-recommender:revealing }

In linear algebra, a matrix can be approximated by another matrix of lower rank while still retaining approximately properties of the matrix that are important for the problem at hand. In other words, we can use another thinner matrix $B$ to approximate the original fat $A$. The corresponding technique is referred to as rank-revealing QR estimation.
We won't talk about this method in detail, but the basic idea is that, the columns are sparse and quite likely linearly dependent. If we can find the rank $r$ of a matrix $A$ and find suitable $r$ columns to replace the original matrix, we can then approximate it.
A TF-IDF model having $10^5$ features often contains a lot of redundant information. Therefore, we can effectively thin the matrix $A$ then fit $C$ into the memory.
Figure [@fig:case-recommender:revealing] illustrates the algorithmic logic in DLSA, which is essentially a distributed stochastic SVD implementation.

To sum up, we propose to reduce the size of TF-IDF model matrix to fit it into the memory, so that we can get a LSA model, where we know the document-topic and topic-word probability distribution.

## Index Text Corpus

With a LSA model at hand, finding the most relevant document is equivalent to finding the nearest neighbours for a given point in the derived vector space, which is often referred to as k-NN problem. The distance is usually measured with the cosine similarity of two vectors.
In the NLP chapter we have seen how to use linear search in the LSA model.
However, neither naive linear search nor conventional `k-d` tree is capable of performing efficient search in such high dimensional space even though the dimensionality has been significantly reduced from $10^5$ to $10^3$ by LSA.

The key observation is that, we need not locate the exact nearest neighbours in practice. In most cases, slight numerical error (reflected in the language context) is not noticeable at all, i.e., the returned documents still look relevant from the user's perspective. By sacrificing some accuracy, we can obtain a significant gain in searching speed.

### Random Projection

To optimise the search, the basic idea is that, instead of searching in all the existing vectors, we can pre-cluster the vectors according to their distances, each cluster with only a small number of vectors.
For an incoming query, as long as we can put this vector into a suitable cluster, we can then search for close vectors only in that cluster.

![Projection on different random lines](images/case-recommender/plot_01.png "plot_01"){ width=90%, #fig:case-recommender:projection }

[@fig:case-recommender:projection] gives a naive example on a 2-dimension vector space.
First, a random vector $x$ is drawn and all the points are projected onto $x$.
Then we divide the whole space into half at the mean value of all projections (i.e., the blue circle on $x$) to reduce the problem size.
For each new subspace, we draw another random vector for projection, and this process continues recursively until the number of points in the space reaches the predefined threshold on cluster size.

In the implementation, we can construct a binary tree to facilitate the search.
Technically, this can be achieved by any tree-based algorithms. Given a tree built from a database, we answer a nearest neighbour query $q$ in an efficient way, by moving $q$ down the tree to its appropriate leaf cell, and then return the nearest neighbour in that cell.
In Kvasir, we use the Randomised Partition tree (RP-tree) introduced in [@dasgupta2013randomized] to do it.
The general idea of RP-tree algorithm used here is clustering the points by partitioning the space into smaller subspaces recursively.

![Construct a binary search tree from the random projection](images/case-recommender/plot_02.png "plot_01"){ width=90%, #fig:case-recommender:search }

The [@fig:case-recommender:search] illustrates how binary search can be built according to the dividing steps shown above.
You can see the five nodes in the vector space are put into five clusters/leaves step by step.
The information of the random vectors such as `x`, `y`, and `z` are also saved.
Once we have this tree, given another query vector, we can put it into one of the clusters along the tree to find the cluster of vectors that are close to it.

Of course, we have already said that this efficiency is traded-off with search accuracy.
One type of common misclassification is that it is possible that we can separate close vectors into different clusters.
As we can see in the first subfigure of [@fig:case-recommender:projection], though the projections of $A$, $B$, and $C$ seem close to each other on $x$, $C$ is actually quite distant from $A$ and $B$.
The reverse can also be true: two nearby points are unluckily divided into different subspaces, e.g., points $B$ and $D$ in the left panel of [@fig:case-recommender:projection].

![Aggregate clustering result from multipel RP-trees](images/case-recommender/plot_03.png "plot_03"){ width=100% #fig:case-recommender:union }

It has been shown that such misclassifications become arbitrarily rare as the iterative procedure continues by drawing more random vectors and performing corresponding splits.
In the implementation, we follow this path and build multiple RP-trees. We expect that the randomness in tree construction will introduce extra variability in the neighbours that are returned by several RP-trees for a given query point. This can be taken as an advantage in order to mitigate the second kind of misclassification while searching for the nearest neighbours of a query point in the combined search set.
As shown in [@fig:case-recommender:union], given an input query vector `x`, we find its neighbour in three different RP-trees, and the final set of neighbour candidates comes from the union of these three different sets.

### Optimising Vector Storage

You may have noticed that, in this method, we need to store all the random vectors that are generated in the non-leaf nodes of the tree. That means storing a large number of random vectors at every node of the tree, each with a large number features.
It introduces significant storage overhead.
For a corpus of 4 million documents, if we use $10^5$ random vectors (i.e., a cluster size of $\frac{4\times~10^6}{2\times~10^5} = 20$ on average), and each vector is a $10^3$-dimension real vector (32-bit float number), the induced storage overhead is about 381.5~MB for each RP-tree.
Therefore, such a solution leads to a huge index of $47.7$~GB given $128$ RP-trees are included, or $95.4$~GB given $256$ RP-trees.

The huge index size not only consumes a significant amount of storage resources, but also prevents the system from scaling up after more and more documents are collected.
One possible solution to reduce the index size is reusing the random vectors. Namely, we can generate a pool of random vectors once, and then randomly choose one from the pool each time when one is needed. However, the immediate challenge emerges when we try to parallelise the tree building on multiple nodes, because we need to broadcast the pool of vectors onto every node, which causes significant network traffic.

![Use a random seed to generate on the fly](images/case-recommender/plot_04.png "plot_04"){ width=100% #fig:case-recommender:randomseed }

To address this challenge, we propose to use a pseudo random seed in building and storing search index. Instead of maintaining a pool of random vectors, we just need a random seed for each RP-tree.
As shown in [@fig:case-recommender:randomseed], in a leaf cluster, instead of storing all the vectors, only the indices of vectors in the original data set are stored. The computation node can build all the random vectors on the fly from the given seed according to the random seed.

From the model building perspective, we can easily broadcast several random seeds with negligible traffic overhead instead of a large matrix in the network. In this way we improve the computation efficiency.
From the storage perspective, we only need to store one 4-byte random seed for each RP-tree. In such a way, we are able to successfully reduce the storage overhead from $47.7$~GB to $512$~B for a search index consisting of $128$ RP-trees (with cluster size 20), or from $95.4$~GB to only $1$~KB if $256$ RP-trees are used.


### Optimise Data Structure

Let's consider a bit more about using multiple RP-trees.
Regarding the design of PANNS, we have two design options in order to improve the searching accuracy. Namely, given the size of the aggregated cluster which is taken as the union of all the target clusters from every tree, we can either use fewer trees with larger leaf clusters, or use more trees with smaller leaf clusters.
Increasing cluster size is intuitive: if we increase it to so large that includes all the vectors, then it is totally accurate.

On the other hand, we expect that when using more trees the probability of a query point to fall very close to a splitting hyperplane should be reduced, thus it should be less likely for its nearest neighbours to lie in a different cluster.
By reducing such misclassifications, the searching accuracy is supposed to be improved.
Based on our knowledge, although there are no previous theoretical results that may justify such a hypothesis	in the field of nearest neighbour search algorithms, this concept could be considered as a combination strategy similar to those appeared in ensemble clustering, a very well established field of research.
Similar to our case, ensemble clustering algorithms improve clustering solutions by fusing information from several data partitions.

To experimentally investigate this hypothesis we employ a subset of the Wikipedia database for further analysis.
In what follows, the data set contains $500,000$ points and we always search for the $50$ nearest neighbours of a query point.
Then we measure the searching accuracy by calculating the amount of actual nearest neighbours found.

![The number of true nearest neighbours found for different number of trees](images/case-recommender/exp01.png "exp01"){width=60% #fig:case-recommender:exp01}

We query $1,000$ points in each experiment.
The results presented in [@fig:case-recommender:exp01] correspond to the mean values of the aggregated nearest neighbours of the $1,000$ query points discovered by PANNS out of $100$ experiment runs.
Note that $x$-axis represents the "size of search space" which is defined by the number of unique points within the union of all the leaf clusters that the query point falls in.
Therefore, given the same search space size, using more tress indicates that the leaf clusters become smaller.
As we can see in [@fig:case-recommender:exp01], for a given $x$ value, the curves move upwards as we use more and more trees, indicating that the accuracy improves.
As shown in the case of 50 trees, almost $80\%$ of the actual nearest neighbours are found by performing a search over the $10\%$ of the data set.

Our empirical results clearly show *the benefits of using more trees instead of using larger clusters for improving search accuracy*. Moreover, regarding the searching performance, since searching can be easily parallelised, using more trees will not impact the searching time.

### Optimise Index Algorithm

![Illustration of parallelising the computation.](images/case-recommender/plot_05.png "plot_05"){ width=90% #fig:case-recommender:parallel}

In classic RP trees we have introduced above, a different random vector is used at each inner node of a tree.
In this approach, the computations in the child-branches cannot proceed without finishing the computation in the parent node, as show in the left figure of [@fig:case-recommender:parallel].
Here the blue dotted lines are critical boundaries.
Instead, we propose to use the same random vector for all the sibling nodes of a tree.
This choice does not affect the accuracy at all because a query point is routed down each of the trees only once; hence, the query point is projected onto a random vector $r_i$ sampled from the same distribution at each level of a tree. This means that we don't need all the inner non-leaf node to be independent random vectors. Instead, the query point is projected onto only $l$ i.i.d. random vectors $r_1, \ldots, r_l$.
An RP-tree has $2^l-1$ inner nodes. Therefore, if each node of a tree had a different random vector as in classic RP-trees, $2^l-1$ different random vectors would be required for one tree.
However, when a single vector is used on each level, only $l$ vectors are required.
This reduces the amount of memory required by the random vectors from exponential to linear with respect to the depth of the trees.

Besides, another extra benefit of using one random vector for one layer is that it speeds up the index construction significantly, since we can vectorise the computation.
Let's first look at the projection of vector $a$ on $b$.
The projected length on $b$ can be expressed as:

$$\|a\|\cos~\theta = a.\frac{b}{\|b\|}.$$ {#eq:case-recommender:project}

Here $\|a\|$ means the length of vector $\mathbf{a}$.
If we requires that all the random vectors $\mathbf{b}$ has to be normalised, [@eq:case-recommender:project] becomes $a.b$, the vector dot.
Now we can perform the projection at this layer by computing: $Xb_l$.
Here $X$ is the dataset, and each row is a document and each column is a feature; $b_l$ is a random vector that we use for this layer.
In this way, we don't have to wait for the left tree to finish to start cutting the right tree.

Now here is the tricky bit: we don't even have to wait for the upper layer to start cutting the lower layer!
The reason is that, at each layer, we do random projection of *all the nodes* in the dataset on one single random vector $b$.
We don't really care the random clustering result from the previous layer.
Therefore, we can perform $Xb_1$, $Xb_2$, ..., $Xb_l$ at the same time.
That means, the projected data set $P$ can be computed directly from the dataset $X$ and a random matrix $B$ as $P = XB$ with only one pass of matrix multiplication.
Here each column of $B$ is just the random vector we use at a layer.

In this approach there is not boundary, and all the projections can be done in just one matrix multiplication.
While some of the observed speed-up is explained by a decreased amount of the random vectors that have to be generated, mostly it is due to enabling efficient computation of all the projections.
Although the total amount of computation stays the same, in practice this speeds up the index construction significantly due to the cache effects and low-level parallelisation through vectorisation.
The matrix multiplication is a basic linear algebra operation and many low level numerical libraries, such as OpenBLAS and MKL, provide extremely high-performance implementation of it.

## Search Articles

By using RP-tree we have already limit the search range from the whole text corpus to only a cluster of small number of documents (vectors), where we can do a linear searching.
We have also introduced several optimisations on the RP-tree itself, including using multiple trees, using random seed to remove the storage of random vectors, improving computation efficiency etc.
But we don't stop here: can we further improve the linear searching itself?
It turns out, we can.

To select the best candidates from a cluster of points, we need to use the coordinates in the original space to calculate their relative distance to the query point. This however, first increases the storage overhead since we need to keep the original high dimensional data set which is usually huge; second increases the query overhead since we need to access such data set.
The performance becomes more severely degraded if the original data set is too big to load into the physical memory.
Moreover, computing the distance between two points in the high dimensional space per se is very time-consuming.

Nonetheless, we will show that it is possible to completely get rid of the original data set while keeping the accuracy at a satisfying level.
The core idea of is simple.
Let's look at the second subfigure in [@fig:case-recommender:projection]. Imagine that we add a new point to search for similar vectors. The normal approach is that we compute the distance between this node and `A`, `B`, `C` etc.
But if you look at it close, all the existing nodes are already projected on the vector `y`, and we can also project the incoming query vector on `y`, and check to which of these points it is close to.
Instead of computing the distances of two vectors, now we only compute the absolute value of subtraction of two numbers (since we can always project a vector onto another one and get a real number as result) as the distance.
By replacing the original space with the projected one, we are able to achieve a significant reduction in storage and non-trivial gains in searching performance.

Of course, it is not always an accurate estimation. In the first subfigure of [@fig:case-recommender:projection], a node can be physically close to `A` or `B`, but its projection could be closest to that of `C`.
That again requires us to consider using multiple RP-trees.
But instead of the actual vector content, in the leaf node of the trees we store only `(index, projected value)`.
Now for the input query vector, we run it in the $N$ RP-trees and get $N$ set of `(index, value)` pairs.
Here each `value` is the absolute value of the difference of projected values between the vector in the tree and the query vector itself.
Each vector of course is label by a unique index.

For each index, we propose to use this metric: $\frac{\sum~d_i}{\sum~c_i}$ to measure how close it is to the query vector.
Here $d_i$ is the distance between node $i$ and query node on projected space, and $c_i$ is the count of total number of node $i$ in all the candidate sets from all the RP-trees.
Smaller measurement means closer distance.
The intuition is that, if distance value of a node on the projected space is small, then it is possibly close to the query node; or, if a node appears many times from the candidate sets of different RP-trees, it is also quite likely a possible close neighbour.

As a further improvement, we update this metric to $\frac{\sum~d_i}{(\sum~c_i)^3}$.
By so doing, we give much more weight on the points which have multiple occurrences from different RP-trees by assuming that such points are more likely to be the true k-NN.
Experiment results confirm that by using this metric it is feasible to use only the projected space in the actual system implementation. Please refer to the original paper if you are interested with more detail.


## Code Implementation

What we have introduced is the main theory behind the [Kvasir](https://kvasira.com/product), a smart content discovery tool to help you manage this rising information flood.
In this chapter, we will show some naive code implementation in OCaml and Owl to help you better understand what we have introduced so far.

First, we show the simple random projection along a RP-tree.

```ocaml env=case-recommender-00
let make_projection_matrix seed m n =
  Owl_stats_prng.init seed;
  Mat.gaussian m n |> Mat.to_arrays

let make_projected_matrix m n =
  Array.init m (fun _ -> Array.make n 0.)
```

These two functions make projection matrix and the matrix to save projected results, both return as row vectors.

```ocaml env=case-recommender-00
let project i j s projection projected =
  let r = ref 0. in
  Array.iter (fun (w, a) ->
    r := !r +. a *. projection.(w).(j);
  ) s;
  projected.(j).(i) <- !r
```

Based on these two matrices, the `project` function processes document `i` on the level `j` in the tree.
The document vector is `s`.
The projection is basically a dot multiplication between `s` and matrix `projection`.


```ocaml env=case-recommender-00
let random seed cluster tfidf =
  let num_doc = Nlp.Tfidf.length tfidf in
  let vocab_len = Nlp.Tfidf.vocab_len tfidf in
  let level = Maths.log2 (float_of_int num_doc /. cluster) |> ceil |> int_of_float in

  let projection = make_projection_matrix seed vocab_len level in
  let projected = make_projected_matrix level num_doc in

  Nlp.Tfidf.iteri (fun i s ->
    for j = 0 to level - 1 do
      project i j s projection projected;
    done;
  ) tfidf;

  vocab_len, level, projected
```

The `random` function performs a random projection of sparse data set, based a built TF-IDF model. Technically, a better way is to use LSA model as the vectorised representation of documents as we have introduced above, since a LSA model acquired based on TF-IDF represents more abstract idea of topics and has less features.
However, here it suffices to use the TF-IDF model to show the random projection process.
This function projects all the document vectors in the model to the `projected` matrix, level by level.
Recall that the result only contains the projected value instead of the whole vector.

As we have explained in the "Search Articles" section, this process can be accelerated to use matrix multiplication.
The code below shows this implementation for the random projection function.
It also returns the shape of projection and the projected result.


```ocaml env=case-recommender-00
let make_projection_matrix seed m n =
  Owl_stats_prng.init seed;
  Mat.gaussian m n

let random seed cluster data =
  let m = Mat.row_num data in
  let n = Mat.col_num data in
  let level = Maths.log2 (float_of_int m /. cluster) |> ceil |> int_of_float in

  let projection = make_projection_matrix seed n level in
  let projected = Mat.dot data projection |> Mat.transpose in

  n, level, projected, projection
```

After getting the projection result, we need to build a RP-tree accordingly.
The following is about how to build the index in the form of a binary search tree.
The tree is defined as:

```ocaml env=case-recommender-00
type t =
  | Node of float * t * t  (* intermediate nodes: split, left, right *)
  | Leaf of int array      (* leaves only contains doc_id *)
```

An intermediate node includes three parts: split, left, right, and the leaves only contain document index.

```ocaml env=case-recommender-00
let split_space_median space =
  let space_size = Array.length space in
  let size_of_l = space_size / 2 in
  let size_of_r = space_size - size_of_l in
  (* sort into increasing order for median value *)
  Array.sort (fun x y -> Pervasives.compare (snd x) (snd y)) space;
  let median =
    match size_of_l < size_of_r with
    | true  -> snd space.(size_of_l)
    | false -> (snd space.(size_of_l-1) +. snd space.(size_of_l)) /. 2.
  in
  let l_subspace = Array.sub space 0 size_of_l in
  let r_subspace = Array.sub space size_of_l size_of_r in
  median, l_subspace, r_subspace
```

The `split_space_median` function divides the projected space into subspaces to assign left and right subtrees.
The passed in `space` is the projected values on a specific level.
The criterion of division is the median value.
The `Array.sort` function sorts the space into increasing order for median value.

```ocaml env=case-recommender-00
let filter_projected_space level projected subspace =
  let plevel = projected.(level) in
  Array.map (fun (doc_id, _) -> doc_id, plevel.(doc_id)) subspace
```

Based on the document id of the points in the subspace, `filter_projected_space` function filters the projected space.
The purpose of this function is to update the projected value using a  specified level so the recursion can continue.
Both the space and the returned result are of the same format: `(doc_id, projected value)`.


```ocaml env=case-recommender-00
let rec make_subtree level projected subspace =
  let num_levels = Array.length projected in
  match level = num_levels with
  | true  -> (
      let leaf = Array.map fst subspace in
      Leaf leaf
    )
  | false -> (
      let median, l_space, r_space = split_space_median subspace in
      let l_space = match level < num_levels - 1 with
        | true  -> filter_projected_space (level+1) projected l_space
        | false -> l_space
      in
      let r_space = match level < num_levels - 1 with
        | true  -> filter_projected_space (level+1) projected r_space
        | false -> r_space
      in
      let l_subtree = make_subtree (level+1) projected l_space in
      let r_subtree = make_subtree (level+1) projected r_space in
      Node (median, l_subtree, r_subtree)
    )
```

Based on these functions, the `make_subtree` recursively grows the binary subtree to make a whole tree.
The `projected` is the projected points we get from the first step. It is of shape `(level, document_number)`.
The `subspace` is a vector of shape `(1, document_number)`.


```ocaml env=case-recommender-00
let grow projected =
  let subspace = Array.mapi (fun doc_id x -> (doc_id, x)) projected.(0) in
  let tree_root = make_subtree 0 projected subspace in
  tree_root
```

The `grow` function calls `make_subtree` to build the binary search tree.
It initialises the first subspace at level 0, and then start recursively making the subtrees from level 0. Currently everything is done in memory for efficiency consideration.


```ocaml env=case-recommender-00
let rec traverse node level x =
  match node with
  | Leaf n         -> n
  | Node (s, l, r) -> (
      match x.(level) < s with
      | true  -> traverse l (level+1) x
      | false -> traverse r (level+1) x
    )
```

Now that the tree is built, we can perform search on it.
The recursive `traverse` function traverses the whole tree to locate the cluster for a projected vector `x` starting from a given level.


```ocaml env=case-recommender-00
let rec iter_leaves f node =
  match node with
  | Leaf n         -> f n
  | Node (s, l, r) -> iter_leaves f l; iter_leaves f r

let search_leaves node id =
  let leaf = ref [||] in
  (
    try iter_leaves (fun l ->
      if Array.mem id l = true then (
        leaf := l;
        failwith "found";
      )
    ) node
    with exn -> ()
  );
  Array.copy !leaf
```

Finally, `search_leaves` returns the leaves/clusters which have the given `id` inside it. It mainly depends on the `iter_iterate` function which iterates all the leaves in a tree and applies function, to perform this search.

All these code above is executed on one tree.
When we collect the k-NN candidates from all the trees, instead of calculating the vector similarity, we utilise the frequency/count of the vectors in the union of all the candidate sets from all the RP-trees.


```ocaml env=case-recommender-00
let count_votes nn =
  let h = Hashtbl.create 128 in
  Owl_utils.aarr_iter (fun x ->
    match Hashtbl.mem h x with
    | true  -> (
        let c = Hashtbl.find h x in
        Hashtbl.replace h x (c + 1)
      )
    | false -> Hashtbl.add h x 1
  ) nn;
  let r = Array.make (Hashtbl.length h) (0,0) in
  let l = ref 0 in
  Hashtbl.iter (fun doc_id votes ->
    r.(!l) <- (doc_id, votes);
    l := !l + 1;
  ) h;
  Array.sort (fun x y -> Pervasives.compare (snd y) (snd x)) r;
  r
```

The `count_votes` function takes in an array of array `nn` as input. Each inner array contains the indexes of candidate nodes from one RP-tree.
These nodes are collected into a hash table, using index as key and the count as value. Then the results are sorted according to the count number.

## Make It Live

We provide a [live demo](https://kvasira.com/demo) of Kvasir.
Here we briefly introduce the implementation of the demo with OCaml.
This demo mainly relies on [Lwt](https://ocsigen.org/lwt/). The Lwt library implements cooperative threads.
It is often used as web server in OCaml.

This demo takes in document in the form of web query API and returns similar documents in the text corpus already included in our backend.
First, we need to do some simple preprocessing using regular expression.
This of course needs some fine tuning in the final product, but needs to be simple and fast.

```text
let simple_preprocess_query_string s =
  let regex = Str.regexp "[=+%0-9]+" in
  Str.global_replace regex " " s
```

The next function `extract_query_params` parse the web query, and retrieves parameters.

```text
let extract_query_params s =
  let regex = Str.regexp "num=\\([0-9]+\\)" in
  let _ = Str.search_forward regex s 0 in
  let num = Str.matched_group 1 s |> int_of_string in

  let regex = Str.regexp "mode=\\([a-z]+\\)" in
  let _ = Str.search_forward regex s 0 in
  let mode = Str.matched_group 1 s in

  let regex = Str.regexp "doc=\\(.+\\)" in
  let _ = Str.search_forward regex s 0 in
  let doc = Str.matched_group 1 s in

  (num, mode, doc)
```

Finally, `start_service` function includes the core query service that keeps running.
It preprocesses the input document and processed with similar document searching according to different search mode.
We won't cover the details of web server implementation details using Lwt. Please refer to its [documentation](https://ocsigen.org/lwt) for more details.

```text
let start_service lda idx =
  let num_query = ref 0 in
  let callback _conn req body =
    body |> Cohttp_lwt_body.to_string >|= (fun body ->
      let query_len = String.length body in
      match query_len > 1 with
      | true  -> (
          try (
            let num, mode, doc = extract_query_params body in
            Log.info "process query #%i ... %i words" !num_query query_len;
            num_query := !num_query + 1;

            let doc = simple_preprocess_query_string doc in
            match mode with
            | "linear" -> query_linear_search ~k:num lda doc
            | "kvasir" -> query_kvasir_idx ~k:num idx lda doc
            | _        -> failwith "kvasir:unknown search mode"
          )
          with exn -> "something bad happened :("
      )
      | false -> (
          Log.warn "ignore an empty query";
          ""
        )
    )
    >>= (fun body -> Server.respond_string ~status:`OK ~body ())
  in
  Server.create ~mode:(`TCP (`Port 8000)) (Server.make ~callback ())
```

## Summary

In this chapter, we presented Kvasir which provides seamless integration of LSA-based content provision into web browsing.
To build Kvasir as a scalable Internet service, we addressed various technical challenges in the system implementation.
Specifically, we proposed a parallel RP-tree algorithm and implemented stochastic SVD on Spark to tackle the scalability challenges in index building and searching.
We have introduced the basic algorithm and how it can optimised step by step,  from storage to computation.
These optimisations include aggregating results from multiple trees, replacing random variable with a single random seed, removing the projection computation boundary between different layers, using count to approximate vector distance, etc.
Thanks to its novel design, Kvasir can easily achieve millisecond query speed for a 14 million document repository.
Kvasir is an open-source project and is currently under active development. The key components of Kvasir are implemented as an Apache Spark library, and all
the [source code](https://www.cl.cam.ac.uk/~lw525/kvasir/#code) are publicly accessible on GitHub.

## References
