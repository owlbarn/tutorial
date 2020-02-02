# Case - Recommender System

Refer to [@wang2016kvasir]

## Architecture {#arch}

At the core, Kvasir implements an LSA-based index and search service, and its architecture can be divided into two subsystems as \textit{frontend} and \textit{backend}. Figure \ref{fig:general} illustrates the general workflow and internal design of the system. The frontend is currently implemented as a lightweight extension in Chrome browser. The browser extension only sends the page URL back to the KServer whenever a new tab/window is created. The KServer running at the backend retrieves the content of the given URL then responds with the most relevant documents in a database. The results are formatted into JSON strings. The extension presents the results in a friendly way on the page being browsed. From user perspective, a user only interacts with the frontend by checking the list of recommendations that may interest him. 

To connect to the frontend, the backend exposes one simple \textit{RESTful API} as below, which gives great flexibility to all possible frontend implementations. By loosely coupling with the backend, it becomes easy to mash-up new services on top of Kvasir. Line 1 and 2 give an example request to Kvasir service. \texttt{type=0} indicates that \texttt{info} contains a URL, otherwise \texttt{info} contains a piece of text if \texttt{type=1}. Line 4-9 present an example response from the server, which contains the metainfo of a list of similar articles. Note that the frontend can refine or rearrange the results based on the metainfo (e.g., similarity or timestamp).

```json
POST
https://api.kvasir/query?type=0&info=url

{"results": [
  {"title": document title,
   "similarity": similarity metric,
   "page_url": link to the document,
   "timestamp": document create date}
]}
```

The backend system implements indexing and searching functionality which consist of five components: Crawler, Cleaner, DLSA, PANNS and KServer. Three components (i.e., Cleaner, DLSA and PANNS) are wrapped into one library since all are implemented on top of Apache Spark. The library covers three phases as text cleaning, database building, and indexing. We briefly present the main tasks in each component as below.

\textbf{Crawler} collects raw documents from the Web then compiles them into two data sets. One is the English Wikipedia dump, and another is compiled from over 300 news feeds of the high-quality content providers such as BBC, Guardian, Times, Yahoo News, MSNBC, and etc. Table \ref{tab:dataset} summarizes the basic statistics of the data sets. Multiple instances of the Crawler run in parallel on different machines. Simple fault-tolerant mechanisms like periodical backup have been implemented to improve the robustness of crawling process. In addition to the text body, the Crawler also records the timestamp, URL and title of the retrieved news as metainfo, which can be further utilized to refine the search results.
%%% The raw text corpus is copied to HDFS periodically to reduce the risk of data loss.

\textbf{Cleaner} cleans the unstructured text corpus and converts the corpus into term frequency-inverse document frequency (TF-IDF) model. In the preprocessing phase, we clean the text by removing HTML tags and stopwords, deaccenting, tokenization, etc. The dictionary refers to the vocabulary of a language model, its quality directly impacts the model performance. To build the dictionary, we exclude both extremely rare and extremely common terms, and keep $10^5$ most popular ones as \textit{features}. More precisely, a term is considered as rare if it appears in less than 20 documents, while a term is considered as common if it appears in more than 40\% of documents.


\textbf{DLSA} builds up an LSA-based model from the previously constructed TF-IDF model. Technically, the TF-IDF itself is already a vector space language model. The reason we seldom use TF-IDF directly is because the model contains too much noise and the dimensionality is too high to process efficiently even on a modern computer. To convert a TF-IDF to an LSA model, DLSA's algebraic operations involve large matrix multiplications and time-consuming SVD. We initially tried to use MLib to implement DLSA. However, MLlib is unable to perform SVD on a data set of $10^5$ features with limited RAM, we have to implement our own stochastic SVD on Apache Spark using rank-revealing technique. Section \ref{sec:dlsa} discusses DLSA in details.

\textbf{PANNS}\footnote{PANNS is becoming a popular choice of Python-based approximate k-NN library for application developers. According to the PyPI's statistics, PANNS has achieved over 27,000 downloads since it was first published in October 2014. The source code is hosted on the Github at https://github.com/ryanrhymes/panns .} builds the search index to enable fast $k$-NN search in high dimensional LSA vector spaces. Though dimensionality has been significantly reduced from TF-IDF ($10^5$ features) to LSA ($10^3$ features), $k$-NN search in a $10^3$-dimension space is still a great challenge especially when we try to provide responsive services. Naive linear search using one CPU takes over 6 seconds to finish in a database of 4 million entries, which is unacceptably long for any realistic services. PANNS implements a parallel RP-tree algorithm which makes a reasonable tradeoff between accuracy and efficiency. PANNS is the core component in the backend system and Section \ref{sec:panns} presents its algorithm in details.

\textbf{KServer} runs within a web server, processes the users requests and replies with a list of similar documents. KServer uses the index built by PANNS to perform fast search in the database. The ranking of the search results is based on the cosine similarity metric. A key performance metric for KServer is the service time. We wrapped KServer into a Docker\footnote{Docker is a virtualization technology which utilizes Linux container to provide system-level isolation. Docker is an open source project and its webiste is https://www.docker.com/} image and deployed multiple KServer instances on different machines to achieve better performance. We also implemented a simple round-robin mechanism to balance the request loads among the multiple KServers.

Kvasir architecture provides a great potential and flexibility for developers to build various interesting applications on different devices, e.g., semantic search engine, intelligent Twitter bots, context-aware content provision, and etc.\footnote{We provide the live demo videos of the seamless integration of Kvasir into web browsing at the official website. The link is http://www.cs.helsinki.fi/u/lxwang/kvasir/\#demo}


## Preprocess Data


## Build Topic Models

The vector space model belongs to algebraic language models, where each document is represented with a row vector. Each element in the vector represents the weight of a term in the dictionary calculated in a specific way. E.g., it can be simply calculated as the frequency of a term in a document, or slightly more complicated TF-IDF. The length of the vector is determined by the size of the dictionary (i.e., number of features). A text corpus containing $m$ documents and a dictionary of $n$ terms will be converted to an $A = m \times n$ row-based matrix. Informally, we say that $A$ grows taller if the number of documents (i.e., $m$) increases, and grows fatter if we add more terms (i.e., $n$) in the dictionary. LSA utilizes SVD to reduce $n$ by only keeping a small number of linear combinations of the original features. To perform SVD, we need to calculate the covariance matrix $C = A^T \times A$, which is a $n \times n$ matrix and is usually much smaller than $A$.

![Rank-revealing reduces dimensionality to perform in-memory SVD](images/case-recommender/plot_06.png "plot_06"){ width=90%, #fig:case-recommender:revealing }

We can easily parallelize the calculation of $C$ by dividing $A$ into $k$ smaller chunks of size $[\frac{m}{k}] \times n$, so that the final result can be obtained by aggregating the partial results as $C = A^T \times A = \sum_{i=1}^{k} A^T_i \times A_i \label{eq:1}$. However, a more serious problem is posed by the large number of columns, i.e., $n$. The SVD function in MLlib is only able to handle tall and thin matrices up to some hundreds of features. For most of the language models, there are often hundreds of thousands features (e.g., $10^5$ in our case). The covariance matrix $C$ becomes too big to fit into the physical memory, hence the native SVD operation in MLlib of Spark fails as the first subfigure of Figure [@fig:case-recommender:revealing] shows.

In linear algebra, a matrix can be approximated by another matrix of lower rank while still retaining approximately properties of the matrix that are important for the problem at hand. In other words, we can use another thinner matrix $B$ to approximate the original fat $A$. The corresponding technique is referred to as rank-revealing QR estimation \cite{Halko:2011:FSR}. A TF-IDF model having $10^5$ features often contains a lot of redundant information. Therefore, we can effectively thin the matrix $A$ then fit $C$ into the memory. Figure [@fig:case-recommender:revealing] illustrates the algorithmic logic in DLSA, which is essentially a distributed stochastic SVD implementation.


## Index Text Corpus

![Projection on different random lines](images/case-recommender/plot_01.png "plot_01"){ width=90%, #fig:case-recommender:projection }

![Construct a binary search tree from the reandom projection](images/case-recommender/plot_02.png "plot_01"){ width=90%, #fig:case-recommender:search }

Figure [@fig:case-recommender:search] illustrates how binary search can be built.


With an LSA model at hand, finding the most relevant document is equivalent to finding the nearest neighbours for a given point in the derived vector space, which is often referred to as k-NN problem. The distance is usually measured with the cosine similarity of two vectors. However, neither naive linear search nor conventional \textit{k-d} tree is capable of performing efficient search in such high dimensional space even though the dimensionality has been significantly reduced from $10^5$ to $10^3$ by LSA.

Nonetheless, we need not locate the exact nearest neighbours in practice. In most cases, slight numerical error (reflected in the language context) is not noticeable at all, i.e., the returned documents still look relevant from the user's perspective. By sacrificing some accuracy, we can obtain a significant gain in searching speed.

The general idea of RP-tree algorithm used here is clustering the points by partitioning the space into smaller subspaces recursively.
 Technically, this can be achieved by any tree-based algorithms. Given a tree built from a database, we answer 
 a nearest neighbour query $q$ in an efficient way, by moving $q$ down the tree to its appropriate leaf cell, and then return the
 nearest neighbour in that cell. However in several cases $q$'s nearest neighbour may well lie within a different cell.

Figure [@fig:case-recommender:projection] gives a naive example on a 2-dimension vector space. First, a random vector $x$ is drawn and all the points are projected onto $x$. Then we divide the whole space into half at the mean value of all projections (i.e., the blue circle on $x$) to reduce the problem size. For each new subspace, we draw another random vector for projection, and this process continues recursively until the number of points in the space reaches the predefined threshold on cluster size. We can construct a binary tree to facilitate the search. As we can see in the first subfigure of Figure [@fig:case-recommender:projection], though the projections of $A$, $B$, and $C$ seem close to each other on $x$, $C$ is actually quite distant from $A$ and $B$.

However, it has been shown that such misclassifications become arbitrarily rare as the iterative procedure continues by
drawing more random vectors and performing corresponding splits.
More precisely, in \cite{Dasgupta:2008:RPT:1374376.1374452} the authors show that under the assumption of some intrinsic
dimensionality of a subcluster (i.e., nodes of a tree structure), its descendant clusters will have a much smaller diameter, hence can include the points that are expected to be more similar to each other. Herein the diameter is defined as the distance between the furthest pair of data points in a cell. Such an example is given in Figure [@fig:case-recommender:projection], where $y$ successfully separates $C$ from $A$ and $B$. 

Another kind of misclassification is that two nearby points are unluckily divided into different subspaces, e.g., points $B$ and $D$ in the left panel of Figure [@fig:case-recommender:projection]. 
To get around this issue, the authors in \cite{Liu04aninvestigation} proposed a tree structure
(i.e., spill tree) where each data point is stored in  multiple leaves, by following overlapping splits.
Although the query time remains essentially the same, the required space is significantly increased.  
In this work we choose to improve the accuracy by building multiple RP-trees. We expect that the randomness in tree construction will introduce extra variability in the neighbours that are returned by several RP-trees for a given query point. This can be taken as an advantage in order to mitigate the second kind of misclassification while searching for the nearest neighbours of a query point in the combined search set.
However, in this case one would  need to store a large number of random vectors at every node of the tree,
introducing significant storage overhead as well.
For a corpus of 4 million documents, if we use $10^5$ random vectors (i.e., a cluster size of 20), and each vector is a $10^3$-dimension real vector (32-bit float number), the induced storage overhead is about 381.5~MB for each RP-tree. Therefore, such a solution leads to a huge index of $47.7$~GB given $128$ RP-trees are included, or $95.4$~GB given $256$ RP-trees.

The huge index size not only consumes a significant amount of storage resources, but also prevents the system from scaling up after more and more documents are collected.
One possible solution to reduce the index size is reusing the random vectors. Namely, we can generate a pool of random vectors once, then randomly choose one from the pool each time when one is needed. However, the immediate challenge emerges when we try to parallelize the tree building on multiple nodes, because we need to broadcast the pool of vectors onto every node, which causes significant network traffic.

To address this challenge, we propose to use a pseudo random seed in building and storing search index. Instead of maintaining a pool of random vectors, we just need a random seed for each RP-tree. The computation node can build all the random vectors on the fly from the given seed. From the model building perspective, we can easily broadcast several random seeds with negligible traffic overhead instead of a large matrix in the network, therefore we improve the computation efficiency. From the storage perspective, we only need to store one 4-byte random seed for each RP-tree. In such a way, we are able to successfully reduce the storage overhead from $47.7$~GB to $512$~B for a search index consisting of $128$ RP-trees (with cluster size 20), or from $95.4$~GB to only $1$~KB if $256$ RP-trees are used.


### Optimise Data Structure

![Increase either leaf size or number of trees, but which is better?](images/case-recommender/plot_03.png "plot_03"){ width=100% }

A RP-tree helps us to locate a cluster which is likely to contain some of the $k$ nearest neighbours for a given query point. Within the cluster, a linear search is performed to identify the best candidates. Regarding the design of PANNS, we have two design options in order to improve the searching accuracy. Namely, given the size of the aggregated cluster which is taken as the union of all the target clusters from every tree, we can
%: we can either use \textit{less trees with bigger cluster}, or use \textit{more trees with smaller cluster}.


* either use \textit{less trees with larger leaf clusters},
* or use \textit{more trees with smaller leaf clusters}.



We expect that when using more trees the probability of a query point to fall very close to a splitting hyperplane
should be reduced, thus it should be less likely for its nearest neighbours to lie in a different cluster. By reducing such misclassifications, the searching accuracy is supposed to be improved.
  Based on our knowledge, although there are no previous theoretical results that may justify such a hypothesis	in the field of nearest neighbour search algorithms, this concept could be considered as a combination strategy similar to those appeared in ensemble clustering, a very well established field of research \cite{okun2008}. 
 Similar to our case, ensemble clustering algorithms improve clustering solutions by	fusing information from several data partitions.
	In our further study on this particular part of the proposed system we intend to extend the probabilistic schemes developed in \cite{DBLP:journals/corr/abs-1302-1948} in an attempt to discover the underlying theoretical properties suggested by our empirical findings. In particular, we intend to similarly provide theoretical bounds for failure probability and show that such failures can be reduced by using more RP-trees.

To experimentally investigate this hypothesis we employ a subset of the Wikipedia database for further analysis.
In what follows, the data set contains $500,000$ points and we always search for the $50$ nearest neighbours of a query point.
Then we measure the searching accuracy by calculating the amount of actual nearest neighbours found.

We query $1,000$ points in each experiment.
The results presented in Figure \ref{fig:test1} correspond to the mean values of the aggregated nearest neighbours of the $1,000$ query points discovered by PANNS out of $100$ experiment runs.
Note that $x$-axis represents the "size of search space" which is defined by the number of unique points within the union of all the leaf clusters that the query point fall in. Therefore, given the same search space size, using more tress indicates that the leaf clusters become smaller.

%%% figure

As we can see in Figure \ref{fig:test1}, for a given $x$ value, the curves move upwards as we use more and more trees, indicating that the accuracy improves.
As shown in the case of 50 trees, almost $80\%$ of the actual nearest neighbours are found by performing a search over the $10\%$ of the data set.

To further illustrate the benefits of using as many RP-trees as possible, we present in Figure \ref{fig:test2} the results where the size of search space remains approximately constant while the number of trees grows and 
subsequently the cluster size shrinks accordingly. As shown, a larger number of trees leads to the better accuracy. E.g., the accuracy is improved about $62.5\%$ by increasing the number of trees from $2$ to $18$. 

%%% figure
 
 
 	Finally in Figure \ref{fig:test3} similar outcome is observed when the average size of the leaf clusters remains approximately constant and the number of trees increases. In these experiments, we choose two specific cluster sizes for comparisons, i.e., cluster size $77$ and $787$. Both are just average leaf cluster sizes resulted from the termination criterion in the tree construction procedure which pre-sets a maximum allowed size of a leaf cluster (here $100$ and $1000$ respectively, selected for illustration purposes as any other relative set up gives similar results). 
 	In addition, we also draw a random subset for any given size from the whole data set to serve as a baseline. As we see, the accuracy of the random subset has a linear improvement rate which is simply due to the linear growth of its search space. As expected, the RP-tree solutions are significantly better than the random subset, and cluster size $77$ consistently outperforms cluster size $787$ especially when the search space is small. 


%%% figure

Our empirical results clearly show the benefits of using more trees instead of using larger clusters for improving search accuracy. Moreover, regarding the searching performance, since searching can be easily parallelized, using more trees will not impact the searching time.


![We do not need to store the actual vector at each node. Instead, we can use a random seed to generate on the fly. In a leaf cluster, only the indices of vectors in the original data set are stored.](images/case-recommender/plot_04.png "plot_04"){ width=100% }


### Optimise Index Algorithm

![Illustration of parallelising the computation.](images/case-recommender/plot_05.png "plot_05"){ width=90% }

Blue dotted lines are critical boundaries. The computations in the child-branches cannot proceed without finishing the computation in the parent node. There is no critical boundary. All the projections can be done in just one matrix multiplication. Therefore, the parallelism can be maximised.


## Search Articles


## Make It Live


## References
