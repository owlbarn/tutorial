# Natural Language Processing

Text is a dominant media type on the Internet along with images, videos, and audios. Many of our day-to-day tasks involve text analysis. Natural language processing (NLP) is a powerful tool to extract insights from text corpora.

NLP is a large topic that covers many different advanced problems, such as speech tagging, named entity recognition, machine translation, speech recognition, etc.
We surely cannot cover all of them in this one single chapter, perhaps not even a whole book.
To this end, in this chapter we mainly focus on the vector space models and topic modelling.

TODO: Explain Topic modelling briefly

TODO: this chapter now mainly lacks general text introduction of NLP.


## Introduction

Survey the literature, give a high-level picture of NLP. Talk about classic NLP ... structured and unstructured text ...

In this chapter, we will use a [news dataset](https://github.com/ryanrhymes/owl_dataset/raw/master/news.txt.gz) crawled from the Internet. It contains 130000 pieces of news from various sources, each line in the file represents one entry.
For example we the first line/document is:

```text
a brazilian man who earns a living by selling space for tattoo adverts on his body is now looking for a customer for his forehead , it appears ... borim says when clients do n't pay or cancel an ad , he crosses them out . " skinvertising " caused a stir in the mid-2000s , when many dot.com companies experimented with it...
```

## Text Corpus

Normally we call a collection of documents a *text corpus*, which contains a large and structured set of texts.
For example, for the English language there are the [Corpus of Contemporary American English](https://www.english-corpora.org/coca/), [Georgetown University Multilayer Corpus](https://corpling.uis.georgetown.edu/gum), etc.
Our news collection is also one such example.
To perform NLP tasks such as topic modelling, the first and perhaps the most important thing is to represent a text corpus as format that the models can process, instead of directly using natural language.

TODO: A survey of annotation methods.

For the task of topic modelling, we perform the tokenisation on the input English text.
The target is to represent each word as an integer index so that we can further process the numbers instead of words.
This is called the *tokenisation* of the text.
Of course we also need to have a mapping function that from index to word.

### Step-by-step Operation

The NLP module in Owl supports building a proper text corpus from given text dataset.
In this section we will show how we can build a corpus from a collection of documents, in a step step way.

In the first step, remove the special characters. We define a regular expression `regexp_split` for special characters such as `,`, `?`, `\t` etc.
First remove them, and then convert all the text into lower-case.
The code below define such a process function, and the `Nlp.Corpus.preprocess` apply it to all the text. Note this function will not change the number of lines in a corpus.

```ocaml
let simple_process s =
  Str.split Owl_nlp_utils.regexp_split s
  |> List.filter (fun x -> String.length x > 1)
  |> String.concat " "
  |> String.lowercase_ascii
  |> Bytes.of_string

let preprocess input_file =
  let output_file = input_file ^ ".output" in
  Nlp.Corpus.preprocess simple_process input_file output_file
```

Based on the processed text corpus, we can build the *vocabulary*.
Each word is assigned a number id, or index, and we have the dictionary to map word to index, and index to word.
This is achieved by using the `Nlp.Vocabulary.build` function.

```ocaml
let build_vocabulary input_file =
  let vocab = Nlp.Vocabulary.build input_file in
  let output_file = input_file ^ ".vocab" in
  Nlp.Vocabulary.save vocab output_file
```

The `build` function returns a vocabulary. It contains three `Hasthtbl`s.
The first maps a word to an index, and the second index to word.
The last hash table is a map between index and its frequency, i.e. number of occurrence in the whole text body.
We can check out the words of highest frequency with:

```ocaml
let print_freq vocab =
  Nlp.Vocabulary.top vocab 10 |>
  Owl.Utils.Array.to_string ~sep:", " fst
```

Unsurprisingly, the "the"'s and "a"'s are most frequently used:

```text
- : string =
"the, to, of, a, and, in, \", s, that, on"
```

Change `Nlp.Vocabulary.top` to `Nlp.Vocabulary.bottom` can shows the words of lowest frequency:

```text
"eichorst, gcs, freeross, depoliticisation, humping, shopable, appurify, intersperse, vyaecheslav, raphaelle"
```

However, in a topic modelling task, we don't want these too frequent but meaningless words and perhaps also the least frequent words that are not about the topic of this document.
Now let's trim off some most and least frequency words. You can trim either by absolute number or by percent. We use percent here, namely trimming off top and bottom 1% of the words.

```ocaml
let trim_vocabulary vocab =
  Nlp.Vocabulary.trim_percent ~lo:0.01 ~hi:0.01 vocab
```

With a proper vocabulary at hands, now we are ready to tokenise a piece of text.

```ocaml
let tokenise vocab text =
  String.split_on_char ' ' text |>
  List.map (Nlp.Vocabulary.word2index vocab)
```

For example, if we tokenise "this is owl book", you will get the following output.

```text
tokenise vocab "this is an owl book";;
- : int list = [55756; 18322; 109456; 90661; 22362]
```

Furthermore, we can now tokenise the whole news collection.

```ocaml
let tokenise_all vocab input_file =
  let doc_s = Owl_utils.Stack.make () in
  Owl_io.iteri_lines_of_file
    (fun i s ->
      let t =
        Str.split Owl_nlp_utils.regexp_split s
        |> List.filter (Owl_nlp_vocabulary.exits_w vocab)
        |> List.map (Owl_nlp_vocabulary.word2index vocab)
        |> Array.of_list
      in
      Owl_utils.Stack.push doc_s i)
    input_file;
  doc_s
```

The process is simple: in the text corpus each line is a document and we iterate through the text line by line.
For each line/document, we remove the special characters, filter out the words that exist in the vocabulary, and map each word to an integer index accordingly.
Even though this is a simplified case, it well illustrates the typical starting point of text analysis before delving into any topic modelling.

### Use the Corpus Module

But we don't have to build a text corpus step by step. We provide the `NLP.Corpus` module for convenience.
By using the `Nlp.Corpus.build` we perform both tasks we have introduced: building vocabulary, and tokenising the text corpus.
With this function we can also specify how to trim off the high-frequency and low-frequency words.
Here is an example:

```ocaml
let main () =
  let ids = Nlp.Corpus.unique "news.txt" "clean.txt" in
  Printf.printf "removed %i duplicates." (Array.length ids);
  let corpus = Nlp.Corpus.build ~lo:0.01 ~hi:0.01 "clean.txt" in
  Nlp.Corpus.print corpus
```

The `Nlp.Corpus.unique` function is just one more layer of pre-processing. It removes the possible duplicated lines/documents.
The output prints out the processing progress, and then a summary of the corpus is printed out.

```text
2020-01-28 19:07:05.461 INFO : build up vocabulary ...
2020-01-28 19:07:10.461 INFO : processed 13587, avg. 2717 docs/s
2020-01-28 19:07:15.463 INFO : processed 26447, avg. 2644 docs/s
...
2020-01-28 19:08:09.125 INFO : convert to binary and tokenise ...
2020-01-28 19:08:34.130 INFO : processed 52628, avg. 2104 docs/s
2020-01-28 19:08:39.132 INFO : processed 55727, avg. 1857 docs/s
...
corpus info
  file path  : news.txt
  # of docs  : 129968
  doc minlen : 10
- : unit = ()
```

The corpus contains xxx parts: the vocabulary, token, and text string.
By calling the `build` function, we also save them for later use.
It creates several files in the current directory.
First, there is the vocabulary file `news.txt.voc` and `news.txt.voc.txt`. They are the same; only that the latter is in a human-readable format that has each line a word and the corresponding index number.
We can get the vocabulary with `Corpus.get_vocab`.

The tokenised text corpus is marshalled to the `news.txt.tok` file, and the string format content is saved as binary file to `news.txt.bin`.
We choose to save the content as binary format to save file size.
To get the i-th document, we can use `Corpus.get corpus i` to get the text string, or `Corpus.get_tok corpus i` to get an integer array that is tokenised version of this document.  

To efficiently access different documents by the document index (line number), we keep track of the accumulated length of text corpus and token array after processing each document. These two type of indexes are saved in the `news.txt.mdl` file.
This file also contains the document id. We have seen the `minlen` value in the output of corpus information.
Each document with less than 10 words will not be included in the corpus.
The document id is an int array that shows the index (line number) of each document in the original text corpus so that it can be traced back.
The document id can be retrieved by `Corpus.get_docid corpus`


In the `Corpus` module, we provide three mechanisms to iterate through the text corpus: `next`, `iteri`, `mapi`.
The `next` function is a generator that yields the next line of text document string in the text corpus until it hits the end of file.
The `iteri` and `mapi` functions work exactly like in the normal Array module.
The first function iterates all the documents one by one in the corpus, and the second maps all the documents in a corpus into another array.
The `iteri_tok` and `mapi_tok` work the same, except that the function should work on integer array instead of string.
Their signatures is shown below:

```text
val iteri : (int -> string -> unit) -> t -> unit

val iteri_tok : (int -> int array -> unit) -> t -> unit

val mapi : (int -> string -> 'a) -> t -> 'a array

val mapi_tok : (int -> 'a -> 'b) -> t -> 'b array
```

The `Corpus` module is designed to support a large number of text corpus. With this tool in hand, we can further proceed with the discussion of topic modelling.

## Vector Space Models

Based on the tokenised text corpus, the next thing we need is a mathematical model to express abstract ideas such as "this sentence makes sense and that one does not", "these two documents are similar", or "the key word in that paragraph is such and such".
To perform NLP tasks such as text retrieval and topic modelling, we use the *Vector Space Model* (VSM) to do that.

According to the wikipedia, a VSM is "an algebraic model for representing text documents (and any objects, in general) as vectors of identifiers".
It may sounds tricky but the basic idea is actually very simple.
For example, let's assume we only care about three topics in any news: covid19, economics, and election.
Then we can represent any news article with a three-element vector, each representing the weight of this topic in it.
For the BBC news ["Coronavirus: Millions more to be eligible for testing"](https://www.bbc.co.uk/news/uk-52462928), we can represent it with vector `(100, 2.5, 0)`.
The specific value does not actually matter here. The point is that now instead of a large chunks of text corpus, we only need to deal with this vector for further processing.

The vector space model proposes a framework that maps a document to a vector $d = (x_1, x_1, \ldots, x_N)$. This N-dimensional vector space is defined by $N$ basic terms.
Under this framework, we mainly have to decide on three factors.
The first is to choose the meaning of each dimension, or the $N$ basic concepts in the vector space.
The second is to specify the weight of each dimension for a document. In our simple example, why do we assign the first weight to `100` instead of `50`? There should be rules about it.
That means we need a proper mapping function $f$ defined.
Finally, after learning the vector representation, how should we measure their similarity?
The similarity of document is a basic idea in text processing.
For topic modelling, we can cluster the documents based on their similarity.
(TODO: Extend this point)

In this chapter we focusing on mapping a document to a vector space. However, VSM is not limited to only documents.
We can also map a word into a vector that represents a point in a certain vector space. This vector is also called *word embedding*.
In a proper representation, the similar words should be cluster together, and can even be used for calculation such as:

$$V_\textrm{king} - V_\textrm{man} + V_\textrm{women} \approx V_\textrm{queen}.$$

One of the most widely used method for word embedding is the `word2vec` proposed in [@mikolov2013exploiting]. It includes different algorithms such as the skip-gram for computing the vector representation of words.
For general purpose use, Google has already published a [pre-trained](https://code.google.com/archive/p/word2vec/) word2vec-based word embedding vector set based on part of the GoogleNews dataset.
This vector set contains 300-dimensional vectors for 3 million words and phrases.

Back to the theme of mapping documents to vector space. In the next chapter, we will start with a simple method that instantiate the VSM: the Bag of Words.

## Bag of Words (BOW)

The Bag of Words is a simple way to map docs into a vector space.
This space uses all the vocabulary as the dimensions.
Suppose there are totally $N$ different words in the vocabulary, then the vector space is of $N$ dimension.
The mapping function is simply counting how many times each word in the vocabulary appears in a document.

For example, let's use the five words "news", "about", "coronavirus", "test", "cases" as the five dimensions in the vector space.
Then if a document is `"...we heard news a new coronavirus vaccine is being developed which is expected to be tested about September..."` will be represented as `[1, 1, 1, 1, 0]` and the document `"...number of positive coronavirus cases is 100 and cumulative cases are 1000..."` will be projected to vector `[0, 0, 1, 0, 2]`.

This Bag of Words method is easy to implement based on the text corpus.
We first define a function that count the term occurrence in a document and return a hash table:

```ocaml env=nlp:bow01
let term_count htbl doc =
  Array.iter
    (fun w ->
      match Hashtbl.mem htbl w with
      | true  ->
        let a = Hashtbl.find htbl w in
        Hashtbl.replace htbl w (a +. 1.)
      | false -> Hashtbl.add htbl w 1.)
    doc
```

The hash table contains all the counts of words in this document. Of course, we can also represent the returned results as an array of integers, though the array would likely be sparse.
Then we can apply this function to all the documents in the corpus using the map function:

```ocaml env=nlp:bow01
let build_bow corpus =
  Nlp.Corpus.mapi_tok
    (fun i doc ->
      let htbl = Hashtbl.create 128 in
      term_count htbl doc;
      htbl)
    corpus
```

Based on this bag of words, the similarity between two vectors can be measured using different methods, e.g. with a simple dot product.

This method is easy to implement and the computation is inexpensive.
It maybe simple, but for some tasks, especially those that has no strict requirement for context or position of words, this method proves to work well.
For example, to cluster spam email, we only need to specify proper keywords as dimensions, such as "income", "bonus", "extra", "cash", "free", "refund", "promise" etc.
We can expect that the spam email texts will be clustered closely and easy to recognise in this vector space using the bag of words.

Actually, one even simpler method is called Boolean model. Instead of term frequency (count of word), the table only contains 1 or 0 to indicate if a word is present in a document.
This approach might also benefit from its simplicity and proved to be useful in certain tasks, but it loses the information about the importance of the word. One can easily construct a document that is close to everyone else, by putting all the vocabulary together.
The bag of word method fixes this problem.

On the other hand, this simple approach does have its own problems.
Back to the previous example, if we want to get how close the a document is to `"news about coronavirus test cases"`, then the doc `"...number of positive coronavirus cases is 100 and cumulative cases are 1000..."` is scored the same as `"hey, I got some good news about your math test result..."`.
This is not what we expected.
Intuitively, words like "coronavirus" should matter more than the more normal words like "test" and "about".
That's why we are going to introduce an improved method in the next section.

## Term Frequency–Inverse Document Frequency (TF-IDF)

In this previous section, we use the count of each term in representing document as vector.
It is a way to represent the frequency the term in the document, and we can call it *term frequency*.
In the previous section we have seen the intuition that the meaning of different word should be different.
This cannot be fixed by simply using term count.
In this section we introduce the idea of *Inverse Document Frequency* (IDF) to address this problem.

The basic idea is simple.
The IDF is used to represent how common a word is across all the documents. You can imagine that if a word is used throughout all the documents, then it must be of less importance in determining a feature of a document.
On the other hand, if a word exists in only 1-2 documents, and where it exists, this word must be of crucial importance to determine its topic.
Therefore, the IDF factor can be multiplied with the term frequency to present a more accurate metric for representing a document as vector.
This approach is called TF-IDF.

Actually, the two parts TF and IDF are just a framework for different computation methods.
To compute the term frequency, we can use the count of words $c$, or the percentage of word in the current document $\frac{c}{N}$ where $N$ is the total number of words in the document.
Another computation method is logarithm normalisation which use $\textrm{log}(c + 1)$.
We can even use the boolean count that take the frequency of word that exists to be 1 that the ones that are not to be 0.
These methods are all defined in the `Owl_nlp.Tfidf` module.

```ocaml
type tf_typ =
  | Binary
  | Count
  | Frequency
  | Log_norm
```

The same goes for the IDF. To measure how common a word $w$ is across all the document, a common way to compute is to do: $log(\frac{N_D}{n_w})$, where $N_D$ is the total number of documents and $n_w$ is the number of documents with term $w$ in it.
This metric is within the range of $[0, \infty]$. It increases with larger total document number or smaller number of documents that contain a specific word.
An improved version is called `Idf_Smooth`. It is calculated as $log(\frac{N_D}{n_w + 1})$.
This method avoid the $n_w$ to be zero to cause divide error, and also avoid getting a `0` for a word just because it used across all the documents.
In Owl they are included in the type `df_typ`.
Here the `Unary` method implies not using IDF, only term frequency.

```ocaml
type df_typ =
  | Unary
  | Idf
  | Idf_Smooth
```

We provide the `Owl_nlp.Tfidf` module to perform the TF-IDF method.
The corpus we have built in the previous section is used as input to it.
Specifically, we use the `Nlp.Tfidf.build` function to build the TFIDF model:

```ocaml
let build_tfidf corpus =
  let tf = Nlp.Tfidf.Count in
  let df = Nlp.Tfidf.Idf in
  let model = Nlp.Tfidf.build ~tf ~df corpus in
  Nlp.Tfidf.save model "news.tfidf";
  model
```

In this code, we configure to use the bag-of-words style word count method to calculate term frequency, and use the normal logarithm method to compute inverse document frequency.
The model can be saved for later use.
After the model is build, we can search similar documents according to a given string.
As a random example, let's just use the first sentence in our first piece of news in the dataset as search target: `"a brazilian man who earns a living by selling space for tattoo adverts on his body is now looking for a customer for his forehead"`.

```ocaml
let query model doc k =
  let typ = Owl_nlp_similarity.Cosine in
  let vec = Nlp.Tfidf.apply model doc in
  let knn = Nlp.Tfidf.nearest ~typ model vec k in
  knn
```

Recall the three gradients in vector space model: choosing dimension topic words, mapping document to vector, and the measurement of similarity.
Here we use the *consine similarity* as a way to measure how aligned two vectors $A$ and $B$ are.
It is defined as:

$$cos(\theta) = \frac{A.B}{\|A\|~\|B\|}$$.

We will talk about the similarity measurement in detail later.
Next, the `vec` returned by the `apply` functions return an array of `(int * float)` tuples. For each item, the integer is the tokenised index of a word in the input document `doc`, and the float number is the corresponding TF-IDF value, based on the `model` we get from previous step.
Finally, the `nearest` function search all the documents and find the vectors that has the largest similarity with the target document.
Let's show the top-10 result by setting `k` to 10:

```text
val knn : (int * float) array =
  [|(11473, -783.546068863270875); (87636, -669.76533603535529);
    (121966, -633.92555577720907); (57239, -554.838541799660675);
    (15810, -550.95468134048258); (15817, -550.775276912183131);
    (15815, -550.775276912183131); (83282, -547.322385552312426);
    (44647, -526.074567425088844); (0, -496.924176137374445)|]
```

The returned result shows the id of the matched documents. We can retrieve each document by running e.g. `Owl_nlp.Corpus.get corpus 11473`.
(TODO: what is the second number?)
To save you some effort to do that, here we list link to some of the original news that are matched to be similar to the target document:

1. *Every tatto tells a story*, doc id: 11473. [[Link](https://www.bbc.co.uk/news/magazine-27831231)]
1. *The Complete Guide to Using Social Media for Customer Service*, doc id: 87636. [[Link](https://buffer.com/resources/social-media-for-customer-service-guide)]
1. *Murder ink? Tattoos can be tricky as evidence*, doc id: 57239. [[Link](https://www.gazettenet.com/Archives/2014/06/tattoos-hg-060514)]
1. *Scottish independence: Cinemas pull referendum adverts*, doc id: 15810. [[Link](https://www.bbc.co.uk/news/uk-scotland-scotland-politics-27602601)]
1. *The profusion of temporarily Brazilian-themed products*, doc id:44647.
[[Link](https://www.bbc.co.uk/news/magazine-27879430)]

If you are interested, the input document comes from [this](https://www.bbc.co.uk/news/blogs-news-from-elsewhere-27051009) BBC news: *Brazil: Man 'earns a living' from tattoo ads*.
Then you can see that, the searched result is actually quite related to the input document, especially the first one, which is exactly the same story written in another piece of news.
The second result is somewhat distant. The word "customer" is heavily used in this document, and we can guess that it is also not frequently seen throughout the text corpus.
The fourth news is not about the tattoo guy, but this news features the topic of "customer" and "adverts".
The fifth news is chosen apparently because of the non-frequent word "brazilian" carries a lot of weight in TF-IDF.
The interesting thing is that the same document, the first document, is ranked only 10th closest.
Note that we just simply take a random sentence without any preprocessing or keyword design, also we use the un-trimmed version of text corpus.
Even so, we can still achieves a somewhat satisfactory matching result, and the result fits nicely with the working mechanisms of the TF-IDF method.


## Latent Dirichlet Allocation (LDA)

In the previous section, we have seen that by specifying a document and using it as a query, we can find out the similar documents as the query.
The query document itself is actually seen as a collection of words.  
However, the real world text, article or news, are rarely as simple as collections of words.
More often than not, an article contains one or more *topics*. For example, it can involves the responsibility of government, the protection of environment, and a recent protest in the city, etc.
Moreover, each of these topics can hardly be totally covered by just one single word.
To this end we introduce the problem *topic modelling*: instead of proposing a search query to find similar content in text corpus, we hope to automatically cluster the documents according to several topics, and each topic is represented by several words.

### Topic Modelling Example

One of such method to do topic modelling is called *Latent Dirichlet Allocation* (LDA).
We have implemented the `Owl_nlp.Lda` module to perform this method.
Without diving into the theory behind this method, let's first use an example to demonstrate how LDA works.

```ocaml
let build_lda corpus topics =
  let model = Nlp.Lda.init ~iter:1000 topics corpus in
  Nlp.Lda.(train SimpleLDA model);
  model
```

The input to LDA is still the text corpus we have built. We also need to specify how many topics we want the text corpus to be divided into. Let's say we set the number of topics to 8.
The process is simple, we first initialise the model using the `init` function and then we can train the model.

The trained model contain two matrices. The first is called the "document-topic", which contains the number of tokens assigned to each topic in each doc. It looks like this:

```text
val dk : Arr.arr =
    C0  C1  C2  C3  C4  C5  C6  C7
R0  13  13   4   7  11  12  14  16
R1  35 196  15  42  31  23 122   4
R2   7   9   3   1   3 163   2   4
R3  10  22  23 140  18  11  17 143
...
```

This matrix shows the distribution of topics in each document, represented by a row.
Each column represents a topic.
For example, you can see that the fifth column of in the third document (R2) is obviously larger than the others.
It means that dominantly talks about only the topic 6.
Similarly, in the fourth document, the topic 4 and topic 8 are of equal coverage.

However, what do these topics look like any way? This concerns the other trained matrix in the model: the "word-topic table". It contains the number of tokens assigned to each topic for each word.
Here is what it looks like:

```text
val wk : Arr.arr =
    C0  C1  C2  C3  C4  C5  C6  C7
R0   1   0   0   0   0   0   0   0
R1   0   0   0   1   0   0   0   0
R2   0   0   0   0   3   0   0   0
R3   0   0   0   0   0   0   0   3
...
```

This is sparse matrix. Each row represents a word from the vocabulary. A topic in a column can thus be represented as the words that have the largest numbers in that column.
For example, we can set that a topic be represented by 10 words. The translation from the word-topic table to text representation is straightforward:

```ocaml
let get_topics vocab wt =
  Mat.map_cols (fun col ->
    Mat.top col 10
    |> Array.map (fun ws ->
      Owl_nlp.Vocabulary.index2word vocab ws.(0))
  ) wt
```

As an example, we can take a look at the topics generated by the "A Million News Headlines" [dataset](https://www.kaggle.com/therohk/million-headlines). [[Link](https://www.kaggle.com/therohk/million-headlines)].


```text
Topic 1:  police child calls day court says abuse dead change market
Topic 2:  council court coast murder gold government face says national police
Topic 3:  man charged police nsw sydney home road hit crash guilty
Topic 4:  says wa death sa abc australian report open sex final
Topic 5:  new qld election ban country future trial end industry hour
Topic 6:  interview australia world cup china south accused pm hill work
Topic 7:  police health govt hospital plan boost car minister school house
Topic 8:  new water killed high attack public farmers funding police urged
```

Here each topic is represented by ten of its highest ranked words in the vocabulary.
but you can "feel" a common theme by connecting these dots together, even though some words may stride away a bit far away from this theme.
We cannot directly observe the topic, only documents and words. Therefore the topics are latent.
The word-topic matrix shows that. each word have different weight in the topic and the words in a topic is ranked according to the weight.
Now that we know what each topic talks about, we can cluster the documents by their most prominent topic, or just discover what topics are covered in a document, with about how much percentage each.


### Gibbs Sampling

Next, we will briefly introduce how the training algorithm works to get the topics using LDA.
The basic idea is that we go through the documents one by one. Each word is initially assigned a random topic -- just a wild guess, since currently there is not way we can know which topic this word belongs to at this stage.
After that, we iterate over all the documents again and again.
In each iterate, we look at each word, and try to find a hopefully a bit more proper topic for this word.
In this process, we assume that all the other topic assignments in the whole text corpus are correct except for the current word we are looking at.
Then we move forward to the next word in this document.
In one iteration, we process all the words in all the documents in the same way.
After enough number of iterations, we can get a quite accurate assignment for each word.
And then of course the topics of each document would be clear.

We need to further explain some details in this general description.
The most important question is, in the sampling of a document, how exactly do we update the topic assignment of a word?
We use the *Gibbs Sampling* algorithm.
For this word, we expect to get a vector of length $k$ where $k$ is the number of topics. It represents a conditional probability distribution of a one word topic assignment conditioned on the rest of the model.
In this distribution vector, the k-th element is:

$$p(z_{d,n}=k | Z_{-d,n}, W, \alpha, \beta) = \frac{n_{d,k} + \alpha_k}{\sum_{i=1}^K~(n_{d,i} + \alpha_i)}~\frac{v_{k,w_{d.n}} + \beta_{w_{d,n}}}{\sum_iv_{k, i} + \beta_i}.$$

Her $w_{d,n}$ is the current word we are looking at. $K$ is the total number of topics.
To perform the sampling, we assume that only the current topic assignment to $w_{d,n}$ is wrong, so we remove the current assignment from the model before this round of iteration.
$Z$ is the topic assignment of all words in all documents, and $W$ is the text corpus.

This computations is multiplication of two parts.
In the first part, $n_{d,k}$ show how many times the document $d$ uses topic $k$, and $\alpha_k$ is the prior weight of topic $k$ in document.
Therefore, this item means the percentage of words that are also assigned the same topic in the whole document $p(t|d)$.
To put it more simply, it shows how much this document likes topic $k$.
The larger it is, the more likely we will assign the current word to topic $k$ again.
In the second part, $v_{k,w}$ is the number of times topic $k$ uses word $w$. $\beta_w$ is the prior weight of word $w$ in a topic.
This item is the percentage of words that are also assigned the same topic in the whole document.
Therefore, this item indicate how a topic likes the word $w$.
Larger number means $w$ will continue be assigned this topic $k$ again.

Finally, we multiply these two items to get the final distribution of probability for the word $w_{d,n}$, in the form of a vector of length $K$.
Then we can uniformly draw a topic from this vector.
As we have said, we iterate this sampling process again and again until the model is good enough.


### Dirichlet Distribution

Now that we know how the model is trained, let's step back to know a bit more about how the LDA think about the world.
The core idea here is that each document can be described by the distribution of topics, and each topic can be described by distribution of words.
This makes sense, since we don't need the text in order to find the topics in an article.
In the LDA model, it can be used to generate a document in this way.
Assume this document contains 10 words and three topics, and the weights of the topics are: 50% of topic one, 30% topic 2, and 20% topic 3.
Then using the LDA model, we can first picks one of these topics randomly, and then according to the words this topic contains, we can pick a word randomly.
That generates one word. We can do the same for the next nine words.
Voila! We now have a "fake" document. The LDA hopes to make this fake document to be close to a real document as much as possible.
In another word, when we are looking at real document, LDA tries to maximise the possibility that this document can be generated from a set of topics.

In the previous chapter we say that at the beginning we just give some random guess about the probability distribution of words and topics etc., but that's not exactly true.
Think about what would happen if we randomly initialise the document-topic table: each document will be equally likely to contain any topic.
But that's rarely the case. An article cannot talk about all the topics at the same time.
What we really hope however, is that a single document belongs to a single topic, which is a more real-world scenario.  
The same goes for the word-topic table.

To that end, LDA uses the [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) to perform this task.
It is a family of continuous multivariate probability distributions parameterised by a vector $\alpha$.
For example, suppose we have only two topics in the whole world.
The tuple `(0, 1)` means it's totally about one topic, and `(1,0)` means its totally about the other.
We can run the `Stats.dirichlet_rvs` function to generate such a pair of float numbers.
The results are shown in [@fig:nlp:dirichlet].
Both figures have the same number of dots.
It shows that with smaller $\alpha$ value, the distribution are pushed to the corners, where it is obviously about one topic or the other.
A larger $\alpha$ value, however, makes the topic concentrate around the middle where it's a mixture of both topics.

![Two dimensional dirichlet distribution with different alpha parameters](images/nlp/dirichlet.png "dirichlet"){width=100% #fig:nlp:dirichlet}

We have introduced the basic mechanism of LDA.
There are many work that extend based on it, such as the SparseLDA in  [@yao2009efficient], and LightLDA in [@yuan2015lightlda].
They may differ in details but share similar basic theory.

## Latent Semantic Analysis (LSA)

Besides LDA, another common technique in performing topic modelling is the Latent Semantic Analysis (LSA).
Its purpose is the same as LDA, which is to get two matrices: the document-topic table, and the word-topic table to show the probability distribution of topics in documents and words in topics.
The difference is that, instead of using an iterative update approach, LSA explicitly builds the *document-word matrix* and then performs the singular value decomposition (SVD) on it to get the two aforementioned matrices.

Assume the text corpus contains $n$ documents, and the vocabulary contains $m$ words, then the document-word matrix is of size $n\times~m$.
We can use the simple word count as the element in this matrix. But as we have discussed in previous section, the count of words does not reflect the significance of a word, so a better way to fill in the document-word matrix is to use the TF-IDF approach for each word in a document.
Apparently, this matrix would be quite sparse. Also its row vectors are in a very high dimension.
The SVD is then used to reduce the dimension and redundancy in this matrix.

perform SVD.
Explain: why the first represent dk table and the third represent the tk table.

Now the truncation, with only the $k$ largest singular values preserved.

![Applying SVD and then truncation on document-word matrix to retrieve topic model](images/nlp/svd.png "svd"){width=100% #fig:nlp:svd}

Explain clearly the intuition of SVD: tilt the coordinate system until cover most of the existing points.
Maybe illustrate this point with 2D example.

Once we have the two tables, the model is trained, and using the model will be the same as in LDA example.

Compare to LDA, this process is easy to understand and implement.
Cons: SVD is computationally intensive and hard to iterate with new data.
The result is decent, but as [this blog](https://www.kaggle.com/rcushen/topic-modelling-with-lsa-and-lda) shows, it may not be as good as LDA in separating out the topic categories.

Application of topic modelling is wide. It can be used for summarising the large corpus of text data, or automatic tagging of articles, etc.

## Indexing and Searching

Topic models are effective tools for clustering documents based on their similarity or relevance. We can further use this tool to query relevant document given an input one. In this section, we will go through some techniques on how to index and query model built using the previous topic modelling method.

### Euclidean and Cosine Similarity

Define what is euclidean and cosine similarity. Emphasise both are correlated on a high-dimensional ball model.

TODO: use an image to illustrate.


### Linear Searching

First implement linear search, in this case, we do not need index at all, but it is very slow. In the following, the corpus is an array of arrays (each of which is an document).

```ocaml
(* calculate pairwise distance for the whole model, format (id,dist) *)
let all_pairwise_distance typ corpus x =
  let dist_fun = Owl_nlp_similarity.distance typ in
  let l = Array.mapi (fun i y -> i, dist_fun x y) corpus in
  Array.sort (fun a b -> Stdlib.compare (snd a) (snd b)) l;
  l

(* return the k most relevant documents *)
let query corpus doc k =
  let typ = Owl_nlp_similarity.Cosine in
  let l = all_pairwise_distance typ corpus doc in
  Array.sub l 0 k
```


### Use Matrix Multiplication

Show that pairwise distance can be done in a matrix multiplication, which is often highly-optimised GEMM operation. We assume the corpus has been converted into a dense matrix, wherein each row vector represents a document.

```ocaml
let query corpus doc k =
  let vec = Mat.transpose doc in
  let l = Mat.(corpus *@ vec) in
  Mat.bottom l k
```


### Random Projection

NOTE: give an image illustration on what is random project, but no need to implement. We will reserve in-depth discussion in Recommender System Chapter.

Only give a teaser here which goes like:
"Linear Search does not rely on data structure, but is slow; matrix multiplication is paralleled version LS, faster, but still compute all the pairwise. If you want to simplify computation, you need to use index, and Random Projection is a widely used index method in industry. It is a big topic, and we will cover that in detail in the case chapter."

![Random projection on 2D plane](images/nlp/plot_01.png "plot_01"){ width=90% #fig:nlp:plot_01 }


## Summary

TBD

## References
