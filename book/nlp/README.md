# Natural Language Processing

Text is a dominant media type on the Internet along with images, videos, and audios. Many of our day-to-day tasks involve text analysis. Natural language processing is a powerful tool to extract insights from text copora.

This chapter focusses on the vector space models and topic modelling.

TODO: this chapter now mainly lacks general text introduction of NLP.


## Introduction

Survey the literature, give a high-level picture of NLP. Talk about classic NLP ... structured and unstructured text ...

In this chapter, we will use a [news dataset](https://github.com/ryanrhymes/owl_dataset/raw/master/news.txt.gz) crawled from the Internet. It contains 130000 pieces of news from various sources, each line in the file represents one entry.
For example we the first line/document is:

```text
a brazilian man who earns a living by selling space for tattoo adverts on his body is now looking for a customer for his forehead , it appears . edson aparecido borim already has 49 adverts tattooed on his chest , back and arms , the g1 news portal reports. he says it all started eight years ago with " a dare in a bar " , but now the ads are his main source of income. " my goal now is to get a big company to tattoo my forehead , but it would have to be a good contract , " he says . he walks around bare-chested in the small town of tabani in the state of sao paulo , but says he 's not obliged to do so all the time. the brazilian charges between 50 and 400 reals ( $ 14- $ 110 ) a month for a tattoo , depending on its size and location on his body , and on the client. borim says when clients do n't pay or cancel an ad , he crosses them out . " skinvertising " caused a stir in the mid-2000s , when many dot.com companies experimented with it. the practice left behind a trail of ads for companies that do n't exist any more , buzzfeed reports . use # newsfromelsewhere to stay up-to-date with our reports via twitter .
```

## Text Corpus

We call a collection of documents a *text corpus*, which is normally a large and structured set of texts.
Our news collection is also one such example.
However, to perform NLP tasks such as topic modelling, the first and perhaps the most important thing is to represent a text corpus as format that the models can process, instead of directly using natural language.

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

```text
2020-01-28 20:29:41.592 INFO : processed 13485, avg. 2696 docs/s
2020-01-28 20:29:46.593 INFO : processed 24975, avg. 2497 docs/s
2020-01-28 20:29:51.593 INFO : processed 39728, avg. 2648 docs/s
...
```

The returned result `vocab` contains three `Hasthtbl`.
The first maps a word to an index, and the second index to word.
The last hash table is a map between index and its frequency, i.e. number of occurrence in the whole text body.
We can check out the words of highest frequency with:

```ocaml
let print_freq vocab =
  Nlp.Vocabulary.top vocab 10 |>
  Owl.Utils.Array.to_string ~sep:", " fst
```

We can see that unsurprisingly, the "the"'s and "a"'s are most frequently used:

```text
- : string =
"the, to, of, a, and, in, \", s, that, on"
```

Change `Nlp.Vocabulary.top` to `Nlp.Vocabulary.bottom` can shows the words of lowest frequency:

```text
"eichorst, gcs, freeross, depoliticisation, humping, shopable, appurify, intersperse, vyaecheslav, raphaelle"
```

Now let's trim off some most and least frequency words. You can trim either by absolute number or by percent. We use percent here, namely trimming off top and bottom 1% of the words.


```ocaml
let trim_vocabulary vocab =
  Nlp.Vocabulary.trim_percent ~lo:0.01 ~hi:0.01 vocab
```

With a vocabulary at hands, now we are ready to tokenise a piece of text.

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

Even though this is a simplified case, it well illustrates the typical starting point of text analysis before delving into any topic modelling.


### Use Corpus Module

But we don't have to build a text corpus step by step. We provide the `NLP.Corpus` module.
A corpus is defined to contain these parts:

- `uri` : path of the binary corpus
- `bin_ofs` : an int array, index of the string corpus
- `tok_ofs` : int array, index of the tokenised corpus
- `bin_fh` : a file descriptor of the binary corpus
- `tok_fh` : a file descriptor of the tokenised corpus
- `vocab` : vocabulary of the corpus, as we have seen before
- `minlen` : minimum length of document to save
- `docid` : int array to show the document id, which can refer to original data

Show how to build corpus using a convenient function in `Nlp.Corpus`. The convenient function builds the dictionary and tokenise the text corpus at the same time. You can further specify how to trim off the high-frequency and low-frequency words when calling the function.



```ocaml
let main () =
  (* remove duplicates *)
  let ids = Nlp.Corpus.unique "news.txt" "clean.txt" in
  Printf.printf "removed %i duplicates." (Array.length ids);

  (* build vocabulary and tokenise *)
  let corpus = Nlp.Corpus.build ~lo:0.01 ~hi:0.01 "clean.txt" in
  Nlp.Corpus.save corpus "news.corpus";
  Nlp.Corpus.print corpus
```

The output is like this ...

```text
2020-01-28 19:07:05.461 INFO : build up vocabulary ...
2020-01-28 19:07:10.461 INFO : processed 13587, avg. 2717 docs/s
2020-01-28 19:07:15.463 INFO : processed 26447, avg. 2644 docs/s
2020-01-28 19:07:20.463 INFO : processed 43713, avg. 2913 docs/s
2020-01-28 19:07:25.463 INFO : processed 57699, avg. 2884 docs/s
2020-01-28 19:07:30.463 INFO : processed 65537, avg. 2621 docs/s
2020-01-28 19:07:35.463 INFO : processed 76199, avg. 2539 docs/s
...
2020-01-28 19:08:09.125 INFO : convert to binary and tokenise ...
2020-01-28 19:08:14.126 INFO : processed 16756, avg. 3350 docs/s
2020-01-28 19:08:19.126 INFO : processed 32888, avg. 3288 docs/s
2020-01-28 19:08:24.127 INFO : processed 43110, avg. 2873 docs/s
2020-01-28 19:08:29.127 INFO : processed 48362, avg. 2417 docs/s
2020-01-28 19:08:34.130 INFO : processed 52628, avg. 2104 docs/s
2020-01-28 19:08:39.132 INFO : processed 55727, avg. 1857 docs/s
...
corpus info
  file path  : news.txt
  # of docs  : 129968
  doc minlen : 10
- : unit = ()
```

The `save` function will create several files, explain these files ...


### Iterate Documents

Teach how to use `Nlp.Corpus.next` and etc. to iterate and map documents.


## Vector Space Models

Survey, explain what is VSM, documents become vectors, i.e. a point in high-dimensional space. With VSM, we can cluster the documents based on their proximity, i.e. similarity.


## Bag of Words (BOW)

Explain what is BOW, simply counting the frequency. What are the pros and cons of this method?

```ocaml
(* count the term occurrency in a document *)
let term_count htbl doc =
  Array.iter
    (fun w ->
      match Hashtbl.mem htbl w with
      | true  ->
        let a = Hashtbl.find htbl w in
        Hashtbl.replace htbl w (a +. 1.)
      | false -> Hashtbl.add htbl w 1.)
    doc

(* build bag-of-words for the corpus *)
let build_bow corpus =
  Nlp.Corpus.mapi_tok
    (fun i doc ->
      let htbl = Hashtbl.create 128 in
      term_count htbl;
      htbl)
    corpus
```


## Term Frequency–Inverse Document Frequency (TFIDF)

Explain what is TFIDF, mention Cambridge Wolfson fellow. The corpus we have built in the previous section is used as input to the following function.

Explain why TFIDF is better than BOW, what is the motivation. Give an examble to illustrate.

```ocaml
let build_tfidf corpus =
  (* configure and build the model *)
  let tf = Nlp.Tfidf.Count in
  let df = Nlp.Tfidf.Idf in
  let model = Nlp.Tfidf.build ~tf ~df corpus in

  (* print and save model *)
  Nlp.Tfidf.print model;
  Nlp.Tfidf.save model "news.tfidf";

  model
```

After the model is build, illustrate how to find k similar documents. The following exmaple uses consine similarity, then convert a document into vector using previously trained TFIDF model. Note do NOT teach how to index and how the similarity is calculted here, teach in Indexing and Searching section.

```ocaml
let query model doc k =
  (* TODO: change api *)
  let typ = Owl_nlp_similarity.Cosine in
  let vec = Nlp.Tfidf.apply model doc in
  let knn = Nlp.Tfidf.nearest ~typ model vec k in
  knn
```


## Latent Dirichlet Allocation (LDA)

Explain what is LDA. `topics` is the number of topics. Owl supports the following types of LDA algorithms.


```ocaml
type lda_typ =
  | SimpleLDA
  | FTreeLDA
  | LightLDA
  | SparseLDA
```

How to train an LDA model.

```text
(* change to ocaml when image progation finished *)
let build_lda corpus topics =
  let model = Nlp.Lda.init ~iter:1000 topics in
  let lda_typ = Nlp.Lda.SparseLDA in
  Nlp.Lda.train lda_typ model;
  Owl.Log.info "LDA training finished."
```


## Latent Semantic Analysis (LSA)

Explain what is LSA, and how it differs from LDA wrt to derived topics.

Read
  * https://en.wikipedia.org/wiki/Latent_semantic_analysis
  * https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/


## Indexing and Searching

Topic models are effective tools for clustering documents based on their similarity or relevance. We can further use this tool to query relevant document given an input one. In this section, we will go through some techniques on how to index and query model built using the previous topic modeling method.

### Euclidean and Consine Similarity

Define what is euclidean and consine similarity. Emphasise both are correlated on a high-dimensional ball model.

TODO: use an image to illustrate.


### Liear Searching

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


## Conclusion

TBD
