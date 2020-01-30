# Natural Language Processing

Text is a dominant media type on the Internet along with images, videos, and audios. Many of our day-to-day tasks involve text analysis. Natural language processing is a powerful tool to extract insights from text copora. 

This chapter focusses on the vector space models and topic modelling.


## Introduction

Survey the literature, give a high-level picture of NLP. Talk about classic NLP ... structured and unstructured text ...

In this chapter, we will use a [news dataset](https://github.com/ryanrhymes/owl_dataset/raw/master/news.txt.gz) crawled from the Internet. It contains 130000 pieces of news from various sources, each line in the file represents one entry.


## Text Corpus

We call a collection of documents a text corpus. 

### Step-by-step Operation

Show how to build a corpus from a collection of documents, in a step step way.

Preprocess the text, convert all the text into lowercase.

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

Then let's build vocabulary out of the text corpus.

```ocaml
let build_vocabulary input_file =
  let vocab = Nlp.Vocabulary.build input_file in

  (* print out the words of highest frequency *)
  Nlp.Vocabulary.top vocab 10 |>
  Owl.Utils.Array.to_string ~sep:", " fst;

  (* print out the words of lowest frequency*)
  Nlp.Vocabulary.bottom vocab 10 |>
  Owl.Utils.Array.to_string ~sep:", " fst;

  (* save the vocabulary *)
  let output_file = input_file ^ ".vocab" in 
  Nlp.Vocabulary.save vocab output_file
```

The progress of building the vocabulary is printed out. After the vocabular is built, the token of the highest frequency is printed out.

```text
2020-01-28 20:29:41.592 INFO : processed 13485, avg. 2696 docs/s
2020-01-28 20:29:46.593 INFO : processed 24975, avg. 2497 docs/s
2020-01-28 20:29:51.593 INFO : processed 39728, avg. 2648 docs/s
2020-01-28 20:29:56.595 INFO : processed 53277, avg. 2663 docs/s
2020-01-28 20:30:01.595 INFO : processed 62908, avg. 2516 docs/s
2020-01-28 20:30:06.595 INFO : processed 69924, avg. 2330 docs/s
...
- : string =
"the, to, of, a, and, in, \", s, that, on"
"eichorst, gcs, freeross, depoliticisation, humping, shopable, appurify, intersperse, vyaecheslav, raphaelle"
```

Now let's trim off some most and least frequency words. You can trim either by absolute number or by percent. We use percent here, namely triming off top and bottom 1% of the words.

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


### Use Convenient Function

Show how to build corpus using a convenient function in `Nlp.Corpus`. The convenient function builds the dictionary and tokienise the text corpus at the same time. You can further specify how to trim off the high-frequency and low-frenquency words when calling the function.

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

After the model is build, illustrate how to find k similar documents. The following exmaple uses consine similarity, then convert a document into vector using previously trained TFIDF model.

```ocaml
let query model doc k =
  (* TODO: change api *)
  let typ = Owl_nlp_similarity.Cosine in
  let vec = Nlp.Tfidf.apply model doc in
  let knn = Nlp.Tfidf.nearest ~typ model vec k in
  knn
```


## Latent Dirichlet Allocation (LDA)

Explain what is LDA.

```ocaml
(* TODO: I need to change NLP interface a bit *)
let build_lda corpus =
  ()
```


## Latent Semantic Analysis (LSA)

Explain what is LSA, and how it differs from LDA wrt to derived topics.


## Indexing and Searching

First implement linear search, then explain random projection and implement a naive version.


## Conclusion

TBD
