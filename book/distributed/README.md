# Distributed Computing

Background: decentralised computation

In this chapter, we will cover two topics:

1. Actor Engine
2. Barrier control, especially PSP

Refer to [@wang2017probabilistic] for more detail.

## Actor System

Introduction: Distributed computing engines etc.

### Design

(TODO: the design of actor's functor stack; how network/barrier etc. are implemented as separated as different module. Connection with Mirage etc.)

### Actor Engines

A key choice when designing systems for decentralised machine learning is the organisation of compute nodes. In the simplest case, models are trained in a centralised fashion on a single node leading to use of hardware accelerators such as GPUs and the TPU. For reasons indicated above, such as privacy and latency, decentralised machine learning is becoming more popular where data and model are spread across multiple compute nodes. Nodes compute over the data they hold, iteratively producing model updates for incorporation into the model, which is subsequently disseminated to all nodes.
These compute nodes can be organised in various ways.

The Actor system has implemented core APIs in both map-reduce engine and parameter sever engine. Both map-reduce and parameter server engines need a (logical) centralised entity to coordinate all the nodes' progress.  To demonstrate PSP's capability to transform an existing barrier control method into its fully distributed version, we also extended the parameter server engine to peer-to-peer (p2p) engine. The p2p engine can be used to implement both data and model parallel applications, both data and model parameters can be (although not necessarily) divided into multiple parts then distributed over different nodes.

Each engine has its own set of APIs. E.g., map-reduce engine includes `map`, `reduce`, `join`, `collect`, etc.; whilst the peer-to-peer engine provides four major APIs: push, pull, schedule, barrier. It is worth noting there is one function shared by all the engines, i.e. barrier function which implements various barrier control mechanisms.

Next we will introduce these three different kinds of engines of Actor.

### Map-Reduce Engine

Following MapReduce [@dean2008mapreduce] programming model, nodes can be divided by tasks: either *map* or *reduce*.
A map function processes a key/value pair to generate a set of intermediate key/value pairs, and a reduce function aggregates all the intermediate key/value paris with the same key.
Execution of this model can automatically be paralleled. Mappers compute in parallel while reducers receive the output from all mappers and combine to produce the accumulated result.
This parameter update is then broadcast to all nodes.
Details such as distributed scheduling, data divide, and communication in the cluster  are mostly transparent to the programmers so that they can focus on the logic of mappers and reducers in solving a problem within a large distributed system.


We can use a simple example to demonstrate this point. (with illustration, not code)

This simple functional style can be applied to a surprisingly wide range of applications.

Interfaces in Actor:

```
val map : ('a -> 'b) -> string -> string

val reduce : ('a -> 'a -> 'a) -> string -> 'a option

val fold : ('a -> 'b -> 'a) -> 'a -> string -> 'a

val filter : ('a -> bool) -> string -> string

val shuffle : string -> string

val union : string -> string -> string

val join : string -> string -> string

val collect : string -> 'a list
```


Example of using Map-Reduce in Actor: we use the classic wordcount example.

```
module Ctx = Actor.Mapre

let print_result x = List.iter (fun (k,v) -> Printf.printf "%s : %i\n" k v) x

let stop_words = ["a";"are";"is";"in";"it";"that";"this";"and";"to";"of";"so";
  "will";"can";"which";"for";"on";"in";"an";"with";"the";"-"]

let wordcount () =
  Ctx.init Sys.argv.(1) "tcp://localhost:5555";
  Ctx.load "unix://data/wordcount.data"
  |> Ctx.flatmap Str.(split (regexp "[ \t\n]"))
  |> Ctx.map String.lowercase_ascii
  |> Ctx.filter (fun x -> (String.length x) > 0)
  |> Ctx.filter (fun x -> not (List.mem x stop_words))
  |> Ctx.map (fun k -> (k,1))
  |> Ctx.reduce_by_key (+)
  |> Ctx.collect
  |> List.flatten |> print_result;
  Ctx.terminate ()

let _ = wordcount ()
```

### Parameter Server Engine

The Parameter Server topology proposed by [@li2014scaling] is similar: nodes are divided into servers holding the shared global view of the up-to-date model parameters, and workers, each holding its own view of the model and executing training. The workers and servers communicate in the format of key-value pairs.
It is proposed to address of challenge of sharing large amount of parameters within a cluster.
The parameter server paradigm applies an asynchronous task model to educe the overall network bandwidth, and also allows for flexible consistency, resource management, and fault tolerance.

Simple Example (distributed training) with illustration:
[IMAGE](https://miro.medium.com/max/371/1*6VRMmXkY3On-PJh8vNRHww.png): Distributed training with Parameter Server (Src:[@li2014scaling])

According to this example, we can see that the Parameter Server paradigm mainly consists of four APIs for the users.

- `schedule`: decide what model parameters should be computed to update in this step. It can be either a local decision or a central decision.

- `pull`: retrieve the updates of model parameters from somewhere then applies them to the local model. Furthermore, the local updates will be computed based on the scheduled model parameter.

- `push`: send the updates to the model plane. The updates can be sent to either a central server or to individual nodes depending on which engine is used(e.g., map-reduce, parameter server, or peer-to-peer).

- `barrier`: decide whether to advance the local step. Various synchronisation methods can be implemented. Besides the classic BSP, SSP, and ASP, we also implement the proposed PSP within this interface.


The interfaces in Actor:

```
val start : ?barrier:barrier -> string -> string -> unit
(** start running the model loop *)

val register_barrier : ps_barrier_typ -> unit
(** register user-defined barrier function at p2p server *)

val register_schedule : ('a, 'b, 'c) ps_schedule_typ -> unit
(** register user-defined scheduler *)

val register_pull : ('a, 'b, 'c) ps_pull_typ -> unit
(** register user-defined pull function executed at master *)

val register_push : ('a, 'b, 'c) ps_push_typ -> unit
(** register user-defined push function executed at worker *)

val register_stop : ps_stop_typ -> unit
(** register stopping criterion function *)

val get : 'a -> 'b * int
(** given a key, get its value and timestamp *)

val set : 'a -> 'b -> unit
(** given a key, set its value at master *)

val keys : unit -> 'a list
(** return all the keys in a parameter server *)

val worker_num : unit -> int
(** return the number of workers, only work at server side *)
```

EXPLAIN

Example of using PS in Actor :

```
module PS = Actor_param

let schedule workers =
  let tasks = List.map (fun x ->
    let k, v = Random.int 100, Random.int 1000 in (x, [(k,v)])
  ) workers in tasks

let push id vars =
  let updates = List.map (fun (k,v) ->
    Owl_log.info "working on %i" v;
    (k,v) ) vars in
  updates

let test_context () =
  PS.register_schedule schedule;
  PS.register_push push;
  PS.start Sys.argv.(1) Actor_config.manager_addr;
  Owl_log.info "do some work at master node"

let _ = test_context ()
```

### Peer-to-Peer Engine

In the above approaches the model parameter storage is managed by a set of centralised servers. In contrast, Peer-to-Peer (P2P) is a fully distributed structure, where each node contains its own copy of the model and nodes communicate directly with each other.
The benefit of this approach.

Illustrate how distributed computing can be finished with P2P model, using a figure.

To obtain the aforementioned two pieces of information, we can organise the nodes into a structured overlay (e.g., chord or kademlia), the total number of nodes can be estimated by the density of each zone (i.e., a chunk of the name space with well-defined prefixes), given the node identifiers are uniformly distributed in the name space. Using a structured overlay in the design guarantees the following sampling process is correct, i.e., random sampling.

Implementation in Actor:

```
open Actor_types

(** start running the model loop *)
val start : string -> string -> unit

(** register user-defined barrier function at p2p server *)
val register_barrier : p2p_barrier_typ -> unit

(** register user-defined pull function at p2p server *)
val register_pull : ('a, 'b) p2p_pull_typ -> unit

(** register user-defined scheduler at p2p client *)
val register_schedule : 'a p2p_schedule_typ -> unit

(** register user-defined push function at p2p client *)
val register_push : ('a, 'b) p2p_push_typ -> unit

(** register stopping criterion function at p2p client *)
val register_stop : p2p_stop_typ -> unit

(** given a key, get its value and timestamp *)
val get : 'a -> 'b * int

(** given a key, set its value at master *)
val set : 'a -> 'b -> unit
```

EXPLAIN

Example of using P2P in Actor (SGD):

```
open Owl
open Actor_types


module MX = Mat
module P2P = Actor_peer

...

let schedule _context =
  Owl_log.debug "%s: scheduling ..." !_context.master_addr;
  let n = MX.col_num !_model in
  let k = Stats.Rnd.uniform_int ~a:0 ~b:(n - 1) () in
  [ k ]

let push _context params =
  List.map (fun (k,v) ->
    Owl_log.debug "%s: working on %i ..." !_context.master_addr k;
    let y = MX.col !data_y k in
    let d = calculate_gradient 10 !data_x y v !gradfn !lossfn in
    let d = MX.(d *$ !step_t) in
    (k, d)
  ) params

let barrier _context =
  Owl_log.debug "checking barrier ...";
  true

let pull _context updates =
  Owl_log.debug "pulling updates ...";
  List.map (fun (k,v,t) ->
    let v0, _ = P2P.get k in
    let v1 = MX.(v0 - v) in
    k, v1, t
  ) updates

let stop _context = false

let start jid =
  P2P.register_barrier barrier;
  P2P.register_schedule schedule;
  P2P.register_push push;
  P2P.register_pull pull;
  P2P.register_stop stop;
  Owl_log.info "P2P: sdg algorithm starts running ...";
  P2P.start jid Actor_config.manager_addr
```

EXPLAIN


## Classic Synchronise Parallel

To ensure the correctness of computation, normally we need to make sure a correct order of updates. For example, one worker can only proceed when the model has been updated with all the workers' updates from previous round.
However, the iterative-convergent nature of ML programmes means that they are error-prone to a certain degree.

Most consistent system often leads to less than ideal system throughput.
This error-proneness means that, the consistency can be relaxed a bit without sacrificing accuracy, and gains system performance at the same time.
This trade-off is decided by the ``barrier'' in distributed ML.
A lot of research on it, both theoretically and practically.

Existing distributed processing systems operate at various points in the space of consistency/speed trade-offs. However, all effectively use one of three different synchronisation mechanisms: Bulk Synchronise Parallel (BSP), Stale Synchronise Parallel (SSP), and Asynchronous Parallel (ASP). These are depicted in [@fig:distributed:barriers].

![Barrier control methods used for synchronisation](images/distributed/barriers.png){#fig:distributed:barriers}

### Bulk Synchronous Parallel

Bulk Synchronous Parallel (BSP) is a deterministic scheme where workers perform a computation phase followed by a synchronisation/communication phase where they exchange updates.
The method ensures that all workers are on the same iteration of a computation by preventing any worker from proceeding to the next step until all can. Furthermore, the effects of the current computation are not made visible to other workers until the barrier has been passed. Provided the data and model of a distributed algorithm have been suitably scheduled, BSP programs are often serialisable -- that is, they are equivalent to sequential computations. This means that the correctness guarantees of the serial program are often realisable making BSP the strongest barrier control method. Unfortunately, BSP does have a disadvantage. As workers must wait for others to finish, the presence of *stragglers*, workers which require more time to complete a step due to random and unpredictable factors, limit the computation efficiency to that of the slowest machine. This leads to a dramatic reduction in performance. Overall, BSP tends to offer high computation accuracy but suffers from poor efficiency in unfavourable environments.

BSP is the most strict lockstep synchronisation; all the nodes are coordinated by a central server.
BSP is sensitive to stragglers so is very slow. But it is simple due to its deterministic nature, easy to write application on top of it.

### Asynchronous Parallel

Asynchronous Parallel (ASP)} takes the opposite approach to BSP, allowing computations to execute as fast as possible by running workers completely asynchronously. In homogeneous environments (e.g. data centers), wherein the workers have similar configurations, ASP enables fast convergence because it permits the highest iteration throughputs. Typically, $P$-fold speed-ups can be achieved by adding more computation/storage/bandwidth resources. However, such asynchrony causes delayed updates: updates calculated on an old model state which should have been applied earlier but were not. Applying them introduces noise and error into the computation. Consequently, ASP suffers from decreased iteration quality and may even diverge in unfavourable environments. Overall, ASP offers excellent speed-ups in convergence but has a greater risk of diverging especially in a heterogeneous context.

ASP is the Least strict synchronisation, no communication among workers for barrier synchronisation all all. Every computer can progress as fast as it can. It is fast and scalable, but often produces noisy updates. No theoretical guarantees on consistency and algorithm’s convergence.

### Stale Synchronous Parallel

Stale Synchronous Parallel (SSP)  is a bounded asynchronous model which can be viewed as a relaxation of BSP. Rather than requiring all workers to be on the same iteration, the system decides if a worker may proceed based on how far behind the slowest worker is, i.e. a pre-defined bounded staleness. Specifically, a worker which is more than $s$ iterations behind the fastest worker is considered too slow. If such a worker is present, the system pauses faster workers until the straggler catches up. This $s$ is known as the *staleness* parameter. More formally, each machine keeps an iteration counter, $c$, which it updates whenever it completes an iteration. Each worker also maintains a local view of the model state. After each iteration, a worker commits updates, i.e., $\Delta$, which the system then sends to other workers, along with the worker's updated counter. The bounding of clock differences through the staleness parameter means that the local model cannot contain updates older than $c -s - 1$ iterations. This limits the potential error. Note that systems typically enforce a *read-my-writes* consistency model. The staleness parameter allows SSP to provide deterministic convergence guarantees.
Note that SSP is a generalisation of BSP: setting $s = 0$ yields the BSP method, whilst setting $s = \infty$ produces ASP. Overall, SSP offers a good compromise between fully deterministic BSP and fully asynchronous ASP~\cite{ho2013}, despite the fact that the central server still needs to maintain the global state to guarantee its determinism nature.

SSP relaxes consistency by allowing difference in iteration rate. The difference is controlled by the bounded staleness.
SSP is supposed to mitigate the negative effects of stragglers. But the server still requires global state.


## Probabilistic Synchronise Parallel

Existing barrier methods allow us to balance consistency against iteration rate in attempting to achieve a high rate of convergence. In particular, SSP parameterises the spectrum between ASP and BSP by introducing a staleness parameter, allowing some degree of asynchrony between nodes so long as no node lags too far behind. Figure~\ref{f:axis} depicts this trade-off.

However, in contrast to a highly reliable and homogeneous datacentre context, let's assume a distributed system consisting of a larger amount of heterogeneous nodes that are distributed at a much larger geographical areas.
The network is unreliable since links can break down, and the bandwidth is heterogeneous. The nodes are not static, they can join and leave the system at any time.
We observe that BSP and SSP are not suitable for this scenario, since both are centralised mechanisms: a single server is responsible for receiving updates from each node and declaring when the next iteration can proceed.

In contrast, ASP is fully distributed and no single node is responsible for the system making progress.
We examine this trade-off in more detail, and suggest that the addition of a new *sampling* primitive exposes a second axis on which solutions can be placed, leading to greater flexibility in how barrier controls can be implemented. We call the resulting barrier control method (or, more precisely, family of barrier control methods) *Probabilistic Synchronous Parallel*, or PSP.

### Basic idea: sampling

The idea of PSP is simple: in a unreliable environment, we can minimise the impact of outliers and stragglers by guaranteeing the majority of the system have synchronised boundary.
The reason we can drop the results from certain portion of workers is that, practically many iterative learning algorithms can tolerate certain level of errors in the process of converging to final solutions. Given a well-defined boundary, if most of the nodes have passed it, the impact of those lagged nodes should be minimised.

Therefore, in PSP, all we need to do is to estimate what percent of nodes have passed a given synchronisation barrier.
Two pieces of information is required to answer this question:
- an estimate on the total number of nodes in the system;
- an estimate of the distribution of current steps of the nodes.

In PSP, either a central oracle tracks the progress of each worker or the workers each hold their own local view.
In a centralised system, without considering the difficulty of monitoring state of all the nodes as system grows bigger, these two pieces of information is apparently trivial to get at a central server.
However, in a distributed system, where each node does not have global knowledge of other nodes, how can it get these information?
In that case, a node randomly selects a subset of nodes in the system and query their individual current local step. By so doing, it obtains a sample of the current nodes' steps in the whole system.
By investigating the distribution of these observed steps, it can derive an estimate of the percentage of nodes which have passed a given step.
After deriving the estimate on the step distribution, a node can choose to either pass the barrier by advancing its local step if a given threshold has been reached (with certain probability) or simply holds until certain condition is satisfied.
Each node only depends on several other nodes to decide its own barrier.

### Compatibility

![Probabilistic Synchronous Parallel example](images/distributed/psp_00.png){#fig:distributed:psp_00}

One great advantage of PSP is its compatibility with existing synchronisation methods.
For classic BSP and SSP, their barrier functions are called by the centralised server to check the synchronisation condition with the given inputs. The output of the function is a boolean decision variable on whether or not to cross the synchronisation barrier, depending on whether the criterion specified in the algorithm is met.
With the proposed sampling primitive, almost nothing needs to be changed in aforementioned algorithms except that only the sampled states instead of the global states are passed into the barrier function. Therefore, we can easily derive the probabilistic version of BSP and SSP, namely *pBSP* and *pSSP*.

PSP improves on ASP by providing probabilistic guarantees about convergence with tighter bounds and less restrictive assumptions. pSSP relaxes SSP's inflexible staleness parameter by permitting some workers to fall further behind.
pBSP relaxes BSP by allowing some workers to lag slightly, yielding a BSP-like method which is more resistant to stragglers but no longer deterministic.

[@fig:distributed:psp_00](a) depicts PSP showing that different subsets of the population of nodes operate in (possibly overlapping) groups, applying one or other barrier control method within their group. In this case, [@fig:distributed:psp_00](b) shows PSP using BSP within group (or pBSP), and [@fig:distributed:psp_00](c) shows PSP using SSP within group (or pSSP).

Formally, at the barrier control point, a worker samples $\beta$ out of $P$ workers without replacement. If a single one of these lags more than $s$ steps behind the current worker then it waits. This process is pBSP (based on BSP) if $s = 0$ and pSSP (based on SSP) if $s > 0$. However, if $s = \infty$ then PSP reduces to ASP.

### Barrier Trade-off Dimensions

![Extra trade-off exposed through PSP](images/distributed/psp_01.png){#fig:distributed:psp_01}

This allows us to decouple the degree of synchronisation from the degree of distribution, introducing *completeness* as a second axis by having each node sample from the population. Within each sampled subgroup, traditional mechanisms can be applied allowing overall progress to be robust against the effect of stragglers while also ensuring a degree of consistency between iterations as the algorithm progresses.
As [@fig:distributed:psp_01](a-b) depicts, the result is a larger design space for synchronisation methods when operating distributed data processing at scale.

As [@fig:distributed:psp_01](c) summarises, probabilistic sampling allows us greater flexibility in designing synchronisation mechanisms in distributed processing systems at scale. When compared with BSP and SSP, we can obtain faster convergence through faster iterations and with no dependence on a single centralised server. When compared with ASP, we can obtain faster convergence with stronger guarantees by providing greater consistency between updates.

Besides its compatibility with existing synchronisation methods, it is also worth emphasising that applying sampling leads to the biggest difference between the classic synchronisation control and probabilistic control: namely the original synchronisation control requires a centralised node to hold the global state whereas the derived probabilistic ones no longer require such information thus can be executed independently on each individual node, further leading to a fully distributed solution.

### Convergence

At the barrier control point, every worker samples $\beta$ out of $P$ workers without replacement. If a single one of these lags more than $s$ steps behind the current worker then it waits.
The probabilities of a node lagging $r$ steps are drawn from a distribution with probability mass function $f(r)$ and cumulative distribution function (CDF) $F(r)$. Both $r$ and $\beta$ can be thought of as constant value.

In a distributing machine learning process, these $P$ workers keep generating updates, and the model is updated with them continuously. In this sequence of updates, each one is indexed by $t$ (which does not mean clock time), and the total length of this sequence is $T$.
Ideally, in a fully deterministic barrier control system, such as BSP, the ordering of updates in this sequence should be fixed. We call it a *true sequence*.
However, in reality, what we get is often a *noisy sequence*, where updates are reordered due to sporadic and random network and system delays.
The difference, or lag, between the order of these two sequence, is denoted by $\gamma_{t}$.

Without talking too much about math in detail in this book, to prove the convergence of PSP requires to construct and idea sequence of updates, each generated by workers in the distributed learning, and compare it with the actual sequence after applying PSP.
The target of proof is to show that, given sufficient time $t$, the difference between these two sequences $\gamma_t$ is limited.

The complete proof of convergence is too long to fit into this chapter. Please refer to [@wang2017probabilistic] for more detail if you are interested in the math.
One key step in the proof is to show that the mean and variance of $\gamma_t$ are bounded.
The average of the mean is bounded by:

$$\frac{1}{T} \sum_{t=0}^{T} E(\gamma_{t}) \leq  S \left( \frac{r(r+1)}{2} + \frac{a(r + 2)}{(1-a)^{2}} \right).$$ {#eq:distributed:bound_mean}

The average of the variance has a similar bound:

$$\frac{1}{T}  \sum_{t=0}^{T} E(\gamma_{t}^{2}) < S \left(\frac{r(r+1)(2r+1)}{6} + \frac{a(r^{2} + 4)}{(1-a)^{3}} \right),$$ {#eq:distributed:bound_var}

where
$$S = \frac{1-a}{F(r)(1-a) + a - a^{T-r+1}}.$$ {#eq:distributed:bound_s}

The intuition is that, when applying PSP, the update sequence we get will not be too different from the true sequence.
To demonstrate the impact of the sampling primitive on bounds quantitatively,
[@fig:distributed:proof_exp] shows how increasing the sampling count, $\beta$, (from 1, 5, to 100, marked with different line colours on the right) yields tighter bounds.
The sampling count $\beta$ is varied between 1 and 100 and marked with different line colours on the right. The staleness, $r$, is set to 4 with $T$ equal to 10000.
The bounds on $\gamma_t$ mean that what a true sequence achieves, in time, a noisy sequence can also achieve, regardless of the order of updates.
Notably, only a small number of nodes need to be sampled to yield bounds close to the optimal. This result has an important implication to justify using sampling primitive in large distributed learning systems due to its effectiveness.

![Plot showing the bound on the average of the means and variances of the sampling distribution.](images/distributed/proof_exp.png){#fig:distributed:proof_exp}

## A Distributed Training Example

In this section, we investigate performance of the proposed PSP in experiments.
We focus on two common metrics in evaluating barrier strategies: the step progress and accuracy.
We use the training of a DNN as an example, using a 9-layer structure used in the Neural Network chapter, and for the training we also use the MNIST handwritten digits dataset.
The learning rate has a decay factor of $1e4$.
The network structure is shown below:

```
let make_network () =
  input [|28;28;1|]
  |> normalisation ~decay:0.9
  |> conv2d [|5;5;1;32|] [|1;1|] ~act_typ:Activation.Relu
  |> max_pool2d [|2;2|] [|2;2|]
  |> dropout 0.1
  |> fully_connected 1024 ~act_typ:Activation.Relu
  |> linear 10 ~act_typ:Activation.(Softmax 1)
  |> get_network
```

We use both real-world experiments and simulations to evaluate different barrier control methods.
The experiments run on 6 nodes using Actor.
To extend the scale, we have also built a simulation platform.
For both cases, we implement the Parameter Server framework. It consists of one server and many worker nodes. In each step, a worker takes a chunk of training data, calculates the weight value, and then aggregates these updates to the parameter server, thus updating the shared model iteratively. A worker pulls new model from server after it is updated.

### Step Progress

First, we are going to investigate if PSP achieves faster iteration speed.
We use 200 workers, and run the simulation for 200 simulated seconds.
[@fig:distributed:exp_step_01] shows the distribution of all nodes' step progress when simulation is finished.

![Progress distribution in steps](images/distributed/exp_step_01.png){#fig:distributed:exp_step_01}

As expected, the most strict BSP leads to a tightly bounded step distribution, but at the same time, using BSP makes all the nodes progress slowly. At the end of simulation, all the nodes only proceed to about 30th step.
As a comparison, using ASP leads to a much faster progress of around 100 steps. But the cost is a much loosely spread distribution, which shows no synchronisation at all among nodes.
SSP allows certain staleness (4 in our experiment) and sits between BSP and ASP.

PSP shows another dimension of performance tuning. We set sample size $\beta$ to 10, i.e. a sampling ratio of only 5\%. The result shows that pBSP is almost as tightly bound as BSP and also much faster than BSP itself.
The same is also true when comparing pSSP and SSP.
In both cases, PSP improves the iteration efficiency while limiting dispersion.

![pBSP parameterised by different sample sizes, from 0 to 64.](images/distributed/exp_step_02.png){width=60% #fig:distributed:exp_step_02}

To further investigate the impact of sample size, we focus on BSP, and choose different sample sizes.
In [@fig:distributed:exp_step_02] we vary the sample size from 0 to 64. As we increase the sample size step by step, the curves start shifting from right to left with tighter and tighter spread, indicating less variance in nodes' progress.
With sample size 0, the pBSP exhibits exactly the same behaviour as that of ASP; with increased sample size, pBSP starts becoming more similar to SSP and BSP with tighter requirements on synchronisation. pBSP of sample size 4 behaves very close to SSP.

Another interesting thing we notice in the experiment is that, with a very small sample size of one or two (i.e., very small communication cost on each individual node), pBSP can already effectively synchronise most of the nodes comparing to ASP. The tail caused by stragglers can be further trimmed by using larger sample size.
This observation confirms our convergence analysis, which explicitly shows that a small sample size can effectively push the probabilistic convergence guarantee to its optimum even for a large system size, which further indicates the superior scalability of the proposed solution.

### Accuracy

Next, we evaluate barrier control methods in training a deep neural network using MNIST dataset. We use inference accuracy on test dataset as measurement of performance.

To run the code, we mainly implement the interfaces we mentioned before: `sche`, `push`, and `pull`.
`sche` and `pull` are performed on server, and `push` is on worker.

```
  let schd nodes =
    Actor_log.info "#nodes: %d" (Array.length nodes);
    if (Array.length nodes > 0) then (
      eval_server_model ()
    );

    let server_value = (get [|key_name|]).(0) in
    Array.map (fun node ->
      Actor_log.info "node: %s schd" node;
      let wid = int_of_uuid node in
      let v = make_task server_value.weight wid in
      let tasks = [|(key_name, v)|] in
      (node, tasks)
    ) nodes

let pull kv_pairs =
    Gc.compact ();
    (* knowing that there is only one key...*)
    let key = key_name in
    Actor_log.info "pull: %s (length: %d) " key (Array.length kv_pairs);

    let s_value = (get [|key|]).(0) in
    let s_weight = ref s_value.weight in
    Array.iter (fun (_, v) ->
      s_weight := add_weight !s_weight v.weight;
      model.clocks.(v.wid) <- model.clocks.(v.wid) + 1
    ) kv_pairs;
    let value = make_task !s_weight 0 in (* wid here is meaningless *)
    [|(key, value)|]

```

And on worker:

```
let push kv_pairs =
    Gc.compact ();
    Array.map (fun (k, v) ->
      Actor_log.info "push: %s" k;
      (* simulated communication delay *)

      (* let t = delay.(v.wid) in *)
      let t = Owl_stats.gamma_rvs ~shape:1. ~scale:1. in
      Unix.sleepf t;

      let nn = make_network () in
      Graph.init nn;
      Graph.update nn v.weight;
      let x, y = get_next_batch v.wid in
      let x = pack x in
      let y = pack y in
      let _ = CGCompiler.train ~params ~init_model:false nn x y in
      let delta = delta_nn nn v.weight in
      let value = make_task delta v.wid in
      (k, value)
    ) kv_pairs
```

![MNIST training using Actor](images/distributed/exp_accuracy_01.png){#fig:distributed:exp_accuracy_01}

In [@fig:distributed:exp_accuracy_01], we conduct the real-world experiments  using 6 worker nodes with the Parameter Server framework we have implemented.
We run the training process for a fixed amount of time, and observe the performance of barrier methods given the same number of updates.
It shows that BSP achieves the highest model accuracy with the least of number of updates, while SSP and ASP achieve lower efficiency. With training progressing, both methods show a tendency to diverge.
By applying sampling, pBSP and pSSP achieve smoother accuracy progress.


## Summary

## References
