# Distributed Computing

Background:

In this chapter, we will cover two topics:

## Actor System

refer to [@wang2017probabilistic]

## Map-Reduce Engine

Introduction 

All the 

## Parameter Server Engine

Introduction 


## Peer-to-Peer Engine

Introduction 

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


## Probabilistic Sampling in Synchronise Parallel

Existing barrier methods allow us to balance consistency against iteration rate in attempting to achieve a high rate of convergence. In particular, SSP parameterises the spectrum between ASP and BSP by introducing a staleness parameter, allowing some degree of asynchrony between nodes so long as no node lags too far behind. Figure~\ref{f:axis} depicts this trade-off.

However, in contrast to a highly reliable and homogeneous datacenter context, let's assume a distributed system consisting of a larger amount of heterogeneous nodes that are distributed at a much larger geographical areas.
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


## A Distributed Training Example


## References
