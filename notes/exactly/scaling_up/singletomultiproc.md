ok simple version:

* **Tensor Parallelism (TP)** → split **each layer’s math** across multiple GPUs.

  * e.g. one layer’s giant matrix multiply is sliced into chunks, each GPU does part of it, then they combine results.
  * Used when a **single layer** is too big for one GPU’s VRAM.
  * “within-layer” splitting.

* **Pipeline Parallelism (PP)** → split **layers themselves** across GPUs.

  * e.g. GPU0 runs layers 0–11, GPU1 runs layers 12–23.
  * Tokens flow through the “pipeline” — one GPU finishes its part, passes activations to the next.
  * Used when the **whole model** doesn’t fit on one GPU, but each layer can fit.

* **Data Parallelism (DP)** → run **whole model copies** on multiple GPUs, each gets a different batch of data, then they sync gradients or outputs.

  * Used to speed up throughput (more requests at once).

* **Expert Parallelism (EP)** → used in **Mixture-of-Experts (MoE)** models.

  * Only some “experts” (sub-networks) run per token, and they can live on different GPUs.

* **Sequence Parallelism (SP)** → split along the **sequence length** dimension.

  * Each GPU handles part of a long sequence of tokens.

---

**Quick mental map:**

> TP = split math inside layers
> PP = split layers themselves
> DP = replicate model for more data
> EP = split model into optional experts
> SP = split the sequence itself

--------------------------------------

**Simply:**

* **Intranode** = communication **within one machine** (between GPUs inside the same box).
  → Uses super-fast links like **NVLink / PCIe** → **very high bandwidth**, **low latency**.

* **Internode** = communication **between machines** (over a network).
  → Uses **Ethernet / InfiniBand** → **much slower**, **higher latency**.

So:

> **Intranode = fast (inside one node)**
> **Internode = slower (across nodes)**

-----------------------------------------

From the engine's perspective, nothing has changed — all of this multiprocessing complexity is abstracted away through a call to model executor's execute_model.

In the UniProcExecutor case: execute_model directly leads to calling execute_model on the worker
In the MultiProcExecutor case: execute_model indirectly leads to calling execute_model on each worker through rpc_broadcast_mq

----------------------------------------

When vLLM says the processes use “shared memory,” they mean shared CPU RAM, not shared VRAM on the GPUs.

They share certain data structures (like message queues) using the OS’s shared memory APIs — e.g., multiprocessing.shared_memory, mmap, or POSIX shm.

3. GPU memory is still separate
Each GPU process:


Has its own CUDA context.


Loads its own slice of the model weights into that GPU’s VRAM.


Keeps its own KV cache blocks for its assigned attention heads.


The shared memory queues never touch GPU tensors; they just coordinate who should run what.

4. Literal picture
CPU RAM (shared)
 ├── rpc_broadcast_mq  ← all processes can read/write here
 ├── worker_response_mq ← each worker writes results here
GPU 0 VRAM — only worker 0 can access
GPU 1 VRAM — only worker 1 can access
...


So in plain words:

“Shared memory” = a shared chunk of CPU RAM used for fast messaging between processes.
The GPUs themselves still have separate VRAM, each controlled by its own process.

CPU Shared Memory Region
 ├── rpc_broadcast_mq      (1 shared inbox → all workers read)
 ├── worker_response_mq_0  (GPU0’s outbox)
 ├── worker_response_mq_1  (GPU1’s outbox)
 ├── ...
 └── worker_response_mq_7  (GPU7’s outbox)

---------------

Essentially, each work item is:

“Hey worker, run your slice of the model on these inputs and return your partial outputs.”

Summary
Step	Queue	Direction	Contents
1	rpc_broadcast_mq	Parent → All workers	“Run forward pass on these tensors.”
2	worker_response_mq[i]	Worker → Parent	“Here’s my result (partial output / status).”

Yes — in **vLLM’s MultiProcExecutor setup**, **each GPU has its own KV cache**, but the details depend on the type of parallelism being used. Let’s break it down precisely:

---

### 🧠 1. The baseline: UniProcExecutor (single GPU)

In the single-GPU case, there’s only **one** KV cache:

* It’s a big contiguous buffer in GPU memory (paged attention allocator).
* Every request maps tokens → KV blocks within that single cache.

So:

> **UniProcExecutor → one process, one GPU, one KV cache.**

---

### ⚙️ 2. MultiProcExecutor (multi-GPU)

When vLLM uses **tensor parallelism (TP)** or **pipeline parallelism (PP)**, the model is split across GPUs.
Each **worker process** (one per GPU) initializes its own:

* CUDA context
* Model shard
* **KV cache allocator**

So effectively:

> **Each GPU process owns a separate KV cache region for its shard of the model.**

---

### 🧩 3. What’s actually stored in each GPU’s KV cache

In **tensor parallelism (TP)** mode:

* Every GPU holds **a slice of the K and V tensors** for *each layer*.
* Example: if TP=4, each GPU holds ¼ of the attention heads (or ¼ of each head’s dimension, depending on the implementation).
* During attention, all GPUs must exchange partial results to reconstruct the full context attention, but each GPU’s KV cache only needs to store **its local partition** of the keys and values.

So each cache looks like:

```
GPU0: KV for heads [0..N/4)
GPU1: KV for heads [N/4..N/2)
GPU2: KV for heads [N/2..3N/4)
GPU3: KV for heads [3N/4..N)
```

These partitions are synchronized via **collective ops** (e.g., NCCL all-gather).

---

### 🧵 4. In pipeline parallelism (PP)

If you go beyond TP and use **pipeline parallelism**, each GPU handles a *different group of layers*.
Then:

* The KV cache for layer 0–11 might live on GPU0.
* The KV cache for layer 12–23 lives on GPU1.
* etc.

Each GPU’s KV cache stores K/V only for the layers it’s responsible for.

---

### 🔄 5. During multi-GPU inference

Each GPU’s process does the same general thing:

1. Allocates its local KV blocks during initialization (`init_device` → `initialize_kv_cache`).
2. Stores its shard’s K/V tensors in VRAM.
3. When decoding, loads the relevant blocks (via paged attention) for its subset of heads/layers.
4. Synchronizes with other GPUs when needed for tensor ops.

So yes, **each GPU absolutely has its own KV cache**, but:

* The caches are **logically linked** (they represent different *parts* of the full model state).
* They are **not redundant copies** unless you’re doing **data parallelism (DP)** — where each GPU runs a full copy of the model for separate requests.

---

### Summary table

| Parallelism type           | KV cache per GPU | What it contains                       | Shared or independent?         |
| -------------------------- | ---------------- | -------------------------------------- | ------------------------------ |
| **UniProc (1 GPU)**        | 1                | Full KV cache                          | N/A                            |
| **Tensor Parallel (TP)**   | 1 per GPU        | Slice of keys/values (e.g. some heads) | Shared logically (collectives) |
| **Pipeline Parallel (PP)** | 1 per GPU        | Layers assigned to that stage          | Shared via pipeline flow       |
| **Data Parallel (DP)**     | 1 per GPU        | Full KV cache (for different requests) | Independent                    |

---

Would you like me to diagram this (how the KV caches are split under tensor parallelism vs pipeline parallelism)? It’s much clearer visually.

Yeah — let’s break **MultiProcExecutor** down in plain terms, using the **vLLM** context.

---

### 1. The core idea

When you run a large model, it may not fit into one GPU’s memory.
**MultiProcExecutor** is the system that lets vLLM run that single model across *multiple GPUs* by spawning **multiple worker processes**, one per GPU.

Each process:

* Controls its own GPU (`cuda:0`, `cuda:1`, …)
* Loads *its shard* of the model weights (for tensor parallelism)
* Communicates with the others via message queues (fast shared-memory pipes)

You can think of it as:

```
Main process
 ├─ Worker 0  -> GPU 0
 ├─ Worker 1  -> GPU 1
 ├─ Worker 2  -> GPU 2
 └─ ...
```

All these workers together act like one big GPU running the model in parallel.

---

### 2. Why it exists

A normal vLLM engine uses **UniProcExecutor** (single GPU).
But when a model’s weights exceed a single GPU’s VRAM (say 24GB), we have to split it up.

**MultiProcExecutor** does this splitting *and manages coordination*:

* Keeps track of which process has which part of the model
* Ensures all GPUs work together for each forward pass
* Gathers results back to the main process

---

### 3. How it actually works

Here’s the simplified lifecycle:

#### (1) Initialization

The main process:

* Creates a **shared message queue** (`rpc_broadcast_mq`) for sending work to workers.
* Spawns `N` worker processes (where `N = tensor_parallel_size`).
* For each worker, also creates a private **response queue** (`worker_response_mq`) for results.

Each worker:

* Initializes its GPU (loads its shard of the model, sets up CUDA context).
* Waits for work from the main process on the broadcast queue.

```
main process
  ├── rpc_broadcast_mq   → shared work queue
  ├── worker_response_mq[0]
  ├── worker_response_mq[1]
  └── ...
```

---

#### (2) Execution

When the engine wants to run a forward pass:

1. The main process puts a work item (like “run this batch of tokens”) into the **broadcast queue**.
   → All workers see the same message (since they all read from it).

2. Each worker:

   * Runs its assigned part of the forward pass (its model shard).
   * Sends back its output through its **response queue**.

3. The main process waits on the designated output worker’s response queue (usually rank 0) to collect results.

---

### 4. Analogy

Imagine you have an 8-part orchestra:

* The **main process** is the conductor.
* Each **worker process** is a musician (GPU) playing one part of the piece (a shard of the model).
* The **broadcast queue** is the conductor’s sheet music (everyone reads from the same instructions).
* The **response queues** are what each musician sends back (their notes/results).
* The conductor combines them into the full sound (final model output).

---

### 5. Compared to UniProcExecutor

| Feature          | UniProcExecutor | MultiProcExecutor                        |
| ---------------- | --------------- | ---------------------------------------- |
| GPUs             | 1               | Many                                     |
| Processes        | 1               | 1 main + N workers                       |
| Communication    | None            | Shared-memory message queues             |
| Use case         | Small model     | Large model that needs model parallelism |
| Parallelism type | None            | Tensor / pipeline parallelism            |

---

### 6. From the blog diagram (what you saw)

That figure shows exactly this setup:

```
MultiProcExecutor
 ├─ main process
 │   └─ rpc_broadcast_mq  (sends work to all)
 │
 ├─ Worker 0 (cuda:0)
 │   ├─ worker_response_mq
 │   └─ rpc_broadcast_mq
 ├─ Worker 1 (cuda:1)
 │   ├─ worker_response_mq
 │   └─ rpc_broadcast_mq
 └─ ...
```

---

### 7. In short

> **MultiProcExecutor** = “One manager, many GPU workers, synchronized through message queues.”

It makes **multi-GPU inference** look like a single process to the rest of vLLM — the rest of the engine doesn’t have to care whether the model is on one GPU or eight.

---

Would you like me to show a concrete example of how it splits and runs a tensor-parallel forward pass across two GPUs (like what actually gets computed on each)?

------------

Here’s a clear **pseudo-code sketch** showing what happens inside vLLM’s **`MultiProcExecutor`**, and how the **input/output queues** are used by parent and worker processes during a forward pass:

---

### 🧩 Simplified structure

```python
class MultiProcExecutor:
    def __init__(self, world_size):
        # shared queue parent → workers
        self.rpc_broadcast_mq = SharedMemoryQueue()
        # per-worker queue worker → parent
        self.worker_response_mqs = [Queue() for _ in range(world_size)]

        # spawn workers
        for rank in range(world_size):
            Process(target=worker_main,
                    args=(rank, self.rpc_broadcast_mq, self.worker_response_mqs[rank])).start()

    def execute_model(self, input_batch):
        # build work item for all workers
        work_item = {
            "command": "execute_model",
            "input_batch": serialize(input_batch),
        }

        # broadcast work to all GPU workers
        self.rpc_broadcast_mq.put(work_item)

        # wait for result from the designated output rank (usually 0)
        result = self.worker_response_mqs[0].get()
        return result["output"]
```

---

### 🧠 Each worker process

```python
def worker_main(rank, broadcast_mq, response_mq):
    # initialize device, load model shard, join NCCL group
    model = init_worker(rank)

    while True:
        # block on broadcast queue
        msg = broadcast_mq.get()              # wait for parent
        if msg["command"] == "shutdown":
            break

        if msg["command"] == "execute_model":
            # deserialize and run local shard of the forward pass
            batch = deserialize(msg["input_batch"])
            output = model.forward(batch)

            # optionally participate in collectives (all_reduce, all_gather)
            sync_distributed(rank, output)

            # only one worker returns the final combined output
            if rank == 0:
                response_mq.put({"output": serialize(output)})
```

---

### 🧮 Queue contents at runtime

| Queue                          | Direction            | Typical contents                                                   |
| ------------------------------ | -------------------- | ------------------------------------------------------------------ |
| **`rpc_broadcast_mq`**         | Parent → All workers | `{ "command": "execute_model", "input_batch": <tensor metadata> }` |
| **`worker_response_mq[rank]`** | Worker → Parent      | `{ "output": <logits/tokens/hidden states> }`                      |
| **(internal CUDA queues)**     | Within worker        | GPU copy + kernel launch tasks (async)                             |

---

### 🕓 Execution timeline

1. **Parent** builds the forward-pass payload and puts it into `rpc_broadcast_mq`.
2. **All workers** dequeue the message simultaneously.
3. Each worker runs its part of the model on its GPU.
4. Workers synchronize through NCCL (tensor parallel / pipeline stages).
5. **Output rank 0** sends the merged result to its `worker_response_mq`.
6. **Parent** collects from that queue and returns to the `EngineCore`.

---

This mirrors exactly how vLLM’s multiprocessing executor avoids Python GIL contention and lets each GPU process run its own CUDA stream independently, while the shared-memory queues provide lightweight IPC instead of sockets or gRPC.

Would you like me to extend this to show **how the MultiProcExecutor plugs into the `EngineCore` loop** (with `input_queue` → `MultiProcExecutor` → `output_queue`), as it appears in the DP diagram you showed earlier?

Excellent question — this goes to the heart of **why vLLM can scale from a single process (UniProcExecutor)** to **MultiProcExecutor (and distributed DP replicas)** *without changing the engine interface at all*.

Let’s unpack how **KV cache**, **paged attention**, and **continuous batching** behave and “appear” to the outside world across these setups.

---

## 1️⃣  From the Outside: Nothing Changes

At the **API / Engine interface level**, both UniProc and MultiProc executors expose the same contract:

```python
engine_core.execute_model(input_batch)
```

That function signature doesn’t change whether you’re running:

* on 1 GPU (UniProcExecutor),
* on 8 GPUs (MultiProcExecutor with TP=8),
* or across nodes (DP replicas coordinated by the DP coordinator).

From the outside:

* You enqueue requests (`engine_core.add_request(...)`)
* You get results (`engine_core.step()`)
* You never touch CUDA streams, shards, or caches manually.

Internally though, the engine core and executor adapt the same abstractions — **input batch**, **KV cache manager**, and **scheduler** — to their parallel environment.

So from the outside, all three systems (KV cache, paged attention, continuous batching) behave identically — but the *implementation underneath* changes.

---

## 2️⃣  KV Cache — Same abstraction, different plumbing

### In UniProcExecutor:

* There’s a **single KV cache allocator** (one process, one GPU, one paged memory pool).
* All blocks live in one contiguous GPU address space.
* The `KVCacheManager` directly allocates and indexes blocks from a single `free_block_queue`.

### In MultiProcExecutor:

* Each worker process has its **own KV-cache allocator**, but all workers form one logical cache space.
* When tensor parallelism is used (TP>1), each rank only holds a *shard* of the KV tensors (e.g., key/value heads are split across GPUs).
* Paged attention metadata (slot mapping) is distributed — every rank gets its local slice.
* Coordination happens via NCCL collectives so all ranks stay consistent (each shard knows which tokens belong to which slot).

**To the EngineCore, though:**

> “KV cache” still looks like a single addressable space with a `.allocate_slots()` call.
> It just happens that `allocate_slots()` now performs NCCL group communication internally instead of local memory operations.

So:

* **Externally:** identical interface (`kv_cache_manager.allocate_slots`, `free`, `reuse`).
* **Internally:** distributed allocator across GPU ranks.

---

## 3️⃣  Paged Attention — identical logic, distributed memory

Paged attention is the memory layout mechanism that maps **token positions → KV cache blocks**.

### In UniProc:

* There’s one attention kernel that sees the full sequence and uses local KV memory.
* The scheduler flattens all active sequences into a single “super sequence” tensor.
* Paged attention ensures each sequence only attends to its own tokens.

### In MultiProc:

* Each rank runs the same logic on its local shard of the KV cache.
* The model’s attention heads are sharded (each GPU holds `num_heads / TP` heads).
* When performing attention, ranks use NCCL all-gather or reduce-scatter to combine attention results across GPUs.

But to higher layers (scheduler, step loop), paged attention is just part of the forward pass inside `execute_model`.
The **input/output tensors** look identical — it’s the executor that calls `MultiProcExecutor.forward()` which wraps these distributed attention kernels.

So again:

> From the outside, paged attention “looks the same.”
> Under the hood, it’s just distributed across GPUs instead of single-device memory.

---

## 4️⃣  Continuous Batching — identical semantics, more parallel threads

Continuous batching is managed entirely by the **scheduler**, not the executor.
It decides which requests to batch in the next step.

### In UniProc:

* The scheduler builds a batch of requests and runs one forward pass.
* The engine_core’s main thread executes this synchronously.

### In MultiProc / DP:

* The scheduler runs in *each replica or worker process*.
* Each replica has its own request queue and local batch.
* The DP coordinator synchronizes which replicas are active (lockstep if needed, or independent for standard models).
* Tensor-parallel shards still receive the same batch via the shared broadcast queue.

So, continuous batching logic doesn’t change at all — the only difference is:

> The same batch description (token IDs, slot mapping, metadata) is now **broadcast to multiple GPU workers** via `rpc_broadcast_mq` instead of being executed in one process.

---

## 5️⃣  Summary Table

| Concept                 | SingleProcExecutor           | MultiProcExecutor                                            | Appearance to EngineCore                   |
| ----------------------- | ---------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| **KV Cache**            | Single allocator, single GPU | Distributed allocator (each rank has shard, synced via NCCL) | Same API (`allocate_slots`, `free_blocks`) |
| **Paged Attention**     | Local KV memory              | Sharded KV + all-reduce across ranks                         | Same semantics, different backend kernel   |
| **Continuous Batching** | Local scheduler batch        | Each replica batches locally; DP coord syncs waves           | Scheduler logic identical                  |

---

## 6️⃣  Key Takeaway

The brilliance of vLLM’s architecture is that:

* **UniProcExecutor** and **MultiProcExecutor** share the same interface;
* all “advanced features” (paged attention, continuous batching, KV caching) live in **EngineCore and Scheduler** layers, which don’t need to know if the executor is local or distributed;
* only the **execution backend** (the Executor) changes its internal mechanics — how it runs forward passes and manages memory across devices.

So from the outside:

> whether you run on 1 GPU or 100 GPUs, vLLM behaves like the same engine — the executor just scales the plumbing underneath.

---

If you want, I can sketch a short ASCII diagram showing exactly where each layer “stops caring” — how `EngineCore` hands to `Executor`, which hands to workers, but the KV cache and scheduler stay identical. Want that?
