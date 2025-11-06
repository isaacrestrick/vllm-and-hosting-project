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