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

