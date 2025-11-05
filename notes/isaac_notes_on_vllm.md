Paged Attention

 3. How it works in practice

Each request (sequence) is represented by a list of pointers to its KV blocks, rather than a single large contiguous tensor.

the attention kernels have been modified to index into a block table (a page table) to fetch K/Vs from wherever they physically reside in GPU memory.

Standard attention:
  K, V = [num_tokens, dim]
  Attention(Q, K, V)

Paged attention:
  K_pages = [[block0], [block1], [block2], ...]
  page_table = { seq1: [0, 2, 5], seq2: [1, 3] }
  Attention(Q, K_pages, V_pages, page_table)

Good question — “kernel” here doesn’t mean a whole operating system kernel.
It means a GPU kernel — a small, highly optimized function that runs in parallel on the GPU.

In the context of paged attention:

The attention kernel is the chunk of GPU code that actually computes

softmax(Q Kᵀ / √d) V


for all tokens in a batch.

In regular attention, that kernel assumes K and V are stored contiguously in memory.

In paged attention, vLLM replaces those with a custom kernel that can follow an indirection table — the “page table” — to load K/V blocks that might be scattered around GPU memory.

So when the blog says

“custom paged attention kernels,”
it means: specialized CUDA (or Triton) GPU kernels that efficiently gather K/Vs according to the page layout, compute attention, and write results — all without copying or re-packing them into a contiguous tensor first.

Essentially:
kernel = the GPU-level function implementing the math of attention, optimized for the paged memory layout.

-----------------------------------------------------------------------------------------------------------

Good question — they’re closely related but sit at **different abstraction levels**.

---

### **1. CUDA = low-level GPU programming platform**

* **What it is:** NVIDIA’s C/C++ API and runtime for writing GPU code.
* **You write:** `__global__` functions (kernels) in C++ that use explicit threads, blocks, and shared memory.
* **You manage:**

  * Launch grid sizes (`<<<blocks, threads>>>`)
  * Memory copies between CPU and GPU
  * Synchronization, streams, etc.
* **Goal:** Full control and maximum performance — but verbose and tricky to optimize.

Example:

```cpp
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

---

### **2. Triton = higher-level compiler for GPU kernels**

* **What it is:** An **open-source DSL from OpenAI** that lets you write custom GPU kernels **in Python**, not C++.
* **You write:** Python functions decorated with `@triton.jit`.
* **You describe operations per “program” (a logical block of threads)** rather than manual warp/thread logic.
* **The compiler (LLVM + PTX backend)** generates efficient CUDA kernels automatically.

Example:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(axis=0)
    offset = pid * 256 + tl.arange(0, 256)
    mask = offset < N
    x = tl.load(X + offset, mask=mask)
    y = tl.load(Y + offset, mask=mask)
    tl.store(Z + offset, x + y, mask=mask)
```

This compiles to CUDA code under the hood, handles tiling, masking, memory coalescing, etc.

---

### **3. How they relate**

| Aspect      | CUDA                            | Triton                                             |
| ----------- | ------------------------------- | -------------------------------------------------- |
| Language    | C/C++                           | Python DSL                                         |
| Level       | Low (hardware-aware)            | Mid-level (auto-optimized)                         |
| Vendor      | NVIDIA                          | OpenAI (built on top of CUDA)                      |
| Compilation | nvcc → PTX → SASS               | Triton → LLVM → PTX → SASS                         |
| Control     | Full control over warps, memory | Easier abstraction; still efficient                |
| Used by     | PyTorch custom ops, NVIDIA SDKs | PyTorch 2.x Inductor, vLLM kernels, FlashAttention |

---

### **4. In vLLM context**

vLLM’s **paged attention kernels** are implemented in **Triton**, not raw CUDA.
That’s because:

* Triton makes it easier to write custom GPU code that fuses operations (QKᵀ, softmax, V multiply, etc.).
* It can still hit near-CUDA performance.
* It’s portable within PyTorch 2’s compilation stack (Inductor uses Triton).

So:

> **CUDA** = the GPU’s native assembly language layer.
> **Triton** = a Pythonic compiler that *generates* CUDA kernels for you.

---

Would you like me to show you how the same small attention operation looks in both CUDA and Triton side-by-side?

-----------------------------------------------------------------------------------------

what is an engine step in your explanation?

In **vLLM** (and other modern LLM inference engines), an **engine step** is the smallest **unit of work** — a single iteration of the model’s forward pass across all active requests.

It’s the heartbeat of continuous batching.

---

### 🔁 What happens in one *engine step*

Each **step** corresponds to generating **one or more tokens** for each active request.

It has **three main stages** (as described in Aleksa Gordić’s article):

---

#### 1. **Schedule**

The scheduler decides **which requests to include** in this step:

* It looks at:

  * **Running requests** (those mid-generation — decode mode)
  * **Waiting requests** (new prompts needing prefill)
* It prioritizes **decode requests** to keep latency low.
* It calls the **KV-cache manager** to allocate memory blocks for each request.
* Requests are then moved between queues:

  * WAITING → RUNNING (when scheduled)
  * RUNNING → FINISHED (when done)

So by the end of this stage, the engine knows which requests are participating in this step and how much KV memory to allocate.

---

#### 2. **Forward pass**

Once the batch for this step is assembled:

* All selected requests’ tokens are **flattened into one “super sequence”**.
* A **single model forward pass** runs on the GPU.
* Each sequence’s KV-cache and attention mask ensure isolation (so different user prompts can coexist).
* The model outputs **logits** (next-token distributions) for all active requests in parallel.

This is where continuous batching shines — instead of restarting per request, the model efficiently handles mixed prefill/decode workloads together.

---

#### 3. **Postprocess**

After the model outputs logits:

* The engine **samples** the next token for each request.
* It **appends** that token to the request’s output buffer.
* If a **stop condition** is met (e.g. EOS, max length), it:

  * Cleans up memory (returns KV-cache blocks to the pool)
  * Marks the request as finished
* Otherwise, it stays in the running queue for the next step.

---

### 🧠 So conceptually:

Each **engine step** = “one global forward pass” over all active sequences.
At each step:

```text
Schedule → Forward Pass → Postprocess
```

Then repeat until all requests are done.

---

### ⚙️ Example

Imagine the system is serving three requests:

| Request | Status  | Tokens to process |
| ------- | ------- | ----------------- |
| A       | running | 1 (decode)        |
| B       | running | 1 (decode)        |
| C       | waiting | 32 (prefill)      |

**Step 1**

* Schedule: A + B (decode), C (prefill)
* Forward pass: runs all three in one GPU call
* Postprocess: A, B get 1 token; C finishes prefill

**Step 2**

* C moves to running (starts decoding)
* A, B, C all decode one token
* GPU stays saturated with continuous batching

---

So when you see lines like:

> “The engine repeatedly calls its `step()` function”

it literally means:

> *Run one complete scheduling–forward–postprocess cycle for all active and new requests.*

---

Would you like me to show a small pseudocode version of what happens in an `engine.step()` loop in vLLM (roughly matching the real scheduler)?

so in continuous batching, after each step, the contents of the batch used in the next forward pass step may change?

--------------------------------

3. What the forward pass sees

Imagine you have:

A (decode) → needs 1 new token
B (prefill) → 4 tokens of new input
C (decode) → needs 1 new token


The engine forms a flattened input like:

[ A_t , B_1 , B_2 , B_3 , B_4 , C_t ]


Each position is tagged with:

request_id

absolute position in that request

KV-cache block indices

Then paged attention uses that metadata to enforce proper boundaries:

A_t can attend only to A’s past (cached)

C_t can attend only to C’s past

B_1–B_4 can attend to each other within B (like normal prefill)

No cross-talk between sequences

. Inside the transformer layers

Each transformer layer runs the same math — but masking and memory layout differ.

For decode tokens:

Each token has:

Q freshly computed (from its hidden state)

K,V fetched from cache

Attention = Q × K_cached^T

Output = softmax(...) × V_cached

For prefill tokens:

Each token in the chunk has:

Q,K,V computed from scratch

Attention = full causal matrix across the chunk

Their K,V get added to the KV cache

So in one fused GPU kernel, you’ve got:

some threads computing Q,K,V for new prefill tokens

some computing only Q (and using cached K,V) for decode tokens !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

everything packed into contiguous tensors, indexed by attention metadata

That’s the magic of paged attention — it makes all these different sequence states coexist efficiently.

. Output after the step

After this forward pass:

Prefill requests finish their chunk (and maybe start decoding next step)

Decode requests each produce 1 next token

Scheduler updates queues (remove finished, keep running, add new)

Then the next engine step repeats with a new mixture.

🔍 Mental model

Think of the GPU as executing:

“One big forward pass that computes all Q/K/Vs it needs for everyone still alive or newly arrived, using metadata to separate them logically.”

Even though half the batch is mid-conversation and half is brand-new prompts, it’s still one batched matmul graph.

-------------------------------------------------------------------

“Returning KV-cache blocks to the pool” = handing their memory back to the allocator so another request can use that space.

In vLLM, KV-cache memory is divided into fixed-size blocks (e.g., each block stores K/V tensors for 16 tokens).

When a request starts, the KV-cache manager allocates enough blocks for that request.

A short prompt might need 2 blocks.

A long prompt might need 100 blocks.

Those blocks are marked as “in use” in a shared structure called the free_block_queue (the pool).

When the request finishes (hits EOS or max length):

The engine frees those blocks — meaning it puts their indices back into the free_block_queue.

They can now be reused for new incoming requests.

hink of the KV-cache as a parking lot:

Each request = a car that parks in several slots (blocks).

The pool = list of empty slots.

“Returning blocks to the pool” = the car leaves → those slots are free again.

so returning blocks to the pool means freeing up the blocks

so free block queue is queue of ids of blocks that are FREE (not loaded with kv cache yet)
---------------------------------------------------------

“How continuous is continuous batching?”

Continuous means the system doesn’t wait for a batch to finish before admitting new work.

In continuous batching:

The engine is in a steady loop of steps (while running_requests or waiting_requests:)

After every engine step, it:

Removes finished requests.

Keeps still-running ones.

Admits new incoming ones immediately.

Then runs the next forward pass.

| Concept                      | Meaning                                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| **Return KV-blocks to pool** | Free the memory used by finished requests so new ones can reuse it                             |
| **Continuous batching**      | The engine always keeps running; after each step, it swaps in/out requests dynamically         |
| **Own KV-cache context**     | Each request’s memory + attention mapping isolates its tokens so mixed batches don’t interfere |

-------------------------------------------------------------------------------------

🧩 Prefill

The model processes many tokens at once (the whole prompt or a chunk of it).

For each token, it computes Q, K, and V.

The K and V vectors for every layer and head are stored in KV cache memory.

After the prefill, the cache contains the full context for that sequence.

So if a prompt is 128 tokens and your block size is 16, the prefill will fill 8 KV blocks for that request — 8 × (K,V) tensors per layer.

🔁 Decode

The model only processes 1 new token (the most recent one).

It computes Q, K, V for that new token.

The new K,V vectors are appended to the end of the cache (usually in-place within the same paged memory blocks).

Attention uses:

Q from the new token

K,V from all previous tokens (cached) + the new one

So each decode step adds exactly one token’s worth of KV vectors.

🧠 Key point

At every step — no matter prefill or decode —

K and V for the new tokens are always stored.

That’s what allows the model to avoid recomputing them later.
When we say “KV cache,” it’s literally a big table of all past K,V tensors indexed by sequence and position.

⚙️ Internally in vLLM:

The scheduler calls allocate_slots() to reserve KV blocks.

The KV-cache manager hands out block indices (from free_block_queue).

During the forward pass:

The GPU computes K,V for new tokens.

The model runner writes them into those block slots in GPU memory.

On the next step, those K,V entries are already there and reused for attention lookups.

--------------------------------------------------

So in short:

Mode	What happens with new tokens	What happens with KV cache
Prefill	Many new tokens processed	Their K,V computed + stored for all of them
Decode	1 new token processed	Its K,V computed + appended to existing cache

2️⃣ Prefill (initial context build)

Prefill means: we’re processing a whole prompt from scratch — none of its K,V exist yet.

Imagine you input 5 tokens [A,B,C,D,E].
During prefill, the model computes Q,K,V for all of them in one big forward pass.

For each token t_i:

Q_i attends to all K_j,V_j where j ≤ i

because causal masking forbids looking ahead.

So:

A: attends to [A]
B: attends to [A,B]
C: attends to [A,B,C]
...
E: attends to [A,B,C,D,E]


But these K,V values are not yet in cache — they’re being computed right now in this same forward pass.
The attention kernel internally handles this: it builds the causal attention matrix within the batch, using those newly computed K,V in memory (not yet written to cache).

During prefill, there is no cached K,V yet — but the attention kernel still uses those K,V vectors as they are computed, within the same forward pass, to let each token attend to all earlier ones.

That’s why we say prefill is compute-heavy (lots of tokens → many Q,K,V matmuls at once) and decode is memory-heavy (few tokens → mostly cache lookups).

-----------------------------------------------------------------------------------
“you run some token(s) through it in parallel, and then either store some stuff (KV) or use the logits to decode”
Exactly:

Prefill = run multiple tokens in parallel → store K,V for all.

Decode = run one token → use logits to pick next token, store its K,V.

Both use the same parallel transformer — we’re just controlling how many tokens go through it at once.

In prefill: you pass a bunch of tokens to the transformer in decode: you pass one token, and a big kv cache to the transformer

4. So what’s really going on conceptually

You can think of the transformer as a state update function:

state_t = transformer_step(token_t, state_{t-1})


Where:

state_{t-1} is the cached K,V context.

transformer_step() is the same giant parallel function that could’ve been used on an entire sequence at once, but we’re now using it incrementally.

That’s the key insight:

The transformer itself isn’t inherently sequential — the autoregressive task is.

--------------------------------------------------

In prefill: you pass a bunch of tokens to the transformer
in decode: you pass one token, and a big kv cache to the transformer

Exactly — that’s the essence of it.

You can think of it like this:

---

## 🧩 Prefill

**Input:**

* A batch of *many tokens* (e.g. a whole prompt)
* No KV cache yet (empty)

**What happens:**

* The model computes **Q, K, V** for every token in the prompt — all at once.
* Each token attends to *all earlier tokens* within that same batch (thanks to causal masking).
* The outputs give you logits for the final token (or all tokens, if you want).

**What’s produced:**

* A **fresh KV cache**: one K and one V vector per token per layer.
* You can now reuse that cache for future decoding steps.

So prefill = *"build the world model of the prompt and save it for reuse."*

---

## ⚙️ Decode

**Input:**

* A *single new token*
* The **big KV cache** from all past tokens (that represents everything the model has “seen”)

**What happens:**

* The model computes Q,K,V for just this new token.
* It uses Q × [all cached Ks]^T to get attention scores, and fetches [all cached Vs] for the weighted sum.
* That gives you the updated hidden state for this new token.
* You sample a new output token from its logits.

**What’s produced:**

* A new single K,V pair for this token, which is **appended to the cache**.

So decode = *"extend the conversation one token at a time using the accumulated memory."*

---

### 🧠 Summary Table

| Stage       | Input Tokens  | KV Cache In     | KV Cache Out          | Parallelism    | Purpose             |
| ----------- | ------------- | --------------- | --------------------- | -------------- | ------------------- |
| **Prefill** | Many (prompt) | Empty           | Fully populated       | High           | Build context       |
| **Decode**  | 1 (new token) | Full (past K,V) | Appended with new K,V | Low (per step) | Generate next token |

---

That’s why:

* Prefill is **compute-heavy** (lots of matmuls at once),
* Decode is **memory-heavy** (loads tons of cached K,V from VRAM).

---

So yeah —

> Prefill = big token batch, no cache yet
> Decode = one token, big cache

That’s the whole story in one line.
-----------------------------------
hold on - is it actually a "new" token inputted, or do we calculate the "new" token by running the last token through the transformer and use the kv cahce of all previous tokens

Perfect question — this is *the* moment where the mental model usually “clicks.”
Let’s separate **what’s input** vs **what’s computed** very clearly.

---

## 🧩 Step by step in decoding

Let’s say you’ve already run a prefill on:

```
[The, cat, sat, on, the]
```

and cached all of their K,V.
Now you want the next word — likely “mat”.

---

### 1️⃣ The “input” to the transformer at this step

You feed in **only the embedding of the *last generated token*** — in this case `"the"` (the final token so far).

So yes — you actually **input that single, last token again**.
It goes through all layers just like before, but this time the transformer layers also get to **use all cached K,V** from previous tokens.

---

### 2️⃣ What the transformer does with it

Each attention layer does:

```
Q_new = token_embedding @ W_Q
K_new = token_embedding @ W_K
V_new = token_embedding @ W_V
```

Then attention is computed as:

```
Attn = softmax(Q_new × [K_cached, K_new]^T)
```

So the new token’s **Q** looks back across **all previous Ks** (cached) and its own `K_new`,
and weights their corresponding **Vs** to produce its next hidden state.

That hidden state is then projected up to **logits**,
and those logits produce the *next* token (the one you haven’t seen yet — e.g., `"mat"`).

---

### 3️⃣ After you pick the new token

You now:

* Append the new token’s **K,V** (from this step) to the cache.
* Use the sampled token (“mat”) as the **input** to the transformer on the next step.

So at every step:

```
Input token = the *last token you just generated*
Output logits = distribution for the *next token to generate*
```

---

### 🔁 Intuitive loop

| Step | Input token | Cache contains           | Output token |
| ---- | ----------- | ------------------------ | ------------ |
| 1    | “The”       | [The]                    | “cat”        |
| 2    | “cat”       | [The, cat]               | “sat”        |
| 3    | “sat”       | [The, cat, sat]          | “on”         |
| 4    | “on”        | [The, cat, sat, on]      | “the”        |
| 5    | “the”       | [The, cat, sat, on, the] | “mat”        |

---

### 🧠 So, to answer directly

> Is it actually a new token inputted, or do we calculate the “new” token by running the last token through the transformer and using the kv cache of all previous tokens?

✅ It’s the latter:

* You **input the last token’s embedding** (the token you just generated).
* You **run it through the transformer**, which uses **all cached K,V** from prior tokens.
* The transformer **outputs logits** for the **next token**.

That’s why decoding is sequential — each new token depends on the model’s output from the previous step.

---

Would you like me to show the same loop in a small diagram that shows what’s “input vs output vs cached” each decode step (so it’s crystal-clear what goes into and comes out of the transformer each time)?

🔁 Intuitive loop
Step	Input token	Cache contains	Output token
1	“The”	[The]	“cat”
2	“cat”	[The, cat]	“sat”
3	“sat”	[The, cat, sat]	“on”
4	“on”	[The, cat, sat, on]	“the”
5	“the”	[The, cat, sat, on, the]	“mat”
🧠 So, to answer directly

Is it actually a new token inputted, or do we calculate the “new” token by running the last token through the transformer and using the kv cache of all previous tokens?

✅ It’s the latter:

You input the last token’s embedding (the token you just generated).

You run it through the transformer, which uses all cached K,V from prior tokens.

The transformer outputs logits for the next token.

That’s why decoding is sequential — each new token depends on the model’s output from the previous step.