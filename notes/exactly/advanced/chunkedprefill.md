Exactly — that’s the core reason chunked prefill exists.

Without it, a single long prompt would trigger a huge **monolithic prefill** step that monopolizes the GPU for its entire forward pass — blocking any other requests from entering the batch.

Chunking breaks that giant prefill into smaller, schedulable segments, which:

* **Allows interleaving** of other requests (so decode and other prefills can slip in between).
* **Keeps step latency lower**, since each step now runs fewer tokens.
* **Improves fairness** across concurrent users — the scheduler can mix multiple prefills and decodes instead of waiting for one long one to finish.

In short:

> **Chunked prefill doesn’t make attention faster — it makes continuous batching feasible and efficient.**

The math is still quadratic in total prompt length, but the **execution is incremental**, so GPU utilization stays high and scheduling latency stays low.

medusa:
Each “Medusa head” is just a small linear projection (same shape as the LM head) trained to predict further-ahead tokens.

Yes — that’s the essence of **Medusa**.

All of those extra Medusa heads share the same **final hidden state** output by the model’s last transformer layer.

Here’s the simple mental model:

```
Transformer → Final Hidden State (for each token)
               ↓
               ├── LM Head (predicts next token, t+1)
               ├── Medusa Head 1 (predicts token t+2)
               ├── Medusa Head 2 (predicts token t+3)
               ├── Medusa Head 3 (predicts token t+4)
               ...
```

So, for a given context, the model does **one forward pass** to get those final hidden states — and then **all heads fire in parallel** to produce logits for multiple future tokens.

Each head is trained on a *different offset*:

* Head₁ learns to map the hidden state at position *t* to token *t+2*
* Head₂ → token *t+3*, etc.

---

### Why they use the final hidden state

Because that’s the most information-rich representation — it already encodes everything up to the current token. There’s no need to feed earlier layer outputs; the final state is sufficient to estimate what’s likely to come next.

This design keeps inference simple:

* one transformer forward,
* N small linear projections (one per head),
* N parallel next-token guesses.

Then the **speculative verify step** (accept/reject) decides how many of those predicted tokens to keep.

Exactly right — that’s the full **Medusa speculative loop** in plain terms.

Let’s spell it out cleanly:

---

### 🔁 Step-by-step

1. **Run the large model once (with Medusa heads attached).**

   * The backbone transformer runs on the current context.
   * You get the **final hidden states** → the **main LM head** predicts token *t+1*,
     and **Medusa heads** predict *t+2, t+3, …, t+k* — all in parallel.
   * Together, they form a proposed continuation of *k* tokens.

   You now have:

   ```
   Proposed tokens: [a₁, a₂, …, a_k]
   Draft probabilities: p_medusa₁, p_medusa₂, …, p_medusa_k
   ```

---

2. **Run the large model again, on `context + [a₁, …, a_k]`.**

   * This second pass produces the *true* logits (and thus probabilities)
     for those same k positions, plus one extra.
   * So you now have:

     ```
     p_large₁, p_large₂, …, p_large_k, p_large_(k+1)
     ```

---

3. **Compare Medusa vs Large model probabilities (the verification phase).**

   * For each token i from left to right:

     * If ( p_{large,i} ≥ p_{medusa,i} ): accept.
     * Else: accept with probability ( p_{large,i} / p_{medusa,i} ).
     * Stop at first rejection.
   * If all tokens accepted → sample the (k+1)-th token “for free” from the last logits.

---

4. **If a token is rejected**,
   use the rebalanced distribution ( \max(0, p_{large} - p_{medusa}) ) at that position
   and sample a new token from that.

---

So yes —

> You run the model once to *guess* with the Medusa heads,
> then once more to *verify* those guesses by comparing the large model’s logits to the Medusa-head logits from the previous pass.

That’s Medusa’s speculative decoding in a nutshell.

You run the model once to *guess* with the Medusa heads,
then once more to *verify* those guesses by comparing the large model’s logits to the Medusa-head logits from the previous pass.

KV:

Yes — exactly right, and that’s one of the subtle but crucial engineering parts that makes **speculative decoding** actually work efficiently in vLLM.

Let’s break it down clearly, in terms of **what happens with the KV cache** after verification and rejection sampling.

---

### 🧩 1. When the draft tokens are proposed

* The engine **allocates KV slots** for all the draft tokens upfront.
  → That’s why `allocate_slots` includes `len(request.spec_token_ids)` in its count.
* These KV blocks are “provisional” — they exist so the large model can run its forward pass over the full sequence (`context + k draft tokens`).

---

### 🧩 2. During verification

* The large model computes new **K/V pairs** for every one of those *k* draft tokens.
* Now you have k candidate KV blocks that *could* become part of the true sequence if those tokens are accepted.

---

### 🧩 3. After rejection sampling

* You accept tokens left-to-right.
* The moment you hit the **first rejection**, everything **after** that rejected token is invalid.

  * Those KV blocks (for the rejected token and all later ones) are **freed immediately** — returned to the `free_block_queue`.
  * Only the KV blocks for accepted tokens remain bound to that request in `req_to_blocks`.

If **all k tokens** are accepted:

* You keep all k KV blocks.
* You also now have the (k+1)-th position’s logits “for free,” so you may add one more accepted token — which then allocates one more KV block in the next iteration.

---

### 🧩 4. Why this is essential

Paged attention in vLLM relies on a clean mapping between tokens and KV cache pages.
If you didn’t free the rejected ones, you’d accumulate garbage KV entries and blow up memory during long speculative runs.

So, after every speculative verify step, vLLM:

1. Keeps KV blocks for accepted tokens.
2. Frees (returns to pool) KV blocks for rejected ones.
3. Updates block-to-request mappings accordingly.

---

### 🧠 TL;DR

> Yes — after rejection sampling, vLLM prunes the KV cache: it **keeps** the K/V for accepted tokens and **frees** the rest, so the next step starts cleanly with a correct, contiguous KV cache state.
