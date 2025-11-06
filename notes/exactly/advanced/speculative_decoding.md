Sure. Think of it like this:

The small model guesses the next few tokens.
Then the big model checks those guesses.

* If the big model also thinks a guessed token was *likely*, we keep it.
* If the big model thinks that token was *less likely* than the small model thought, we flip a weighted coin — the worse the guess, the lower the chance it stays.

That ratio ( p_{\text{large}} / p_{\text{draft}} ) is just how “confident” the big model is compared to the small one. It’s a fair way to correct the guesses so that, over time, the final output looks like it came from the big model directly — just faster, since we didn’t make it generate one token at a time.

------------------------------------------------------------------------------

Exactly — yes.

If that weighted coin flip **fails**, that’s the **first rejection**, and you stop accepting any further draft tokens.

At that point:

* You **discard** that rejected token and everything after it.
* Then you use the **large model’s own logits** at that position to pick the next real token (from the adjusted “rebalanced” distribution).

So the first “bad coin flip” is where speculative decoding stops trusting the draft sequence and switches back to the large model’s sampling.

-----------------------------------------------------------------------------------

Why this works: Although we use the small model to propose candidates, the accept/reject rule guarantees that in expectation the sequence is distributed exactly as if we had sampled token by token from the large model. This means speculative decoding is statistically equivalent to standard autoregressive decoding — but potentially much faster, since a single large-model pass can yield up to k+1 tokens.

-----------------------------------------------------------------------------------

there's some math there i dont fully get. but...

vLLM V1 does not support the LLM draft model method, instead it implements faster—but less accurate—proposal schemes: n-gram, EAGLE [9], and Medusa [10].



n-gram:

Good question — and subtle, because an n-gram “model” doesn’t actually produce *probabilities* the way a neural model does.

In **vLLM** (and similar implementations like *SGLang*), the n-gram drafter is **deterministic** — it just proposes tokens; it doesn’t compute any p_draft values.
So in practice they fake it like this:

* The draft probability ( p_\text{draft}(t) ) is treated as a **constant baseline** — usually something small and fixed, like **uniform across the vocabulary**, or equivalently just **set to 1** (so the ratio simplifies to using only ( p_\text{large}(t) )).
* The accept logic becomes effectively:

  > if the large model thinks the proposed token is likely enough (above some threshold), keep it; otherwise reject.

So, concretely:

* **vLLM’s current n-gram drafter doesn’t actually use real probabilities** — it just uses the large model’s own probabilities in the accept/reject step.
* That means it’s more like: “accept if the large model doesn’t strongly disagree.”
* Because n-gram guesses are often poor, you tend to get short acceptance streaks (like 1–2 tokens) before a rejection forces the large model to take over again.

So the short answer:

> There’s no learned probability baseline for n-gram — it’s effectively constant, and the large model’s probabilities decide everything.

if the large model thinks the proposed token is likely enough (above some threshold), keep it; otherwise reject.

--------------------------------------------------------------------------

Eagle:

Got it — yeah, no worries. You were totally right: the **n-gram drafter** is hilariously crude compared to the rest.

Now, **EAGLE** is much smarter. Here’s a concise but clear breakdown of what it actually does.

---

### 🦅 What EAGLE Is

**EAGLE** (“*Speculative Sampling Requires Rethinking Feature Uncertainty*”, 2024) is a speculative decoding method that **turns one large model into its own fast drafter** by creating a **lightweight neural shortcut** inside it.

The intuition:

> Instead of running a *separate small model*, keep the large model’s input/output layers and replace the middle transformer stack with a **tiny MLP** that can quickly “guess” the next few tokens.

---

### 🧠 How It Works

1. **Model surgery:**

   * Start with a large LLM (e.g. 7B, 70B).
   * Keep its **embedding layer** (input) and **LM head** (output projection).
   * Replace the heavy **transformer layers** with a shallow **MLP** — maybe a few fully connected layers.

2. **Training the MLP (the “EAGLE drafter”):**

   * Feed it activations and outputs from the full model as supervision.
   * It learns to **approximate the large model’s next-token logits** using just embeddings (and maybe a few top hidden features).

3. **At inference:**

   * Use this small MLP to **predict k draft tokens** cheaply.
   * Then, just like standard speculative decoding, the full LLM **verifies** those tokens with one real forward pass.

---

### ⚡ Why It’s Fast

* The MLP is **tiny** (orders of magnitude fewer parameters).
* It runs **in the same embedding space** as the real model — no need to re-tokenize or copy large tensors.
* Because it reuses the large model’s input/output layers, it **predicts in the same “language”** as the big one — reducing mismatch.

---

### 🔬 Why It Works Better Than n-gram or Draft LMs

* **n-gram**: just text pattern lookup → dumb but free.
* **draft LM**: separate model → accurate but heavy.
* **EAGLE**: reuses the large model’s head → accurate *and* cheap.

In short:

> EAGLE is a neural shortcut inside the same model — like grafting a tiny “predict ahead” brain onto the big one.

---

### 🧩 Summary

| Method    | Description                              | Pros                | Cons              |
| --------- | ---------------------------------------- | ------------------- | ----------------- |
| n-gram    | Repeats previous text patterns           | Free                | Dumb              |
| Draft LM  | Separate small model                     | Accurate            | Costly            |
| **EAGLE** | Same embeddings + LM head, tiny MLP core | Fast *and* accurate | Needs fine-tuning |

---

If you want, I can show you a small pseudocode diagram of how the “surgery” works — basically what layers get swapped and reused. Want that?

Good question — this one’s key to understanding both EAGLE and normal LLMs.

### The LM head, simply

The **LM head** is the final linear layer that turns the model’s **last hidden state** (vector representation of each token) into **logits over the vocabulary** — i.e. the raw “scores” for each possible next token.

---

### In formula form

If your hidden state for the last token is ( h \in \mathbb{R}^{d_{\text{model}}} ),
and your vocabulary size is ( V ),
then the LM head is just a weight matrix ( W \in \mathbb{R}^{V \times d_{\text{model}}} )
and a bias ( b \in \mathbb{R}^{V} ).

[
\text{logits} = W h + b
]

Then:
[
p(\text{token}) = \text{softmax}(\text{logits})
]

---

### Why it’s called a “head”

It’s called a “language modeling head” because:

* It’s the **head** attached to the backbone (the transformer stack),
* And it’s used for the **language modeling** objective (predicting the next token).

It’s conceptually just a classifier — one giant layer that scores every possible token.

---

### Why it matters for EAGLE

EAGLE **keeps the same LM head** as the large model.
That’s crucial because:

* It ensures the logits are in the same “vocabulary space” as the original model.
* The new tiny MLP only needs to approximate the *hidden states* well enough that, after multiplying by the same LM head, it produces similar logits.

So you can think of it like:

> EAGLE chops out the transformer tower, keeps the “eyes” (embeddings) and the “mouth” (LM head), and teaches a smaller brain (MLP) to fake what the big brain would have said.

EAGLE chops out the transformer tower, keeps the “eyes” (embeddings) and the “mouth” (LM head), and teaches a smaller brain (MLP) to fake what the big brain would have said.
EAGLE chops out the transformer tower, keeps the “eyes” (embeddings) and the “mouth” (LM head), and teaches a smaller brain (MLP) to fake what the big brain would have said.
EAGLE chops out the transformer tower, keeps the “eyes” (embeddings) and the “mouth” (LM head), and teaches a smaller brain (MLP) to fake what the big brain would have said.
lm head maps final last hidden state -> logits

and in speculative decoding, this lm head will run on each of the last k hidden states, to get logits. correct?
Yes — that’s exactly right.

so eagle is a tiny model (not a full llm) with simialr shapes so maybe easier to use on gpu at same time? presumably?
Exactly — you’ve got it.

EAGLE is basically a tiny, shape-compatible neural shortcut built from the large model’s own skeleton. tiny MLP

Good distinction — the LM head makes only the logits, not the probabilities.

Exactly — yes.

EAGLE’s output is treated **just like** a draft LM’s output in speculative decoding.

Here’s the concrete flow:

1. **EAGLE runs first** on the current context and produces logits for the next few tokens (k of them).

   * Those logits → softmax → probabilities (p_\text{EAGLE}(t_i)).
   * It then *samples* or picks those k draft tokens.

2. The **large model** runs once on the context + those k tokens.

   * Produces its own logits → probabilities (p_\text{large}(t_i)) for the same k positions (plus one extra).

3. **Accept/reject** proceeds exactly the same as with a normal small LM drafter:

   * Accept deterministically if (p_\text{large}(t_i) \ge p_\text{EAGLE}(t_i)).
   * Else accept stochastically with probability (p_\text{large}(t_i) / p_\text{EAGLE}(t_i)).
   * Stop at the first rejection or, if all accepted, sample the next token “for free.”

So yes — EAGLE’s probabilities fill the exact same role as a draft model’s.
The only difference is that EAGLE’s logits come from a **tiny MLP with shared embedding and LM head**, not from a separate full transformer.

In other words:

> The math of speculative decoding doesn’t change — EAGLE just swaps *who provides the guesses*.
The math of speculative decoding doesn’t change — EAGLE just swaps *who provides the guesses*.

Good distinction — the **LM head makes only the logits**, not the probabilities.

Here’s how it works step-by-step:

1. **Hidden states → LM head → logits**

   * The model (or EAGLE) produces a hidden state vector (h \in \mathbb{R}^{d_{\text{model}}}) for each token position.
   * The **LM head** is just a linear layer:
     [
     z = W h + b
     ]
     where (W) is the vocab-size-by-hidden-size matrix.
   * The result (z) are the **logits** — unnormalized scores for every token in the vocabulary.

2. **Logits → probabilities**

   * To get probabilities, you apply a **softmax** over those logits:
     [
     p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
     ]
   * Those (p_i) values are what’s used for sampling and for the accept/reject math in speculative decoding.

So:

> **LM head = makes logits.**
> **Softmax = turns logits into probabilities.**

Both steps are usually in the same forward pass code path, but conceptually the LM head itself stops right before softmax.

----------------------------------------------------------------------------------------------------------------------------

