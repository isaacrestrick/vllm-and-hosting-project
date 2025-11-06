In **vLLM’s guided decoding system**, the **grammar bitmask** is the data structure that represents — for each token — whether it’s *allowed or disallowed* according to a **grammar-based finite state machine (FSM)**.

It’s the **core mechanism** that enforces grammar constraints at the token level during decoding.

---

### 🔍 What it is

The grammar bitmask (`_grammar_bitmask`) is a tensor (on GPU) that stores **binary flags**:

* `1` → token is **allowed** at this step
* `0` → token is **disallowed**

Before the model samples the next token, vLLM uses this bitmask to modify logits:

* Allowed tokens keep their original logit values.
* Disallowed tokens get their logits set to `–∞`, so the softmax gives them probability 0.

This is how vLLM ensures the model can only generate text that conforms to the provided grammar.

---

### 🧠 How it works step-by-step

1. **Compile grammar → FSM**

   * When you specify a grammar (e.g. JSON, regex, choice list), vLLM compiles it into a finite state machine via a backend like **xgrammar**.
   * Each FSM state encodes the set of tokens that are valid next moves.

2. **Initialize `_grammar_bitmask`**

   * When the request starts, the FSM begins at its start state.
   * `_grammar_bitmask` represents which tokens are valid at that state.

3. **Forward pass**

   * The model outputs logits for every token in the vocabulary.
   * The bitmask is used to **mask out invalid tokens** by setting their logits to `–∞`.

4. **Sample + advance FSM**

   * The model samples one of the remaining allowed tokens.
   * The FSM advances to the next state (e.g., if “P” was sampled, the FSM moves along the “Positive” branch).
   * `_grammar_bitmask` is updated to reflect the allowed tokens in this new state.

This cycle repeats every generation step.

---

### 🧩 Implementation detail

Because vocabularies are large (e.g., 32k tokens), vLLM uses a **compact encoding**:

* Each 32-bit integer in `_grammar_bitmask` encodes 32 tokens (one bit per token).
* Example:

  ```
  10100010... → [1,0,1,0,0,0,1,0,...]
  ```
* For vocab size `V`, the bitmask has `ceil(V / 32)` integers.
* During decoding, this compressed mask is expanded back to length `V` and used to filter logits.

This bitmask lives on the GPU for speed and is updated in-place each step.

---

### 🧮 Example

Suppose your vocabulary = `[P, o, s, N, e, g, a, t, i, v, e, <eos>]`

At the start:

```
FSM allows: {P, N}
Bitmask:    [1,0,0,1,0,0,0,0,0,0,0,0]
```

After sampling “P”, FSM moves to the “Positive” path:

```
FSM allows: {o}
Bitmask:    [0,1,0,0,0,0,0,0,0,0,0,0]
```

Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.

---

### 🧩 Summary

| Concept              | Role                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| **Grammar**          | High-level rules (regex, JSON schema, CFG, etc.)                          |
| **FSM**              | Converts grammar into a state machine tracking allowed tokens             |
| **_grammar_bitmask** | Compact binary tensor showing which tokens are valid at the current state |
| **Usage**            | Multiplied into logits to enforce constraints during decoding             |

---

In short:

> **The grammar bitmask is the GPU-side binary map that tells vLLM’s guided decoder exactly which tokens the model is allowed to generate at each step, according to the grammar-derived finite state machine.**

Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.
Each step, `_grammar_bitmask` updates accordingly and masks logits before sampling.

bitmask on the logits