Exactly — that’s the core reason chunked prefill exists.

Without it, a single long prompt would trigger a huge **monolithic prefill** step that monopolizes the GPU for its entire forward pass — blocking any other requests from entering the batch.

Chunking breaks that giant prefill into smaller, schedulable segments, which:

* **Allows interleaving** of other requests (so decode and other prefills can slip in between).
* **Keeps step latency lower**, since each step now runs fewer tokens.
* **Improves fairness** across concurrent users — the scheduler can mix multiple prefills and decodes instead of waiting for one long one to finish.

In short:

> **Chunked prefill doesn’t make attention faster — it makes continuous batching feasible and efficient.**

The math is still quadratic in total prompt length, but the **execution is incremental**, so GPU utilization stays high and scheduling latency stays low.
