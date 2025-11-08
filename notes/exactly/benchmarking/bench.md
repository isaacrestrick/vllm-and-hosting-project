How to benchmark in vLLM
vLLM provides a vllm bench {serve,latency,throughput} CLI that wraps vllm / benchmarks / {server,latency,throughput}.py.

Here is what the scripts do:

latency — uses a short input (default 32 tokens) and samples 128 output tokens with a small batch (default 8). It runs several iterations and reports e2e latency for the batch.
throughput — submits a fixed set of prompts (default: 1000 ShareGPT samples) all at once (aka as QPS=Inf mode), and reports input/output/total tokens and requests per second across the run.
serve — Launches a vLLM server and simulates a real-world workload by sampling request inter-arrival times from a Poisson (or more generally, Gamma) distribution. It sends requests over a time window, measures all the metrics we’ve discussed, and can optionally enforce a server-side max concurrency (via a semaphore, e.g. limiting the server to 64 concurrent requests).
Here is an example of how you can run the latency script:

vllm bench latency
  --model <model-name>
  --input-tokens 32
  --output-tokens 128
  --batch-size 8
Note

Benchmark configs used in CI live under .buildkite/nightly-benchmarks/tests.

There is also an auto-tune script that drives the serve benchmark to find argument settings that meet target SLOs (e.g., “maximize throughput while keeping p99 e2e < 500 ms”), returning a suggested config.