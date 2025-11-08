# Next Steps: Learning About Benchmarking in vLLM

## Current State

I've started exploring benchmarking in vLLM through:
- Basic benchmarking notebook (`benchmarking.ipynb`) measuring TTFT, ITL, TPOT, Latency/E2E, Throughput, and Goodput
- Understanding vLLM's built-in benchmarking CLI tools (`vllm bench`)
- Learning about batch saturation (`B_{sat}`) and the memory-bound vs compute-bound regimes
- Comparing different optimization approaches (guided decoding, prefix caching, disaggregated prefill/decode)

## Learning Goals

### 1. Master vLLM's Built-in Benchmarking Tools

**Priority: High**

- [ ] Deep dive into `vllm bench latency` - understand how it measures latency with different batch sizes
- [ ] Explore `vllm bench throughput` - learn how it handles QPS=Inf mode and measures tokens/second
- [ ] Study `vllm bench serve` - understand Poisson/Gamma request arrival simulation and concurrency limits
- [ ] Experiment with the auto-tune script for finding optimal configurations that meet SLOs
- [ ] Review CI benchmark configs in `.buildkite/nightly-benchmarks/tests` to understand production benchmarking practices

**Resources:**
- vLLM documentation on benchmarking CLI
- Existing notes in `notes/exactly/benchmarking/bench.md`

### 2. Understand Batch Saturation and Performance Characteristics

**Priority: High**

- [ ] Experiment with different batch sizes to find `B_{sat}` for various models
- [ ] Measure performance below `B_{sat}` (memory-bound regime) vs above `B_{sat}` (compute-bound regime)
- [ ] Understand how step time scales with batch size in each regime
- [ ] Learn about GPU utilization metrics and how they relate to batch saturation
- [ ] Study memory bandwidth vs compute utilization trade-offs

**Resources:**
- Notes in `notes/exactly/benchmarking/benchmarking.md` about `B_{sat}`

### 3. Advanced Benchmarking Metrics

**Priority: Medium**

- [ ] Implement accurate TTFT measurement using streaming API (current implementation uses approximations)
- [ ] Measure per-token latency using async API with token-by-token callbacks
- [ ] Understand p50, p95, p99 latency percentiles and their importance
- [ ] Learn about tail latency and how to measure it effectively
- [ ] Study request queuing time vs processing time breakdowns
- [ ] Explore goodput vs throughput differences in real-world scenarios

**Current Limitations:**
- Batch generation API doesn't expose per-token timing
- Need to use streaming/async APIs for accurate token-level metrics

### 4. Production-Grade Benchmarking Practices

**Priority: Medium**

- [ ] Learn about load testing with realistic request patterns
- [ ] Understand how to simulate production workloads (request arrival distributions, concurrency patterns)
- [ ] Study benchmarking at scale (multi-GPU, distributed setups)
- [ ] Learn about benchmarking different model sizes and architectures
- [ ] Understand how to benchmark with different optimization features enabled/disabled
- [ ] Explore continuous benchmarking and regression detection

### 5. Optimization-Specific Benchmarking

**Priority: Low**

- [ ] Benchmark speculative decoding (requires GPU setup)
- [ ] Measure prefix caching benefits with different cache hit rates
- [ ] Benchmark chunked prefill performance
- [ ] Compare continuous batching vs static batching performance
- [ ] Measure disaggregated prefill/decode benefits in multi-GPU setups
- [ ] Benchmark guided decoding overhead vs benefits

### 6. Tools and Infrastructure

**Priority: Low**

- [ ] Set up proper GPU environment for accurate benchmarking (currently limited to CPU/Mac)
- [ ] Learn about profiling tools (PyTorch Profiler, NVIDIA Nsight, etc.)
- [ ] Understand how to benchmark with different hardware configurations
- [ ] Explore distributed benchmarking setups
- [ ] Learn about benchmarking in cloud environments

## Practical Next Steps

1. **Immediate (This Week):**
   - Run `vllm bench latency` with different batch sizes to understand batch saturation
   - Experiment with `vllm bench throughput` to see how it handles large request volumes
   - Try `vllm bench serve` with different concurrency limits

2. **Short Term (Next 2 Weeks):**
   - Refactor benchmarking notebook to use streaming API for accurate TTFT/ITL measurements
   - Create batch size sweep experiments to find `B_{sat}` empirically
   - Study vLLM's CI benchmark configs to understand production practices

3. **Medium Term (Next Month):**
   - Set up GPU environment for more accurate benchmarking
   - Implement comprehensive benchmarking suite covering all major optimizations
   - Create automated benchmarking pipeline for regression testing

## Questions to Answer

- How does batch size affect latency vs throughput trade-offs?
- What's the optimal batch size for different model sizes?
- How do different optimizations (prefix caching, guided decoding, etc.) affect various metrics?
- What are realistic SLO targets for production LLM serving?
- How do you benchmark with realistic production workloads?
- What's the relationship between GPU utilization and batch saturation?

## Resources

- vLLM benchmarking documentation
- Existing notes: `notes/exactly/benchmarking/`
- Benchmarking notebook: `benchmarking.ipynb`
- vLLM source code: `.buildkite/nightly-benchmarks/tests` (CI configs)

