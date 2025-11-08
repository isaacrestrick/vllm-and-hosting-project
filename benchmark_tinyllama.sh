#!/bin/bash

# Benchmark script for TinyLlama model using vLLM benchmark CLI
# Based on vLLM benchmarking documentation

set -e  # Exit on error

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BENCHMARK_TYPE="${1:-all}"  # Default to running all benchmarks

echo "======================================"
echo "vLLM Benchmark Suite for TinyLlama"
echo "======================================"
echo "Model: $MODEL"
echo "Benchmark Type: $BENCHMARK_TYPE"
echo ""

# Function to run latency benchmark
run_latency() {
    echo "--------------------------------------"
    echo "Running LATENCY Benchmark"
    echo "--------------------------------------"
    echo "Configuration:"
    echo "  - Input tokens: 32"
    echo "  - Output tokens: 128"
    echo "  - Batch size: 8"
    echo ""
    
    vllm bench latency \
        --model "$MODEL" \
        --input-len 32 \
        --output-len 128 \
        --batch-size 8
}

# Function to run throughput benchmark
run_throughput() {
    echo "--------------------------------------"
    echo "Running THROUGHPUT Benchmark"
    echo "--------------------------------------"
    echo "Configuration:"
    echo "  - Prompts: 1000 ShareGPT samples (default)"
    echo "  - QPS: Infinity (all at once)"
    echo ""
    
    vllm bench throughput \
        --model "$MODEL" \
        --input-len 32 \
        --output-len 128
}

# Function to run serve benchmark
run_serve() {
    echo "--------------------------------------"
    echo "Running SERVE Benchmark"
    echo "--------------------------------------"
    echo "Configuration:"
    echo "  - Simulates real-world workload"
    echo "  - Poisson/Gamma inter-arrival times"
    echo ""
    
    vllm bench serve \
        --model "$MODEL"
}

# Main execution logic
case "$BENCHMARK_TYPE" in
    latency)
        run_latency
        ;;
    throughput)
        run_throughput
        ;;
    serve)
        run_serve
        ;;
    all)
        echo "Running all benchmark types..."
        echo ""
        run_latency
        echo ""
        run_throughput
        echo ""
        run_serve
        ;;
    *)
        echo "Usage: $0 [latency|throughput|serve|all]"
        echo ""
        echo "Benchmark types:"
        echo "  latency    - Measures end-to-end latency for small batches"
        echo "  throughput - Measures throughput with fixed prompt set"
        echo "  serve      - Simulates real-world workload with Poisson arrivals"
        echo "  all        - Run all benchmark types (default)"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "Benchmarking complete!"
echo "======================================"

