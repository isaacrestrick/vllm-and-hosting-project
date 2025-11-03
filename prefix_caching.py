from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Long prefix that will be cached (shared context across requests)
long_prefix = """
You are a helpful AI assistant. Here is some background information:
The company was founded in 2020 and has grown significantly since then.
We operate in multiple countries and serve millions of customers worldwide.
Our mission is to provide excellent service and innovative solutions.

Now, please respond to this prompt: """

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "Artificial intelligence is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

def main():
    # Initialize LLM with prefix caching enabled
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", enable_prefix_caching=True)
    
    # Create prompts with shared prefix
    prefixed_prompts = [long_prefix + p for p in prompts]
    
    print("Running with prefix caching enabled...")
    print(f"Shared prefix length: ~{len(long_prefix.split())} words\n")
    
    # First batch - will cache the prefix
    print("=" * 60)
    print("FIRST BATCH (building cache)")
    print("=" * 60)
    start = time.time()
    outputs = llm.generate(prefixed_prompts, sampling_params)
    first_time = time.time() - start
    print(f"Time: {first_time:.3f}s\n")
    
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: ...{prompts[i]}")
        print(f"Response: {output.outputs[0].text[:100]}...")
        print()
    
    # Second batch - should be faster due to cache
    print("=" * 60)
    print("SECOND BATCH (using cache)")
    print("=" * 60)
    start = time.time()
    outputs = llm.generate(prefixed_prompts, sampling_params)
    second_time = time.time() - start
    print(f"Time: {second_time:.3f}s\n")
    
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: ...{prompts[i]}")
        print(f"Response: {output.outputs[0].text[:100]}...")
        print()
    
    # Show improvement
    improvement = ((first_time - second_time) / first_time) * 100
    print("=" * 60)
    print(f"Speedup from caching: {improvement:.1f}% faster on second batch")
    print("=" * 60)

if __name__ == "__main__":
    main()