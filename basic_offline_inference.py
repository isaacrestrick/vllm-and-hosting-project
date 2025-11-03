from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os

load_dotenv()

# Standard test prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "Artificial intelligence is",
]

# Standard sampling parameters for consistent benchmarking
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50
)

def main():
    print("Basic Offline Inference - Baseline Performance")
    print(f"Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"Parameters: temp={sampling_params.temperature}, "
          f"top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}")
    print("=" * 60 + "\n")
    
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs, 1):
        print(f"Prompt {i}: {output.prompt}")
        print(f"Response: {output.outputs[0].text}")
        print()

if __name__ == "__main__":
    main()