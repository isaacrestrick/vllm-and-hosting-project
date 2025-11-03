from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os

load_dotenv()

prompts = [
    "hi, my name is",
    "The president of America is"
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    print("Starting the program...")
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()