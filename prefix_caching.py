from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os

load_dotenv()

long_prefix = "<a piece of text that is encoded into more than block_size tokens>"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(long_prefix + prompts[0], sampling_params)
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)

    outputs = llm.generate(long_prefix + prompts[1], sampling_params)
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()