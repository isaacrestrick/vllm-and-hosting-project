import os
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

load_dotenv()

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

speculative_config = {
    "method": "ngram",
    "prompt_lookup_max": 5,
    "prompt_lookup_min": 3,
    "num_speculative_tokens": 3,
}

'''
    This does not work on my macbook because speculative decoding is gpu only. lol!
'''
def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", speculative_config=speculative_config)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()