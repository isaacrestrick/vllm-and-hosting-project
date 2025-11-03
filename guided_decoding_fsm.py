import os
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def main():
    load_dotenv()
    model_name = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompts = [
        "Sentiment: This sucks! This is horrible! This is terrible! Output: ",
        "Sentiment: The weather is beautiful and sunny today! Output: ",
    ]
    guided_decoding_params = GuidedDecodingParams(choice=["Negative", "Neutral", "Positive"])
    sampling_params = SamplingParams(structured_outputs=guided_decoding_params)

    # Use float32 dtype to fix xgrammar compatibility issue
    llm = LLM(model=model_name, dtype="float32")
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)

if __name__ == "__main__":
    main()