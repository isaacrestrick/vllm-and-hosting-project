import os
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def main():
    load_dotenv()
    model_name = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Use JSON schema for structured output (generates more meaningful tokens)
    json_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"}
        },
        "required": ["summary", "sentiment", "confidence"]
    }
    
    prompts = [
        "Analyze this text and provide a JSON response: 'This is absolutely terrible!'",
        "Analyze this text and provide a JSON response: 'The weather is beautiful and sunny today!'",
    ]
    
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        structured_outputs=guided_decoding_params,
        temperature=0.8,
        max_tokens=50
    )

    # Use float32 dtype to fix xgrammar compatibility issue
    llm = LLM(model=model_name, dtype="float32")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"JSON Output: {output.outputs[0].text}")

if __name__ == "__main__":
    main()