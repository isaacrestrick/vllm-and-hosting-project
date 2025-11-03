import os
import time
from multiprocessing import Event, Process
import multiprocessing as mp
from dotenv import load_dotenv

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

load_dotenv()

# Standard test prompts (consistent with other benchmarks)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "Artificial intelligence is",
]

# Standard parameters
STANDARD_TEMPERATURE = 0.8
STANDARD_TOP_P = 0.95
STANDARD_MAX_TOKENS = 50

def run_prefill(prefill_done, stop_event):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES_PREFILL", "0")
    model_name = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Prefill only generates 1 token to pass KV cache
    sampling_params = SamplingParams(temperature=0, top_p=STANDARD_TOP_P, max_tokens=1)

    ktc = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    )

    print("Starting prefill process...")
    llm = LLM(model=model_name, kv_transfer_config=ktc)
    llm.generate(prompts, sampling_params)

    prefill_done.set()  # notify decode instance that KV cache is ready
    print("Prefill completed, KV cache ready")

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    # Exit gracefully when stop_event is set
    while not stop_event.is_set():
        time.sleep(0.1)
    print("Prefill process exiting gracefully.")

def run_decode(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES_DECODE", "1")
    model_name = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Use standard parameters for decode phase
    sampling_params = SamplingParams(
        temperature=STANDARD_TEMPERATURE,
        top_p=STANDARD_TOP_P,
        max_tokens=STANDARD_MAX_TOKENS
    )

    ktc = KVTransferConfig(
        kv_connector="SharedStorageConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": "local_storage"},
    )

    print("Starting decode process, waiting for KV cache...")
    llm = LLM(model=model_name, kv_transfer_config=ktc)

    prefill_done.wait()  # block waiting for KV cache from prefill instance
    print("KV cache received, starting decode...")

    # Internally it'll first fetch KV cache before starting the decoding loop
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(output.prompt)
        print(output.outputs[0].text)
        print()

def main():
    prefill_done = Event()
    stop_event = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, stop_event))
    decode_process = Process(target=run_decode, args=(prefill_done,))

    print("Starting disaggregated prefill/decode processes...")
    prefill_process.start()
    decode_process.start()

    decode_process.join()
    stop_event.set()  # Signal prefill to exit gracefully
    prefill_process.join(timeout=5)  # Wait up to 5 seconds for graceful exit
    if prefill_process.is_alive():
        prefill_process.terminate()  # Force terminate if still running
        prefill_process.join()
    print("All processes completed.")

if __name__ == "__main__":
    main()