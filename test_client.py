"""
Example client for testing the vLLM FastAPI server
"""

import requests
import json

# Server configuration
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_basic_generation():
    """Test basic text generation"""
    print("Testing basic generation...")
    
    payload = {
        "prompt": "Hello, my name is",
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 50
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    result = response.json()
    
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Metrics: {result['metrics']}\n")


def test_json_generation():
    """Test guided decoding with JSON schema"""
    print("Testing JSON structured output...")
    
    json_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"}
        },
        "required": ["summary", "sentiment", "confidence"]
    }
    
    payload = {
        "prompt": "Analyze this text: The weather today is absolutely wonderful!",
        "temperature": 0.8,
        "max_tokens": 100,
        "use_json_schema": True,
        "json_schema": json_schema
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    result = response.json()
    
    print(f"Prompt: {result['prompt']}")
    print(f"Generated JSON: {result['generated_text']}")
    print(f"Metrics: {result['metrics']}\n")


def test_prefix_caching():
    """Test prefix caching with shared context"""
    print("Testing prefix caching...")
    
    shared_prefix = """
    You are a helpful AI assistant. Here is some background information:
    The company was founded in 2020 and has grown significantly since then.
    We operate in multiple countries and serve millions of customers worldwide.
    Our mission is to provide excellent service and innovative solutions.
    
    Now, please respond to this prompt: """
    
    payload = {
        "prompt": "What is our company's mission?",
        "prefix": shared_prefix,
        "temperature": 0.8,
        "max_tokens": 50
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    result = response.json()
    
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Has prefix: {result['settings']['has_prefix']}")
    print(f"Metrics: {result['metrics']}\n")


def test_batch_generation():
    """Test batch generation"""
    print("Testing batch generation...")
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Artificial intelligence is",
    ]
    
    response = requests.post(
        f"{BASE_URL}/batch_generate",
        params={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 30
        },
        json=prompts
    )
    
    result = response.json()
    
    print(f"Batch processed {len(result['results'])} prompts:")
    for i, item in enumerate(result['results'], 1):
        print(f"\n{i}. Prompt: {item['prompt']}")
        print(f"   Generated: {item['generated_text'][:100]}...")
    
    print(f"\nMetrics: {result['metrics']}\n")


def test_model_info():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Model info: {json.dumps(response.json(), indent=2)}\n")


def test_streaming_generation():
    """Test streaming generation with metrics"""
    print("Testing streaming generation with metrics...")
    
    payload = {
        "prompt": "The future of artificial intelligence is",
        "temperature": 0.8,
        "max_tokens": 30
    }
    
    response = requests.post(f"{BASE_URL}/generate_stream", json=payload, stream=True)
    
    print("Streaming response:")
    generated_text = ""
    metrics = None
    
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data = json.loads(line_str[6:])
                if data['type'] == 'token':
                    token = data.get('token', '')
                    generated_text += token
                    if 'ttft_ms' in data:
                        print(f"  First token received (TTFT: {data['ttft_ms']} ms)")
                    elif 'itl_ms' in data:
                        print(f"  Token {data.get('token_index', '?')}: '{token}' (ITL: {data['itl_ms']} ms)")
                elif data['type'] == 'done':
                    metrics = data.get('metrics', {})
                    print(f"\n  Generation complete!")
                    print(f"  Generated text: {generated_text}")
                    print(f"  Metrics: TTFT={metrics.get('ttft_ms', 0)}ms, TPOT={metrics.get('tpot_ms', 0)}ms, E2E={metrics.get('e2e_ms', 0)}ms")
                    print(f"  Output tokens: {metrics.get('output_tokens', 0)}")
                elif data['type'] == 'error':
                    print(f"  Error: {data.get('error', 'Unknown error')}")
    
    print()


def test_benchmark():
    """Test benchmark endpoint"""
    print("Testing benchmark endpoint...")
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]
    
    payload = {
        "prompts": prompts,
        "temperature": 0.8,
        "max_tokens": 50,
        "max_ttft_ms": 500,  # SLO: max 500ms TTFT
        "max_tpot_ms": 100,  # SLO: max 100ms TPOT
        "max_e2e_ms": 2000,  # SLO: max 2000ms E2E
    }
    
    response = requests.post(f"{BASE_URL}/benchmark", json=payload)
    result = response.json()
    
    print(f"\nBenchmark Results:")
    print(f"  Duration: {result['benchmark_duration_seconds']}s")
    print(f"  Requests: {result['num_requests']}")
    print(f"  Total tokens: {result['total_tokens']} (input: {result['total_input_tokens']}, output: {result['total_output_tokens']})")
    
    print(f"\nThroughput:")
    print(f"  Total tokens/sec: {result['throughput']['total_tokens_per_sec']}")
    print(f"  Output tokens/sec: {result['throughput']['output_tokens_per_sec']}")
    print(f"  Requests/sec: {result['throughput']['requests_per_sec']}")
    
    print(f"\nGoodput (SLO-Compliant):")
    print(f"  Compliant requests: {result['goodput']['slo_compliant_requests']}/{result['num_requests']}")
    print(f"  Compliance rate: {result['goodput']['slo_compliance_rate']}%")
    print(f"  Goodput: {result['goodput']['goodput_tokens_per_sec']} tokens/sec")
    
    print(f"\nTTFT Statistics:")
    print(f"  Mean: {result['ttft']['mean_ms']}ms, Median: {result['ttft']['median_ms']}ms")
    print(f"  P95: {result['ttft']['p95_ms']}ms, P99: {result['ttft']['p99_ms']}ms")
    
    print(f"\nTPOT Statistics:")
    print(f"  Mean: {result['tpot']['mean_ms']}ms, Median: {result['tpot']['median_ms']}ms")
    print(f"  P95: {result['tpot']['p95_ms']}ms, P99: {result['tpot']['p99_ms']}ms")
    
    print(f"\nITL Statistics:")
    print(f"  Mean: {result['itl']['mean_ms']}ms, Median: {result['itl']['median_ms']}ms")
    print(f"  P95: {result['itl']['p95_ms']}ms, P99: {result['itl']['p99_ms']}ms")
    
    print(f"\nE2E Statistics:")
    print(f"  Mean: {result['e2e']['mean_ms']}ms, Median: {result['e2e']['median_ms']}ms")
    print(f"  P95: {result['e2e']['p95_ms']}ms, P99: {result['e2e']['p99_ms']}ms")
    
    print()


def main():
    """Run all tests"""
    print("=" * 80)
    print("vLLM FastAPI Server Test Client")
    print("=" * 80 + "\n")
    
    try:
        test_health()
        test_model_info()
        test_basic_generation()
        
        # Optional: uncomment to test JSON generation (may require model support)
        # test_json_generation()
        
        # Optional: uncomment to test prefix caching (requires ENABLE_PREFIX_CACHING=true)
        # test_prefix_caching()
        
        test_batch_generation()
        
        # Test new streaming and benchmark endpoints
        test_streaming_generation()
        test_benchmark()
        
        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print(f"Make sure the server is running on {BASE_URL}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

