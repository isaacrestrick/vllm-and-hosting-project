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

