"""
FastAPI server for vLLM inference with configurable settings
Based on benchmarking tweaks from the vllm-and-hosting-project
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
import time
import json
from dotenv import load_dotenv

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

load_dotenv()

# Global LLM instance (will be initialized on startup)
llm_instance = None
model_name = None


class InferenceRequest(BaseModel):
    """Request model for inference"""
    prompt: str = Field(..., description="The input prompt for generation")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p sampling parameter")
    max_tokens: int = Field(50, ge=1, le=2048, description="Maximum tokens to generate")
    
    # Advanced options
    top_k: int = Field(-1, description="Top-k sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    
    # Guided decoding options
    use_json_schema: bool = Field(False, description="Enable JSON structured output")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema for guided decoding")
    
    # Prefix for shared context (useful for testing prefix caching)
    prefix: Optional[str] = Field(None, description="Prefix to prepend to prompt (for prefix caching)")


class InferenceResponse(BaseModel):
    """Response model for inference"""
    generated_text: str
    prompt: str
    model: str
    settings: Dict[str, Any]
    metrics: Dict[str, float]


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking"""
    prompts: List[str] = Field(..., description="List of prompts to benchmark")
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(50, ge=1, le=2048)
    top_k: int = Field(-1)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    
    # SLO thresholds for goodput calculation
    max_ttft_ms: Optional[float] = Field(None, description="Max TTFT in ms for SLO")
    max_tpot_ms: Optional[float] = Field(None, description="Max TPOT in ms for SLO")
    max_e2e_ms: Optional[float] = Field(None, description="Max E2E latency in ms for SLO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LLM on startup, cleanup on shutdown"""
    global llm_instance, model_name
    
    # Startup
    model_name = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    enable_prefix_caching = os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true"
    
    print(f"Initializing vLLM with model: {model_name}")
    print(f"Prefix caching enabled: {enable_prefix_caching}")
    
    try:
        llm_instance = LLM(
            model=model_name,
            enable_prefix_caching=enable_prefix_caching,
            # dtype="float32" can be set via env var if needed
        )
        print("vLLM initialized successfully!")
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        raise
    
    yield
    
    # Shutdown
    print("Shutting down vLLM...")
    llm_instance = None


app = FastAPI(
    title="vLLM Inference Server",
    description="FastAPI server for vLLM inference with configurable settings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": model_name,
        "service": "vLLM Inference Server"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    return {"status": "healthy", "model": model_name}


@app.post("/generate", response_model=InferenceResponse)
async def generate(request: InferenceRequest):
    """
    Generate text using vLLM
    
    Supports various sampling parameters and optimizations:
    - Temperature and top_p/top_k sampling
    - Guided decoding with JSON schema
    - Prefix caching (if enabled on startup)
    - Custom penalties and constraints
    """
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    try:
        # Build the full prompt (with prefix if provided)
        full_prompt = request.prompt
        if request.prefix:
            full_prompt = request.prefix + request.prompt
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )
        
        # Add guided decoding if JSON schema is provided
        if request.use_json_schema and request.json_schema:
            guided_decoding_params = GuidedDecodingParams(json=request.json_schema)
            sampling_params.structured_outputs = guided_decoding_params
        
        # Measure generation time
        start_time = time.time()
        outputs = llm_instance.generate([full_prompt], sampling_params)
        end_time = time.time()
        
        # Extract results
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        # Calculate metrics
        generation_time = end_time - start_time
        
        # Estimate token count (rough approximation)
        output_tokens = len(generated_text.split())
        throughput = output_tokens / generation_time if generation_time > 0 else 0
        
        return InferenceResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model=model_name,
            settings={
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "max_tokens": request.max_tokens,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "use_json_schema": request.use_json_schema,
                "has_prefix": request.prefix is not None,
            },
            metrics={
                "generation_time_seconds": round(generation_time, 4),
                "estimated_output_tokens": output_tokens,
                "estimated_throughput_tok_per_sec": round(throughput, 2),
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/batch_generate")
async def batch_generate(prompts: List[str], 
                         temperature: float = 0.8,
                         top_p: float = 0.95,
                         max_tokens: int = 50):
    """
    Generate text for multiple prompts in batch
    
    This is more efficient than multiple individual requests
    as vLLM can process them together.
    """
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        start_time = time.time()
        outputs = llm_instance.generate(prompts, sampling_params)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        results = []
        for i, output in enumerate(outputs):
            results.append({
                "prompt": prompts[i],
                "generated_text": output.outputs[0].text,
            })
        
        return {
            "results": results,
            "model": model_name,
            "metrics": {
                "total_generation_time_seconds": round(generation_time, 4),
                "num_prompts": len(prompts),
                "avg_time_per_prompt_seconds": round(generation_time / len(prompts), 4),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation error: {str(e)}")


@app.post("/generate_stream")
async def generate_stream(request: InferenceRequest):
    """
    Generate text using vLLM with streaming to capture per-token metrics
    
    Returns streaming response with tokens and metrics:
    - TTFT (Time to First Token)
    - ITL (Inter-Token Latency) for each token
    - TPOT (Time Per Output Token) - average ITL
    - E2E (End-to-End Latency)
    
    Note: vLLM's generate() yields RequestOutput objects as generation progresses,
    allowing us to measure timing metrics incrementally.
    """
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    async def generate():
        try:
            # Build the full prompt
            full_prompt = request.prompt
            if request.prefix:
                full_prompt = request.prefix + request.prompt
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else -1,
                max_tokens=request.max_tokens,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
            )
            
            # Add guided decoding if JSON schema is provided
            if request.use_json_schema and request.json_schema:
                guided_decoding_params = GuidedDecodingParams(json=request.json_schema)
                sampling_params.structured_outputs = guided_decoding_params
            
            # Get tokenizer for accurate token counting
            tokenizer = llm_instance.llm_engine.tokenizer.tokenizer if hasattr(llm_instance.llm_engine, 'tokenizer') else None
            
            # Track timing metrics
            request_start_time = time.time()
            first_token_time = None
            last_token_time = None
            itl_times = []  # Inter-token latencies
            previous_text = ""
            previous_token_count = 0
            
            # Use vLLM's generator which yields RequestOutput objects as generation progresses
            for output in llm_instance.generate([full_prompt], sampling_params, use_tqdm=False):
                if output.outputs:
                    current_time = time.time()
                    current_text = output.outputs[0].text
                    finish_reason = output.outputs[0].finish_reason
                    
                    # Count tokens accurately if tokenizer available
                    if tokenizer:
                        current_token_count = len(tokenizer.encode(current_text))
                    else:
                        # Fallback to word count approximation
                        current_token_count = len(current_text.split())
                    
                    # Check if we have new tokens
                    if current_text != previous_text:
                        new_text = current_text[len(previous_text):]
                        new_token_count = current_token_count - previous_token_count
                        
                        if first_token_time is None:
                            # First token received
                            first_token_time = current_time
                            ttft = (first_token_time - request_start_time) * 1000  # Convert to ms
                            
                            yield f"data: {json.dumps({'type': 'token', 'token': new_text, 'ttft_ms': round(ttft, 2), 'token_index': 0})}\n\n"
                        else:
                            # Subsequent tokens
                            # Calculate ITL per new token (distribute time across new tokens)
                            if new_token_count > 0:
                                time_since_last = (current_time - last_token_time) * 1000
                                itl_per_token = time_since_last / new_token_count
                                
                                for i in range(new_token_count):
                                    itl_times.append(itl_per_token)
                                    token_index = previous_token_count + i + 1
                                    yield f"data: {json.dumps({'type': 'token', 'token': new_text[i] if len(new_text) == new_token_count else new_text, 'itl_ms': round(itl_per_token, 2), 'token_index': token_index})}\n\n"
                        
                        previous_text = current_text
                        previous_token_count = current_token_count
                        last_token_time = current_time
                    
                    # Check if generation is complete
                    if finish_reason is not None:
                        e2e_time = (last_token_time - request_start_time) * 1000 if last_token_time else 0
                        tpot = sum(itl_times) / len(itl_times) if itl_times else 0
                        
                        # Calculate metrics
                        metrics = {
                            'ttft_ms': round((first_token_time - request_start_time) * 1000, 2) if first_token_time else 0,
                            'tpot_ms': round(tpot, 2),
                            'e2e_ms': round(e2e_time, 2),
                            'itl_times_ms': [round(t, 2) for t in itl_times],
                            'output_tokens': current_token_count,
                            'generated_text': current_text
                        }
                        
                        yield f"data: {json.dumps({'type': 'done', 'metrics': metrics})}\n\n"
                        break
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """
    Run benchmark tests on multiple prompts and calculate aggregate metrics
    
    Returns:
    - Throughput (tokens per second)
    - Goodput (throughput meeting SLOs)
    - Aggregate TTFT, TPOT, ITL, E2E statistics
    """
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            max_tokens=request.max_tokens,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
        )
        
        # Get tokenizer for accurate token counting
        tokenizer = llm_instance.llm_engine.tokenizer.tokenizer if hasattr(llm_instance.llm_engine, 'tokenizer') else None
        
        benchmark_start_time = time.time()
        all_metrics = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Process each prompt
        for prompt in request.prompts:
            request_start_time = time.time()
            first_token_time = None
            last_token_time = None
            itl_times = []
            previous_text = ""
            previous_token_count = 0
            
            # Use vLLM's generator to capture incremental metrics
            for output in llm_instance.generate([prompt], sampling_params, use_tqdm=False):
                if output.outputs:
                    current_time = time.time()
                    current_text = output.outputs[0].text
                    finish_reason = output.outputs[0].finish_reason
                    
                    # Count tokens accurately
                    if tokenizer:
                        current_token_count = len(tokenizer.encode(current_text))
                    else:
                        current_token_count = len(current_text.split())
                    
                    # Track timing for new tokens
                    if current_text != previous_text:
                        new_token_count = current_token_count - previous_token_count
                        
                        if first_token_time is None:
                            first_token_time = current_time
                        else:
                            # Calculate ITL for new tokens
                            if new_token_count > 0:
                                time_since_last = (current_time - last_token_time) * 1000
                                itl_per_token = time_since_last / new_token_count
                                for _ in range(new_token_count):
                                    itl_times.append(itl_per_token)
                        
                        previous_text = current_text
                        previous_token_count = current_token_count
                        last_token_time = current_time
                    
                    if finish_reason is not None:
                        # Calculate metrics for this request
                        ttft_ms = (first_token_time - request_start_time) * 1000 if first_token_time else 0
                        e2e_ms = (last_token_time - request_start_time) * 1000 if last_token_time else 0
                        tpot_ms = sum(itl_times) / len(itl_times) if itl_times else 0
                        
                        total_output_tokens += current_token_count
                        
                        all_metrics.append({
                            'ttft_ms': ttft_ms,
                            'tpot_ms': tpot_ms,
                            'e2e_ms': e2e_ms,
                            'itl_times_ms': itl_times,
                            'output_tokens': current_token_count,
                            'meets_slo': True  # Will be updated below
                        })
                        break
            
            # Count input tokens accurately
            if tokenizer:
                total_input_tokens += len(tokenizer.encode(prompt))
            else:
                total_input_tokens += len(prompt.split())
        
        benchmark_end_time = time.time()
        benchmark_duration = benchmark_end_time - benchmark_start_time
        
        # Calculate aggregate statistics
        if not all_metrics:
            raise HTTPException(status_code=500, detail="No metrics collected")
        
        # Calculate SLO compliance
        for metric in all_metrics:
            meets_slo = True
            if request.max_ttft_ms and metric['ttft_ms'] > request.max_ttft_ms:
                meets_slo = False
            if request.max_tpot_ms and metric['tpot_ms'] > request.max_tpot_ms:
                meets_slo = False
            if request.max_e2e_ms and metric['e2e_ms'] > request.max_e2e_ms:
                meets_slo = False
            metric['meets_slo'] = meets_slo
        
        # Aggregate statistics
        ttft_values = [m['ttft_ms'] for m in all_metrics]
        tpot_values = [m['tpot_ms'] for m in all_metrics]
        e2e_values = [m['e2e_ms'] for m in all_metrics]
        all_itl_values = []
        for m in all_metrics:
            all_itl_values.extend(m['itl_times_ms'])
        
        # Calculate throughput
        total_tokens = total_input_tokens + total_output_tokens
        throughput_tok_per_sec = total_tokens / benchmark_duration if benchmark_duration > 0 else 0
        output_throughput_tok_per_sec = total_output_tokens / benchmark_duration if benchmark_duration > 0 else 0
        request_throughput_req_per_sec = len(request.prompts) / benchmark_duration if benchmark_duration > 0 else 0
        
        # Calculate goodput (only tokens from requests meeting SLOs)
        slo_compliant_metrics = [m for m in all_metrics if m['meets_slo']]
        goodput_tokens = sum(m['output_tokens'] for m in slo_compliant_metrics)
        goodput_tok_per_sec = goodput_tokens / benchmark_duration if benchmark_duration > 0 else 0
        
        def percentile(values, p):
            if not values:
                return 0
            sorted_values = sorted(values)
            index = int(len(sorted_values) * p / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]
        
        return {
            "benchmark_duration_seconds": round(benchmark_duration, 2),
            "num_requests": len(request.prompts),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            
            # Throughput metrics
            "throughput": {
                "total_tokens_per_sec": round(throughput_tok_per_sec, 2),
                "output_tokens_per_sec": round(output_throughput_tok_per_sec, 2),
                "requests_per_sec": round(request_throughput_req_per_sec, 2),
            },
            
            # Goodput metrics
            "goodput": {
                "slo_compliant_requests": len(slo_compliant_metrics),
                "slo_compliant_tokens": goodput_tokens,
                "goodput_tokens_per_sec": round(goodput_tok_per_sec, 2),
                "slo_compliance_rate": round(len(slo_compliant_metrics) / len(all_metrics) * 100, 2) if all_metrics else 0,
            },
            
            # TTFT statistics
            "ttft": {
                "mean_ms": round(sum(ttft_values) / len(ttft_values), 2),
                "median_ms": round(percentile(ttft_values, 50), 2),
                "p95_ms": round(percentile(ttft_values, 95), 2),
                "p99_ms": round(percentile(ttft_values, 99), 2),
                "min_ms": round(min(ttft_values), 2),
                "max_ms": round(max(ttft_values), 2),
            },
            
            # TPOT statistics
            "tpot": {
                "mean_ms": round(sum(tpot_values) / len(tpot_values), 2),
                "median_ms": round(percentile(tpot_values, 50), 2),
                "p95_ms": round(percentile(tpot_values, 95), 2),
                "p99_ms": round(percentile(tpot_values, 99), 2),
                "min_ms": round(min(tpot_values), 2),
                "max_ms": round(max(tpot_values), 2),
            },
            
            # ITL statistics
            "itl": {
                "mean_ms": round(sum(all_itl_values) / len(all_itl_values), 2) if all_itl_values else 0,
                "median_ms": round(percentile(all_itl_values, 50), 2) if all_itl_values else 0,
                "p95_ms": round(percentile(all_itl_values, 95), 2) if all_itl_values else 0,
                "p99_ms": round(percentile(all_itl_values, 99), 2) if all_itl_values else 0,
                "min_ms": round(min(all_itl_values), 2) if all_itl_values else 0,
                "max_ms": round(max(all_itl_values), 2) if all_itl_values else 0,
            },
            
            # E2E statistics
            "e2e": {
                "mean_ms": round(sum(e2e_values) / len(e2e_values), 2),
                "median_ms": round(percentile(e2e_values, 50), 2),
                "p95_ms": round(percentile(e2e_values, 95), 2),
                "p99_ms": round(percentile(e2e_values, 99), 2),
                "min_ms": round(min(e2e_values), 2),
                "max_ms": round(max(e2e_values), 2),
            },
            
            # SLO thresholds used
            "slo_thresholds": {
                "max_ttft_ms": request.max_ttft_ms,
                "max_tpot_ms": request.max_tpot_ms,
                "max_e2e_ms": request.max_e2e_ms,
            },
            
            # Per-request metrics
            "per_request_metrics": all_metrics,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if llm_instance is None:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    return {
        "model_name": model_name,
        "prefix_caching_enabled": os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true",
        "max_model_len": 2048,  # This is the default for TinyLlama
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting vLLM FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

