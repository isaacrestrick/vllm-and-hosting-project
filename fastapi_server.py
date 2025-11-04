"""
FastAPI server for vLLM inference with configurable settings
Based on benchmarking tweaks from the vllm-and-hosting-project
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
import time
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

