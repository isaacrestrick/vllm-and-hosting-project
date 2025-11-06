const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface InferenceRequest {
  prompt: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  use_json_schema?: boolean;
  json_schema?: any;
  prefix?: string;
}

export interface InferenceResponse {
  generated_text: string;
  prompt: string;
  model: string;
  settings: Record<string, any>;
  metrics: {
    generation_time_seconds: number;
    estimated_output_tokens: number;
    estimated_throughput_tok_per_sec: number;
  };
}

export interface BenchmarkRequest {
  prompts: string[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  max_ttft_ms?: number;
  max_tpot_ms?: number;
  max_e2e_ms?: number;
}

export interface BenchmarkResponse {
  benchmark_duration_seconds: number;
  num_requests: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  throughput: {
    total_tokens_per_sec: number;
    output_tokens_per_sec: number;
    requests_per_sec: number;
  };
  goodput: {
    slo_compliant_requests: number;
    slo_compliant_tokens: number;
    goodput_tokens_per_sec: number;
    slo_compliance_rate: number;
  };
  ttft: {
    mean_ms: number;
    median_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
  tpot: {
    mean_ms: number;
    median_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
  itl: {
    mean_ms: number;
    median_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
  e2e: {
    mean_ms: number;
    median_ms: number;
    p95_ms: number;
    p99_ms: number;
    min_ms: number;
    max_ms: number;
  };
  slo_thresholds: {
    max_ttft_ms?: number;
    max_tpot_ms?: number;
    max_e2e_ms?: number;
  };
  per_request_metrics: Array<{
    ttft_ms: number;
    tpot_ms: number;
    e2e_ms: number;
    itl_times_ms: number[];
    output_tokens: number;
    meets_slo: boolean;
  }>;
}

export interface StreamTokenEvent {
  type: 'token' | 'done' | 'error';
  token?: string;
  ttft_ms?: number;
  itl_ms?: number;
  token_index?: number;
  metrics?: {
    ttft_ms: number;
    tpot_ms: number;
    e2e_ms: number;
    itl_times_ms: number[];
    output_tokens: number;
    generated_text: string;
  };
  error?: string;
}

export async function generateText(request: InferenceRequest): Promise<InferenceResponse> {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate text');
  }

  return response.json();
}

export async function batchGenerate(
  prompts: string[],
  temperature: number = 0.8,
  top_p: number = 0.95,
  max_tokens: number = 50
): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/batch_generate?temperature=${temperature}&top_p=${top_p}&max_tokens=${max_tokens}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(prompts),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to batch generate');
  }

  return response.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_BASE_URL}/model_info`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch model info');
  }

  return response.json();
}

export async function checkHealth(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error('Server is not healthy');
  }

  return response.json();
}

export async function generateStream(
  request: InferenceRequest,
  onToken: (event: StreamTokenEvent) => void
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/generate_stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error('Failed to start streaming generation');
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('No response body reader available');
  }

  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          onToken(data);
          
          if (data.type === 'done' || data.type === 'error') {
            return;
          }
        } catch (e) {
          console.error('Failed to parse SSE data:', e);
        }
      }
    }
  }
}

export async function runBenchmark(request: BenchmarkRequest): Promise<BenchmarkResponse> {
  const response = await fetch(`${API_BASE_URL}/benchmark`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to run benchmark');
  }

  return response.json();
}

