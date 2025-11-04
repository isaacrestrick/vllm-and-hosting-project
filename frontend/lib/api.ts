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

export interface ModelInfo {
  model_name: string;
  prefix_caching_enabled: boolean;
  max_model_len: number;
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

