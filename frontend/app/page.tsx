'use client';

import { useState } from 'react';
import { generateText, batchGenerate, getModelInfo, checkHealth, InferenceResponse, ModelInfo } from '../lib/api';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.95);
  const [maxTokens, setMaxTokens] = useState(50);
  const [topK, setTopK] = useState(-1);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [prefix, setPrefix] = useState('');
  
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [healthStatus, setHealthStatus] = useState<string | null>(null);
  
  const [batchPrompts, setBatchPrompts] = useState('');
  const [batchResults, setBatchResults] = useState<any>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await generateText({
        prompt,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        top_k: topK === -1 ? undefined : topK,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        prefix: prefix.trim() || undefined,
      });
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to generate text');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchGenerate = async () => {
    const prompts = batchPrompts.split('\n').filter(p => p.trim());
    if (prompts.length === 0) {
      setError('Please enter at least one prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setBatchResults(null);

    try {
      const response = await batchGenerate(prompts, temperature, topP, maxTokens);
      setBatchResults(response);
    } catch (err: any) {
      setError(err.message || 'Failed to batch generate');
    } finally {
      setLoading(false);
    }
  };

  const handleCheckHealth = async () => {
    try {
      const response = await checkHealth();
      setHealthStatus(`✅ ${response.status} - Model: ${response.model}`);
    } catch (err: any) {
      setHealthStatus(`❌ ${err.message}`);
    }
  };

  const handleGetModelInfo = async () => {
    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch model info');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">vLLM Inference Server</h1>
          <p className="text-gray-600">Interact with your FastAPI vLLM server</p>
        </header>

        {/* Health Check Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">Server Status</h2>
            <button
              onClick={handleCheckHealth}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Check Health
            </button>
          </div>
          {healthStatus && (
            <p className="text-gray-700 font-mono text-sm">{healthStatus}</p>
          )}
        </div>

        {/* Model Info Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-800">Model Information</h2>
            <button
              onClick={handleGetModelInfo}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Get Info
            </button>
          </div>
          {modelInfo && (
            <div className="bg-gray-50 rounded-lg p-4 mt-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Model Name</p>
                  <p className="font-semibold text-gray-900">{modelInfo.model_name}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Prefix Caching</p>
                  <p className="font-semibold text-gray-900">
                    {modelInfo.prefix_caching_enabled ? '✅ Enabled' : '❌ Disabled'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Max Model Length</p>
                  <p className="font-semibold text-gray-900">{modelInfo.max_model_len}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Generation Parameters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Generation Parameters</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature: {temperature}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Top-P: {topP}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Tokens: {maxTokens}
              </label>
              <input
                type="range"
                min="1"
                max="2048"
                step="1"
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Top-K: {topK === -1 ? 'Disabled' : topK}
              </label>
              <input
                type="range"
                min="-1"
                max="100"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Frequency Penalty: {frequencyPenalty}
              </label>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={frequencyPenalty}
                onChange={(e) => setFrequencyPenalty(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Presence Penalty: {presencePenalty}
              </label>
              <input
                type="range"
                min="-2"
                max="2"
                step="0.1"
                value={presencePenalty}
                onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Single Generation */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Text Generation</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prefix (Optional - for prefix caching)
            </label>
            <textarea
              value={prefix}
              onChange={(e) => setPrefix(e.target.value)}
              placeholder="Enter shared prefix context..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={3}
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
            />
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
          >
            {loading ? 'Generating...' : 'Generate Text'}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Generated Text:</h3>
              <p className="text-gray-700 mb-4 whitespace-pre-wrap">{result.generated_text}</p>
              
              <div className="border-t pt-4 mt-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Metrics:</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Generation Time:</span>
                    <span className="ml-2 font-semibold">{result.metrics.generation_time_seconds}s</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Output Tokens:</span>
                    <span className="ml-2 font-semibold">{result.metrics.estimated_output_tokens}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Throughput:</span>
                    <span className="ml-2 font-semibold">{result.metrics.estimated_throughput_tok_per_sec} tok/s</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Batch Generation */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Batch Generation</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompts (one per line)
            </label>
            <textarea
              value={batchPrompts}
              onChange={(e) => setBatchPrompts(e.target.value)}
              placeholder="Enter prompts, one per line..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={6}
            />
          </div>

          <button
            onClick={handleBatchGenerate}
            disabled={loading}
            className="w-full px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
          >
            {loading ? 'Generating...' : 'Batch Generate'}
          </button>

          {batchResults && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Batch Results:</h3>
              {batchResults.results.map((item: any, idx: number) => (
                <div key={idx} className="mb-4 pb-4 border-b last:border-b-0">
                  <p className="text-sm text-gray-600 mb-1">Prompt {idx + 1}:</p>
                  <p className="text-gray-800 mb-2">{item.prompt}</p>
                  <p className="text-sm text-gray-600 mb-1">Generated:</p>
                  <p className="text-gray-700">{item.generated_text}</p>
                </div>
              ))}
              
              <div className="border-t pt-4 mt-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">Batch Metrics:</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Total Time:</span>
                    <span className="ml-2 font-semibold">{batchResults.metrics.total_generation_time_seconds}s</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Prompts:</span>
                    <span className="ml-2 font-semibold">{batchResults.metrics.num_prompts}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Avg Time/Prompt:</span>
                    <span className="ml-2 font-semibold">{batchResults.metrics.avg_time_per_prompt_seconds}s</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

