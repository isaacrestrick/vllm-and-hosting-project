'use client';

import { useState } from 'react';
import { generateText, batchGenerate, getModelInfo, checkHealth, InferenceResponse, ModelInfo, generateStream, runBenchmark, StreamTokenEvent, BenchmarkResponse } from '../lib/api';

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
  
  // Streaming state
  const [streamingText, setStreamingText] = useState('');
  const [streamingMetrics, setStreamingMetrics] = useState<any>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Benchmark state
  const [benchmarkPrompts, setBenchmarkPrompts] = useState('');
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResponse | null>(null);
  const [maxTtftMs, setMaxTtftMs] = useState<number | undefined>(undefined);
  const [maxTpotMs, setMaxTpotMs] = useState<number | undefined>(undefined);
  const [maxE2eMs, setMaxE2eMs] = useState<number | undefined>(undefined);
  const [isBenchmarking, setIsBenchmarking] = useState(false);

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

  const handleStreamGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsStreaming(true);
    setError(null);
    setStreamingText('');
    setStreamingMetrics(null);

    try {
      await generateStream(
        {
          prompt,
          temperature,
          top_p: topP,
          max_tokens: maxTokens,
          top_k: topK === -1 ? undefined : topK,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          prefix: prefix.trim() || undefined,
        },
        (event: StreamTokenEvent) => {
          if (event.type === 'token') {
            setStreamingText((prev) => prev + (event.token || ''));
          } else if (event.type === 'done' && event.metrics) {
            setStreamingMetrics(event.metrics);
            setIsStreaming(false);
          } else if (event.type === 'error') {
            setError(event.error || 'Streaming error');
            setIsStreaming(false);
          }
        }
      );
    } catch (err: any) {
      setError(err.message || 'Failed to stream generation');
      setIsStreaming(false);
    }
  };

  const handleRunBenchmark = async () => {
    const prompts = benchmarkPrompts.split('\n').filter(p => p.trim());
    if (prompts.length === 0) {
      setError('Please enter at least one prompt for benchmarking');
      return;
    }

    setIsBenchmarking(true);
    setError(null);
    setBenchmarkResults(null);

    try {
      const response = await runBenchmark({
        prompts,
        temperature,
        top_p: topP,
        max_tokens: maxTokens,
        top_k: topK === -1 ? undefined : topK,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        max_ttft_ms: maxTtftMs,
        max_tpot_ms: maxTpotMs,
        max_e2e_ms: maxE2eMs,
      });
      setBenchmarkResults(response);
    } catch (err: any) {
      setError(err.message || 'Failed to run benchmark');
    } finally {
      setIsBenchmarking(false);
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

        {/* Streaming Generation with Metrics */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Streaming Generation (with Metrics)</h2>
          
          <button
            onClick={handleStreamGenerate}
            disabled={isStreaming || loading}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold mb-4"
          >
            {isStreaming ? 'Streaming...' : 'Generate with Streaming Metrics'}
          </button>

          {streamingText && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg mb-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Streamed Text:</h3>
              <p className="text-gray-700 whitespace-pre-wrap">{streamingText}</p>
            </div>
          )}

          {streamingMetrics && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Streaming Metrics:</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600 block">TTFT:</span>
                  <span className="font-semibold text-lg">{streamingMetrics.ttft_ms} ms</span>
                </div>
                <div>
                  <span className="text-gray-600 block">TPOT:</span>
                  <span className="font-semibold text-lg">{streamingMetrics.tpot_ms} ms</span>
                </div>
                <div>
                  <span className="text-gray-600 block">E2E Latency:</span>
                  <span className="font-semibold text-lg">{streamingMetrics.e2e_ms} ms</span>
                </div>
                <div>
                  <span className="text-gray-600 block">Output Tokens:</span>
                  <span className="font-semibold text-lg">{streamingMetrics.output_tokens}</span>
                </div>
              </div>
              {streamingMetrics.itl_times_ms && streamingMetrics.itl_times_ms.length > 0 && (
                <div className="mt-4">
                  <span className="text-gray-600 text-sm">ITL Times (ms):</span>
                  <p className="text-xs font-mono text-gray-700 mt-1">
                    {streamingMetrics.itl_times_ms.slice(0, 10).join(', ')}
                    {streamingMetrics.itl_times_ms.length > 10 && ' ...'}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Benchmark Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Benchmark Testing</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Prompts for Benchmarking (one per line)
            </label>
            <textarea
              value={benchmarkPrompts}
              onChange={(e) => setBenchmarkPrompts(e.target.value)}
              placeholder="Enter prompts, one per line..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={6}
            />
          </div>

          <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max TTFT (ms) - SLO Threshold
              </label>
              <input
                type="number"
                value={maxTtftMs || ''}
                onChange={(e) => setMaxTtftMs(e.target.value ? parseFloat(e.target.value) : undefined)}
                placeholder="Optional"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max TPOT (ms) - SLO Threshold
              </label>
              <input
                type="number"
                value={maxTpotMs || ''}
                onChange={(e) => setMaxTpotMs(e.target.value ? parseFloat(e.target.value) : undefined)}
                placeholder="Optional"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max E2E (ms) - SLO Threshold
              </label>
              <input
                type="number"
                value={maxE2eMs || ''}
                onChange={(e) => setMaxE2eMs(e.target.value ? parseFloat(e.target.value) : undefined)}
                placeholder="Optional"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <button
            onClick={handleRunBenchmark}
            disabled={isBenchmarking || loading}
            className="w-full px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
          >
            {isBenchmarking ? 'Running Benchmark...' : 'Run Benchmark'}
          </button>

          {benchmarkResults && (
            <div className="mt-6 space-y-6">
              {/* Summary */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Benchmark Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 block">Duration:</span>
                    <span className="font-semibold">{benchmarkResults.benchmark_duration_seconds}s</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Requests:</span>
                    <span className="font-semibold">{benchmarkResults.num_requests}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Input Tokens:</span>
                    <span className="font-semibold">{benchmarkResults.total_input_tokens}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Output Tokens:</span>
                    <span className="font-semibold">{benchmarkResults.total_output_tokens}</span>
                  </div>
                </div>
              </div>

              {/* Throughput */}
              <div className="p-4 bg-green-50 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Throughput</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 block">Total Tokens/sec:</span>
                    <span className="font-semibold text-lg">{benchmarkResults.throughput.total_tokens_per_sec}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Output Tokens/sec:</span>
                    <span className="font-semibold text-lg">{benchmarkResults.throughput.output_tokens_per_sec}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Requests/sec:</span>
                    <span className="font-semibold text-lg">{benchmarkResults.throughput.requests_per_sec}</span>
                  </div>
                </div>
              </div>

              {/* Goodput */}
              <div className="p-4 bg-yellow-50 rounded-lg">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Goodput (SLO-Compliant)</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 block">Compliant Requests:</span>
                    <span className="font-semibold">{benchmarkResults.goodput.slo_compliant_requests} / {benchmarkResults.num_requests}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Compliance Rate:</span>
                    <span className="font-semibold">{benchmarkResults.goodput.slo_compliance_rate}%</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Compliant Tokens:</span>
                    <span className="font-semibold">{benchmarkResults.goodput.slo_compliant_tokens}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 block">Goodput (tok/s):</span>
                    <span className="font-semibold text-lg">{benchmarkResults.goodput.goodput_tokens_per_sec}</span>
                  </div>
                </div>
              </div>

              {/* Latency Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* TTFT */}
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-semibold text-gray-800 mb-2">TTFT (Time to First Token)</h4>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between"><span>Mean:</span><span className="font-semibold">{benchmarkResults.ttft.mean_ms} ms</span></div>
                    <div className="flex justify-between"><span>Median:</span><span className="font-semibold">{benchmarkResults.ttft.median_ms} ms</span></div>
                    <div className="flex justify-between"><span>P95:</span><span className="font-semibold">{benchmarkResults.ttft.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>P99:</span><span className="font-semibold">{benchmarkResults.ttft.p99_ms} ms</span></div>
                    <div className="flex justify-between"><span>Min:</span><span className="font-semibold">{benchmarkResults.ttft.min_ms} ms</span></div>
                    <div className="flex justify-between"><span>Max:</span><span className="font-semibold">{benchmarkResults.ttft.max_ms} ms</span></div>
                  </div>
                </div>

                {/* TPOT */}
                <div className="p-4 bg-purple-50 rounded-lg">
                  <h4 className="font-semibold text-gray-800 mb-2">TPOT (Time Per Output Token)</h4>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between"><span>Mean:</span><span className="font-semibold">{benchmarkResults.tpot.mean_ms} ms</span></div>
                    <div className="flex justify-between"><span>Median:</span><span className="font-semibold">{benchmarkResults.tpot.median_ms} ms</span></div>
                    <div className="flex justify-between"><span>P95:</span><span className="font-semibold">{benchmarkResults.tpot.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>P99:</span><span className="font-semibold">{benchmarkResults.tpot.p99_ms} ms</span></div>
                    <div className="flex justify-between"><span>Min:</span><span className="font-semibold">{benchmarkResults.tpot.min_ms} ms</span></div>
                    <div className="flex justify-between"><span>Max:</span><span className="font-semibold">{benchmarkResults.tpot.max_ms} ms</span></div>
                  </div>
                </div>

                {/* ITL */}
                <div className="p-4 bg-indigo-50 rounded-lg">
                  <h4 className="font-semibold text-gray-800 mb-2">ITL (Inter-Token Latency)</h4>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between"><span>Mean:</span><span className="font-semibold">{benchmarkResults.itl.mean_ms} ms</span></div>
                    <div className="flex justify-between"><span>Median:</span><span className="font-semibold">{benchmarkResults.itl.median_ms} ms</span></div>
                    <div className="flex justify-between"><span>P95:</span><span className="font-semibold">{benchmarkResults.itl.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>P99:</span><span className="font-semibold">{benchmarkResults.itl.p99_ms} ms</span></div>
                    <div className="flex justify-between"><span>Min:</span><span className="font-semibold">{benchmarkResults.itl.min_ms} ms</span></div>
                    <div className="flex justify-between"><span>Max:</span><span className="font-semibold">{benchmarkResults.itl.max_ms} ms</span></div>
                  </div>
                </div>

                {/* E2E */}
                <div className="p-4 bg-red-50 rounded-lg">
                  <h4 className="font-semibold text-gray-800 mb-2">E2E (End-to-End Latency)</h4>
                  <div className="text-xs space-y-1">
                    <div className="flex justify-between"><span>Mean:</span><span className="font-semibold">{benchmarkResults.e2e.mean_ms} ms</span></div>
                    <div className="flex justify-between"><span>Median:</span><span className="font-semibold">{benchmarkResults.e2e.median_ms} ms</span></div>
                    <div className="flex justify-between"><span>P95:</span><span className="font-semibold">{benchmarkResults.e2e.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>P99:</span><span className="font-semibold">{benchmarkResults.e2e.p99_ms} ms</span></div>
                    <div className="flex justify-between"><span>Min:</span><span className="font-semibold">{benchmarkResults.e2e.min_ms} ms</span></div>
                    <div className="flex justify-between"><span>Max:</span><span className="font-semibold">{benchmarkResults.e2e.max_ms} ms</span></div>
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

