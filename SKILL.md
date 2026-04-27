---
name: graphsignal-sdk
description: >-
  Set up and integrate Graphsignal inference observability into Python applications,
  vLLM, SGLang, and dstack services. Use when the user wants to add profiling, tracing,
  or monitoring to inference workloads, needs help configuring the Graphsignal SDK,
  or asks about graphsignal-run, CUPTI, or GPU profiling setup.
---

# Graphsignal SDK

Graphsignal captures performance profiles, traces, metrics, and errors for inference workloads. Auto-instrumentation covers vLLM, SGLang, and PyTorch out of the box.

## Install

```bash
pip install -U graphsignal
```

For CUPTI-based GPU profiling on Linux, install the extra matching the CUDA version:

```bash
pip install graphsignal[cu12]   # CUDA 12.x
pip install graphsignal[cu13]   # CUDA 13.x
```

## Configure

### Option A: In Python code

```python
import graphsignal

graphsignal.configure(api_key='my-api-key')
```

All `configure()` args can be set via env vars: `GRAPHSIGNAL_API_KEY`, `GRAPHSIGNAL_DEBUG_MODE`, etc.

`configure()` parameters:

| Arg | Env var | Purpose |
|-----|---------|---------|
| `api_key` | `GRAPHSIGNAL_API_KEY` | API key (required) |
| `api_base` | `GRAPHSIGNAL_API_BASE` | On-premise server URL |
| `tags` | `GRAPHSIGNAL_TAG_{KEY}` | Process-level tags |
| `auto_instrument` | `GRAPHSIGNAL_AUTO_INSTRUMENT` | Auto-instrument libraries (default `True`) |
| `debug_mode` | `GRAPHSIGNAL_DEBUG_MODE` | Enable debug logging |

### Option B: graphsignal-run CLI

Wrap any command — no code changes needed:

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run <my-app>
```

## Integrate with vLLM

Graphsignal automatically instruments vLLM (engine, scheduler, KV cache, attention, output processing, Prometheus metrics).

### In a Python app

```python
import graphsignal
graphsignal.configure(api_key='my-api-key')
# then use vLLM normally
```

### vLLM server via graphsignal-run

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run vllm serve Qwen/Qwen1.5-7B-Chat --port 8000
```

### vLLM Docker

```bash
docker run --gpus all \
  -p 8000:8000 --ipc=host \
  -e GRAPHSIGNAL_API_KEY=YOUR_API_KEY \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  vllm/vllm-openai:latest \
  -lc 'pip install --no-cache-dir graphsignal[cu12] \
       && exec graphsignal-run vllm serve \
           --model Qwen/Qwen2-VL-7B-Instruct \
           --trust-remote-code'
```

## Integrate with SGLang

Graphsignal automatically instruments SGLang (operations, OTEL spans, Prometheus metrics).

### In a Python app

```python
import graphsignal
graphsignal.configure(api_key='my-api-key')
# then use SGLang normally
```

### SGLang server via graphsignal-run

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run sglang serve \
  --model-path Qwen/Qwen1.5-7B-Chat \
  --port 8000
```

### SGLang Docker

```bash
docker run --gpus all \
  -p 8000:8000 --ipc=host \
  -e GRAPHSIGNAL_API_KEY=YOUR_API_KEY \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash \
  your-sglang-image:latest \
  -lc 'pip install --no-cache-dir graphsignal[cu12] \
       && exec graphsignal-run sglang serve \
           --model-path Qwen/Qwen2.5-1.5B-Instruct \
           --port 8000'
```

## Integrate via dstack

dstack runs inference as services. Use `graphsignal-run` around the server command, same as bare-metal.

### dstack service config (SGLang example)

```yaml
type: service
name: deepseek-r1

image: lmsysorg/sglang:latest
env:
  - MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  - GRAPHSIGNAL_API_KEY

commands:
  - |
    pip install --no-cache-dir 'graphsignal[cu12]' && \
    graphsignal-run python3 -m sglang.launch_server \
      --model-path $MODEL_ID \
      --port 8000 \
      --trust-remote-code

port: 8000
model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B

resources:
  gpu: 24GB
```

Deploy:

```bash
dstack apply -f service.dstack.yml
```

If your Docker image already includes Graphsignal, skip the `pip install` step. If using the `sglang` CLI instead of `launch_server`, use `graphsignal-run sglang serve ...`.

## Manual Tracing

For code not auto-instrumented, use `trace()` or `@trace_function`:

```python
with graphsignal.trace('inference') as span:
    span.set_counter('prompt_tokens', prompt_tokens)
    span.set_counter('completion_tokens', completion_tokens)

    with span.trace('model-cold-boot') as sub_span:
        ...
```

```python
@graphsignal.trace_function
def my_function():
    ...
```

Record counter metrics within a trace:

```python
with graphsignal.trace('my-function') as span:
    span.inc_counter_metric('call_cost', price_per_call)
```

## Manual Profiling

Profile specific Python functions (3.12+):

```python
graphsignal.profile_function(func=slow_transform, category='transform', op_name='data-transform')
```

Profile by import path (avoids direct import):

```python
graphsignal.profile_function_path(path='myapp.tasks.prepare_data', category='preprocessing')
```

Profile CUDA kernels by pattern (Linux, CUPTI required):

```python
graphsignal.profile_cuda_kernel(kernel_pattern="cublas", op_name="matmul_gemm")
```

## Tags

Tags enable filtering and breakdown of traces, profiles, and metrics.

```python
graphsignal.configure(api_key='my-api-key', tags={'app_version': '1.0'})

graphsignal.set_tag('app_version', '1.0')

graphsignal.set_context_tag('user_id', current_user_id)

with graphsignal.trace('generate', tags=dict(user_id=my_user_id)):
    ...
```

Via CLI:

```bash
env GRAPHSIGNAL_TAG_APP_VERSION="1.0" graphsignal-run <my-app>
```

## Troubleshooting

Enable debug logging: `graphsignal.configure(debug_mode=True)` or `GRAPHSIGNAL_DEBUG_MODE=true`.

Ensure outgoing connections to `https://api.graphsignal.com` are allowed.

## Reference

- Full Python API: https://graphsignal.com/docs/reference/python-api/
- vLLM integration: https://graphsignal.com/docs/integrations/vllm/
- SGLang integration: https://graphsignal.com/docs/integrations/sglang/
- dstack integration: https://graphsignal.com/docs/integrations/dstack/
