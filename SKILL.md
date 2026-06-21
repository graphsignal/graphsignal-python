---
name: graphsignal-profiler
description: >-
  Set up the Graphsignal Profiler for inference workloads — vLLM, SGLang,
  PyTorch, and dstack services. Use when the user wants GPU profiling, tracing,
  or monitoring for inference, asks about `graphsignal-run` or `graphsignal.watch()`,
  or asks about CUPTI / Prometheus / OTLP setup.
---

# Graphsignal Profiler

Graphsignal observes inference workloads from a **sidecar process** — the profiler. It never shares a process with CUDA: the profiler watches the workload externally via `/dev/shm`, OTLP/gRPC, Prometheus scraping, and NVML. Auto-instrumentation covers vLLM, SGLang, and PyTorch out of the box.

## Install

Two install patterns depending on how you'll launch the profiler.

**For `graphsignal-run` (CLI, recommended):** install as a uv tool, isolated from your workload env.

```bash
UV_TOOL_BIN_DIR=/usr/local/bin uv tool install 'graphsignal[cu12]'   # CUDA 12.x
# or
UV_TOOL_BIN_DIR=/usr/local/bin uv tool install 'graphsignal[cu13]'   # CUDA 13.x
```

`UV_TOOL_BIN_DIR=/usr/local/bin` puts `graphsignal-run` in a directory that is already on `PATH` for every shell, including non-interactive scripts and containers.

**For `graphsignal.watch()` (in-process Python entry point):** install into the app's own env.

```bash
uv add 'graphsignal[cu12]'    # or pip install -U 'graphsignal[cu12]'
```

The `cu12` / `cu13` extras are Linux-only and only needed for GPU profiling.

## Configure

The profiler reads its config from environment variables.

| Variable                              | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `GRAPHSIGNAL_API_KEY` *(required)*    | Account API key.                                                        |
| `GRAPHSIGNAL_API_BASE`                | Override the API endpoint (defaults to `https://api.graphsignal.com`). |
| `GRAPHSIGNAL_TAG_<KEY>=<value>`       | Arbitrary tag attached to all signals (e.g. `GRAPHSIGNAL_TAG_DEPLOYMENT=us-prod`). |

Set these before invoking `graphsignal-run` or calling `graphsignal.watch()`.

## Run

### Option A — `graphsignal-run` CLI (recommended)

Wrap the launch command for your workload.

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run vllm serve Qwen/Qwen1.5-7B-Chat --port 8000
```

The CLI sets up CUPTI env vars, spawns a profiler sidecar subprocess, and `execv`'s into the workload.

### Option B — `graphsignal.watch()` from Python

For applications that bootstrap themselves (long-lived servers, scripts, notebooks), call `graphsignal.watch()` once during startup, **before any CUDA work happens**.

```python
import graphsignal

graphsignal.watch()
# ... your application code (PyTorch, vLLM, SGLang, etc.) ...
```

It sets up the CUPTI env vars in this process and spawns the profiler sidecar subprocess targeting `os.getpid()`. Returns the `subprocess.Popen` so the caller can `wait()` or `terminate()` it.

### OpenTelemetry tracing (opt-in)

Distributed traces (engine / scheduler / attention spans over OTLP/gRPC) are **off by default**. Enable them with `--enable-otel`, which must come *before* the workload command:

```bash
graphsignal-run --enable-otel sglang serve --model-path Qwen/Qwen1.5-7B-Chat --port 8000
```

This injects the engine's trace flags and starts a local OTLP collector in the profiler. It requires OpenTelemetry installed in the **engine's** environment (e.g. `pip install opentelemetry-sdk opentelemetry-exporter-otlp`) — graphsignal can't provide it when installed in a separate env (e.g. `uv tool`), and SGLang ≥ 0.5.10 errors at startup if tracing is enabled without it. Prometheus metrics and CUPTI GPU profiling are captured regardless of this flag; OTEL injection applies only to `graphsignal-run` (not `graphsignal.watch()`).

## Engine-specific notes

### vLLM

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run vllm serve Qwen/Qwen1.5-7B-Chat --port 8000
```

Or from Python (before importing vLLM):

```python
import graphsignal
graphsignal.watch()
import vllm
# ...
```

Captures vLLM's Prometheus metrics and CUPTI GPU profiling out of the box. Engine / scheduler / KV-cache / attention / output-processing OTEL spans are added with `--enable-otel` (see [OpenTelemetry tracing](#opentelemetry-tracing-opt-in) above).

vLLM Docker (image without CUPTI):

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

### SGLang

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run sglang serve \
  --model-path Qwen/Qwen1.5-7B-Chat \
  --port 8000
```

Captures SGLang's Prometheus metrics and operation-level GPU profiling out of the box. OTEL spans are added with `--enable-otel` (see [OpenTelemetry tracing](#opentelemetry-tracing-opt-in) above).

SGLang Docker:

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

### PyTorch

Auto-instrumented for common PyTorch operator / module hot paths and CUDA memory metrics. Use either CLI or `watch()`:

```bash
export GRAPHSIGNAL_API_KEY="..."
graphsignal-run python my_app.py
```

```python
import graphsignal
graphsignal.watch()
import torch
# ...
```

### dstack

dstack runs inference as services. Use `graphsignal-run` around the launch command, exactly like bare-metal SGLang/vLLM.

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

## Troubleshooting

Ensure outgoing connections to `https://api.graphsignal.com` are allowed.

## Reference

- Full Profiler API: https://graphsignal.com/docs/reference/profiler-api/
- vLLM integration: https://graphsignal.com/docs/integrations/vllm/
- SGLang integration: https://graphsignal.com/docs/integrations/sglang/
- dstack integration: https://graphsignal.com/docs/integrations/dstack/
