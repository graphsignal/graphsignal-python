# Graphsignal: Inference Profiler

[![License](http://img.shields.io/github/license/graphsignal/graphsignal-profiler)](https://github.com/graphsignal/graphsignal-profiler/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal-profiler?label=version)](https://github.com/graphsignal/graphsignal-profiler)


Graphsignal is a production-scale inference profiling platform that helps engineers optimize AI performance across models, engines, GPUs, and other accelerators. It provides essential visibility across the inference stack, including:

* Continuous, high-resolution profiling timelines exposing operation durations and resource utilization across inference workloads.
* LLM generation tracing with per-step timing, token throughput, and latency breakdowns for major inference frameworks.
* System-level metrics for inference engines and hardware (CPU, GPU, accelerators).
* Error monitoring for device-level failures and inference errors.
* Inference telemetry for AI agents to identify bottlenecks and drive targeted improvements across the inference stack.

[![Dashboards](https://graphsignal.com/external/screenshot-dashboard.png?v=2)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).


## Install

```bash
uv tool install 'graphsignal[cu12]'   # CUDA 12.x
# or
uv tool install 'graphsignal[cu13]'   # CUDA 13.x
```


## Profile

Wrap your launch command with `graphsignal-run`:

```bash
export GRAPHSIGNAL_API_KEY=<my-api-key>
graphsignal-run vllm serve <model> --port 8001
```

Environment variables read by the profiler:

| Variable                              | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `GRAPHSIGNAL_API_KEY` *(required)*    | Your account API key.                                                  |
| `GRAPHSIGNAL_TAG_<KEY>=<value>`       | Arbitrary tag attached to all signals (e.g. `GRAPHSIGNAL_TAG_DEPLOYMENT=us-prod`). |

Sign up for a free account at [graphsignal.com](https://graphsignal.com); you'll find the API key in [Settings / API Keys](https://app.graphsignal.com/settings/api-keys).

See the [Profiler CLI](https://graphsignal.com/docs/reference/profile-cli/) reference for the full set of options.

Applications that bootstrap themselves can call `graphsignal.watch()` from Python instead — see the [Profiler API](https://graphsignal.com/docs/reference/profiler-api/) reference.

See integration documentation for libraries and inference engines:

* [PyTorch](https://graphsignal.com/docs/integrations/pytorch/)
* [vLLM](https://graphsignal.com/docs/integrations/vllm/)
* [SGLang](https://graphsignal.com/docs/integrations/sglang/)


## Optimize

[Log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze your application.

### Optimize with AI

Install the Graphsignal skill to let your AI coding agent (Claude Code, Codex, or Gemini) fetch and analyze signal context directly from your agent. See [AI Optimization](https://graphsignal.com/docs/guides/ai-optimization/) for setup instructions.


## Overhead

The profiler has minimal impact on production performance. CUPTI activity is collected with low-overhead APIs in a sidecar process, and the in-process injection only writes raw activity records — analysis and upload happen in the sidecar.


## Security and Privacy

The profiler only establishes outbound connections to `api.graphsignal.com` to send data; inbound connections or commands are not possible.

Content and sensitive information, such as prompts and completions, are not recorded.


## Troubleshooting

If something doesn't look right, report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://api.graphsignal.com` are allowed.
