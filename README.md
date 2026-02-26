# Graphsignal: Inference Observability

[![License](http://img.shields.io/github/license/graphsignal/graphsignal-python)](https://github.com/graphsignal/graphsignal-python/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal-python?label=version)](https://github.com/graphsignal/graphsignal-python)


Graphsignal is an inference observability platform that helps developers accelerate and troubleshoot AI systems. It provides essential visibility across the inference stack, including:

* Continuous, high-resolution profiling timelines exposing operation durations and resource utilization across inference workloads.
* LLM generation tracing with per-step timing, token throughput, and latency breakdowns for major inference frameworks.
* System-level metrics for inference engines and hardware (CPU, GPU, accelerators).
* Error monitoring for device-level failures, runtime exceptions, and inference errors.


[![Dashboards](https://graphsignal.com/external/screenshot-dashboard.png)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).


## Install

Install the Graphsignal library.

```bash
pip install -U graphsignal
```

**GPU profiling (Linux):** For CUPTI-based GPU profiling, install the extra matching your CUDA version: `pip install graphsignal[cu12]` (CUDA 12.x) or `pip install graphsignal[cu13]` (CUDA 13.x).


## Configure

Configure the Graphsignal SDK by specifying your API key directly or via the `GRAPHSIGNAL_API_KEY` environment variable.

```python
import graphsignal

graphsignal.configure(api_key='my-api-key')
# or pass the API key in GRAPHSIGNAL_API_KEY environment variable
```

See [`configure()`](https://graphsignal.com/docs/reference/python-api/#graphsignalconfigure) API docs for all configuration parameters.


To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api-keys) page.

Alternatively, you can use the Graphsignal runner when running your application. Pass the API key via the `GRAPHSIGNAL_API_KEY` environment variable.

```bash
graphsignal-run <my-app>
```


## Integrate

Graphsignal integrates through tracing - either via auto-instrumentation or manual setup. It automatically captures traces, errors, performance profiles, and data. All insights are available for analysis at [app.graphsignal.com](https://app.graphsignal.com/).

Refer to the guides below for detailed information on:

* [Manual Tracing](https://graphsignal.com/docs/guides/manual-tracing/)
* [Manual Profiling](https://graphsignal.com/docs/guides/manual-profiling/)
* [Using Tags](https://graphsignal.com/docs/guides/using-tags/)

See integration documentation for libraries and inference engines:

* [PyTorch](https://graphsignal.com/docs/integrations/pytorch/)
* [vLLM](https://graphsignal.com/docs/integrations/vllm/)

See the [API reference](https://graphsignal.com/docs/reference/python-api/) for complete documentation.


## Analyze

[Log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze your application.


## Overhead

Graphsignal tracer is highly lightweight. The overhead per trace is measured to be less than 100 microseconds. While profiling can introduce slight overhead, the profiling rate is limited.


## Security and Privacy

The Graphsignal tracer only establishes outbound connections to `api.graphsignal.com` to send data; inbound connections or commands are not possible.

Content and sensitive information, such as prompts and completions, are not recorded.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn’t provide hints for resolving the issue, report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://api.graphsignal.com` are allowed.
