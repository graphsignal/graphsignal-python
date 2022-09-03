# Graphsignal: Inference Monitoring and Profiling 

[![License](http://img.shields.io/github/license/graphsignal/graphsignal)](https://github.com/graphsignal/graphsignal/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal?label=version)](https://github.com/graphsignal/graphsignal)
[![Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


Graphsignal is a machine learning inference observability platform. It allows ML engineers and MLOps teams to:

* Monitor, troubleshoot and optimize inference by analyzing performance bottlenecks, resource utilization and errors.
* Start measuring and profiling server applications and batch jobs automatically by adding a few lines of code.
* Use Graphsignal in local, remote or cloud environment without installing any additional software or opening inbound ports.
* Keep data private; no code or data is sent to Graphsignal cloud, only statistics and metadata.

[![Dashboards](https://graphsignal.com/external/screencast-dashboards.gif)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).

## Documentation

See full documentation at [graphsignal.com/docs](https://graphsignal.com/docs/).


## Getting Started

### 1. Installation

Install Graphsignal agent by running:

```
pip install graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal):

```
git clone https://github.com/graphsignal/graphsignal.git
python setup.py install
```


### 2. Configuration

Configure Graphsignal agent by specifying your API key directly or via environment variable.

```python
import graphsignal

graphsignal.configure(api_key='my-api-key')
```

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api-keys) page.


### 3. Integration

Use the following examples to integrate Graphsignal agent into your machine learning application. See integration documentation and [API reference](https://graphsignal.com/docs/reference/python-api/) for full reference.

Graphsignal agent is **optimized for production**. All inferences wrapped with `inference_span` will be measured, but only a few will be traced and profiled to ensure low overhead.


#### Tracing

To measure and trace inferences, wrap the code with `inference_span` method.

```python
tracer = graphsignal.tracer()

with tracer.inference_span(model_name='my-model'):
    # function call or code segment
```

Other integrations are available as well. See [integration documentation](https://graphsignal.com/docs/) for more information.


#### Profiling

Enable/disable various profilers depending on the code and model runtime by passing `with_profiler` argument to `tracer()` method. By default `with_profiler=True` and Python profiler is enabled. Set to `False` to disable profiling.

```python
tracer = graphsignal.tracer(with_profiler='pytorch')
```

The following values are currently supported: `True` (or `python`), `tensorflow`, `pytorch`, `jax`, `onnxruntime`. See [integration documentation](https://graphsignal.com/docs/) for more information on each profiler.


#### Exception tracking

When `with` context manager is used with `inference_span` method, exceptions are automatically reported. For other cases, use `InferenceSpan.set_exception(exc_info)` method.


### 4. Monitoring

After everything is setup, [log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze inference performance.


## Examples

### Model serving

```python
import graphsignal

graphsignal.configure(api_key='my-api-key')
tracer = graphsignal.tracer()

...

def predict(x):
    with tracer.inference_span(model_name='my-model-prod'):
        return model(x)
```

### Batch job

```python
import graphsignal

graphsignal.configure(api_key='my-api-key')
tracer = graphsignal.tracer()

....

for x in data:
    with tracer.inference_span(model_name='my-model', tags=dict(job_id='job1')):
        preds = model(x)
```

More integration examples are available in [`examples`](https://github.com/graphsignal/examples) repo.


## Overhead

Although profiling may add some overhead to applications, Graphsignal only profiles certain inferences, automatically limiting the overhead.


## Security and Privacy

Graphsignal agent can only open outbound connections to `agent-api.graphsignal.com` and send data, no inbound connections or commands are possible. 

No code or data is sent to Graphsignal cloud, only statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://agent-api.graphsignal.com` are allowed.

For GPU profiling, if `libcupti` agent is failing to load, make sure the [NVIDIA® CUDA® Profiling Tools Interface](https://developer.nvidia.com/cupti) (CUPTI) is installed by running:

```console
/sbin/ldconfig -p | grep libcupti
```
