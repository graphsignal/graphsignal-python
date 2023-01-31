# Graphsignal: AI Application Monitoring

[![License](http://img.shields.io/github/license/graphsignal/graphsignal)](https://github.com/graphsignal/graphsignal/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal?label=version)](https://github.com/graphsignal/graphsignal)
[![Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


Graphsignal is an AI observability platform. It helps ML engineers and MLOps teams make AI applications run faster and reliably by monitoring and analyzing performance, resources, data and errors. Graphsignal's capabilities enable full visibility into AI applications for any model, data and deployment.

* Monitor and analyze inference latency, throughput and resource utilization.
* Track GPU utilization in the context of inference.
* Get notified about errors and exceptions with full machine learning context.
* Monitor data to detect data issues and silent failures.

[![Dashboards](https://graphsignal.com/external/screencast-dashboards.gif)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).


## Install

Install Graphsignal agent by running:

```
pip install graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal):

```
git clone https://github.com/graphsignal/graphsignal.git
python setup.py install
```


## Configure

Configure Graphsignal agent by specifying your API key directly or via `GRAPHSIGNAL_API_KEY` environment variable.

```python
import graphsignal

graphsignal.configure(api_key='my-api-key', deployment='my-model-prod-v1') 
```

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api-keys) page.

To track deployments, versions and environments separately, specify a `deployment` parameter or environment variable `GRAPHSIGNAL_DEPLOYMENT`.


## Integrate

Use the following examples to integrate Graphsignal agent into your machine learning application. See integration documentation and [API reference](https://graphsignal.com/docs/reference/python-api/) for full reference.

Graphsignal agent is **optimized for production**. All executions are measured, but only a few are recorded to ensure low overhead.


### Monitoring and tracing

Some libraries and frameworks, e.g. OpenAI client, are **auto-instrumented** and traced automatically.

To measure and monitor any other executions, e.g. model inference or inference API calls, wrap the code with [`start_trace`](https://graphsignal.com/docs/reference/python-api/#graphsignalstart_trace) method or use [`trace_function`](https://graphsignal.com/docs/reference/python-api/#graphsignaltrace_function) decorator.

```python
with graphsignal.start_trace(endpoint='predict'):
    pred = model(x)
```

```python
@graphsignal.trace_function
def predict(x):
    return model(x)
```

Other integrations and callbacks are available as well. See [integration documentation](https://graphsignal.com/docs/) for more information.

Enable profiling to additionally record code-level statistics. Profiling is disabled by default due to potential overhead. To enable, provide [`TraceOptions`](https://graphsignal.com/docs/reference/python-api/#graphsignaltraceoptions) object.

```python
with graphsignal.start_trace(
        endpoint='predict', 
        options=graphsignal.TraceOptions(enable_profiling=True)):
    pred = model(x)
```

### Exception tracking

For auto-instrumented libraries, or when using `trace_function` decorator, `start_trace` method with `with` context manager or callbacks, exceptions are **automatically** recorded. For other cases, use [`EndpointTrace.set_exception`](https://graphsignal.com/docs/reference/python-api/#graphsignalendpointtraceset_exception) method.


### Data monitoring

Data is automatically monitored for auto-instrumented libraries. To track data metrics and record data profiles for other cases, [`EndpointTrace.set_data`](https://graphsignal.com/docs/reference/python-api/#graphsignalendpointtraceset_data) method can be used.

```python
with graphsignal.start_trace(endpoint='predict') as trace:
    trace.set_data('input', input_data)
```

The following data types are currently supported: `list`, `dict`, `set`, `tuple`, `str`, `bytes`, `numpy.ndarray`, `tensorflow.Tensor`, `torch.Tensor`.

**No raw data is recorded** by the agent, only statistics such as size, shape or number of missing values.


## Observe

After everything is setup, [log in](https://app.graphsignal.com/) to Graphsignal to monitor and analyze execution performance and monitor for issues.


## Examples

### Model serving

```python
import graphsignal

graphsignal.configure(api_key='my-api-key', deployment='my-model-prod')

...

def predict(x):
    with graphsignal.start_trace(endpoint='predict'):
        return model(x)
```

### Batch job

```python
import graphsignal

graphsignal.configure(api_key='my-api-key', deployment='my-model', tags=dict(job_id='job1'))

...

for x in data:
    with graphsignal.start_trace(endpoint='predict'):
        preds = model(x)
```

More integration examples are available in [`examples`](https://github.com/graphsignal/examples) repo.


## Overhead

Graphsignal agent is very lightweight. While all executions are measured, Graphsignal agent only records certain executions, automatically limiting the overhead.


## Security and Privacy

Graphsignal agent can only open outbound connections to `agent-api.graphsignal.com` and send data, no inbound connections or commands are possible.

No code or data is sent to Graphsignal cloud, only statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://agent-api.graphsignal.com` are allowed.
