# Graphsignal: Inference Profiler

[![License](http://img.shields.io/github/license/graphsignal/graphsignal)](https://github.com/graphsignal/graphsignal/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal?label=version)](https://github.com/graphsignal/graphsignal)
[![Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


Graphsignal is a machine learning inference profiler. It helps data scientists and ML engineers make model inference faster and more efficient. It is built for real-world use cases and allows ML practitioners to:

* Optimize inference by benchmarking latency and throughput, analyzing execution trace, operation-level statistics and compute utilization.
* Start profiling scripts and notebooks automatically by adding a few lines of code.
* Use the profiler in local, remote or cloud environment without installing any additional software or opening inbound ports.
* Keep data private; no code or data is sent to Graphsignal cloud, only run statistics and metadata.

[![Dashboards](https://graphsignal.com/external/screencast-dashboards.gif)](https://graphsignal.com/)

Learn more at [graphsignal.com](https://graphsignal.com).

## Documentation

See full documentation at [graphsignal.com/docs](https://graphsignal.com/docs/).


## Getting Started

### 1. Installation

Install the profiler by running:

```
pip install graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal):

```
git clone https://github.com/graphsignal/graphsignal.git
python setup.py install
```


### 2. Configuration

Configure the profiler by specifying your API key and workload name directly or via environment variables.

```python
import graphsignal

graphsignal.configure(api_key='my_api_key', workload_name='job1')
```

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api-keys) page.

`workload_name` identifies the job, application or service that is being profiled.

One workload can be run multiple times, e.g. to benchmark different parameters. To tag each run, use `graphsignal.add_tag('mytag')`.

In case of multiple subsequent runs/experiments executed within a single script or notebook, call `graphsignal.end_run()` to end current run, upload it and initialize a new one.

Graphsignal has a built-in support for distributed inference. See [Distributed Workloads](https://graphsignal.com/docs/profiler/distributed-workloads/) section for more information.

### 3. Profiling

Use the following minimal examples to integrate Graphsignal into your machine learning script. See integration documentation and  [profiling API reference](https://graphsignal.com/docs/profiler/api-reference/) for full reference.

When `profile_inference` method is used repeatedly, all inferences will be measured, but only a few will be profiled to ensure low overhead.


#### [TensorFlow](https://graphsignal.com/docs/integrations/tensorflow/)

```python
from graphsignal.profilers.tensorflow import profile_inference

with profile_inference():
    # single or batch prediction
```

#### [Keras](https://graphsignal.com/docs/integrations/keras/)

```python
from graphsignal.profilers.keras import GraphsignalCallback

model.predict(..., callbacks=[GraphsignalCallback()])
# or model.evaluate(..., callbacks=[GraphsignalCallback()])
```

#### [PyTorch](https://graphsignal.com/docs/integrations/pytorch/)

```python
from graphsignal.profilers.pytorch import profile_inference

with profile_inference():
    # single or batch prediction
```

#### [PyTorch Lightning](https://graphsignal.com/docs/integrations/pytorch-lightning/)

```python
from graphsignal.profilers.pytorch_lightning import GraphsignalCallback

trainer = Trainer(..., callbacks=[GraphsignalCallback()])
trainer.predict() # or trainer.validate() or trainer.test()
```

#### [Hugging Face](https://graphsignal.com/docs/integrations/hugging-face/)

```python
from transformers import pipeline
from graphsignal.profilers.pytorch import profile_inference
# or from graphsignal.profilers.tensorflow import profile_inference

generator = pipeline(task="text-generation")

with profile_inference():
    output = generator('some text')
```

#### [JAX](https://graphsignal.com/docs/integrations/jax/)

```python
from graphsignal.profilers.jax import profile_inference

with profile_inference():
    # single or batch prediction
```

#### [ONNX Runtime](https://graphsignal.com/docs/integrations/onnx-runtime/)

```python
import onnxruntime
from graphsignal.profilers.onnxruntime import initialize_profiler, profile_inference

sess_options = onnxruntime.SessionOptions()
initialize_profiler(sess_options)

session = onnxruntime.InferenceSession('my_model_path', sess_options)
with profile_inference(session):
    session.run(...)
```

#### [Other frameworks](https://graphsignal.com/docs/integrations/other-frameworks/)

```python
from graphsignal.profilers.generic import profile_inference

with profile_inference():
    # single or batch prediction
```

### 4. Logging

Logging parameters and metrics enables benchmarking inference latency and throughput against logged values. For example, logging evaluation accuracy in optimization runs is useful for ensuring that the accuracy is not affected by inference optimizations or to identify the best tradeoff.

```python
graphsignal.log_param('my_param', 'val')
```

```python
graphsignal.log_metric('my_metric', 0.9)
```

Parameters and metrics can also be passed via environment variables. See [profiling API reference](/docs/profiler/api-reference/#graphsignallog_param) for full documentation.


### 5. Dashboards

After profiling is setup, [open](https://app.graphsignal.com/) Graphsignal to analyze recorded profiles.


## Example

```python
# 1. Import Graphsignal modules
import graphsignal
from graphsignal.profilers.pytorch import profile_inference

# 2. Configure
graphsignal.configure(api_key='my_key', workload_name='my_gpu_inference')

....

# 3. Use profile method to measure and profile single or batch predictions
for x in data:
    with profile_inference():
        preds = model(x)
```

More integration examples are available in [`examples`](https://github.com/graphsignal/examples) repo.


## Overhead

Although profiling may add some overhead to applications, Graphsignal Profiler only profiles certain inferences, automatically limiting the overhead.


## Security and Privacy

Graphsignal Profiler can only open outbound connections to `profile-api.graphsignal.com` and send data, no inbound connections or commands are possible. 

No code or data is sent to Graphsignal cloud, only run statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://profile-api.graphsignal.com` are allowed.

For GPU profiling, if `libcupti` library is failing to load, make sure the [NVIDIA® CUDA® Profiling Tools Interface](https://developer.nvidia.com/cupti) (CUPTI) is installed by running:

```console
/sbin/ldconfig -p | grep libcupti
```
