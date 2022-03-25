# Graphsignal Profiler

[![License](http://img.shields.io/github/license/graphsignal/graphsignal)](https://github.com/graphsignal/graphsignal/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal?label=version)](https://github.com/graphsignal/graphsignal)
[![SaaS Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


## Overview

Graphsignal is a machine learning profiler. It helps data scientists and ML engineers make model training and inference faster and more efficient. Learn more at [graphsignal.com](https://graphsignal.com).


### Machine Learning Profiling

* **Training and inference profiling** Graphsignal provides operation and kernel level statistics about machine learning runs, e.g. training steps or prediction calls, to enable optimization and tuning.

* **Integration and support** Add a few lines of code to automatically profile TensorFlow and PyTorch jobs and model serving automatically.

* **Any environment** The profiler only needs outbound connection to Graphsignal. No inbound ports or additional software is required, allowing it run everywhere.

* **Data privacy** No code or data is sent to Graphsignal cloud, only run statistics and metadata.


### Profile dashboard

[![Data Analysis](https://graphsignal.com/external/profile-dashboard.png)](https://graphsignal.com)


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

Import the module in your application:

```python
import graphsignal
```

For GPU profiling, make sure the [NVIDIA® CUDA® Profiling Tools Interface](https://developer.nvidia.com/cupti) (CUPTI) is installed by running:

```console
/sbin/ldconfig -p | grep libcupti
```


### 2. Configuration

Configure the profiler by specifying your API key.

```python
graphsignal.configure(api_key='my_api_key', workload_name='job1')
```

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api_keys) page.

`workload_name` identifies the job, application or service that is being profiled.


### 3. Profiling

To profile TensorFlow or PyTorch, add the following code around a code span, e.g. training step/batch or a prediction call. Only some spans will be profiled; the profiler decides which spans to profile for optimal statistics and low overhead. See [profiling API reference](https://graphsignal.com/docs/profiler/api-reference/) for full documentation.
 

Profile TensorFlow:

```python
from graphsignal.profilers.tensorflow import profile_span

span = profile_span()
    # training step, prediction call, etc.
span.stop()
```

Profile TensorFlow using `with` context manager:

```python
from graphsignal.profilers.tensorflow import profile_span

with graphsignal.profile_span() as span:
    # training step, prediction call, etc.
```

Profile Keras training or inference using a callback:

```python
from graphsignal.profilers.keras import GraphsignalCallback

model.fit(..., callbacks=[GraphsignalCallback()])
# or model.predict(..., callbacks=[GraphsignalCallback()])
```

Profile PyTorch:

```python
from graphsignal.profilers.pytorch import profile_span

span = profile_span()
    # training step, prediction call, etc.
span.stop()
```

Profile PyTorch using `with` context manager:

```python
from graphsignal.profilers.pytorch import profile_span

with profile_span() as span:
    # training step, prediction call, etc.
```

Profile Hugging Face training using a callback:

```python
from graphsignal.profilers.huggingface import GraphsignalPTCallback
# or GraphsignalTFCallback for TensorFlow

trainer = Trainer(..., callbacks=[GraphsignalPTCallback()])
# or trainer.add_callback(GraphsignalPTCallback())
```

Optionally record metadata in the profile:

```python
span.add_metadata('key1', 'value1')
```


### 4. Dashboards

After profiling is setup, [sign in](https://app.graphsignal.com/signin) to Graphsignal to analyze recorded profiles.


## Examples

### Model training

```python
import torch

import graphsignal
from graphsignal.profilers.pytorch import profile_span

graphsignal.configure(api_key='my_key', workload_name='training_example')

x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(10):
    with profile_span():
        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Model serving

```python
from tensorflow import keras
import json
from flask import Flask, request

import graphsignal
from graphsignal.profilers.tensorflow import profile_span

graphsignal.configure(api_key='my_key', workload_name='fraud_detection_prod')

model = keras.models.load_model('fraud_model.h5')
app = Flask(__name__)

@app.route('/predict_fraud', methods = ['POST'])
def predict_digit():
    input_data = request.get_json()

    with profile_span():
      output_data = model.predict([input_data])

    return json.dumps(output_data.tolist())

app.run(port=8090)
```


## Overhead

Although profiling may add some overhead to applications, Graphsignal Profiler only profiles certain spans, e.g. training batches or prediction calls, automatically limiting the overhead.


## Security and Privacy

Graphsignal Profiler can only open outbound connections to `profile-api.graphsignal.com` and send data, no inbound connections or commands are possible. 

No code or data is sent to Graphsignal cloud, only run statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://profile-api.graphsignal.com` are allowed.
