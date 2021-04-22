# Graphsignal Logger


## Overview

Graphsignal is an observability platform for monitoring and troubleshooting production machine learning applications. It helps ML engineers, MLOps teams and data scientists to quickly address issues with data and models as well as proactively analyze model performance and availability. Learn more at [graphsignal.ai](https://graphsignal.ai).


## AI Observability

* **Model monitoring.** Monitor offline and online predictions for *data validity and anomalies*, *sudden data drift and concept drift*, *prediction latency*, *exceptions*, *system metrics* and more.
* **Automatic issue detection.** Get notified on data, model and code issues via email, Slack and other channels.
* **Root cause analysis.** Analyse prediction outliers and issue-related samples for faster problem root cause identification.
* **Model framework and deployment agnostic.** Monitor models serving *online*, in streaming apps, accessed via APIs or *offline*, running batch predictions.
* **Any scale and data size.** Graphsignal logger *only sends data statistics and samples* allowing it to scale with your application and data.
* **Team access.** Easily add team members to your account, as many as you need.


## Documentation

See full documentation at [graphsignal.ai/docs](https://graphsignal.ai/docs/).


## Getting Started

### Installation

Install the Python logger by running

```
pip install graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal).

```
git clone https://github.com/graphsignal/graphsignal.git
python setup.py install
```

And import the package in your application

```python
import graphsignal
```

### Configuration

Configure the logger by specifying the API key.

```python
graphsignal.configure(api_key='my_api_key')
```

To get an API key, sign up for a free trial account at [graphsignal.ai](https://graphsignal.ai). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.ai/settings/api_keys) page.


### Logging session

Get logging session for a deployed model identified by model name and deployment name. Multiple sessions can be used in parallel in case of multi-model scrips or servers.

```python
sess = graphsignal.session(model_name='my_model', deployment_name='production')
```

If a model is versioned you can set the version as a model attribute.

Set model attributes.

```python
sess.set_attribute('my attribute', 'value123')
```

Some system attributes, such as Python version and OS are added automatically.


### Prediction Logging

Log single or batch model prediction/inference data. Pass prediction data according to [supported data formats](https://graphsignal.ai/docs/python-logger/supported-data-formats) using `list`, `dict`, `pandas.DataFrame` or `numpy.ndarray`.

Computed data statistics such as feature and class distributions are uploaded at certain intervals and on process exit. Additionally, random and outlier prediction instances may be uploaded.


```python
# Examples of input features and output classes.
x = pandas.DataFrame(data=[[0.1, 'A'], [0.2, 'B']], columns=['feature1', 'feature2'])
y = numpy.asarray([[0.2, 0.8], [0.1, 0.9]])

sess.log_prediction(input_data=x, output_data=y)
```

Track metrics. The last set value is used when metric is aggregated.

```python
sess.log_metric('my_metric', 1.0)
```

Log any prediction-related event or exception.

```python
sess.log_event(description='My event', attributes={'my_attr': '123'})
```

Measure prediction latency and record any exceptions.

```python
with sess.measure_latency()
    my_model.predict(X)
```

See [prediction logging API reference](https://graphsignal.ai/docs/python-logger/api-reference/) for full documentation.


### Example

```python
import numpy as np
from tensorflow import keras
import graphsignal

# Configure Graphsignal logger
graphsignal.configure(api_key='my_api_key')

# Get logging session for the model
sess = graphsignal.session(model_name='my_mnist_model', deployment_name='production')


model = keras.models.load_model('mnist_model.h5')

(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

# Measure predict call latency
with sess.measure_latency()
    output = model.predict(x_test)

# See supported data formats description at 
# https://graphsignal.ai/docs/python-logger/supported-data-formats
sess.log_prediction(output_data=output)

# Report a metric
sess.log_metric('my_metric', 1.2)
```

See more [examples](https://github.com/graphsignal/graphsignal/tree/main/examples).


## Performance

When logging predictions, the data is windowed and only when certain time interval or window size conditions are met, data statistics are computed and sent along with a few sample and outlier data instances by the **background thread**.

Since only data statistics are sent to our servers, there is **no limitation** on logged data size and it doesn't have a direct effect on logging performance.


## Security and Privacy

Graphsignal logger can only open outbound connections to `log-api.graphsignal.ai` and send data, no inbound connections or commands are possible. 

Please make sure to exclude or anonymize any personally identifiable information (PII) when logging model data and events. If necessary, sending prediction instances can be disabled by setting `log_instances` option to `False` when configuring the logger. This, however, can impair root cause analysis capabilities.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `log-api.graphsignal.ai` are allowed.
