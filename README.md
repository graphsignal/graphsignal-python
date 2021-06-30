# Graphsignal Logger


## Overview

Graphsignal is a machine learning model monitoring platform. It helps ML engineers, MLOps teams and data scientists to quickly address issues with data and models as well as proactively analyze model performance and availability. Learn more at [graphsignal.ai](https://graphsignal.ai).

![Model Dashboard](readme-screenshot.png)


## Model Monitoring

* **Data monitoring.** Monitor offline and online predictions for *data validity and anomalies*, *data drift, model drift*, and more.
* **Automatic issue detection.** Graphsignal automatically detects and notifies on issues with data and models, no need to manually setup and maintain complex rules.
* **Model framework and deployment agnostic.** Monitor models serving *online*, in streaming apps, accessed via APIs or *offline*, running batch predictions.
* **Any scale and data size.** Graphsignal logger *only sends data statistics* allowing it to scale with your application and data.
* **Data privacy.** No raw data is sent to Graphsignal cloud, only data statistics and metadata.
* **Team access.** Easily add team members to your account, as many as you need.


## Documentation

See full documentation at [graphsignal.ai/docs](https://graphsignal.ai/docs/).


## Getting Started

### 1. Installation

Install the Python logger by running

```
pip install graphsignal
```

Or clone and install the [GitHub repository](https://github.com/graphsignal/graphsignal).

```
git clone https://github.com/graphsignal/graphsignal.git
python setup.py install
```

Import the package in your application

```python
import graphsignal
```

### 2. Configuration

Configure the logger by specifying your API key.

```python
graphsignal.configure(api_key='my_api_key')
```

To get an API key, sign up for a free account at [graphsignal.ai](https://graphsignal.ai). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.ai/settings/api_keys) page.


### 3. Logging session

Get logging session for a deployed model identified by deployment name. Multiple sessions can be used in parallel in case of multi-model scrips or servers.

```python
sess = graphsignal.session(deployment_name='model1_prod')
```

You can also use `with` statement, which will also transparently catch exceptions and report as error events:

```python
with graphsignal.session(deployment_name='model1_prod') as sess:
    # prediction code here
```

If a model is versioned you can set the version as a model tag.

Set model tags. Tags can be updated dynamically, for example, when a new model version is dynamically loaded.

```python
sess.set_tag('version', '1.0')
```


### 4. Prediction Logging

Log single or batch model prediction/inference data. Pass prediction data according to [supported data formats](https://graphsignal.ai/docs/python-logger/supported-data-formats) using `list`, `dict`, `pandas.DataFrame` or `numpy.ndarray`.

Computed data statistics are uploaded at certain intervals and on process exit.

```python
# Examples of input features and output classes.
x = pandas.DataFrame(data=[[0.1, 'A'], [0.2, 'B']], columns=['feature1', 'feature2'])
y = numpy.asarray([[0.2, 0.8], [0.1, 0.9]])

sess.log_prediction(input_data=x, output_data=y)
```

Log any prediction-related event and error.

```python
sess.log_event(description='Some event', attributes={'some_attr': '123'}, is_error=True)
```

See [prediction logging API reference](https://graphsignal.ai/docs/python-logger/api-reference/) for full documentation.


### 5. Dashboards and Alerting

After prediction logging is setup, [sign in](https://app.graphsignal.ai/signin) to Graphsignal to check out various dashboards and set up alerts for automatically detected issues.


## Example

```python
import numpy as np
from tensorflow import keras
import graphsignal

# Configure Graphsignal logger
graphsignal.configure(api_key='my_api_key')

# Get logging session for the model
sess = graphsignal.session(deployment_name='mnist_prod')


model = keras.models.load_model('mnist_model.h5')

(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

output = model.predict(x_test)

# See supported data formats description at 
# https://graphsignal.ai/docs/python-logger/supported-data-formats
sess.log_prediction(output_data=output)
```

See more [examples](https://github.com/graphsignal/graphsignal/tree/main/examples).


## Performance

When logging predictions, the data is windowed and only when certain time interval or window size conditions are met, data statistics are computed and sent by the **background thread**.

Since only data statistics are sent to our servers, there is **no limitation** on logged data size and it doesn't have a direct effect on logging performance.


## Security and Privacy

Graphsignal logger can only open outbound connections to `log-api.graphsignal.ai` and send data, no inbound connections or commands are possible. 

No raw data is sent to Graphsignal cloud, only data statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://log-api.graphsignal.ai` are allowed.
