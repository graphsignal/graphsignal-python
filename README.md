# Graphsignal Logger

[![License](http://img.shields.io/github/license/graphsignal/graphsignal)](https://github.com/graphsignal/graphsignal/blob/main/LICENSE)
[![Version](https://img.shields.io/github/v/tag/graphsignal/graphsignal?label=version)](https://github.com/graphsignal/graphsignal)
[![Downloads](https://pepy.tech/badge/graphsignal)](https://pepy.tech/project/graphsignal)
[![SaaS Status](https://img.shields.io/uptimerobot/status/m787882560-d6b932eb0068e8e4ade7f40c?label=SaaS%20status)](https://stats.uptimerobot.com/gMBNpCqqqJ)


## Overview

Graphsignal is a machine learning model monitoring platform. It helps ML engineers and data scientists address data issues and analyze model performance in production. Learn more at [graphsignal.com](https://graphsignal.com).


### Model Monitoring

* **Data monitoring.** Monitor offline and online predictions for *data validity and anomalies*, *data drift, model drift*, *exceptions*, and more.
* **Model performance monitoring.** Monitor model performance for *binary*, *categorical* and *numeric* models and data segments.
* **Automatic issue detection.** Graphsignal automatically detects and notifies on issues with data and models, no need to manually setup and maintain complex rules.
* **Model framework and deployment agnostic.** Monitor models serving *online*, in streaming apps, accessed via APIs or *offline*, running batch predictions.
* **Any scale and data size.** Graphsignal logger *only sends data statistics* allowing it to scale with your application and data.
* **Data privacy.** No raw data is sent to Graphsignal cloud, only data statistics and metadata.

### Dashboards and Alerting

#### Data Analysis
[![Data Analysis](https://graphsignal.com/external/readme-data-analysis.png)](https://graphsignal.com)

#### Model Performance
[![Model Performance](https://graphsignal.com/external/readme-model-performance.png)](https://graphsignal.com)

#### Automatic Alerts
[![Automatic Alerts](https://graphsignal.com/external/readme-alert-timeline.png)](https://graphsignal.com)


## Documentation

See full documentation at [graphsignal.com/docs](https://graphsignal.com/docs/).


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

To get an API key, sign up for a free account at [graphsignal.com](https://graphsignal.com). The key can then be found in your account's [Settings / API Keys](https://app.graphsignal.com/settings/api_keys) page.


### 3. Logging session

Get logging session for a deployed model identified by deployment name. Multiple sessions can be used in parallel in case of multi-model scrips or servers.

```python
sess = graphsignal.session(deployment_name='model1_prod')
```

`with` statement can be used for offline/batch logging to make sure all remaining buffered data is automatically uploaded at the end.

```python
with graphsignal.session(deployment_name='model1_prod') as sess:
```

Log any model metadata such as model version or deployment information.

```python
sess.log_metadata('key1', 'val1')
```


### 4. Prediction Logging

Log single or batch model prediction/inference data to monitor data schema changes and drift. Computed data statistics are uploaded at certain intervals and on process exit.

Log single prediction.

```python
sess.log_prediction(
  features={'feat1': 1.2, 'feat2': 'XX'},
  output=True)
```

Log prediction batch. Pass prediction data using `list`, `dict`, `numpy.ndarray` or `pandas.DataFrame`.

```python
sess.log_prediction_batch(
  features=[[1.2, 70], [3.5, 40]], 
  outputs=[[0.5], [0.75]])
```

See [logging API reference](https://graphsignal.com/docs/python-logger/api-reference/) for full documentation.


### 5. Evaluation Logging

Log prediction and ground truth label to evaluate model performance. Because ground truth is usually available at a later point, **evaluation logging is independent from prediction logging**. Prediction logging is **not** required for model performance monitoring and visualization.

```python
sess.log_evaluation(
  prediction=False,
  label=True)
```

See [logging API reference](https://graphsignal.com/docs/python-logger/api-reference/) for full documentation.

Model output type is inferred from label and prediction types. Model performance metrics such as accuracy, F1-score, MSE, etc. are computed based on the model output type.

To additionally visualize and monitor performance metrics for various data segments, a `segments` list can be provided.

```python
sess.log_evaluation(
  prediction=False,
  label=True, 
  segments=['age_group_2', 'country_US'])
```

Log evaluation batch by passing predictions and labels using `list` or `numpy.ndarray`.

```python
sess.log_evaluation_batch(
  predictions=[True, True, False], 
  labels=[False, True, False],
  segments=[['state_CA'], ['state_CA'], ['state_MA']])
```


### 6. Dashboards and Alerting

After logging is setup, [sign in](https://app.graphsignal.com/signin) to Graphsignal to check out various dashboards and set up alerting for automatically detected issues.


## Examples

### Online prediction logging

```python
from tensorflow import keras
import json
from flask import Flask, request

import graphsignal
graphsignal.configure(api_key='my_key')

sess = graphsignal.session(deployment_name='fraud_detection_prod')
sess.log_metadata('model version', '1.0')

model = keras.models.load_model('fraud_model.h5')
app = Flask(__name__)

@app.route('/predict_fraud', methods = ['POST'])
def predict_digit():
    features = request.get_json()

    # feature extraction code here
    
    output_data = model.predict([input_data])

    sess.log_prediction(
      features=features, 
      prediction=output_data[0])

    return json.dumps(output_data.tolist())

app.run(port=8090)
```

### Offline evaluation logging

```python
import graphsignal
graphsignal.configure(api_key='my_key')

# load predictions and labels here
last_hour_predictions=...
last_hour_labels=...

with graphsignal.session(deployment_name='my_risk_model_prod') as sess:
  sess.log_evaluation_batch(
    predictions=last_hour_predictions,
    labels=last_hour_labels)
```

## Performance

Graphsignal logger uses streaming algorithms for computing data statistics to ensure production-level performance and memory usage. Data statistics are computed for time windows and sent to Graphsignal by the **background thread**.

Since only data statistics is sent to our servers, there is **no limitation** on logged data size.


## Security and Privacy

Graphsignal logger can only open outbound connections to `log-api.graphsignal.com` and send data, no inbound connections or commands are possible. 

No raw data is sent to Graphsignal cloud, only data statistics and metadata.


## Troubleshooting

To enable debug logging, add `debug_mode=True` to `configure()`. If the debug log doesn't give you any hints on how to fix a problem, please report it to our support team via your account.

In case of connection issues, please make sure outgoing connections to `https://log-api.graphsignal.com` are allowed.
