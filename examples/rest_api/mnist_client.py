import time
import threading
import logging
import requests
import random
import numpy as np
from tensorflow import keras

import graphsignal

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Configure Graphsignal
graphsignal.configure(api_key='testkey')

# Get logging session for the model
sess = graphsignal.session(deployment_name='mnist_prod')


(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

idx = np.random.randint(0, x_test.shape[0], size=(1,))
instance = x_test[idx,:].tolist()
logger.debug('Sending request with input: %s', instance)
res = requests.post('http://localhost:8090/predict_digit', json=instance)
output = res.json()
logger.debug('Prediction output: %s', output)

# Log prediction
sess.log_prediction(output=output[0])
