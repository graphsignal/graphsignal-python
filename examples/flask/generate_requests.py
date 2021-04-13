import time
import threading
import logging
import requests
import random
import numpy as np
from tensorflow import keras

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)


def send_prediction_requests():
    while True:
        try:
            idx = np.random.randint(0, x_test.shape[0], size=(1,))
            instance = x_test[idx,:]
            logger.debug('Sending request with input: %s', instance.tolist())
            res = requests.post('http://localhost:8090/predict_digit', json=instance.tolist())
            logger.debug('Prediction output: %s', res.json())
        except:
            logger.error('Prediction request failed', exc_info=True)

        time.sleep(5)

t = threading.Thread(target=send_prediction_requests)
t.start()
