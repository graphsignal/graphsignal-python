import logging
import numpy as np
from tensorflow import keras
import json
from flask import Flask
from flask import request

import graphsignal

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Configure Graphsignal
graphsignal.configure(api_key='testkey', debug_mode=True)

# Get logging session for the model
sess = graphsignal.session(deployment_name='mnist_v1_prod')

model = keras.models.load_model('mnist_model.h5')

app = Flask(__name__)

@app.route('/predict_digit', methods = ['POST'])
def predict_digit():
    input = request.get_json()
    
    logger.debug('Received prediction request: %s', input)

    output = model.predict(input)

    # Log prediction
    sess.log_prediction(output_data=output)

    # Increment global counter
    sess.log_metric('my_metric', 1)

    logger.debug('Returning prediction output: %s', input)
    return json.dumps(output.tolist())

app.run(port=8090)